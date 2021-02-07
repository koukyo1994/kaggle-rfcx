import cv2
import librosa
import numpy as np
import pandas as pd
import torch.utils.data as torchdata

from pathlib import Path

from .constants import N_CLASSES, CLIP_DURATION


def normalize_melspec(X: np.ndarray):
    eps = 1e-6
    mean = X.mean()
    X = X - mean
    std = X.std()
    Xstd = X / (std + eps)
    norm_min, norm_max = Xstd.min(), Xstd.max()
    if (norm_max - norm_min) > eps:
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


class RandomSamplingRateAndDurationSpectrogramDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, tp: pd.DataFrame, fp: pd.DataFrame, datadir: Path,
                 waveform_transforms=None, spectrogram_transforms=None,
                 melspectrogram_parameters={},
                 pcen_parameters={},
                 sampling_rate=32000,
                 img_size=[960, 240],
                 duration=10,
                 duration_randomness_ratio=0.1,
                 sampling_rate_randomness_ratio=0.1):
        if not isinstance(img_size, list):
            raise ValueError("`img_size must be a list`")
        unique_recording_id = df.recording_id.unique().tolist()
        unique_tp_recordin_id = tp.recording_id.unique().tolist()
        intersection = set(unique_recording_id).intersection(unique_tp_recordin_id)
        self.df = df[df.recording_id.isin(intersection)].reset_index(drop=True)
        self.tp = tp[tp.recording_id.isin(intersection)].reset_index(drop=True)
        self.fp = fp  # unused
        self.datadir = datadir
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters
        self.sampling_rate = sampling_rate
        self.img_size = img_size
        self.duration = duration
        self.duration_randomness_ratio = duration_randomness_ratio
        self.sampling_rate_randomness_ratio = sampling_rate_randomness_ratio
        if len(list(datadir.glob("*.flac"))) == 0:
            self.suffix = ".wav"
        else:
            self.suffix = ".flac"

    def __len__(self):
        return len(self.tp)

    def __getitem__(self, idx: int):
        sample = self.tp.loc[idx, :]
        index = sample["index"]
        flac_id = sample["recording_id"]

        t_min = sample["t_min"]
        t_max = sample["t_max"]

        duration_ratio = np.random.choice(np.arange(1.0 - self.duration_randomness_ratio, 1.0 + self.duration_randomness_ratio, 0.01))
        sampling_rate_ratio = np.random.choice(np.arange(1.0 - self.sampling_rate_randomness_ratio, 1.0 + self.sampling_rate_randomness_ratio, 0.01))

        duration = self.duration * duration_ratio
        sr = self.sampling_rate * sampling_rate_ratio

        call_duration = t_max - t_min
        if call_duration > duration:
            offset = np.random.choice(np.arange(max(t_min - call_duration / 2, 0), t_min + call_duration / 2, 0.1))
            offset = min(CLIP_DURATION - duration, offset)
        else:
            offset = np.random.choice(np.arange(max(t_max - duration, 0), t_min, 0.1))
            offset = min(CLIP_DURATION - duration, offset)

        y, sr = librosa.load(self.datadir / f"{flac_id}{self.suffix}",
                             sr=sr,
                             mono=True,
                             offset=offset,
                             duration=duration,
                             res_type="kaiser_fast")
        if self.waveform_transforms:
            y = self.waveform_transforms(y).astype(np.float32)

        length = len(y)
        width_before_resize = int(self.img_size[0] * self.melspectrogram_parameters["n_mels"] / self.img_size[1])
        hop_length = int(length / (width_before_resize - 1))
        self.melspectrogram_parameters["hop_length"] = hop_length

        melspec = librosa.feature.melspectrogram(y, sr=self.sampling_rate, **self.melspectrogram_parameters)
        pcen = librosa.pcen(melspec, sr=sr, **self.pcen_parameters)
        clean_mel = librosa.power_to_db(melspec ** 1.5)
        melspec = librosa.power_to_db(melspec)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(image=melspec)["image"]
            pcen = self.spectrogram_transforms(image=pcen)["image"]
            clean_mel = self.spectrogram_transforms(image=clean_mel)["image"]
        else:
            pass

        norm_melspec = normalize_melspec(melspec)
        norm_pcen = normalize_melspec(pcen)
        norm_clean_mel = normalize_melspec(clean_mel)
        image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)

        height, width, _ = image.shape
        image = cv2.resize(image, tuple(self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        tail = offset + duration
        query_string = f"recording_id == '{flac_id}' & "
        query_string += f"t_min < {tail} & t_max > {offset}"
        all_tp_events = self.tp.query(query_string)

        label = np.zeros(N_CLASSES, dtype=np.float32)

        n_frames = image.shape[2]
        seconds_per_frame = duration / n_frames
        strong_label = np.zeros((n_frames, N_CLASSES), dtype=np.float32)

        for species_id in all_tp_events["species_id"].unique():
            label[int(species_id)] = 1.0

        for _, row in all_tp_events.iterrows():
            t_min = row.t_min
            t_max = row.t_max
            species_id = row.species_id

            start_index = int((t_min - offset) / seconds_per_frame)
            end_index = int((t_max - offset) / seconds_per_frame)

            strong_label[start_index:end_index, species_id] = 1.0

        return {
            "recording_id": flac_id,
            "image": image,
            "targets": {
                "weak": label,
                "strong": strong_label
            },
            "index": index
        }
