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


class SpectrogramDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, tp: pd.DataFrame, fp: pd.DataFrame, datadir: Path,
                 waveform_transforms=None, spectrogram_transforms=None,
                 melspectrogram_parameters={},
                 pcen_parameters={},
                 sampling_rate=32000,
                 img_size=224,
                 duration=10,
                 centering=False):
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
        self.centering = centering
        if self.melspectrogram_parameters.get("hop_length") is not None:
            self.hop_length = self.melspectrogram_parameters["hop_length"]
        else:
            self.hop_length = 512

    def __len__(self):
        return len(self.tp)

    def __getitem__(self, idx: int):
        sample = self.tp.loc[idx, :]
        flac_id = sample["recording_id"]

        t_min = sample["t_min"]
        t_max = sample["t_max"]
        species_id = sample["species_id"]

        if not self.centering:
            offset = np.random.choice(np.arange(max(t_max - self.duration, 0), t_min, 0.1))
            offset = min(CLIP_DURATION - self.duration, offset)
        else:
            call_duration = t_max - t_min
            relative_offset = (self.duration - call_duration) / 2
            offset = min(max(0, t_min - relative_offset), CLIP_DURATION - self.duration)
        y, sr = librosa.load(self.datadir / f"{flac_id}.flac",
                             sr=self.sampling_rate,
                             mono=True,
                             offset=offset,
                             duration=self.duration,
                             res_type="kaiser_fast")
        if self.waveform_transforms:
            y = self.waveform_transforms(y).astype(np.float32)

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
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
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        label = np.zeros(N_CLASSES, dtype=np.float32)

        n_points = len(y)
        n_frames = int(n_points / self.hop_length) + 1
        strong_label = np.zeros((n_frames, N_CLASSES), dtype=np.float32)

        label[species_id] = 1.0

        start_index = int(((t_min - offset) * self.sampling_rate) / self.hop_length) + 1
        end_index = int(((t_max - offset) * self.sampling_rate) / self.hop_length) + 1

        strong_label[start_index:end_index, species_id] = 1.0

        return {
            "recording_id": flac_id,
            "image": image,
            "targets": {
                "weak": label,
                "strong": strong_label
            }
        }


class SpectrogramTestDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path,
                 waveform_transforms=None, spectrogram_transforms=None,
                 melspectrogram_parameters={},
                 pcen_parameters={},
                 sampling_rate=32000,
                 img_size=224,
                 duration=10):
        self.df = df
        self.datadir = datadir
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.img_size = img_size

    def __len__(self):
        return len(self.df) * (CLIP_DURATION // self.duration)

    def __getitem__(self, idx_: int):
        n_chunk_per_clip = CLIP_DURATION // self.duration
        idx = idx_ // n_chunk_per_clip
        segment_id = idx_ % n_chunk_per_clip

        sample = self.df.loc[idx, :]
        flac_id = sample["recording_id"]

        offset = segment_id * self.duration
        y, sr = librosa.load(self.datadir / f"{flac_id}.flac",
                             sr=self.sampling_rate,
                             mono=True,
                             offset=offset,
                             duration=self.duration,
                             res_type="kaiser_fast")
        if self.waveform_transforms:
            y = self.waveform_transforms(y).astype(np.float32)

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
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
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        return {
            "recording_id": flac_id,
            "image": image
        }


class MultiLabelSpectrogramDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, tp: pd.DataFrame, fp: pd.DataFrame, datadir: Path,
                 waveform_transforms=None, spectrogram_transforms=None,
                 melspectrogram_parameters={},
                 pcen_parameters={},
                 sampling_rate=32000,
                 img_size=224,
                 duration=10,
                 centering=False):
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
        self.centering = centering
        if self.melspectrogram_parameters.get("hop_length") is not None:
            self.hop_length = self.melspectrogram_parameters["hop_length"]
        else:
            self.hop_length = 512

    def __len__(self):
        return len(self.tp)

    def __getitem__(self, idx: int):
        sample = self.tp.loc[idx, :]
        flac_id = sample["recording_id"]

        t_min = sample["t_min"]
        t_max = sample["t_max"]

        if not self.centering:
            offset = np.random.choice(np.arange(max(t_max - self.duration, 0), t_min, 0.1))
            offset = min(CLIP_DURATION - self.duration, offset)
        else:
            call_duration = t_max - t_min
            relative_offset = (self.duration - call_duration) / 2
            offset = min(max(0, t_min - relative_offset), CLIP_DURATION - self.duration)
        y, sr = librosa.load(self.datadir / f"{flac_id}.flac",
                             sr=self.sampling_rate,
                             mono=True,
                             offset=offset,
                             duration=self.duration,
                             res_type="kaiser_fast")
        if self.waveform_transforms:
            y = self.waveform_transforms(y).astype(np.float32)

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
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
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        import pdb
        pdb.set_trace()
        tail = offset + self.duration
        query_string = f"recording_id == '{flac_id}' & ("
        query_string += f"(t_max > {offset} & t_min < {offset}) | "
        query_string += f"(t_min < {tail} & t_max > {tail}))"
        all_tp_events = self.tp.query(query_string)

        label = np.zeros(N_CLASSES, dtype=np.float32)

        n_points = len(y)
        n_frames = int(n_points / self.hop_length) + 1
        strong_label = np.zeros((n_frames, N_CLASSES), dtype=np.float32)

        for species_id in all_tp_events["species_id"].unique():
            label[int(species_id)] = 1.0

        for _, row in all_tp_events.iterrows():
            t_min = row.t_min
            t_max = row.t_max
            species_id = row.species_id

            start_index = int(((t_min - offset) * self.sampling_rate) / self.hop_length) + 1
            end_index = int(((t_max - offset) * self.sampling_rate) / self.hop_length) + 1

            strong_label[start_index:end_index, species_id] = 1.0

        return {
            "recording_id": flac_id,
            "image": image,
            "targets": {
                "weak": label,
                "strong": strong_label
            }
        }
