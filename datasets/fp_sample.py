import cv2
import librosa
import numpy as np
import pandas as pd
import torch.utils.data as torchdata

from pathlib import Path

from .constants import N_CLASSES, CLIP_DURATION
from .spectrogram import normalize_melspec


class SampleFPSpectrogramDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, tp: pd.DataFrame, fp: pd.DataFrame, datadir: Path,
                 waveform_transforms=None, spectrogram_transforms=None,
                 melspectrogram_parameters={},
                 pcen_parameters={},
                 sampling_rate=32000,
                 img_size=224,
                 duration=10,
                 centering=False):
        unique_recording_id = df.recording_id.unique().tolist()
        unique_tp_recording_id = tp.recording_id.unique().tolist()
        unique_fp_recording_id = fp.recording_id.unique().tolist()
        intersection_tp = set(unique_recording_id).intersection(unique_tp_recording_id)
        intersection_fp = set(unique_recording_id).intersection(unique_fp_recording_id)
        self.tp = tp[tp.recording_id.isin(intersection_tp)].reset_index(drop=True)
        self.fp = fp[fp.recording_id.isin(intersection_fp)].reset_index(drop=True)
        self.datadir = datadir
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters
        self.sampling_rate = sampling_rate
        self.img_size = img_size
        self.duration = duration
        self.centering = centering

    def __len__(self):
        return len(self.tp) * 2

    def __getitem__(self, idx: int):
        sample_tp = idx % 2 == 0
        if sample_tp:
            idx = idx // 2
            sample = self.tp.loc[idx, :]
        else:
            sample_again = True
            while sample_again:
                sample = self.fp.sample(1).reset_index(drop=True).loc[0]
                flac_id = sample["recording_id"]
                if len(self.tp.query(f"recording_id == '{flac_id}'")) == 0:
                    break

        flac_id = sample["recording_id"]
        index = sample["index"]

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

        label = np.zeros(N_CLASSES, dtype=np.float32)
        n_frames = image.shape[2]
        seconds_per_frame = self.duration / n_frames
        strong_label = np.zeros((n_frames, N_CLASSES), dtype=np.float32)

        tail = offset + self.duration
        query_string = f"recording_id == '{flac_id}' & "
        query_string += f"t_min < {tail} & t_max > {offset}"

        if sample_tp:
            all_tp_events = self.tp.query(query_string)

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
