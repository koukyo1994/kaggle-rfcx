import cv2
import librosa
import numpy as np
import pandas as pd
import torch.utils.data as torchdata

from pathlib import Path

from .constants import CLIP_DURATION


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


class SpectrogramTTADataset(torchdata.Dataset):
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
        return len(self.df) * ((CLIP_DURATION // self.duration) * 2 - 1)

    def __getitem__(self, idx: int):
        n_chunk_per_clip = CLIP_DURATION // self.duration
        clip_idx = idx // (n_chunk_per_clip * 2 - 1)
        segment_id = idx % (n_chunk_per_clip * 2 - 1)

        sample = self.df.loc[clip_idx, :]
        flac_id = sample["recording_id"]

        if segment_id < n_chunk_per_clip:
            offset = segment_id * self.duration
        else:
            offset = (segment_id - n_chunk_per_clip) * self.duration + self.duration / 2
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
