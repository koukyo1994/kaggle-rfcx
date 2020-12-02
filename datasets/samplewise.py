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


class SampleWiseSpectrogramDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, tp: pd.DataFrame, fp: pd.DataFrame, datadir: Path,
                 waveform_transforms=None, spectrogram_transforms=None,
                 melspectrogram_parameters={},
                 pcen_parameters={},
                 sampling_rate=32000,
                 img_size=224,
                 duration=10):
        unique_recording_id = df.recording_id.unique().tolist()
        unique_tp_recordin_id = tp.recording_id.unique().tolist()
        intersection = set(unique_recording_id).intersection(unique_tp_recordin_id)
        self.df = df[df.recording_id.isin(intersection)].reset_index(drop=True)
        self.tp = tp[tp.recording_id.isin(intersection)].reset_index(drop=True)  # unused
        self.fp = fp  # unused
        self.datadir = datadir
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters
        self.sampling_rate = sampling_rate
        self.img_size = img_size
        self.duration = duration

        all_flacs = list(self.datadir.glob("*.flac"))
        if len(all_flacs) > 0:
            self.format = "flac"
        else:
            self.format = "wav"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        flac_id = sample["recording_id"]

        y, sr = librosa.load(self.datadir / f"{flac_id}.{self.format}",
                             sr=self.sampling_rate,
                             mono=True,
                             res_type="kaiser_fast")
        if self.waveform_transforms:
            y = self.waveform_transforms(y).astype(np.float32)

        images = []
        n_images = CLIP_DURATION // self.duration
        for i in range(n_images):
            y_patch = y[i * self.duration * sr:(i + 1) * self.duration * sr]

            melspec = librosa.feature.melspectrogram(y_patch, sr=sr, **self.melspectrogram_parameters)
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

            images.append(image)

        return {
            "recording_id": flac_id,
            "image": np.asarray(images)
        }


class SampleWiseSpectrogramTestDataset(torchdata.Dataset):
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
        self.img_size = img_size
        self.duration = duration

        all_flacs = list(self.datadir.glob("*.flac"))
        if len(all_flacs) > 0:
            self.format = "flac"
        else:
            self.format = "wav"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        flac_id = sample["recording_id"]

        y, sr = librosa.load(self.datadir / f"{flac_id}.{self.format}",
                             sr=self.sampling_rate,
                             mono=True,
                             res_type="kaiser_fast")
        if self.waveform_transforms:
            y = self.waveform_transforms(y).astype(np.float32)

        images = []
        n_images = CLIP_DURATION // self.duration
        for i in range(n_images):
            y_patch = y[i * self.duration * sr:(i + 1) * self.duration * sr]

            melspec = librosa.feature.melspectrogram(y_patch, sr=sr, **self.melspectrogram_parameters)
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

            images.append(image)

        return {
            "recording_id": flac_id,
            "image": np.asarray(images)
        }
