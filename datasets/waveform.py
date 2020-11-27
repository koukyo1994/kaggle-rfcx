import librosa
import numpy as np
import pandas as pd
import torch.utils.data as torchdata

from pathlib import Path

from .constants import N_CLASSES, CLIP_DURATION


class WaveformDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, tp: pd.DataFrame, fp: pd.DataFrame, datadir: Path,
                 transforms=None, sampling_rate=32000, duration=10, hop_length=320):
        unique_recording_id = df.recording_id.unique().tolist()
        unique_tp_recordin_id = tp.recording_id.unique().tolist()
        intersection = set(unique_recording_id).intersection(unique_tp_recordin_id)
        self.df = df[df.recording_id.isin(intersection)].reset_index(drop=True)
        self.tp = tp[tp.recording_id.isin(intersection)].reset_index(drop=True)
        self.fp = fp  # unused
        self.datadir = datadir
        self.transforms = transforms
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.hop_length = hop_length

    def __len__(self):
        return len(self.tp)

    def __getitem__(self, idx: int):
        sample = self.tp.loc[idx, :]
        flac_id = sample["recording_id"]

        t_min = sample["t_min"]
        t_max = sample["t_max"]
        species_id = sample["species_id"]

        offset = np.random.choice(np.arange(max(t_max - self.duration, 0), t_min, 0.1))
        offset = min(CLIP_DURATION - self.duration, offset)
        y, sr = librosa.load(self.datadir / f"{flac_id}.flac",
                             sr=self.sampling_rate,
                             mono=True,
                             offset=offset,
                             duration=self.duration,
                             res_type="kaiser_fast")
        if self.transforms:
            y = self.transforms(y).astype(np.float32)

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
            "waveform": y,
            "targets": {
                "weak": label,
                "strong": strong_label
            }
        }


class WaveformValidDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, tp: pd.DataFrame, fp: pd.DataFrame, datadir: Path,
                 transforms=None, sampling_rate=32000, duration=10, hop_length=320):
        self.df = df
        self.tp = tp
        self.fp = fp
        self.datadir = datadir
        self.transforms = transforms
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.hop_length = hop_length

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
        if self.transforms:
            y = self.transforms(y).astype(np.float32)

        tp = self.tp.query(f"recording_id == '{flac_id}'")[["species_id", "t_min", "t_max"]].values
        label = np.zeros(N_CLASSES, dtype=np.float32)

        n_points = len(y)
        n_frames = int(n_points / self.hop_length) + 1
        strong_label = np.zeros((n_frames, N_CLASSES), dtype=np.float32)
        if len(tp) != 0:
            for row in tp:
                t_min, t_max = row[1], row[2]
                species_id = int(row[0])
                if t_min > offset and t_max < offset + self.duration:
                    label[species_id] = 1.0
                    start_index = int(((t_min - offset) * self.sampling_rate) / self.hop_length) + 1
                    end_index = int(((t_max - offset) * self.sampling_rate) / self.hop_length) + 1
                    strong_label[start_index:end_index, species_id] = 1.0

        return {
            "recording_id": flac_id,
            "waveform": y,
            "targets": {
                "weak": label,
                "strong": strong_label
            }
        }


class WaveformTestDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path,
                 transforms=None, sampling_rate=32000, duration=10, hop_length=320):
        self.df = df
        self.datadir = datadir
        self.transforms = transforms
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.hop_length = hop_length

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
        if self.transforms:
            y = self.transforms(y).astype(np.float32)

        return {
            "recording_id": flac_id,
            "waveform": y
        }


class MultiLabelWaveformDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, tp: pd.DataFrame, fp: pd.DataFrame, datadir: Path,
                 transforms=None, sampling_rate=32000, duration=10, hop_length=320,
                 centering=False):
        unique_recording_id = df.recording_id.unique().tolist()
        unique_tp_recordin_id = tp.recording_id.unique().tolist()
        intersection = set(unique_recording_id).intersection(unique_tp_recordin_id)
        self.df = df[df.recording_id.isin(intersection)].reset_index(drop=True)
        self.tp = tp[tp.recording_id.isin(intersection)].reset_index(drop=True)
        self.fp = fp  # unused
        self.datadir = datadir
        self.transforms = transforms
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.hop_length = hop_length
        self.centering = centering

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
        if self.transforms:
            y = self.transforms(y).astype(np.float32)

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
            "waveform": y,
            "targets": {
                "weak": label,
                "strong": strong_label
            }
        }
