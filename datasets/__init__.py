import pandas as pd
import torch.utils.data as torchdata

import transforms

from pathlib import Path

from .fp_sample import SampleFPSpectrogramDataset
from .freq_limit_input import (LimitedFrequencySpectrogramDataset, LimitedFrequencySequentialValidationDataset,
                               LimitedFrequencySpectrogramTestDataset, SPECIES_RANGE_MAP, RANGE_SPECIES_MAP,
                               LimitedFrequencySampleWiseSpectrogramTestDataset)
from .mixup import LogmelMixupDataset, LogmelMixupWithFPDataset
from .samplewise import SampleWiseSpectrogramDataset, SampleWiseSpectrogramTestDataset
from .sequential import SequentialValidationDataset
from .spectrogram import (SpectrogramDataset, SpectrogramTestDataset, MultiLabelSpectrogramDataset, TorchAudioMLDataset,
                          TorchAudioMLTestDataset, FasterMLSpectrogramDataset, FasterSpectrogramTestDataset)
from .waveform import (WaveformDataset, WaveformValidDataset, WaveformTestDataset,
                       MultiLabelWaveformDataset)
from .random_crop import RandomFasterMLSpectrogramDataset


__DATASETS__ = {
    "SampleFPSpectrogramDataset": SampleFPSpectrogramDataset,
    "LimitedFrequencySpectrogramDataset": LimitedFrequencySpectrogramDataset,
    "LimitedFrequencySequentialValidationDataset": LimitedFrequencySequentialValidationDataset,
    "LimitedFrequencySpectrogramTestDataset": LimitedFrequencySpectrogramTestDataset,
    "LimitedFrequencySampleWiseSpectrogramTestDataset": LimitedFrequencySampleWiseSpectrogramTestDataset,
    "LogmelMixupDataset": LogmelMixupDataset,
    "LogmelMixupWithFPDataset": LogmelMixupWithFPDataset,
    "SpectrogramDataset": SpectrogramDataset,
    "SpectrogramTestDataset": SpectrogramTestDataset,
    "MultiLabelSpectrogramDataset": MultiLabelSpectrogramDataset,
    "FasterMLSpectrogramDataset": FasterMLSpectrogramDataset,
    "FasterSpectrogramTestDataset": FasterSpectrogramTestDataset,
    "TorchAudioMLDataset": TorchAudioMLDataset,
    "TorchAudioMLTestDataset": TorchAudioMLTestDataset,
    "WaveformDataset": WaveformDataset,
    "WaveformValidDataset": WaveformValidDataset,
    "WaveformTestDataset": WaveformTestDataset,
    "MultiLabelWaveformDataset": MultiLabelWaveformDataset,
    "SampleWiseSpectrogramDataset": SampleWiseSpectrogramDataset,
    "SampleWiseSpectrogramTestDataset": SampleWiseSpectrogramTestDataset,
    "SequentialValidationDataset": SequentialValidationDataset,
    "RandomFasterMLSpectrogramDataset": RandomFasterMLSpectrogramDataset
}


def get_metadata(config: dict):
    data_config = config["data"]

    tp = pd.read_csv(data_config["train_tp_path"])
    fp = pd.read_csv(data_config["train_fp_path"])
    train_audio = Path(data_config["train_audio_path"])
    test_audio = Path(data_config["test_audio_path"])

    train_audios = list(train_audio.glob("*.flac"))
    if len(train_audios) == 0:
        train_audios = list(train_audio.glob("*.wav"))

    test_audios = list(test_audio.glob("*.flac"))
    if len(test_audios) == 0:
        test_audios = list(test_audio.glob("*.wav"))

    train_audio_lists = [audio.name.replace(".flac", "").replace(".wav", "") for audio in train_audios]
    test_audio_lists = [audio.name.replace(".flac", "").replace(".wav", "") for audio in test_audios]
    train_all = pd.DataFrame({
        "recording_id": train_audio_lists
    })
    test_all = pd.DataFrame({
        "recording_id": test_audio_lists
    })
    clip_level_tp = tp.groupby("recording_id")["species_id"].apply(list)
    clip_level_fp = fp.groupby("recording_id")["species_id"].apply(list)

    tp["species_id_song_id"] = tp["species_id"].map(str) + "_" + tp["songtype_id"].map(str)
    fp["species_id_song_id"] = fp["species_id"].map(str) + "_" + fp["songtype_id"].map(str)

    tp = tp.reset_index(drop=False)
    fp = fp.reset_index(drop=False)

    clip_level_tp_joint = tp.groupby("recording_id")["species_id_song_id"].apply(list)
    clip_level_fp_joint = fp.groupby("recording_id")["species_id_song_id"].apply(list)

    train_all = train_all.merge(clip_level_tp, on="recording_id", how="left").rename(
        columns={"species_id": "tp"})
    train_all = train_all.merge(clip_level_fp, on="recording_id", how="left").rename(
        columns={"species_id": "fp"})
    train_all = train_all.merge(clip_level_tp_joint, on="recording_id", how="left").rename(
        columns={"species_id_song_id": "tp_species_id_song_id"})
    train_all = train_all.merge(clip_level_fp_joint, on="recording_id", how="left").rename(
        columns={"species_id_song_id": "fp_species_id_song_id"})

    train_all["n_tp"] = train_all.tp.map(lambda x: len(x) if isinstance(x, list) else 0)
    train_all["n_fp"] = train_all.fp.map(lambda x: len(x) if isinstance(x, list) else 0)
    train_all["n_tp_fp"] = train_all["n_tp"] + train_all["n_fp"]

    return tp, fp, train_all, test_all, train_audio, test_audio


def get_train_loader(df: pd.DataFrame,
                     tp: pd.DataFrame,
                     fp: pd.DataFrame,
                     datadir: Path,
                     config: dict,
                     phase: str):
    dataset_config = config["dataset"]
    loader_config = config["loader"][phase]
    if dataset_config[phase]["name"] in ["WaveformDataset", "WaveformValidDataset",
                                         "MultiLabelWaveformDataset"]:
        transform = transforms.get_waveform_transforms(config, phase)
        params = dataset_config[phase]["params"]

        dataset = __DATASETS__[dataset_config[phase]["name"]](
            df, tp, fp, datadir, transform, **params)
    elif dataset_config[phase]["name"] in ["SpectrogramDataset", "MultiLabelSpectrogramDataset",
                                           "SampleFPSpectrogramDataset", "TorchAudioMLDataset",
                                           "FasterMLSpectrogramDataset", "LogmelMixupDataset",
                                           "SampleWiseSpectrogramDataset", "LogmelMixupWithFPDataset",
                                           "SequentialValidationDataset", "LimitedFrequencySpectrogramDataset",
                                           "LimitedFrequencySequentialValidationDataset",
                                           "RandomFasterMLSpectrogramDataset"]:
        waveform_transforms = transforms.get_waveform_transforms(config, phase)
        spectrogram_transforms = transforms.get_spectrogram_transforms(config, phase)
        params = dataset_config[phase]["params"]

        dataset = __DATASETS__[dataset_config[phase]["name"]](
            df, tp, fp, datadir,
            waveform_transforms,
            spectrogram_transforms,
            **params)
    else:
        raise NotImplementedError
    loader = torchdata.DataLoader(dataset, **loader_config)
    return loader


def get_test_loader(df: pd.DataFrame,
                    datadir: Path,
                    config: dict):
    dataset_config = config["dataset"]
    loader_config = config["loader"]["test"]
    if dataset_config["test"]["name"] in ["WaveformTestDataset"]:
        transform = transforms.get_waveform_transforms(config, "test")
        params = dataset_config["test"]["params"]

        dataset = __DATASETS__[dataset_config["test"]["name"]](
            df, datadir, transform, **params)
    elif dataset_config["test"]["name"] in ["SpectrogramTestDataset", "TorchAudioMLTestDataset",
                                            "FasterSpectrogramTestDataset", "SampleWiseSpectrogramTestDataset",
                                            "LimitedFrequencySpectrogramTestDataset",
                                            "LimitedFrequencySampleWiseSpectrogramTestDataset"]:
        waveform_transforms = transforms.get_waveform_transforms(config, "test")
        spectrogram_transforms = transforms.get_spectrogram_transforms(config, "test")
        params = dataset_config["test"]["params"]

        dataset = __DATASETS__[dataset_config["test"]["name"]](
            df, datadir, waveform_transforms, spectrogram_transforms, **params)
    else:
        raise NotImplementedError
    loader = torchdata.DataLoader(dataset, **loader_config)
    return loader
