import argparse

import librosa

import soundfile as sf

from pathlib import Path
from tqdm import tqdm


def resample(audio_path: Path, save_path: Path, target_sr=32000):
    y, _ = librosa.load(audio_path,
                        sr=target_sr,
                        mono=True,
                        res_type="kaiser_best")
    sf.write(save_path, y, samplerate=target_sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=32000)

    args = parser.parse_args()

    target_sr = args.sr

    audio_dir = Path("./train")
    save_dir = Path(f"./train_{int(target_sr / 1000)}k")
    save_dir.mkdir(exist_ok=True, parents=True)

    audio_paths = list(audio_dir.glob("*.flac"))
    for audio_path in tqdm(audio_paths, desc="train"):
        audio_name = audio_path.name
        save_path = save_dir / audio_name.replace(".flac", ".wav")
        resample(audio_path, save_path, target_sr)

    audio_dir = Path("./test")
    save_dir = Path(f"./test_{int(target_sr)}k")
    save_dir.mkdir(parents=True, exist_ok=True)

    audio_path = list(audio_dir.glob("*.flac"))
    for audio_path in tqdm(audio_paths, desc="test"):
        audio_name = audio_path.name
        save_path = save_dir / audio_name.replace(".flac", ".wav")
        resample(audio_path, save_path, target_sr)
