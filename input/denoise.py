import nussl
import soundfile as sf

from joblib import delayed, Parallel
from pathlib import Path


def denoise(audio_path: Path, save_dir: Path):
    history = nussl.AudioSignal(audio_path)
    separator = nussl.separation.primitive.Repet(history)
    estimates = separator()
    foreground = estimates[1].audio_data[0]
    audio_name = audio_path.name
    save_path = save_dir / audio_name.replace(".flac", ".wav")
    sf.write(save_path, foreground, samplerate=48000)


if __name__ == "__main__":
    audio_dir = Path("./test")
    save_dir = Path("./test_denoised")
    save_dir.mkdir(exist_ok=True, parents=True)

    audio_paths = list(audio_dir.glob("*.flac"))
    Parallel(n_jobs=20, verbose=10)(delayed(denoise)(
        audio_path, save_dir
    ) for audio_path in audio_paths)
