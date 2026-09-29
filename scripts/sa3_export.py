"""Run the pinned SA3 MLX CLI and reduce gain before PCM16 quantization."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from local_model import RUNTIME

sys.path.insert(0, str(RUNTIME / "scripts"))
import sa3_mlx


def save_wav_with_headroom(path, audio, sample_rate=44100):
    audio = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("Cannot export non-finite audio")
    peak = float(np.max(np.abs(audio)))
    gain = min(1.0, 0.944060876 * (1.0 - 1e-6) / peak) if peak > 0 else 1.0
    sf.write(path, (audio * gain).T, sample_rate, subtype="PCM_16")
    if gain < 1:
        print(f"PCM headroom: {gain:.5f}x gain, {20*np.log10(gain):.2f} dB")


def main():
    sa3_mlx.save_wav = save_wav_with_headroom
    sys.argv = [str(RUNTIME / "scripts/sa3_mlx.py"), *sys.argv[1:]]
    sa3_mlx.main()


if __name__ == "__main__":
    main()
