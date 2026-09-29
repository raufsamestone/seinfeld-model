"""Prepare an isolated training set from exactly one local YouTube WAV."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/sources/seinfeld-transition-music-only.wav"
DEST = ROOT / "experiments/seinfeld-transition-reference"
CONFIG = json.loads((ROOT / "config/local-training.json").read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if DEST.exists():
        raise FileExistsError(f"Refusing to overwrite experiment data: {DEST}")

    # Decode only the requested source. No directory globbing or other inputs.
    pcm = DEST / "source-44100-stereo.wav"
    pcm.parent.mkdir(parents=True)
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(SOURCE), "-ar", "44100",
        "-ac", "2", "-c:a", "pcm_s16le", str(pcm),
    ], check=True)
    audio, sr = sf.read(pcm, dtype="float32", always_2d=True)
    if sr != 44100 or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise RuntimeError(f"Unexpected decoded audio shape/rate: {audio.shape}, {sr}")

    chunk_samples = round(3.6 * sr)
    source_sha256 = sha256(SOURCE)
    clips = []
    index = 0
    for start in range(0, len(audio) - chunk_samples + 1, chunk_samples):
        index += 1
        # Sparse, deterministic temporal holdout. All material still comes
        # exclusively from the configured single source WAV.
        split = "holdout" if index % 9 == 0 else "train"
        name = f"cue-{index:03}.wav"
        target = DEST / "audio" / split / name
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, audio[start:start + chunk_samples], sr, subtype="PCM_16")
        reread, reread_sr = sf.read(target, dtype="float32", always_2d=True)
        if reread_sr != sr or reread.shape != (chunk_samples, 2):
            raise RuntimeError(f"Invalid generated segment: {target}")
        target.with_suffix(".txt").write_text(CONFIG["prompt"] + "\n")
        relative = target.relative_to(ROOT).as_posix()
        clips.append({
            "id": f"{index:03}",
            "split": split,
            "file": relative,
            "source_sha256": source_sha256,
            "start_seconds": round(start / sr, 6),
            "seconds": round(len(reread) / sr, 6),
            "pcm_sha256": hashlib.sha256(reread.tobytes()).hexdigest(),
        })

    if not any(c["split"] == "train" for c in clips) or not any(c["split"] == "holdout" for c in clips):
        raise RuntimeError("Need both training and held-out segments")
    (DEST / "source-44100-stereo.wav").unlink()
    manifest = {
        "source_file": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_sha256,
        "source_duration_seconds": round(len(audio) / sr, 6),
        "split_note": "All segments derive only from seinfeld-transition-music-only.wav. Every ninth non-overlapping 3.6s segment is held out; temporal neighbours may share musical context.",
        "clips": clips,
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for split in ("train", "holdout"):
        subset = [c for c in clips if c["split"] == split]
        print(f"{split}: {len(subset)} clips, {sum(c['seconds'] for c in subset):.1f}s")
    print(f"Only source: {SOURCE}")
    print(f"Dataset: {DEST}")


if __name__ == "__main__":
    main()
