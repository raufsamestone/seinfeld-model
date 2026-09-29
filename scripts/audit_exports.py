"""Signal/provenance audit, NOT a perceptual quality or copyright clearance test."""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import soundfile as sf

from local_model import DATA, ROOT, digest, save_json


def header(path):
    with Path(path).open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if size > 16_000_000:
            raise ValueError("Unexpected safetensors header size")
        return json.loads(f.read(size))


def max_copy_similarity(query, source, sr=4410):
    """Absolute normalized PCM cross-correlation of 1 s windows at original speed.

    A limited exact/near-copy screen: misses pitch shifts, time stretches and
    musical quotation. Not proof of originality. Inputs already downsampled.
    """
    length = sr
    best = 0.0
    for start in range(0, len(query) - length + 1, sr // 2):
        window = query[start:start + length]
        window = window - window.mean()
        norm = np.linalg.norm(window)
        if norm < 0.001 or len(source) < length:
            continue
        size = 1 << (len(source) + length - 1).bit_length()
        conv = np.fft.irfft(np.fft.rfft(source, size) * np.fft.rfft(window[::-1], size), size)
        sums = np.concatenate(([0.0], np.cumsum(source)))
        squares = np.concatenate(([0.0], np.cumsum(source ** 2)))
        energy = squares[length:] - squares[:-length] - (sums[length:] - sums[:-length]) ** 2 / length
        scores = np.abs(conv[length - 1:len(source)]) / (norm * np.sqrt(np.maximum(energy, 1e-12)))
        best = max(best, float(scores.max()))
    return best


def audit(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    checkpoint = Path(manifest["adapter"]) if manifest["adapter"] else None
    evidence = {"is_finetuned": bool(checkpoint), "adapter_sha256": manifest["adapter_sha256"],
                "manifest_sha256": digest(directory / "manifest.json"), "clips": []}
    if checkpoint:
        if digest(checkpoint) != manifest["adapter_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        metadata = header(checkpoint)["__metadata__"]
        config = json.loads(metadata["lora_config"])
        if config["step"] < 1:
            raise ValueError("Adapter has no training steps")
        evidence["adapter_config"] = config
    dataset = json.loads((DATA / "manifest.json").read_text())
    references = []
    for record in dataset["clips"]:
        samples, _ = sf.read(ROOT / record["file"], always_2d=True)
        # Decimation is only a lightweight matching heuristic, not audio export.
        references.append((record, samples.mean(axis=1)[::10]))
    hashes = set()
    for item in manifest["clips"]:
        path = directory / item["file"]
        if digest(path) != item["sha256"] or item["sha256"] in hashes:
            raise ValueError("Incorrect/duplicate output hash")
        hashes.add(item["sha256"])
        samples, sr = sf.read(path, always_2d=True)
        if sr != 44100 or samples.shape != (158760, 2) or not np.isfinite(samples).all():
            raise ValueError("Unexpected audio format")
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < 0.001:
            raise ValueError("Silent output")
        mono = samples.mean(axis=1)[::10]
        matches = sorted(({"id": r["id"], "split": r["split"],
                           "correlation": max_copy_similarity(mono, signal)}
                          for r, signal in references), key=lambda v: v["correlation"], reverse=True)
        peaks = [round(float(np.max(chunk)), 4) for chunk in np.array_split(np.max(np.abs(samples), axis=1), 96)]
        if peaks != item["peaks"]:
            raise ValueError("Waveform does not match audio")
        evidence["clips"].append({"file": item["file"], "rms": rms,
                                  "peak": float(np.max(np.abs(samples))),
                                  "closest_pcm_window": matches[0],
                                  "copy_flag": matches[0]["correlation"] > 0.92})
    evidence["limits"] = "PCM-window heuristic only; no pitch/time invariant or semantic originality guarantee. Human listening required."
    save_json(directory / "audit.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    audit(parser.parse_args().directory)
