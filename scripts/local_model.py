"""Local Apple-Silicon fine-tuning. No hosted inference, uploads or paid jobs.

Run with .runtime/mlx-venv/bin/python. See README for environment setup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/local-training.json").read_text())
RUNTIME = ROOT / ".runtime/stable-audio-3/optimized/mlx"
DATA = Path(os.environ.get("SEINFELD_DATA_DIR", ROOT / "experiments/seinfeld-transition-reference")).resolve()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_dataset():
    manifest = json.loads((DATA / "manifest.json").read_text())
    required_source = "research/sources/seinfeld-transition-music-only.wav"
    if manifest.get("source_file") != required_source:
        raise RuntimeError(f"Training source must be exactly {required_source}")
    source = ROOT / required_source
    if not source.is_file() or digest(source) != manifest.get("source_sha256"):
        raise RuntimeError("Single-source WAV is missing or its checksum changed")
    clips = manifest["clips"]
    if any(c.get("source_sha256") != manifest["source_sha256"] for c in clips):
        raise RuntimeError("Dataset contains a segment from another source")
    if not clips or len({c["pcm_sha256"] for c in clips}) != len(clips):
        raise RuntimeError("Empty or duplicate dataset")
    for split in ("train", "holdout"):
        expected = {ROOT / c["file"] for c in clips if c["split"] == split}
        actual = set((DATA / "audio" / split).glob("*.wav"))
        if not expected or actual != expected:
            raise RuntimeError(f"Unexpected or missing {split} WAVs")
    return manifest


def run(args, log=None, cwd=ROOT):
    print("+", " ".join(map(str, args)), flush=True)
    if log is None:
        subprocess.run(list(map(str, args)), check=True, cwd=cwd)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as out:
            subprocess.run(list(map(str, args)), check=True, cwd=cwd,
                           stdout=out, stderr=subprocess.STDOUT)


def verify_runtime():
    revision = subprocess.check_output(
        ["git", "-C", str(RUNTIME), "rev-parse", "HEAD"], text=True).strip()
    if revision != CONFIG["upstream_revision"]:
        raise RuntimeError(f"Unexpected upstream revision: {revision}")


def download():
    from huggingface_hub import hf_hub_download
    verify_runtime()
    files = ["same_l_encoder_f32.npz", "t5gemma_f16.npz",
             "dit_medium-base_f16.npz", "dit_medium_f16.npz", "same_l_decoder_f32.npz"]
    for name in files:
        print(f"Downloading/checking {name}", flush=True)
        cached = Path(hf_hub_download(CONFIG["weights_repository"], "MLX/" + name,
                                     revision=CONFIG["weights_revision"]))
        link = RUNTIME / "models/mlx" / name
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and link.resolve() != cached.resolve():
            raise RuntimeError(f"Conflicting existing weights: {link}")
        if not link.exists():
            link.symlink_to(cached)
        print(f"Ready: {name} ({cached.stat().st_size / 1e9:.2f} GB)", flush=True)


def encode():
    verify_runtime()
    validate_dataset()
    for split in ("train", "holdout"):
        run([sys.executable, "-u", RUNTIME / "scripts/pre_encode_mlx.py",
             "--audio-dir", DATA / "audio" / split,
             "--output-dir", DATA / "latents" / split, "--codec", CONFIG["codec"]])


def train(args):
    verify_runtime()
    manifest = validate_dataset()
    expected = {"cue-" + c["id"] for c in manifest["clips"] if c["split"] == "train"}
    if {p.stem for p in (DATA / "latents/train").glob("*.npy")} != expected:
        raise RuntimeError("Unexpected or missing training latents; run encode first")
    for cue in expected:
        metadata = json.loads((DATA / "latents/train" / (cue + ".json")).read_text())
        if metadata.get("prompt") != CONFIG["prompt"] or metadata["latent_shape"][1] > CONFIG["latent_crop_length"]:
            raise RuntimeError("Stale caption or truncated training phrase; re-encode/check crop length")
    if args.steps < 1:
        raise ValueError("Steps must be positive")
    label = f"slap-bass-{args.steps}-{time.time_ns()}"
    log = ROOT / "output/training" / (label + ".log")
    cmd = [sys.executable, "-u", RUNTIME / "scripts/lora_train_mlx.py",
           "--dit", CONFIG["model"], "--latents-dir", DATA / "latents/train",
           "--lr", CONFIG["learning_rate"], "--name", label,
           "--save-dir", ROOT / "models/runs", "--batch-size", 1,
           "--seed", CONFIG["seed"], "--adapter-type", "dora-rows",
           "--rank", CONFIG["rank"], "--max-steps", args.steps,
           "--checkpoint-every", min(100, args.steps),
           "--latent-crop-length", CONFIG["latent_crop_length"],
           "--no-random-crop", "--grad-checkpoint", "--no-compile",
           "--gradient-clip-val", 1, "--prompt-config", ROOT / "config/prompts.json",
           "--demo-every", 0]
    if args.resume:
        cmd.extend(["--lora-ckpt-path", Path(args.resume).resolve()])
    save_json(log.with_suffix(".json"), {"config": CONFIG, "command": list(map(str, cmd)),
              "dataset_sha256": digest(DATA / "manifest.json"), "status": "started"})
    print(f"Training log: {log}", flush=True)
    start = time.time()
    telemetry_dir = log.with_suffix("")
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(cmd, log, cwd=telemetry_dir)
    except BaseException:
        save_json(log.with_suffix(".json"), {"config": CONFIG, "command": list(map(str, cmd)),
                  "dataset_sha256": digest(DATA / "manifest.json"), "status": "failed/interrupted",
                  "wall_seconds": time.time() - start})
        raise
    checkpoints = sorted((ROOT / "models/runs" / label).rglob("*.safetensors"))
    save_json(log.with_suffix(".json"), {"config": CONFIG, "command": list(map(str, cmd)),
              "dataset_sha256": digest(DATA / "manifest.json"), "status": "completed",
              "wall_seconds": time.time() - start,
              "checkpoints": [str(p.relative_to(ROOT)) for p in checkpoints]})
    print(f"Training completed; {len(checkpoints)} checkpoint(s).", flush=True)


def export(args):
    import numpy as np
    import soundfile as sf
    verify_runtime()
    adapter = Path(args.adapter).resolve() if args.adapter else None
    if adapter and not adapter.is_file():
        raise FileNotFoundError(adapter)
    destination = ROOT / "output" / (args.name or ("trained-showcase" if adapter else "base-comparison"))
    destination.mkdir(parents=True, exist_ok=True)
    old_manifest_path = destination / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text()) if args.resume and old_manifest_path.exists() else None
    if args.resume:
        expected_hash = digest(adapter) if adapter else None
        if not old_manifest or old_manifest.get("adapter_sha256") != expected_hash:
            raise RuntimeError("Cannot resume exports: existing manifest is missing or belongs to another adapter")
    elif old_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite export set: {old_manifest_path}")
    clips = []
    for i, seed in enumerate(CONFIG["export_seeds"][:args.count], 1):
        target = destination / f"riff-{i:02}.wav"
        if target.exists():
            prior = next((c for c in old_manifest["clips"] if c["file"] == target.name), None) if old_manifest else None
            if not prior or prior["seed"] != seed or digest(target) != prior["sha256"]:
                raise FileExistsError(f"Existing export does not match resumable manifest: {target}")
            audio, sr = sf.read(target, always_2d=True)
            if sr != 44100 or audio.shape != (158760, 2) or not np.isfinite(audio).all():
                raise RuntimeError(f"Existing export is invalid: {target}")
            clips.append(prior)
            continue
        cmd = [sys.executable, "-u", ROOT / "scripts/sa3_export.py", "--dit", CONFIG["model"],
               "--decoder", CONFIG["codec"], "--prompt", CONFIG["prompt"], "--seconds", 3.6,
               "--steps", 8, "--seed", seed, "--cfg", 1, "--out", target]
        if adapter:
            cmd.extend(["--lora", adapter, "--lora-strength", args.strength])
        run(cmd, target.with_suffix(".log"))
        audio, sr = sf.read(target, always_2d=True)
        if sr != 44100 or audio.shape != (158760, 2) or not np.isfinite(audio).all():
            raise RuntimeError(f"Unexpected export format: {target}")
        mono = np.max(np.abs(audio), axis=1)
        peaks = [round(float(np.max(chunk)), 4) for chunk in np.array_split(mono, 96)]
        clips.append({"id": f"riff-{i:02}", "file": target.name, "seed": seed,
                      "duration": len(audio) / sr, "sha256": digest(target), "peaks": peaks,
                      "rms": float(np.sqrt(np.mean(audio ** 2))),
                      "clipped_fraction": float(np.mean(np.abs(audio) >= 0.9999))})
        save_json(destination / "manifest.json", {"adapter": str(adapter) if adapter else None,
                  "adapter_sha256": digest(adapter) if adapter else None, "strength": args.strength,
                  "config": CONFIG, "clips": clips,
                  "quality": "Unreviewed experimental outputs; distinct hashes do not prove musical novelty."})
    if len({c["sha256"] for c in clips}) != len(clips):
        raise RuntimeError("Identical output detected")
    print(f"Exported {len(clips)} clips to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("download", "encode"):
        sub.add_parser(name)
    training = sub.add_parser("train")
    training.add_argument("--steps", type=int, required=True)
    training.add_argument("--resume")
    exporting = sub.add_parser("export")
    exporting.add_argument("--adapter")
    exporting.add_argument("--name")
    exporting.add_argument("--strength", type=float, default=1)
    exporting.add_argument("--count", type=int, choices=range(1, 7), default=6)
    exporting.add_argument("--resume", action="store_true", help="Continue an interrupted export set with its verified manifest")
    args = parser.parse_args()
    if args.action in ("train", "export"):
        globals()[args.action](args)
    else:
        globals()[args.action]()


if __name__ == "__main__":
    main()
