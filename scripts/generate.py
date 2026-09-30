"""Generate one WAV using config/generation.json (local Apple-Silicon MLX)."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from urllib.request import urlopen
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/generation.json"
REQUIRED = {
    "dit", "decoder", "prompt", "seconds", "sampling_steps", "seed",
    "cfg", "adapter", "adapter_url", "adapter_sha256", "adapter_strength", "output",
}


def ensure_adapter(config: dict) -> Path:
    """Fetch the published adapter once, then verify it before inference."""
    adapter = Path(config["adapter"])
    adapter = adapter if adapter.is_absolute() else ROOT / adapter
    expected = config["adapter_sha256"].lower()
    if adapter.exists():
        actual = hashlib.sha256(adapter.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Adapter SHA-256 mismatch for {adapter}: {actual}")
        return adapter

    adapter.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=adapter.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
        digest = hashlib.sha256()
        try:
            with urlopen(config["adapter_url"], timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    tmp.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(
                    f"Downloaded adapter SHA-256 mismatch: {digest.hexdigest()}"
                )
            temporary.replace(adapter)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH,
                        help="Generation JSON (default: config/generation.json)")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text())
    missing = REQUIRED - config.keys()
    if missing:
        raise ValueError(f"Missing config values: {', '.join(sorted(missing))}")

    adapter = ensure_adapter(config)

    output_name = str(config["output"]).format(seed=config["seed"])
    output = Path(output_name)
    output = output if output.is_absolute() else ROOT / output
    output = output.resolve()
    generated_dir = (ROOT / "output/generated").resolve()
    if not output.is_relative_to(generated_dir):
        raise ValueError("Config output must be inside output/generated/")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing WAV: {output}\n"
                              "Change the seed or output name in config/generation.json.")
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, "-u", str(ROOT / "scripts/sa3_export.py"),
        "--dit", str(config["dit"]), "--decoder", str(config["decoder"]),
        "--prompt", str(config["prompt"]), "--seconds", str(config["seconds"]),
        "--steps", str(config["sampling_steps"]), "--seed", str(config["seed"]),
        "--cfg", str(config["cfg"]), "--lora", str(adapter),
        "--lora-strength", str(config["adapter_strength"]), "--out", str(output),
    ]
    print(f"Config: {config_path}", flush=True)
    print(f"Output: {output}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
