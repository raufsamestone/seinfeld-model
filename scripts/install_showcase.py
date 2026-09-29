"""Copy audited generated WAVs into a LOCAL site checkout. Does not deploy/upload."""
import argparse
import json
import shutil
from pathlib import Path

from local_model import digest, save_json


def install(source, site):
    source, site = Path(source).resolve(), Path(site).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    audit = json.loads((source / "audit.json").read_text())
    if not manifest.get("adapter_sha256") or not audit["is_finetuned"]:
        raise ValueError("Only actual fine-tuned model exports belong in this gallery")
    if audit["manifest_sha256"] != digest(source / "manifest.json") or audit["adapter_sha256"] != manifest["adapter_sha256"]:
        raise ValueError("Stale audit; rerun audit_exports.py")
    if len(manifest["clips"]) not in (5, 6) or any(c["copy_flag"] for c in audit["clips"]):
        raise ValueError("Expected 5–6 recordings with no PCM-copy flags")
    index = site / "src/components/seinfeld/recordings.json"
    if not index.is_file() or not (site / "src/app/seinfeld/page.tsx").is_file():
        raise ValueError("Not the expected website checkout")
    destination = site / "public/audio/seinfeld"
    destination.mkdir(parents=True, exist_ok=True)
    for item in manifest["clips"]:
        name = item["file"]
        if Path(name).name != name or not name.endswith(".wav"):
            raise ValueError("Unsafe export filename")
        wav = source / name
        if digest(wav) != item["sha256"]:
            raise ValueError("Export hash mismatch")
        target = destination / name
        if target.exists() and digest(target) != item["sha256"]:
            raise FileExistsError(f"Existing gallery differs; archive it before replacing: {target}")
    for item in manifest["clips"]:
        shutil.copy2(source / item["file"], destination / item["file"])
    save_json(index, {"adapter_sha256": manifest["adapter_sha256"],
                     "training_steps": audit["adapter_config"]["step"],
                     "quality_review": "pending",
                     "clips": [{k: clip[k] for k in ("id", "file", "duration", "sha256", "peaks", "seed")}
                               for clip in manifest["clips"]]})
    print(f"Installed {len(manifest['clips'])} real model exports into local site: {site}. Not deployed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("site")
    args = parser.parse_args()
    install(args.source, args.site)
