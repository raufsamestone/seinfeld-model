"""Fast tests: no model downloads, GPU use, cloud access or proprietary audio needed."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("local_model", ROOT / "scripts/local_model.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class TrainingContract(unittest.TestCase):
    def test_pins_are_immutable_commits(self):
        for key in ("upstream_revision", "weights_revision"):
            self.assertRegex(model.CONFIG[key], r"^[0-9a-f]{40}$")

    def test_window_fits_selected_cues(self):
        self.assertGreaterEqual(model.CONFIG["latent_crop_length"] * 4096 / 44100, 7.4)

    def test_seeds_distinct_and_six(self):
        self.assertEqual(len(set(model.CONFIG["export_seeds"])), 6)

    def test_prompt_is_actually_consumed(self):
        config = json.loads((ROOT / "config/prompts.json").read_text())
        self.assertEqual(config["tag_keys"], ["prompt"])
        self.assertFalse(config["use_paths"])
        self.assertTrue(config["use_tags"])

    def test_digest(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture"
            path.write_bytes(b"fixture")
            self.assertEqual(model.digest(path), hashlib.sha256(b"fixture").hexdigest())

    def test_no_extra_training_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            expected = root / "research/sources/seinfeld-transition-music-only.wav"
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"single-reference")
            source_hash = model.digest(expected)
            records = []
            for split in ("train", "holdout"):
                directory = data / "audio" / split
                directory.mkdir(parents=True)
                path = directory / "cue.wav"
                path.touch()
                records.append({"file": str(path.relative_to(root)), "split": split,
                                "pcm_sha256": split, "source_sha256": source_hash})
            manifest = {"source_file": "research/sources/seinfeld-transition-music-only.wav",
                        "source_sha256": source_hash, "clips": records}
            (data / "manifest.json").write_text(json.dumps(manifest))
            with patch.object(model, "ROOT", root), patch.object(model, "DATA", data):
                model.validate_dataset()
                (data / "audio/train/accidental-reference.wav").touch()
                with self.assertRaisesRegex(RuntimeError, "Unexpected"):
                    model.validate_dataset()

    def test_no_generation_api_or_old_engine(self):
        for name in ("src/neural-server.mjs", "src/composer.mjs", "worker/musicgen.py", "data/bank"):
            self.assertFalse((ROOT / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
