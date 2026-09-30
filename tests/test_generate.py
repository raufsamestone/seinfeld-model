import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.generate import ensure_adapter


class DownloadAdapter(unittest.TestCase):
    def test_downloads_and_verifies_adapter(self):
        payload = b"test adapter weights"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "adapter.safetensors"
            config = {
                "adapter": str(target),
                "adapter_url": "https://example.test/adapter.safetensors",
                "adapter_sha256": hashlib.sha256(payload).hexdigest(),
            }
            with patch("scripts.generate.urlopen", return_value=io.BytesIO(payload)):
                self.assertEqual(ensure_adapter(config), target)
            self.assertEqual(target.read_bytes(), payload)

    def test_rejects_bad_digest_and_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "adapter.safetensors"
            config = {
                "adapter": str(target),
                "adapter_url": "https://example.test/adapter.safetensors",
                "adapter_sha256": "0" * 64,
            }
            with patch("scripts.generate.urlopen", return_value=io.BytesIO(b"wrong")):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    ensure_adapter(config)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_refuses_modified_cached_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "adapter.safetensors"
            target.write_bytes(b"tampered")
            config = {
                "adapter": str(target),
                "adapter_url": "https://example.test/adapter.safetensors",
                "adapter_sha256": "0" * 64,
            }
            with self.assertRaisesRegex(ValueError, "Adapter SHA-256 mismatch"):
                ensure_adapter(config)


if __name__ == "__main__":
    unittest.main()
