import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
    import soundfile
except ImportError:
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
if np is not None:
    from audit_exports import max_copy_similarity


@unittest.skipIf(np is None, "Optional local audio dependencies not installed")
class CopyScreen(unittest.TestCase):
    def test_detects_shifted_scaled_pcm_excerpt(self):
        source = np.random.default_rng(1).normal(0, 0.1, 12000)
        query = source[2000:7000] * 0.7 + 0.01
        self.assertGreater(max_copy_similarity(query, source), 0.99)

    def test_independent_noise_does_not_match(self):
        rng = np.random.default_rng(2)
        self.assertLess(max_copy_similarity(rng.normal(size=6000), rng.normal(size=12000)), 0.1)

    def test_silence_is_not_a_copy(self):
        self.assertEqual(max_copy_similarity(np.zeros(5000), np.zeros(12000)), 0)
