# Local single-source experiment

## Active dataset

The experiment uses only `research/sources/seinfeld-transition-music-only.wav`, a 244.053-second stereo 48 kHz recording titled “Seinfeld Scene Transition Music (Music Only) - v3”. The original YouTube video ID is `B7IRCudlYPg`; it is retained here as provenance, not as a folder name. The source SHA-256 and segment timestamps are recorded in `experiments/seinfeld-transition-reference/manifest.json`.

The preparation script creates 67 non-overlapping 3.6-second PCM16 clips: 60 training clips (~216 s) and 7 temporal holdouts (~25.2 s). All clips derive from that one source. This is not an independent episode-level test; adjacent musical context can cross segment boundaries.

## Training status

Training uses pinned Stable Audio 3 medium BASE + MLX DoRA settings in `config/local-training.json`; inference uses medium ARC and SAME-L. The current run completed 1,000 optimizer steps, rank 16, learning rate 1e-4 (1,611 seconds; 6.90 GB reported peak memory). Run ID: `slap-bass-1000-1790717937566268000`. Checkpoints are under `models/runs/slap-bass-1000-1790717937566268000/d62c9775/checkpoints/`; logs are under `output/training/`.

Step-500 adapter SHA-256: `bcf84019d4b551b00285e54aa8cf59a8484956752d53df50b760849a59b3cfca`.
Step-1000 adapter SHA-256: `218b9a841ed4208fe26a55b1f30301dfe42511c3671fba008508abd6ff2515c2`.

The preferred generation configuration is `config/generation.json` (medium, SAME-L, 3.6 seconds, 8 inference steps, seed 307, step-1000 adapter). Run `.runtime/mlx-venv/bin/python scripts/generate.py`; generated WAVs go to `output/generated/`. Previous audition WAVs were removed from the project. Seed 307 remains configured so the preferred setup can be reproduced.

The source and clip WAV bytes were preserved during the directory rename (source SHA-256 remains `417c16cbfa0145c3040e0b8b59a4cb9d28c6e6c7069b16ebbb24799bea0a72dc`). Dataset manifest paths were updated to the clearer layout, so the current manifest checksum differs from the historical value recorded by the completed training run; this is a path/metadata change, not a re-encoded or retrained dataset.

The old multi-source pilot was rejected for poor resemblance and is not an accepted model. Its records/checkpoints are retained outside this working project in the local legacy archive.

## Limits

This is fine-tuning, not training from scratch. One four-minute compilation does not guarantee genuine style learning, generalization, unique outputs, or realistic output. Training/holdout loss is not a listening score. Source and generated-output rights are unverified. Keep the experiment private; do not upload the recording, segments, adapter or generated WAVs until rights and perceptual quality have been reviewed.
