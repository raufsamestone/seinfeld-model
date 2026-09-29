<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/0d/Seinfeld.svg" alt="Seinfeld" width="330">
</p>

<h1 align="center">Seinfeld Bass</h1>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="Stable Audio 3 Medium" src="https://img.shields.io/badge/Stable%20Audio-3%20Medium-6A5ACD">
  <img alt="Apple MLX" src="https://img.shields.io/badge/Apple-MLX-111111?logo=apple&logoColor=white">
  <img alt="DoRA rank 16" src="https://img.shields.io/badge/Adapter-DoRA%20%7C%20rank%2016-D97706">
</p>

Local fine-tuning and generation experiments for short, Seinfeld-inspired slap-bass cues.

## Samples

- [Sample 1 · seed 307 · 3.6 seconds](https://raw.githubusercontent.com/raufsamestone/seinfeld-model/main/samples/seinfeld-1.wav)
- [Sample 2 · seed 410 · 3.6 seconds](https://raw.githubusercontent.com/raufsamestone/seinfeld-model/main/samples/seinfeld-2.wav)

More samples are on the [Seinfeld showcase](https://berkay.fyi/seinfeld).

## Model and training

Stable Audio 3 Medium was fine-tuned locally on Apple Silicon with Python, MLX and a rank-16 DoRA adapter. The experiment used one 244-second WAV split into 67 non-overlapping clips: 60 training and 7 holdout. Training ran for 1,000 steps at a `1e-4` learning rate. Inference uses the SAME-L codec and T5Gemma text encoder.

This is adapter fine-tuning, not training a foundation model from scratch. The training recording, derived clips and adapter are not bundled. The two sample WAVs are in `samples/`. See [`research/local-training.md`](research/local-training.md) and [`THIRD_PARTY.md`](THIRD_PARTY.md) for the workflow and provenance.

## Generate locally

Requires an Apple Silicon Mac, Python 3.12, `uv`, FFmpeg and FFprobe.

```sh
git clone https://github.com/Stability-AI/stable-audio-3.git .runtime/stable-audio-3
git -C .runtime/stable-audio-3 checkout 3a82c807b69cf4b7c5c05270011a5d5e47abac18
uv venv --python 3.12 .runtime/mlx-venv
uv pip install --python .runtime/mlx-venv/bin/python -r requirements-mlx.txt
.runtime/mlx-venv/bin/python scripts/local_model.py download
```

Provide your locally trained adapter at the path in `config/generation.json`, edit that config, then generate:

```sh
.runtime/mlx-venv/bin/python scripts/generate.py
```

The WAV is written under `output/generated/`. Training and generation settings live in `config/`.

## Checks

```sh
python3 -m unittest discover -s tests -v
```

See [`LICENSE`](LICENSE) for this repository's code license. Third-party model terms and media rights are separate.
