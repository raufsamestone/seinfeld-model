# Third-party provenance

## Active model and runtime

- Runtime: https://github.com/Stability-AI/stable-audio-3 at
  3a82c807b69cf4b7c5c05270011a5d5e47abac18, optimized/mlx/.
- Weights: https://huggingface.co/stabilityai/stable-audio-3-optimized at
  da6edc54ddba10bfd79a077102ded687f80e882b.
- Training: Stable Audio 3 medium BASE. Inference: medium ARC.
- Codec: SAME-L. Text encoder: Google's T5Gemma.
- Stable Audio weights: Stability AI Community License, https://stability.ai/license .
  T5Gemma has its own upstream terms.
- Not covered by this repository's MIT code license. Preserve required upstream
  notices/license files when distributing an adapter/package.
- Runtime versions: requirements-mlx.txt.

## Private single-source research recording

The active experiment uses one local file:
`research/sources/seinfeld-transition-music-only.wav`, from the YouTube video
“Seinfeld Scene Transition Music (Music Only) - v3” (video ID `B7IRCudlYPg`).
The video ID is provenance only, not a project directory name. Its SHA-256 is
recorded in the derived dataset manifest. Its license, training rights, and
generated-output publication rights are unverified. A public YouTube download
is not a grant of those rights.

Other downloaded sources and the earlier multi-source pilot are not inputs to
the active run. No source audio, derived segments, or model adapter is uploaded
or bundled with the website. Before public release, resolve source/output
rights and have a human review the output for similarity and memorization.

## Retired experiments

The archived Markov/CC0 sampler, JohnSlap pack and MusicGen continuation code
are not dependencies of the local fine-tuning pipeline. Their original
attribution files survive in the legacy archive and repository history.
They must not be presented as this trained model.
