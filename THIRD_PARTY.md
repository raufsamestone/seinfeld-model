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
- The published adapter is an experimental rank-16 DoRA fine-tune derived from
  Stable Audio 3 Medium. Its distribution follows the Stability AI Community
  License; see `NOTICE` and `STABILITY_COMMUNITY_LICENSE.md`. The repository's
  MIT license covers code only.
- Runtime versions: requirements-mlx.txt.

## Private single-source research recording

The active experiment uses one local file:
`research/sources/seinfeld-transition-music-only.wav`, from the YouTube video
“Seinfeld Scene Transition Music (Music Only) - v3” (video ID `B7IRCudlYPg`).
The video ID is provenance only, not a project directory name. Its SHA-256 is
recorded in the derived dataset manifest. Its license and training rights have
not been independently verified. A public YouTube download is not a grant of
those rights. No source audio or derived training clips are included in Git or
in the model release. Publishing adapter weights does not resolve rights in the
material used to train them.

Other downloaded sources and the earlier multi-source pilot are not inputs to
the active run. Source audio and derived segments are not uploaded or bundled.
The adapter is distributed separately as a GitHub Release asset. The two curated generated examples in
`samples/` are published for listening. Source and output rights have not been
independently verified; review them before distributing additional material.

## Retired experiments

The archived Markov/CC0 sampler, JohnSlap pack and MusicGen continuation code
are not dependencies of the local fine-tuning pipeline. Their original
attribution files survive in the legacy archive and repository history.
They must not be presented as this trained model.
