"""Matched-noise held-out RF loss before/after the adapter. Not an audio realism score."""
import argparse
import importlib
import json
import sys

import numpy as np

from local_model import CONFIG, DATA, ROOT, RUNTIME, digest, save_json, verify_runtime


def evaluate(checkpoint):
    verify_runtime()
    sys.path.insert(0, str(RUNTIME / "scripts"))
    import lora_train_mlx as upstream
    import mlx.core as mx
    from pathlib import Path
    checkpoint = Path(checkpoint).resolve()
    state, adapter_config = upstream.load_lora_checkpoint(checkpoint)
    window = CONFIG["latent_crop_length"]
    weights = str(RUNTIME / "models/mlx/dit_medium-base_f16.npz")
    loader = importlib.import_module(upstream.DIT_CHOICES["medium"]["loader"])
    model = loader.load_dit(weights, T_lat=window, dtype=mx.float16, compile_=False)
    padding, seconds = upstream.load_conditioner_from_npz(weights, prefix="cond.")
    text = upstream.T5Gemma.from_npz(str(RUNTIME / upstream.T5GEMMA_NPZ_REL))
    prompt = upstream.build_conditioning(text, padding, [CONFIG["prompt"]])
    mx.eval(prompt)
    del text
    mx.clear_cache()
    upstream.inject_from_lora_config(model, adapter_config, checkpoint_prefix="model.")
    sec = upstream.TrainableSecondsEmbedder(seconds.W, seconds.b)
    upstream.inject_from_lora_config(sec, adapter_config, checkpoint_prefix="conditioners.seconds_total.")
    bundle = upstream.TrainBundle(model, sec)
    local_cond = mx.array(np.concatenate([np.ones((1, 1, window), np.float32),
                                         np.zeros((1, 256, window), np.float32)], axis=1)
                          .transpose(0, 2, 1)).astype(mx.float16)
    dataset = upstream.PreEncodedLatentDataset(DATA / "latents/holdout", window,
               random_crop=False, prompt_config=json.loads((ROOT / "config/prompts.json").read_text()), seed=123)
    results = {}
    for phase in ("base", "trained"):
        if phase == "trained":
            restored = upstream.load_trainable_lora_state(bundle, state)
            print(f"Restored {restored} adapter layers", flush=True)
        rows = []
        for i, batch in enumerate(upstream.iterate_batches(dataset, 1, shuffle=False, seed=123)):
            clean = mx.array(batch["latents"]).astype(mx.float16)
            mask = mx.array(batch["padding_mask"])
            token = bundle.secs(mx.array(np.asarray(batch["seconds_total"], dtype=np.float32)))
            cross = mx.concatenate([prompt, token.astype(mx.float32)], axis=1).astype(mx.float16)
            global_cond = token[:, 0, :].astype(mx.float16)
            for t in (0.2, 0.5, 0.8):
                noise = mx.array(np.random.default_rng(900 + i).normal(size=clean.shape).astype(np.float32)).astype(mx.float16)
                def forward(noised, time):
                    return bundle.dit(noised, time.astype(noised.dtype), cross, global_cond,
                                      local_add_cond=local_cond)
                loss = upstream.rectified_flow_loss(forward, clean, mx.array([t]), noise=noise, loss_mask=mask)
                value = float(loss.item())
                if not np.isfinite(value):
                    raise RuntimeError("Nonfinite evaluation loss")
                rows.append({"holdout_index": i, "t": t, "loss": value})
                print(phase, i, t, value, flush=True)
        results[phase] = {"mean_loss": float(np.mean([r["loss"] for r in rows])), "rows": rows}
    results["adapter_sha256"] = digest(checkpoint)
    results["relative_loss_change"] = results["trained"]["mean_loss"] / results["base"]["mean_loss"] - 1
    results["limits"] = "Three file-level holdout cues, three fixed diffusion timesteps, one noise draw per cue. Near-duplicate sources possible. Not a perceptual realism or generalization benchmark."
    save_json(ROOT / "output/evaluation/holdout.json", results)
    print(json.dumps({k: v["mean_loss"] for k, v in results.items() if k in ("base", "trained")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    evaluate(parser.parse_args().checkpoint)
