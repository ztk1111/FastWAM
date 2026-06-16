# Goal Alignment Sandbox

This folder is a small standalone experiment for aligning frozen Wan/T5 text
latents with frozen Wan VAE final-image latents on:

```text
/data/ztk/datasets/libero_goal_image/episode_last_frames
```

The experiment does not train T5 or VAE. It trains only lightweight projection
heads:

```text
T5 context tokens      -> learned text queries  -> g_text [M,D]
final-frame VAE latent -> learned visual queries -> g_vis  [M,D]
```

The current training loss keeps the original positive-pair alignment, but adds
simple anti-collapse terms:

```text
loss = pooled_cosine(mean(g_text), mean(g_vis))
     + 0.1 * pooled_mse(mean(g_text), mean(g_vis))
     + set_alignment(g_text, g_vis)
     + 0.1 * variance(flatten(g_text), flatten(g_vis))
     + 0.05 * token_diversity(g_text, g_vis)
     + 0.5 * task_ce(mean(g_text), task_index)
     + 0.5 * task_ce(mean(g_vis), task_index)
```

No InfoNCE is used because one task text can correspond to many visually
different final frames.

## Full LIBERO Dataset

For the full LIBERO LeRobot dataset, first materialize one final-frame goal
image per episode. This follows the FastWAM LIBERO camera convention by taking
the final frame from `observation.images.image` and
`observation.images.wrist_image`, horizontally concatenating them, and writing a
metadata file with globally unique task ids across all suites.

```bash
python experiments/goal_alignment/materialize_libero_goal_images.py \
  --libero-root /data/ztk/code/FastWAM/data/libero_mujoco3.3.2 \
  --output-dir runs/goal_alignment/libero_episode_last_frames
```

For a quick smoke test:

```bash
python experiments/goal_alignment/materialize_libero_goal_images.py \
  --libero-root /data/ztk/code/FastWAM/data/libero_mujoco3.3.2 \
  --output-dir runs/goal_alignment/libero_episode_last_frames_smoke \
  --max-episodes-per-suite 2
```

Use the materialized directory as `--dataset-dir` in the following steps.

## 1. Cache Text Latents

This defaults to CPU so it can run without occupying GPU memory.

```bash
python experiments/goal_alignment/cache_text_latents.py \
  --dataset-dir runs/goal_alignment/libero_episode_last_frames \
  --output-dir runs/goal_alignment/text_latents \
  --device cpu \
  --context-len 128
```

## 2. Cache Final-Image VAE Latents

VAE encoding is much faster on GPU, but `--device cpu` also works if needed.

```bash
python experiments/goal_alignment/cache_image_latents.py \
  --dataset-dir runs/goal_alignment/libero_episode_last_frames \
  --output-dir runs/goal_alignment/image_latents \
  --height 224 \
  --width 448 \
  --device cuda
```

## 3. Train Alignment Heads

```bash
python experiments/goal_alignment/train_alignment.py \
  --dataset-dir runs/goal_alignment/libero_episode_last_frames \
  --text-cache-dir runs/goal_alignment/text_latents \
  --image-cache-dir runs/goal_alignment/image_latents \
  --output-dir runs/goal_alignment/alignment \
  --epochs 60 \
  --batch-size 32
```


## 4. Sanity Check

After training, run the collapse and task-cluster checks:

```bash
python experiments/goal_alignment/sanity_check.py \
  --ckpt runs/goal_alignment/alignment/best.pt
```

For the full LIBERO token-bank run:

```bash
python experiments/goal_alignment/sanity_check.py \
  --ckpt runs/goal_alignment/alignment_full_libero_tokens/best.pt
```

The script reads `dataset_dir`, cache dirs, `context_len`, `goal_dim`, and
`num_goal_tokens` from the checkpoint config by default. Useful signs:

```text
set text-image cosine: high is better for token-bank alignment
text/image offdiag cosine: should not be near 1.0
text/image token mean feature std: should not collapse toward 0
cross text->image same-task acc: high means text goals retrieve matching visual goals
```

Outputs:

```text
runs/goal_alignment/alignment/
├── config.json
├── history.jsonl
├── latest.pt
└── best.pt
```

The learned `GoalProjector` can later be reused to append a compact goal token bank
to FastWAM/Wan context:

```text
goal_tokens = encode_text(t5_context, context_mask)  # [B, M, 4096]
context' = concat([t5_context, goal_tokens], dim=1)
```

For this first probe, the VAE branch uses channel-pooled final-frame latents.
If this signal looks useful, the next step is to replace `P_vis(pool(VAE))` with
a projector over final-frame video-DiT tokens, which is closer to the action
conditioning path used by IDM.
