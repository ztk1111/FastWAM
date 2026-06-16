# Action VAE

Standalone action-only VAE training for FastWAM-style action chunks.

For the current LIBERO IDM setup, FastWAM uses `num_frames=33`, so one action chunk is
`num_frames - 1 = 32` actions. This VAE compresses `[32, action_dim]` into 3 latent tokens.

Run:

```bash
python -m action_vae.train --config action_vae/libero_action_vae.yaml
```

Common overrides:

```bash
python -m action_vae.train --config action_vae/libero_action_vae.yaml train.batch_size=256 train.max_steps=10000
```

Outputs are written to `output_dir`:

- `final.pt`: model, optimizer, config, step, and optional action normalization stats.
- `step_*.pt`: periodic checkpoints.
- `action_norm.pt`: action mean/std when `data.normalize=true`.
- `metadata.json`: chunk and latent dimensions.

This path uses only action keys from LeRobot and sets `during_training=False`, so it does not decode videos.
