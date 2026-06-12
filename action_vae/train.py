from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from action_vae.dataset import ActionOnlyLeRobotDataset, compute_action_normalization
from action_vae.model import ActionChunkVAE, action_vae_loss


def _load_cfg(path: str, overrides: list[str]):
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_checkpoint(
    path: Path,
    model: ActionChunkVAE,
    optimizer: torch.optim.Optimizer,
    cfg,
    step: int,
    epoch: int,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "action_mean": mean.detach().cpu() if mean is not None else None,
        "action_std": std.detach().cpu() if std is not None else None,
    }
    torch.save(payload, path)


def _evaluate(model: ActionChunkVAE, loader: DataLoader, device: torch.device, beta: float, max_batches: int):
    model.eval()
    totals = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            action = batch["action"].to(device=device, dtype=torch.float32)
            action_is_pad = batch["action_is_pad"].to(device=device)
            output = model(action, action_is_pad=action_is_pad)
            _, metrics = action_vae_loss(output, action, action_is_pad=action_is_pad, beta=beta)
            for key in totals:
                totals[key] += metrics[key]
            count += 1
    model.train()
    if count == 0:
        return {key: float("nan") for key in totals}
    return {key: value / count for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Train a standalone action chunk VAE.")
    parser.add_argument("--config", default="action_vae/libero_action_vae.yaml")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. train.batch_size=512")
    args = parser.parse_args()

    cfg = _load_cfg(args.config, args.overrides)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    action_meta = OmegaConf.to_container(cfg.data.action_meta, resolve=True)
    train_dataset = ActionOnlyLeRobotDataset(
        dataset_dirs=list(cfg.data.dataset_dirs),
        action_meta=action_meta,
        chunk_len=int(cfg.data.chunk_len),
        global_sample_stride=int(cfg.data.global_sample_stride),
        val_set_proportion=float(cfg.data.val_set_proportion),
        is_training_set=True,
        seed=int(cfg.seed),
    )

    mean = None
    std = None
    if bool(cfg.data.normalize):
        print("[action_vae] computing action normalization stats")
        mean, std = compute_action_normalization(
            train_dataset,
            batch_size=int(cfg.train.stats_batch_size),
            num_workers=int(cfg.train.num_workers),
        )
        train_dataset.set_normalization(mean, std)
        torch.save({"mean": mean, "std": std}, output_dir / "action_norm.pt")

    val_loader = None
    if float(cfg.data.val_set_proportion) > 1e-6:
        val_dataset = ActionOnlyLeRobotDataset(
            dataset_dirs=list(cfg.data.dataset_dirs),
            action_meta=action_meta,
            chunk_len=int(cfg.data.chunk_len),
            global_sample_stride=int(cfg.data.global_sample_stride),
            val_set_proportion=float(cfg.data.val_set_proportion),
            is_training_set=False,
            seed=int(cfg.seed),
            mean=mean,
            std=std,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(cfg.train.batch_size),
            shuffle=False,
            num_workers=int(cfg.train.num_workers),
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    device = _device()
    model = ActionChunkVAE(
        action_dim=train_dataset.action_dim,
        chunk_len=int(cfg.data.chunk_len),
        latent_tokens=int(cfg.model.latent_tokens),
        latent_dim=int(cfg.model.latent_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        dropout=float(cfg.model.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
        foreach=False,
    )

    max_steps = int(cfg.train.max_steps)
    beta = float(cfg.train.beta)
    log_every = int(cfg.train.log_every)
    save_every = int(cfg.train.save_every)
    eval_every = int(cfg.train.eval_every)
    grad_clip = float(cfg.train.grad_clip)

    metadata = {
        "action_dim": train_dataset.action_dim,
        "chunk_len": int(cfg.data.chunk_len),
        "latent_tokens": int(cfg.model.latent_tokens),
        "latent_dim": int(cfg.model.latent_dim),
        "dataset_size": len(train_dataset),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        "[action_vae] "
        f"device={device} size={len(train_dataset)} action_dim={train_dataset.action_dim} "
        f"chunk_len={cfg.data.chunk_len} latent_tokens={cfg.model.latent_tokens} max_steps={max_steps}"
    )

    step = 0
    epoch = 0
    start = time.perf_counter()
    model.train()
    while step < max_steps:
        for batch in train_loader:
            action = batch["action"].to(device=device, dtype=torch.float32)
            action_is_pad = batch["action_is_pad"].to(device=device)

            output = model(action, action_is_pad=action_is_pad)
            loss, metrics = action_vae_loss(output, action, action_is_pad=action_is_pad, beta=beta)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            step += 1
            if log_every > 0 and step % log_every == 0:
                elapsed = max(time.perf_counter() - start, 1e-6)
                speed = step / elapsed
                eta = int((max_steps - step) / max(speed, 1e-9))
                eta_h, eta_rem = divmod(eta, 3600)
                eta_m, eta_s = divmod(eta_rem, 60)
                print(
                    "[action_vae] "
                    f"epoch={epoch} step={step}/{max_steps} "
                    f"loss={metrics['loss']:.6f} recon={metrics['recon_loss']:.6f} kl={metrics['kl_loss']:.6f} "
                    f"speed={speed:.2f} step/s eta={eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
                )

            if val_loader is not None and eval_every > 0 and step % eval_every == 0:
                val_metrics = _evaluate(
                    model,
                    val_loader,
                    device=device,
                    beta=beta,
                    max_batches=int(cfg.train.eval_batches),
                )
                print(
                    "[action_vae][eval] "
                    f"step={step} loss={val_metrics['loss']:.6f} "
                    f"recon={val_metrics['recon_loss']:.6f} kl={val_metrics['kl_loss']:.6f}"
                )

            if save_every > 0 and step % save_every == 0:
                _save_checkpoint(output_dir / f"step_{step:06d}.pt", model, optimizer, cfg, step, epoch, mean, std)

            if step >= max_steps:
                break
        epoch += 1

    _save_checkpoint(output_dir / "final.pt", model, optimizer, cfg, step, epoch, mean, std)
    print(f"[action_vae] saved final checkpoint to {output_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
