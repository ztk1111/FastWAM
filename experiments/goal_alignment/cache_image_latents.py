from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs

from common import (
    DEFAULT_DATASET_DIR,
    DEFAULT_PROMPT,
    atomic_torch_save,
    image_cache_path,
    preprocess_goal_image,
    read_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen Wan VAE latents for final-frame goal images.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/goal_alignment/image_latents"))
    parser.add_argument("--model-id", type=str, default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--tokenizer-model-id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--redirect-common-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-template", type=str, default=DEFAULT_PROMPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_metadata(args.dataset_dir, prompt_template=args.prompt_template)

    _, _, vae_config, _ = _resolve_configs(
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        redirect_common_files=bool(args.redirect_common_files),
    )
    vae_config.download_if_necessary()
    dtype = torch.bfloat16
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=dtype,
        device=args.device,
    ).eval()

    cached = 0
    skipped = 0
    with torch.no_grad():
        for record in tqdm(records, desc="Caching image latents"):
            path = image_cache_path(args.output_dir, record)
            if path.exists() and not args.overwrite:
                skipped += 1
                continue
            image_tensor = preprocess_goal_image(record.image_path, height=args.height, width=args.width)
            latent = vae.encode(
                [image_tensor.to(device=args.device, dtype=dtype)],
                device=args.device,
                tiled=False,
            )[0]
            payload = {
                "latent": latent.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                "image_path": str(record.image_path),
                "task": record.task,
                "task_index": record.task_index,
                "episode_index": record.episode_index,
                "frame_index": record.frame_index,
                "height": int(args.height),
                "width": int(args.width),
                "model_id": args.model_id,
            }
            atomic_torch_save(payload, path)
            cached += 1

    print(f"records={len(records)} cached={cached} skipped={skipped} output={args.output_dir}")


if __name__ == "__main__":
    main()
