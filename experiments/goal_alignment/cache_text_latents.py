from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

from common import (
    DEFAULT_DATASET_DIR,
    DEFAULT_PROMPT,
    atomic_torch_save,
    read_metadata,
    text_cache_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen Wan/T5 text latents for goal alignment.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/goal_alignment/text_latents"))
    parser.add_argument("--model-id", type=str, default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--tokenizer-model-id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--redirect-common-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-template", type=str, default=DEFAULT_PROMPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_metadata(args.dataset_dir, prompt_template=args.prompt_template)
    prompts = sorted({record.prompt for record in records})

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        redirect_common_files=bool(args.redirect_common_files),
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    dtype = torch.bfloat16
    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=dtype,
        device=args.device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=int(args.context_len),
        clean="whitespace",
    )

    to_encode = []
    for prompt in prompts:
        path = text_cache_path(args.output_dir, prompt, args.context_len)
        if args.overwrite or not path.exists():
            to_encode.append(prompt)

    print(f"records={len(records)} unique_prompts={len(prompts)} to_encode={len(to_encode)} output={args.output_dir}")
    with torch.no_grad():
        for start in tqdm(range(0, len(to_encode), args.batch_size), desc="Caching text latents"):
            batch_prompts = to_encode[start : start + args.batch_size]
            ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
            ids = ids.to(args.device)
            mask = mask.to(device=args.device, dtype=torch.bool)
            context = text_encoder(ids, mask)
            for i, prompt in enumerate(batch_prompts):
                payload = {
                    "prompt": prompt,
                    "context": context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                    "mask": mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                    "model_id": args.model_id,
                    "tokenizer_model_id": args.tokenizer_model_id,
                    "context_len": int(args.context_len),
                }
                atomic_torch_save(payload, text_cache_path(args.output_dir, prompt, args.context_len))


if __name__ == "__main__":
    main()
