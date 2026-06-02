"""
预编码 LIBERO 所有任务的文本 prompt，保存为 T5 嵌入缓存文件。

使用方法:
    python scripts/precompute_libero_text_embeds.py \
        --output-dir checkpoints/libero_text_embeds \
        --context-len 128

说明:
    - T5 加载到 CPU（e.g., 使用 ~22GB 系统内存），不占用 GPU 显存
    - 遍历所有 LIBERO 套件的全部任务，去重后批量编码
    - 生成与训练数据缓存相同格式的 .pt 文件
    - eval 时直接加载缓存，无需加载 T5 到 GPU
"""

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# 确保可以 import fastwam
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CONTEXT_LEN = 128


def _model_id_to_enc_id(model_id: str) -> str:
    """从 model_id 派生出文本编码器 ID，与训练缓存文件名保持一致。"""
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def main():
    parser = argparse.ArgumentParser(
        description="预编码 LIBERO 所有任务 prompt 的 T5 文本嵌入"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="缓存输出目录（e.g., checkpoints/libero_text_embeds）"
    )
    parser.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-model-id", default=DEFAULT_TOKENIZER_MODEL_ID)
    parser.add_argument(
        "--suite-names", nargs="+",
        default=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="要编码的 LIBERO 套件名称列表"
    )
    parser.add_argument("--batch-size", type=int, default=16,
                        help="T5 CPU 编码批大小")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enc_id = _model_id_to_enc_id(args.model_id)

    # ---- 1. 加载 T5 到 CPU ----
    print("[1/4] Loading T5 encoder on CPU ...")
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        redirect_common_files=True,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device="cpu",
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=args.context_len,
        clean="whitespace",
    )
    print("  T5 encoder loaded on CPU.")

    # ---- 2. 枚举所有 LIBERO 任务并收集唯一 prompt ----
    print("[2/4] Enumerating LIBERO tasks ...")
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()

    # 去重：相同 prompt 只编码一次，但记录其在哪些任务中出现
    prompt_to_tasks = {}   # prompt → [(suite_name, task_id), ...]
    all_tasks = []         # [(suite_name, task_id, prompt), ...]

    for suite_name in args.suite_names:
        if suite_name not in benchmark_dict:
            print(f"  [WARN] Suite '{suite_name}' not found in LIBERO benchmark, skipping.")
            continue
        task_suite = benchmark_dict[suite_name]()
        n_tasks = task_suite.n_tasks
        for task_id in range(n_tasks):
            task = task_suite.get_task(task_id)
            task_language = task.language  # 自然语言指令
            prompt = DEFAULT_PROMPT.format(task=task_language)
            all_tasks.append((suite_name, task_id, prompt))

            if prompt not in prompt_to_tasks:
                prompt_to_tasks[prompt] = []
            prompt_to_tasks[prompt].append((suite_name, task_id))

    unique_prompts = list(prompt_to_tasks.keys())
    print(f"  Total tasks: {len(all_tasks)}, unique prompts: {len(unique_prompts)}")

    # ---- 3. 批量编码 ----
    print("[3/4] Encoding prompts on CPU ...")
    skipped = 0
    encoded = 0

    for start in tqdm(range(0, len(unique_prompts), args.batch_size),
                      desc="Encoding", unit="batch"):
        batch_prompts = unique_prompts[start:start + args.batch_size]

        with torch.no_grad():
            ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
            ids = ids.to("cpu")
            mask = mask.to(device="cpu", dtype=torch.bool)
            context = text_encoder(ids, mask)  # [B, L, 4096]

            for j, prompt in enumerate(batch_prompts):
                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                cache_path = output_dir / f"{prompt_hash}.t5_len{args.context_len}.{enc_id}.pt"

                if cache_path.exists():
                    skipped += 1
                    continue

                context_j = context[j].detach().to(dtype=torch.bfloat16).contiguous()
                mask_j = mask[j].detach().to(dtype=torch.bool).contiguous()
                payload = {
                    "context": context_j,    # [L, 4096]
                    "mask": mask_j,          # [L]
                }
                torch.save(payload, str(cache_path))
                encoded += 1

    # ---- 4. 保存 prompt → hash 映射（eval 时快速查找） ----
    print("[4/4] Saving prompt→hash mapping ...")
    import json
    mapping = {}
    for prompt, tasks in prompt_to_tasks.items():
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for suite_name, task_id in tasks:
            key = f"{suite_name}__{task_id}"
            mapping[key] = {
                "hash": prompt_hash,
                "prompt": prompt,
                "cache_file": f"{prompt_hash}.t5_len{args.context_len}.{enc_id}.pt",
            }

    mapping_path = output_dir / "prompt_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Encoded={encoded}, skipped={skipped}, total_tasks={len(all_tasks)}")
    print(f"Caches saved to: {output_dir}")
    print(f"Mapping saved to: {mapping_path}")


if __name__ == "__main__":
    main()
