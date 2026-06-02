"""
文本嵌入预计算脚本 —— 使用 T5 文本编码器批量编码提示词并缓存结果。

该脚本预先计算所有数据集中任务指令（tasks.jsonl）对应的 T5 文本嵌入，
并将其缓存到磁盘，避免训练时重复运行文本编码器。

核心功能：
    1. 扫描数据集目录，收集所有唯一的任务提示词
    2. 使用 Wan2.2 的 T5 文本编码器批量编码提示词
    3. 将编码结果（context + mask）通过 SHA256 哈希保存到指定缓存目录
    4. 支持分布式编码（多 GPU torchrun）
    5. 支持覆盖模式（overwrite）和跳过已缓存条目（overwrite=false）
    6. 支持 override_instruction 模式，仅编码单个指令提示词
    7. 原子写入（先写入 .tmp 文件再 rename，防止文件损坏）

用法:
    # 单卡
    python scripts/precompute_text_embeds.py

    # 多卡分布式
    torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py

    # 仅编码一个指令，跳过数据集扫描
    python scripts/precompute_text_embeds.py override_instruction="pick up the object"

配置依赖:
    - cfg.data: 数据集配置（必须包含 text_embedding_cache_dir）
    - cfg.model: 模型配置（model_id, tokenizer_model_id 等）
    - cfg.overwrite: 是否覆盖已有缓存
"""

import hashlib
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

# 默认的模型和编码参数常量
DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_BATCH_SIZE = 16


def _init_distributed():
    """
    初始化分布式训练环境（如果配置了 WORLD_SIZE > 1）。

    通过环境变量 WORLD_SIZE 和 LOCAL_RANK 检测分布式环境，
    使用 NCCL（GPU）或 Gloo（CPU）后端初始化进程组。

    返回:
        tuple: (is_distributed, rank, world_size, local_rank)
            - is_distributed (bool): 是否为分布式模式
            - rank (int): 当前进程的全局 rank
            - world_size (int): 总进程数
            - local_rank (int): 当前进程的本地 rank
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _to_bool(value: Any) -> bool:
    """
    将各种格式的值解析为布尔值。

    支持字符串形式的 "1"/"0", "true"/"false", "yes"/"no", "y"/"n",
    也直接接受 Python 的 bool 类型。

    参数:
        value (Any): 要解析的值

    返回:
        bool: 布尔结果

    异常:
        ValueError: 无法解析的值
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _iter_dataset_nodes(node: Any, path: str = "data"):
    """
    递归遍历配置树，找到所有包含 "dataset_dirs" 字段的数据集节点。

    用于从复杂的嵌套配置中提取所有数据集配置路径。

    参数:
        node (Any): 当前配置节点（DictConfig 或 ListConfig）
        path (str): 当前路径字符串，用于日志和错误定位

    产生:
        tuple: (node_path, node) 其中 node_path 是配置路径，node 是包含 dataset_dirs 的节点
    """
    if isinstance(node, DictConfig):
        if "dataset_dirs" in node and node.get("dataset_dirs") is not None:
            yield path, node
        for key, value in node.items():
            yield from _iter_dataset_nodes(value, f"{path}.{key}")
    elif isinstance(node, ListConfig):
        for idx, value in enumerate(node):
            yield from _iter_dataset_nodes(value, f"{path}[{idx}]")


def _collect_dataset_settings(data_cfg: DictConfig):
    """
    收集所有数据集配置中的数据集目录、缓存目录和上下文长度。

    从嵌套的数据配置中递归搜索，收集每个数据节点的配置信息。

    参数:
        data_cfg (DictConfig): 数据配置对象

    返回:
        tuple: (dataset_dirs, cache_dirs, context_lens)
            - dataset_dirs (list[str]): 去重后的数据集目录列表
            - cache_dirs (list[Path]): 去重后的文本嵌入缓存目录列表
            - context_lens (set[int]): 所有上下文的长度集合

    异常:
        ValueError: 节点定义了 dataset_dirs 但未定义 text_embedding_cache_dir
    """
    dataset_dirs: list[str] = []
    cache_dirs: list[Path] = []
    context_lens = set()

    for node_path, node in _iter_dataset_nodes(data_cfg, path="data"):
        raw_dirs = node.get("dataset_dirs")
        if raw_dirs is None:
            continue

        cache_dir = node.get("text_embedding_cache_dir")
        if cache_dir is None or not str(cache_dir).strip():
            raise ValueError(
                f"Missing `text_embedding_cache_dir` for dataset node `{node_path}` "
                "(this node defines `dataset_dirs`)."
            )

        for ds in raw_dirs:
            ds_str = str(ds)
            if ds_str not in dataset_dirs:
                dataset_dirs.append(ds_str)

        cache_dir_path = Path(str(cache_dir)).expanduser()
        if cache_dir_path not in cache_dirs:
            cache_dirs.append(cache_dir_path)

        context_len = node.get("context_len")
        if context_len is not None:
            context_lens.add(int(context_len))

        logger.info("Discovered dataset node `%s` with %d dataset_dirs.", node_path, len(raw_dirs))

    return dataset_dirs, cache_dirs, context_lens


def _resolve_context_len(context_lens: set[int]) -> int:
    """
    检查并返回唯一的上下文长度值。

    所有数据集节点的 context_len 必须一致，否则抛出错误。

    参数:
        context_lens (set[int]): 所有节点 context_len 值的集合

    返回:
        int: 唯一的上下文长度值

    异常:
        ValueError: 存在多个不同的 context_len 值
    """
    if len(context_lens) != 1:
        raise ValueError(
            f"Found multiple context_len values in data config: {sorted(context_lens)}. "
            "Please keep them consistent."
        )
    return next(iter(context_lens))


def _read_unique_prompts(dataset_dirs: list[str]) -> list[str]:
    """
    从所有数据集目录的 meta/tasks.jsonl 文件中读取并去重任务提示词。

    每个任务的原始 task 字符串会被用 DEFAULT_PROMPT 格式化（如
    "Produce a video of a robot arm that follows the instruction: {task}"），
    然后去重并返回所有唯一的完整提示词。

    输入输出示例:
        输入 dataset_dirs: ["/data/bridge", "/data/fractal"]
        读取 tasks.jsonl: {"task": "pick up the block"}, ...
        输出 prompts: ["Produce a video of...pick up the block", ...]

    参数:
        dataset_dirs (list[str]): 数据集目录列表

    返回:
        list[str]: 去重后的提示词列表

    异常:
        FileNotFoundError: tasks.jsonl 不存在
        KeyError: JSONL 行缺少 "task" 字段
    """
    prompts: list[str] = []
    seen = set()
    total_task_rows = 0

    for ds_dir in dataset_dirs:
        tasks_path = Path(ds_dir) / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            raise FileNotFoundError(f"Missing tasks file: {tasks_path}")

        with tasks_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "task" not in record:
                    raise KeyError(f"Missing `task` field at {tasks_path}:{line_idx}")
                task = str(record["task"])
                prompt = DEFAULT_PROMPT.format(task=task)
                total_task_rows += 1
                if prompt not in seen:
                    seen.add(prompt)
                    prompts.append(prompt)

    logger.info(
        "Loaded %d task rows from %d datasets, deduplicated to %d prompts.",
        total_task_rows,
        len(dataset_dirs),
        len(prompts),
    )
    return prompts


def _get_override_prompt(override_instruction: Any) -> str | None:
    """
    如果配置中指定了 override_instruction，则生成单个覆盖提示词。

    在 override 模式下，跳过数据集扫描，仅编码这一个提示词。

    参数:
        override_instruction (Any): 配置中的 override_instruction 值

    返回:
        str | None: 格式化后的提示词，如果未指定则返回 None
    """
    if override_instruction is None:
        return None
    task = str(override_instruction).strip()
    if task == "":
        return None
    return DEFAULT_PROMPT.format(task=task)


def _model_id_to_enc_id(model_id: str) -> str:
    """
    从模型 ID 中提取简洁的编码器标识符，用于缓存文件名。

    例如 "Wan-AI/Wan2.2-TI2V-5B" -> "wan22ti2v5b"

    参数:
        model_id (str): HuggingFace 模型 ID

    返回:
        str: 简化后的编码器标识符
    """
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path):
    """
    原子方式保存 torch 张量到文件。

    先写入一个临时文件（UUID 后缀），然后再用 os.replace 原子替换，
    避免写入过程中断导致文件损坏。

    参数:
        payload (dict): 包含 "context" 和 "mask" 张量的字典
        output_path (Path): 目标文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    """
    主函数：使用 T5 文本编码器批量编码任务提示词并缓存到磁盘。

    完整流程：
        1. 初始化分布式环境（如果可用）
        2. 从配置中收集数据集目录和缓存目录
        3. 从 tasks.jsonl 读取所有任务提示词（或使用 override_instruction）
        4. 加载 Wan2.2 T5 文本编码器
        5. 检查缓存状态（overwrite=false 时跳过已缓存的提示词）
        6. 在分布式模式下对提示词进行分片（每个 rank 处理一部分）
        7. 批量编码：通过 tokenizer 分词 -> T5 编码 -> 保存 context 和 mask
        8. 跨 rank 聚合统计信息（新缓存数、覆盖数、跳过数）
        9. 记录汇总日志并清理分布式进程组

    配置依赖:
        cfg.model: 包含 model_id 和 tokenizer_model_id
        cfg.data: 包含数据集配置（dataset_dirs, text_embedding_cache_dir, context_len）
        cfg.overwrite: 是否覆盖已有缓存（默认 True）
        cfg.override_instruction: 可选的单指令覆盖（跳过数据集扫描）

    缓存文件命名:
        {sha256(prompt)}.t5_len{context_len}.{enc_id}.pt
    """
    setup_logging(log_level=logging.INFO)

    # 初始化分布式环境
    is_distributed, rank, world_size, local_rank = _init_distributed()
    if is_distributed and rank == 0:
        logger.info("Distributed enabled: world_size=%d", world_size)
    if (not is_distributed) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(
            "Multi-GPU available. To use it, run: torchrun --standalone --nproc_per_node=%d scripts/precompute_text_embeds.py",
            torch.cuda.device_count(),
        )

    overwrite = _to_bool(cfg.get("overwrite", True))
    model_cfg = cfg.model
    if model_cfg is None:
        raise ValueError("`cfg.model` is required.")
    if cfg.data is None:
        raise ValueError("`cfg.data` is required.")

    # 从嵌套的数据配置中收集数据集目录和缓存位置
    dataset_dirs, cache_dirs, context_lens = _collect_dataset_settings(cfg.data)
    if not cache_dirs:
        raise ValueError("No `text_embedding_cache_dir` found under `cfg.data`.")

    context_len = _resolve_context_len(context_lens)
    override_prompt = _get_override_prompt(cfg.get("override_instruction"))
    if override_prompt is not None:
        # 覆盖模式：只编码单个指令，跳过数据集扫描
        prompts = [override_prompt]
        logger.info("Using override_instruction; skipping dataset scan and encoding exactly 1 prompt.")
    else:
        if not dataset_dirs:
            raise ValueError("No `dataset_dirs` found under `cfg.data`.")
        # 正常模式：从所有数据集的 tasks.jsonl 读取提示词
        prompts = _read_unique_prompts(dataset_dirs)
    if not prompts:
        logger.warning("No prompts found from tasks.jsonl; nothing to do.")
        return

    # 确定设备和数据类型
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    torch_dtype = torch.bfloat16
    model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
    tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID))
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
    enc_id = _model_id_to_enc_id(model_id)

    logger.info(
        "Preparing text encoder with model_id=%s tokenizer_model_id=%s device=%s dtype=%s context_len=%d overwrite=%s",
        model_id,
        tokenizer_model_id,
        device,
        torch_dtype,
        context_len,
        overwrite,
    )

    # 加载 T5 文本编码器和分词器
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch_dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=context_len,
        clean="whitespace",
    )

    # 初始化统计计数器（每个缓存目录独立统计）
    stats = {
        str(cache_dir): {"new": 0, "overwrite": 0, "skip": 0}
        for cache_dir in cache_dirs
    }

    # 分布式分片：每个 rank 只处理自己分到的提示词
    prompts = prompts[rank::world_size] if is_distributed else prompts

    # 非覆盖模式：检查哪些提示词已在所有缓存目录中存在，跳过它们
    if not overwrite:
        fully_cached_local = 0
        prompts_to_encode: list[str] = []
        for prompt in prompts:
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            filename = f"{hashed}.t5_len{context_len}.{enc_id}.pt"
            fully_cached = True
            for cache_dir in cache_dirs:
                cache_path = cache_dir / filename
                if not cache_path.exists():
                    fully_cached = False
                    break
            if fully_cached:
                fully_cached_local += 1
                for cache_dir in cache_dirs:
                    stats[str(cache_dir)]["skip"] += 1
            else:
                prompts_to_encode.append(prompt)

        prompts = prompts_to_encode

        # 跨 rank 聚合缓存统计
        fully_cached_global = fully_cached_local
        to_encode_global = len(prompts)
        if is_distributed:
            reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
            count_tensor = torch.tensor([fully_cached_local, len(prompts)], device=reduce_device, dtype=torch.long)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            fully_cached_global = int(count_tensor[0].item())
            to_encode_global = int(count_tensor[1].item())

        if (not is_distributed) or rank == 0:
            logger.info(
                "overwrite=false: fully cached prompts=%d, prompts to encode=%d",
                fully_cached_global,
                to_encode_global,
            )

    logger.info("Writing caches to %d directories.", len(cache_dirs))
    prompts_encoded_local = len(prompts)
    prompts_encoded_global = prompts_encoded_local
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        count_tensor = torch.tensor([prompts_encoded_local], device=reduce_device, dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        prompts_encoded_global = int(count_tensor.item())

    # 主编码循环：批量处理提示词
    over_length_prompts = 0
    with tqdm(
        total=len(prompts),
        desc=f"Encoding prompts (rank {rank}/{world_size})" if is_distributed else "Encoding prompts",
        unit="prompt",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(prompts), DEFAULT_BATCH_SIZE):
                batch_prompts = prompts[start : start + DEFAULT_BATCH_SIZE]
                # 分词：将文本转换为 token ID 和注意力掩码
                ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                # 统计超出最大长度的提示词（mask 全部为 True = 无 padding，即超出 context_len）
                over_length_prompts += int(mask.all(dim=1).sum().item())
                # T5 编码：输出 text_encoder embedding
                context = text_encoder(ids, mask)

                for i, prompt in enumerate(batch_prompts):
                    # 使用 SHA256 哈希作为缓存文件名，保证唯一性
                    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    context_i = context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    mask_i = mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous()
                    payload = {
                        "context": context_i,
                        "mask": mask_i,
                    }

                    # 写入所有缓存目录（支持多个数据源共享缓存）
                    for cache_dir in cache_dirs:
                        cache_path = cache_dir / f"{hashed}.t5_len{context_len}.{enc_id}.pt"
                        key = str(cache_dir)
                        if cache_path.exists() and not overwrite:
                            stats[key]["skip"] += 1
                            continue

                        if cache_path.exists():
                            stats[key]["overwrite"] += 1
                        else:
                            stats[key]["new"] += 1

                        _atomic_torch_save(payload, cache_path)

                pbar.update(len(batch_prompts))

    # 跨 rank 聚合最终统计信息
    over_length_global = over_length_prompts
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        over_tensor = torch.tensor([over_length_prompts], device=reduce_device, dtype=torch.long)
        dist.all_reduce(over_tensor, op=dist.ReduceOp.SUM)
        over_length_global = int(over_tensor.item())

        counts_tensor = torch.tensor(
            [
                [stats[str(cache_dir)]["new"], stats[str(cache_dir)]["overwrite"], stats[str(cache_dir)]["skip"]]
                for cache_dir in cache_dirs
            ],
            device=reduce_device,
            dtype=torch.long,
        )
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            for idx, cache_dir in enumerate(cache_dirs):
                key = str(cache_dir)
                stats[key]["new"] = int(counts_tensor[idx, 0].item())
                stats[key]["overwrite"] = int(counts_tensor[idx, 1].item())
                stats[key]["skip"] = int(counts_tensor[idx, 2].item())

    # 主进程输出汇总信息
    if (not is_distributed) or rank == 0:
        logger.info("Finished precomputing text embeddings.")
        logger.info(
            "Over-length prompts (mask all True, i.e. no padding after truncation/max_length=%d): %d/%d",
            context_len,
            over_length_global,
            prompts_encoded_global,
        )
        for cache_dir in cache_dirs:
            key = str(cache_dir)
            logger.info(
                "Cache dir: %s | new=%d overwrite=%d skip=%d",
                key,
                stats[key]["new"],
                stats[key]["overwrite"],
                stats[key]["skip"],
            )

    # 清理分布式进程组
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
