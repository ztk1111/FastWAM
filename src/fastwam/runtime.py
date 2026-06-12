"""
FastWAM 运行时模块 —— 模型工厂函数与训练/推理入口。

该模块提供了 FastWAM 系列模型（FastWAM, FastWAMJoint, FastWAMIDM）以及
原始 Wan22 模型的创建工厂函数，并封装了训练流程（run_training）和推理流程（run_inference）
的顶层调用逻辑。用户通常通过 Hydra/OmegaConf 配置驱动这些函数。

典型流程：
    1. 通过配置文件定义模型、数据、训练/推理参数
    2. 调用 run_training(cfg) 启动训练，或 run_inference(cfg) 执行推理
    3. 训练流程内部调用 create_fastwam / create_fastwam_joint / create_fastwam_idm 等工厂函数构建模型
"""

import logging
import os
import inspect
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf

from .trainer import Wan22Trainer
from .utils.logging_config import get_logger, setup_logging
from .utils.video_io import save_mp4
from .utils import misc

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    """
    规范化混合精度字符串配置。

    将用户传入的混合精度参数（如 "no", "fp16", "bf16"）统一转为小写并校验合法性。

    参数:
        mixed_precision (str): 原始混合精度字符串，如 "FP16", "Bf16", "no"

    返回:
        str: 规范化后的值，只能是 "no" / "fp16" / "bf16"

    异常:
        ValueError: 如果输入不是字符串或值不在合法范围内
    """
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    """
    将混合精度配置字符串映射为 PyTorch 数据类型。

    参数:
        mixed_precision (str): 混合精度类型，如 "no", "fp16", "bf16"

    返回:
        torch.dtype: 对应的 PyTorch 数据类型
            - "no"   -> torch.float32
            - "fp16" -> torch.float16
            - "bf16" -> torch.bfloat16
    """
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def create_wan22_model(
    model_id: str,
    tokenizer_model_id: str,
    dit_config,
    tokenizer_max_len: int = 512,
    train_shift: float = 5.0,
    infer_shift: float = 5.0,
    num_train_timesteps: int = 1000,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    创建原始 Wan22 扩散模型核心（Wan22Core）。

    该函数从 HuggingFace 仓库加载 Wan2.2 预训练权重，并基于传入的 DiT 配置
    构建核心模型。通常用于纯视频生成的场景，不包含动作/机械臂控制相关模块。

    参数:
        model_id (str): Wan22 模型在 HuggingFace 上的 ID，如 "Wan-AI/Wan2.2-TI2V-5B"
        tokenizer_model_id (str): 分词器模型 ID，如 "Wan-AI/Wan2.1-T2V-1.3B"
        dit_config (dict | DictConfig): DiT 模型配置字典，包含 hidden_dim, num_layers, num_heads 等
        tokenizer_max_len (int): 分词器最大序列长度，默认 512
        train_shift (float): 训练时 flow matching 的 shift 参数，默认 5.0
        infer_shift (float): 推理时 flow matching 的 shift 参数，默认 5.0
        num_train_timesteps (int): 训练时扩散步数，默认 1000
        redirect_common_files (bool): 是否重定向通用文件（缓存），默认 True
        model_dtype (torch.dtype): 模型权重数据类型，默认 torch.bfloat16
        device (str): 模型所在设备，默认 "cuda"

    返回:
        Wan22Core: 构建好的 Wan22 模型实例
    """
    from .models.wan22.wan22 import Wan22Core

    if isinstance(dit_config, DictConfig):
        dit_config = OmegaConf.to_container(dit_config, resolve=True)
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must resolve to a dict, got {type(dit_config)}")

    return Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        redirect_common_files=bool(redirect_common_files),
        dit_config=dit_config,
        train_shift=float(train_shift),
        infer_shift=float(infer_shift),
        num_train_timesteps=int(num_train_timesteps),
    )


def create_fastwam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    创建 FastWAM 模型（标准版）。

    FastWAM 是带有动作调节（action conditioning）的视频扩散模型。它在视频 DiT 的基础上
    引入了一个额外的动作 DiT（ActionDiT），用于处理机械臂动作序列的扩散建模。
    视频 DiT 和动作 DiT 通过混合注意力（Mixed Attention, MoT）共享跨模态信息。

    输入输出维度示例:
        - video_dit_config: {"hidden_dim": 1536, "num_layers": 30, "num_heads": 24, ...}
        - action_dit_config: {"hidden_dim": 128, "action_dim": 7, "num_layers": 30, ...}
        - proprio_dim: 机械臂本体感知维度，如 12（关节角 + 夹爪状态等）
        - 模型内部: 视频特征 [B, T, C, H, W] <-> 动作特征 [B, T', D]

    参数:
        model_id (str): HuggingFace 模型 ID
        tokenizer_model_id (str): 分词器模型 ID
        video_dit_config (dict | DictConfig): 视频 DiT 配置
        tokenizer_max_len (int): 分词器最大长度，默认 512
        load_text_encoder (bool): 是否加载文本编码器，默认 True
        proprio_dim (int | None): 本体感知维度，None 表示不使用
        action_dit_config (dict | DictConfig | None): 动作 DiT 配置
        action_dit_pretrained_path (str | None): 动作 DiT 预训练权重路径
        skip_dit_load_from_pretrain (bool): 是否跳过 DiT 预训练权重加载，默认 False
        video_scheduler (dict | DictConfig | None): 视频调度器参数
        action_scheduler (dict | DictConfig | None): 动作调度器参数（必需，需包含 train_shift, infer_shift, num_train_timesteps）
        loss (dict | DictConfig | None): 损失权重配置，如 {"lambda_video": 1.0, "lambda_action": 1.0}
        mot_checkpoint_mixed_attn (bool): 是否对混合注意力层使用梯度检查点，默认 True
        redirect_common_files (bool): 是否重定向通用文件，默认 True
        model_dtype (torch.dtype): 模型数据类型，默认 torch.bfloat16
        device (str): 模型所在设备，默认 "cuda"

    返回:
        FastWAM: 构建好的 FastWAM 模型实例
    """
    from .models.wan22.fastwam import FastWAM

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return FastWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


def create_fastwam_joint(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    创建 FastWAMJoint 模型（联合版）。

    FastWAMJoint 是 FastWAM 的联合训练变体，在标准 FastWAM 基础上进一步优化了
    视频 DiT 与动作 DiT 之间的信息融合方式。适用于需要更紧密耦合的视频-动作联合建模场景。

    参数:
        与 create_fastwam 相同，详见其文档。

    返回:
        FastWAMJoint: 构建好的 FastWAMJoint 模型实例
    """
    from .models.wan22.fastwam_joint import FastWAMJoint

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return FastWAMJoint.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


def create_fastwam_idm(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    goal_token: dict | None = None,
    goal_token_config: dict | None = None,
    bidirectional: dict | None = None,
    bidirectional_config: dict | None = None,
):
    """
    创建 FastWAMIDM 模型（逆动力学模型版）。

    FastWAMIDM 专注于从视频帧序列中推断动作（逆动力学建模），
    适用于需要从视频观测中提取动作指令的应用场景。
    与标准 FastWAM 相比，其动作建模模块的设计更偏向于逆推理。

    参数:
        与 create_fastwam 相同，详见其文档。

    返回:
        FastWAMIDM: 构建好的 FastWAMIDM 模型实例
    """
    from .models.wan22.fastwam_idm import (
        FastWAMIDM,
    )

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    # Goal-token conditioning is intentionally disabled for this IDM training variant.
    goal_token_config = None

    if isinstance(bidirectional, DictConfig):
        bidirectional = OmegaConf.to_container(bidirectional, resolve=True)
    if isinstance(bidirectional_config, DictConfig):
        bidirectional_config = OmegaConf.to_container(bidirectional_config, resolve=True)
    if bidirectional_config is None:
        bidirectional_config = bidirectional
    if bidirectional_config is not None and not isinstance(bidirectional_config, dict):
        raise ValueError(f"bidirectional must be dict-like, got {type(bidirectional_config)}")

    return FastWAMIDM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        goal_token_config=goal_token_config,
        subgoal_latent_config=None,
        bidirectional_config=bidirectional_config,
    )


def build_datasets(data_cfg: DictConfig):
    """
    从配置构建训练集和验证集数据集。

    如果配置中未提供验证集（data_cfg.val），则复用训练集作为验证集。
    若验证集有独立的 pretrained_norm_stats，则优先使用；否则回退到训练集或默认路径。

    参数:
        data_cfg (DictConfig): 数据配置，需包含 train 子配置，可选包含 val 子配置
            - data_cfg.train: 训练集配置（将被传递给 hydra.utils.instantiate）
            - data_cfg.val: 验证集配置（可选）

    返回:
        tuple: (train_dataset, val_dataset) 两个数据集实例
    """
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def _resolve_train_device() -> str:
    """解析训练设备，支持 CUDA/NPU 分布式训练环境。"""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        device_count = npu.device_count()
        if device_count <= 1:
            return "npu:0"
        if local_rank < 0 or local_rank >= device_count:
            return "npu:0"
        return f"npu:{local_rank}"

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count <= 1:
            return "cuda:0"
        if local_rank < 0 or local_rank >= device_count:
            return "cuda:0"
        return f"cuda:{local_rank}"

    return "cpu"


def run_training(cfg: DictConfig):
    """
    运行完整的训练流程。

    这是训练入口函数，执行以下步骤：
        1. 初始化日志和输出目录，保存配置快照
        2. 根据配置创建模型（通过 Hydra instantiate）
        3. 构建训练集和验证集
        4. 创建 Wan22Trainer 训练器并启动训练循环

    参数:
        cfg (DictConfig): 完整的 Hydra/OmegaConf 配置对象，需包含:
            - cfg.output_dir: 输出目录
            - cfg.mixed_precision: 混合精度设置
            - cfg.model: 模型配置
            - cfg.data: 数据集配置
            - cfg.learning_rate, cfg.batch_size, cfg.num_epochs 等训练超参数
    """
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()

def run_inference(cfg: DictConfig):
    """
    运行推理流程，从输入图像生成视频。

    执行以下步骤：
        1. 加载模型和可选的微调检查点
        2. 读取输入图像，进行中心裁剪并缩放到目标尺寸
        3. 将图像归一化到 [-1, 1] 范围并增加 batch 维度
        4. 调用模型进行推理生成视频
        5. 保存生成的视频为 MP4 文件

    参数:
        cfg (DictConfig): 完整的 Hydra 配置，需包含 cfg.inference 子配置:
            - cfg.inference.device: 推理设备
            - cfg.inference.checkpoint_path: 检查点路径（可选）
            - cfg.inference.input_image_path: 输入图像路径
            - cfg.inference.width, cfg.inference.height: 目标尺寸
            - cfg.inference.prompt: 文本提示
            - cfg.inference.negative_prompt: 负面提示
            - cfg.inference.text_cfg_scale: 文本 CFG 缩放系数
            - cfg.inference.action_cfg_scale: 动作 CFG 缩放系数
            - cfg.inference.num_frames: 生成视频帧数
            - cfg.inference.num_inference_steps: 推理步数
            - cfg.inference.seed: 随机种子
            - cfg.inference.output_mp4: 输出视频路径

    返回:
        str: 输出视频文件的路径
    """
    setup_logging(log_level=logging.INFO)
    inference_cfg = cfg.inference
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(inference_cfg.device))
    checkpoint_path = inference_cfg.get("checkpoint_path")
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            logger.info("Loading finetuned checkpoint: %s", checkpoint_path)
            model.load_checkpoint(checkpoint_path)
        else:
            logger.warning("Checkpoint not found, skipping load: %s", checkpoint_path)
    model.eval()

    def center_crop_resize(img: Image, width: int, height: int) -> Image.Image:
        """
        对输入图像执行中心裁剪并缩放到目标尺寸。

        先按最长边等比放大，使得目标尺寸完全包含在图像内，
        然后从中心区域裁剪出精确的目标尺寸。

        参数:
            img (Image): 输入 PIL Image
            width (int): 目标宽度
            height (int): 目标高度

        返回:
            Image.Image: 裁剪缩放后的图像
        """
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        resized = img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        return resized.crop((left, top, left + width, top + height))

    input_image = Image.open(str(inference_cfg.input_image_path)).convert("RGB")
    input_image = center_crop_resize(input_image, width=inference_cfg.width, height=inference_cfg.height)
    arr = np.array(input_image, dtype=np.float32)
    x = torch.from_numpy(arr)
    x = x.to(device=model.device, dtype=model.torch_dtype)
    x = x * (2.0 / 255.0) - 1.0
    x = repeat(x, "H W C -> B C H W", B=1)
    output_mp4 = str(inference_cfg.output_mp4)

    infer_kwargs = {
        "prompt": str(inference_cfg.prompt),
        "negative_prompt": str(inference_cfg.negative_prompt),
        "text_cfg_scale": float(inference_cfg.text_cfg_scale),
        "action_cfg_scale": float(inference_cfg.action_cfg_scale),
        "input_image": x,
        "num_frames": int(inference_cfg.num_frames),
        "num_inference_steps": int(inference_cfg.num_inference_steps),
        "sigma_shift": None if inference_cfg.get("sigma_shift") is None else float(inference_cfg.sigma_shift),
        "seed": int(inference_cfg.seed),
        "rand_device": str(inference_cfg.rand_device),
        "tiled": bool(inference_cfg.tiled),
    }

    infer_out = model.infer(**infer_kwargs)
    video = infer_out["video"]
    save_mp4(video, output_mp4, fps=15)
    logger.info("Saved inference video to %s", output_mp4)
    return output_mp4
