"""
FastWAM 项目中 Wan2.2 系列模型的加载工具模块。

本模块提供了加载 Wan2.2 TI2V-5B 模型各组件的核心函数，
包括 DiT（扩散 Transformer）、VAE（变分自编码器）和文本编码器。
基于模型文件的哈希值进行自动注册和类型检测，兼容 DiffSynth 框架的加载模式。
"""

from dataclasses import dataclass
import inspect
from typing import Any

import torch
import time

from .io import ModelConfig, hash_model_file, load_state_dict
from .state_dict_converters import (
    wan_video_vae_state_dict_converter,
)
from ..wan_video_dit import WanVideoDiT
from ..wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..wan_video_vae import WanVideoVAE38
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)
# 当跳过预训练权重加载时，使用该哨兵值标记 DiT 路径
SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"


@dataclass
class Wan22LoadedComponents:
    """
    存储 Wan2.2-TI2V-5B 模型所有已加载组件的数据类。

    该数据类将加载好的各个子模块及其文件路径封装在一起，
    方便在训练/推理流程中传递和使用完整的模型管线。

    Attributes:
        dit: 视频 DiT（扩散 Transformer）模型
        vae: 视频 VAE（变分自编码器）模型
        text_encoder: 文本编码器，若不加载则为 None
        tokenizer: HuggingFace 分词器，若不加载则为 None
        dit_path: DiT 权重的文件路径
        vae_path: VAE 权重的文件路径
        text_encoder_path: 文本编码器权重的文件路径
        tokenizer_path: 分词器配置的路径
    """
    dit: WanVideoDiT
    vae: WanVideoVAE38
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: str
    vae_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None


# 模型注册表：通过文件哈希值自动识别模型类型。
# 每个条目包含：
#   - model_hash: 模型文件的唯一哈希标识
#   - model_name: 模型逻辑名称
#   - model_class: 对应的 PyTorch 模块类
#   - state_dict_converter（可选）: 用于转换状态字典键名的函数
WAN22_MODEL_REGISTRY = [
    {
        # 示例: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": WanTextEncoder,
    },
    {
        # 示例: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "1f5ab7703c6fc803fdded85ff040c316",
        "model_name": "wan_video_dit",
        "model_class": WanVideoDiT,
    },
    {
        # 示例: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth")
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": WanVideoVAE38,
        "state_dict_converter": wan_video_vae_state_dict_converter,
    },
]


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    """
    校验 DiT 配置字典的合法性。

    通过反射检查 WanVideoDiT.__init__ 的签名，确保传入的配置：
      - 不含构造函数不认识的键（unknown keys）
      - 包含所有必需的参数（required keys，即没有默认值的参数）

    Args:
        dit_config: 用户传入的 DiT 配置字典。

    Returns:
        校验通过后的配置字典（原始输入的一份拷贝）。

    Raises:
        ValueError: 如果包含未知键或缺少必需键。
    """
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")

    validated = dict(dit_config)

    # 通过 inspect 签名提取 WanVideoDiT 允许的参数列表
    signature = inspect.signature(WanVideoDiT.__init__)
    allowed_keys = set()
    required_keys = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        allowed_keys.add(name)
        if param.default is inspect.Signature.empty:
            required_keys.add(name)

    # 检查是否有不在构造函数签名中的键
    unknown_keys = sorted(set(validated) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in `dit_config`: {unknown_keys}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )

    # 检查是否缺少无默认值的必需参数
    missing_keys = sorted(required_keys - set(validated))
    if missing_keys:
        raise ValueError(
            f"Missing required keys in `dit_config`: {missing_keys}. "
            "Please specify all required WanVideoDiT constructor args."
        )

    return validated


def _load_registered_model(
    path,
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
    model_kwargs_override: dict[str, Any] | None = None,
):
    """
    根据模型文件路径和注册表自动加载已注册的模型。

    工作流程：
      1. 计算模型文件的哈希值。
      2. 在 WAN22_MODEL_REGISTRY 中匹配 model_hash + model_name。
      3. 根据匹配到的配置实例化模型类。
      4. 加载权重文件（.safetensors 或 .bin），如有需要则应用 state_dict 键名转换。
      5. 将权重加载到模型，并移动到指定设备和数据类型。

    Args:
        path: 模型权重文件的路径。
        model_name: 注册表中对应的模型名称（用于二次确认）。
        torch_dtype: 权重的目标数据类型（如 torch.bfloat16）。
        device: 模型需要加载到的设备（如 "cuda"）。
        model_kwargs_override: 额外覆盖的模型构造参数。

    Returns:
        已加载完毕并移动到目标设备的 PyTorch 模型实例。

    Raises:
        ValueError: 如果无法根据哈希值匹配到已知模型。
    """
    # 计算文件哈希用于注册表匹配
    model_hash = hash_model_file(path)

    matched_config = None
    for config in WAN22_MODEL_REGISTRY:
        if config["model_hash"] == model_hash and config["model_name"] == model_name:
            matched_config = config
            break
    if matched_config is None:
        raise ValueError(
            f"Cannot detect model type for {model_name}. File: {path}. "
            f"Model hash: {model_hash}. This standalone package follows DiffSynth hash-based loading."
        )

    model_class = matched_config["model_class"]
    # 合并注册表中的额外 kwargs 与用户传入的覆盖参数
    model_kwargs = dict(matched_config.get("extra_kwargs", {}))
    if model_kwargs_override is not None:
        model_kwargs.update(model_kwargs_override)
    state_dict_converter = matched_config.get("state_dict_converter")

    # 实例化模型（仅结构，不含权重）
    model = model_class(**model_kwargs)
    # 加载权重数据到 CPU
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    # 如果有注册的键名转换器，则应用转换（例如 DiffSynth -> FastWAM 格式）
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)

    # strict=False 允许忽略不匹配的键（如某些辅助结构）
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch_dtype)
    return model


def _resolve_configs(model_id: str, tokenizer_model_id: str, redirect_common_files: bool = True):
    """
    解析模型组件对应的 ModelConfig 配置。

    根据传入的 model_id 和 tokenizer_model_id，分别为 DiT、文本编码器、
    VAE 和分词器构建对应的 ModelConfig 对象。当 redirect_common_files=True 时，
    将 .pth 格式的公共文件重定向到 DiffSynth-Studio 已转换的 .safetensors 格式。

    Args:
        model_id: 主模型 ID（如 "Wan-AI/Wan2.2-TI2V-5B"）。
        tokenizer_model_id: 分词器模型 ID（如 "Wan-AI/Wan2.1-T2V-1.3B"）。
        redirect_common_files: 是否将 .pth 文件重定向到 .safetensors 版本。

    Returns:
        包含 (dit_config, text_config, vae_config, tokenizer_config) 的四元组。
    """
    dit_config = ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors")
    text_config = ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
    vae_config = ModelConfig(model_id=model_id, origin_file_pattern="Wan2.2_VAE.pth")
    tokenizer_config = ModelConfig(model_id=tokenizer_model_id, origin_file_pattern="google/umt5-xxl/")

    if redirect_common_files:
        # 将部分 .pth 文件重定向到 DiffSynth-Studio 已转换为 safetensors 格式的版本
        redirect_dict = {
            "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
            "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
        }
        text_config.model_id, text_config.origin_file_pattern = redirect_dict[text_config.origin_file_pattern]
        vae_config.model_id, vae_config.origin_file_pattern = redirect_dict[vae_config.origin_file_pattern]
    return dit_config, text_config, vae_config, tokenizer_config


def load_wan22_ti2v_5b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = True,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
):
    """
    加载 Wan2.2-TI2V-5B（图生视频）模型的所有组件。

    主入口函数，依次加载：
      1. DiT（扩散 Transformer）—— 主生成网络
      2. VAE（变分自编码器）—— 图像/视频潜在空间编解码
      3. 文本编码器（T5-XXL）—— 文本条件编码
      4. HuggingFace 分词器 —— 文本预处理

    关键特性：
      - 支持跳过 DiT 预训练权重加载，用于从零训练或加载自定义 checkpoint。
      - 支持跳过文本编码器加载，配合预缓存的文本特征使用。
      - 自动将公共文件（如 .pth）重定向到 .safetensors 格式以加速加载。

    Args:
        device: 模型加载目标设备（默认 "cuda"）。
        torch_dtype: 模型权重和计算的数据类型（默认 torch.bfloat16）。
        model_id: HuggingFace / ModelScope 上的模型 ID。
        tokenizer_model_id: 分词器模型 ID。
        tokenizer_max_len: 分词器最大序列长度。
        redirect_common_files: 是否将 .pth 重定向到 .safetensors。
        dit_config: DiT 模型构造参数（必须提供，包含 num_blocks、hidden_size 等结构参数）。
        skip_dit_load_from_pretrain: 若为 True，则跳过 DiT 预训练权重加载。
        load_text_encoder: 若为 True，加载文本编码器和分词器；否则只返回 None。

    Returns:
        Wan22LoadedComponents 数据类，包含所有加载完成的组件和文件路径。

    典型用法:
        >>> components = load_wan22_ti2v_5b_components(
        ...     dit_config={"num_blocks": 40, "hidden_size": 3072, ...}
        ... )
        >>> components.dit  # WanVideoDiT 实例
        >>> components.vae  # WanVideoVAE38 实例
        >>> components.text_encoder  # WanTextEncoder 实例 或 None
    """
    logger.info("Loading Wan2.2-TI2V-5B components...")
    start = time.time()

    if dit_config is None:
        raise ValueError("`dit_config` is required for Wan2.2-TI2V-5B loading.")
    # 校验用户传入的 DiT 配置，确保包含所有必需参数且无不认识参数
    validated_dit_config = _validate_dit_config(dit_config)

    # 解析各个组件的下载配置（model_id + 文件模式）
    dit_model_config, text_config, vae_config, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )

    # VAE 总是需要下载/加载
    vae_config.download_if_necessary()
    if load_text_encoder:
        # 文本编码器和分词器按需下载
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()

    if skip_dit_load_from_pretrain:
        # 跳过预训练权重：仅用配置初始化随机权重，适用于从头训练或 checkpoint 覆盖
        logger.info(
            "Skipping pretrained video DiT load (`skip_dit_load_from_pretrain=True`); "
            "initializing video expert randomly and expecting checkpoint override."
        )
        dit: WanVideoDiT = WanVideoDiT(**validated_dit_config).to(device=device, dtype=torch_dtype)
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        # 标准流程：下载权���并通过注册表加载
        dit_model_config.download_if_necessary()
        dit = _load_registered_model(
            dit_model_config.path,
            "wan_video_dit",
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs_override=validated_dit_config,
        )
        dit_path = str(dit_model_config.path)

    # 文本编码器与分词器的加载（可选）
    text_encoder: WanTextEncoder | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    if load_text_encoder:
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=torch_dtype,
            device=device,
        )
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=int(tokenizer_max_len),
            clean="whitespace",
        )
        text_encoder_path = str(text_config.path)
        tokenizer_path = str(tokenizer_config.path)
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); "
            "training must provide cached `context/context_mask`."
        )
    # VAE 始终从注册表加载
    vae: WanVideoVAE38 = _load_registered_model(vae_config.path, "wan_video_vae", torch_dtype=torch_dtype, device=device)
    logger.info("Finished loading Wan2.2-TI2V-5B components in %.2f seconds.", time.time() - start)
    return Wan22LoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=dit_path,
        vae_path=str(vae_config.path),
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )
