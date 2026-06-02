"""
ActionDiT 骨干网络预处理脚本 —— 从 WanVideoDiT 权重插值生成 ActionDiT 初始权重。

该脚本将 Wan2.2 视频 DiT（WanVideoDiT）的骨干网络权重通过形状适配插值，
转换为 ActionDiT（动作条件 DiT）的初始权重。核心逻辑包括：
    1. 加载视频 DiT 和创建 ActionDiT 空网络
    2. 对骨干网络中的每个张量，如果形状不同则进行多维插值
    3. 可选的 alpha 缩放（alpha = sqrt(d_v / d_a)），用于修正维度变化引起的方差偏移
    4. 保存为 .pt 格式的载荷文件，供训练时加载

典型用法:
    python scripts/preprocess_action_dit_backbone.py \
        --model-config configs/model/fastwam.yaml \
        --output data/backbones/action_dit_backbone.pt \
        --dtype bfloat16 --device cuda

背景:
    ActionDiT 与 WanVideoDiT 共享相同的骨干网络结构（层数、头数、注意力头维度），
    但部分张量的最后一维（如 hidden_dim）可能不同（视频 DiT 为 1536，动作 DiT 通常更小）。
    该脚本通过顺序的 1D 线性插值适配这些维度差异。
"""

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components


def _parse_dtype(name: str) -> torch.dtype:
    """
    将字符串格式的数据类型名称解析为 PyTorch 数据类型。

    参数:
        name (str): 数据类型名称，支持 "float32", "float16", "bfloat16"

    返回:
        torch.dtype: 对应的 PyTorch 数据类型

    异常:
        ValueError: 不支持的数据类型名称
    """
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}. Expected one of: float32, float16, bfloat16.")


def _parse_bool(name: str) -> bool:
    """
    将字符串解析为布尔值。

    支持多种格式：1/0, true/false, yes/no, y/n。

    参数:
        name (str): 要解析的字符串

    返回:
        bool: 解析后的布尔值

    异常:
        ValueError: 无法解析的输入
    """
    value = str(name).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {name}")


def _is_unresolved_interpolation(value: Any) -> bool:
    """
    检查值是否为未解析的 OmegaConf 插值表达式。

    OmegaConf 中形如 "${video_dit_config.hidden_dim}" 的字符串
    在配置加载时可能未被解析（resolve=False 模式）。

    参数:
        value (Any): 待检查的值

    返回:
        bool: 如果是未解析的插值表达式则返回 True
    """
    return isinstance(value, str) and "${" in value and "}" in value


def _resolve_from_video_cfg(value: Any, video_cfg: dict[str, Any]) -> Any:
    """
    如果值是引用 video_dit_config 的插值表达式，则从 video_cfg 中解析其值。

    例如，若 action 配置中的 "num_heads" 为 "${video_dit_config.num_heads}"，
    则从 video_cfg 字典中取出对应的整数值。

    参数:
        value (Any): 可能包含插值表达式的值
        video_cfg (dict): 视频 DiT 配置字典

    返回:
        Any: 解析后的值（如果无法解析则返回原始值）
    """
    if not _is_unresolved_interpolation(value):
        return value
    text = str(value).strip()
    if not (text.startswith("${") and text.endswith("}")):
        return value
    expr = text[2:-1]
    if not expr.startswith("video_dit_config."):
        return value
    key = expr.split(".", 1)[1]
    if key not in video_cfg:
        return value
    resolved = video_cfg[key]
    return value if _is_unresolved_interpolation(resolved) else resolved


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    """
    对张量的最后一维进行 1D 线性插值。

    将任意形状的张量展平为 [N, 1, last_dim] 进行线性插值，
    然后恢复原始形状（最后一维替换为 new_size）。

    输入维度示例:
        - 输入: [N, 1536] -> 插值到新大小 -> [N, 128]
        - 输入: [L, D]  = [30, 1536] -> 插值 -> [30, 128]

    参数:
        tensor (torch.Tensor): 输入张量
        new_size (int): 目标最后一维大小

    返回:
        torch.Tensor: 插值后的张量，shape 为 (*tensor.shape[:-1], new_size)
    """
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """
    将源张量插值到目标形状，支持多维度的逐维插值。

    对每个维度逐一检查，如果当前大小与目标不同，则通过 permute + 1D 插值调整。
    该方法会处理维度不一致的情况（增加或减少维度）。

    算法：
        对每个维度 dim:
            1. 将该维度 permute 到最后一维
            2. 调用 _interpolate_last_dim 进行 1D 线性插值
            3. 用逆 permute 恢复原始维度顺序

    输入输出示例:
        - 源 [1536] -> 目标 [128]
        - 源 [1536, 1536] -> 目标 [128, 128]
        - 源 [30, 1536] -> 目标 [30, 128]

    参数:
        src (torch.Tensor): 源张量
        target_shape (tuple): 目标形状

    返回:
        torch.Tensor: 插值到目标形状的张量

    异常:
        ValueError: 无法减少 batch 维度或插值结果形状不匹配
    """
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        # Permute the target dimension to the end for interpolation
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        # Construct inverse permutation to restore original order
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        # Permute, interpolate, and restore original order
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape for tensor. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def _load_model_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    从 YAML 配置文件加载视频 DiT 和 Action DiT 的配置字典。

    解析配置文件中顶层 key "video_dit_config" 和 "action_dit_config"。
    如果 action_dim 是未解析的插值表达式，则默认设为 7。
    同时解析 action 配置中引用 video 配置的字段（如 num_heads, attn_head_dim, num_layers 等）。

    参数:
        path (Path): YAML 配置文件路径（如 configs/model/fastwam.yaml）

    返回:
        tuple: (video_cfg, action_cfg, full_cfg)
            - video_cfg (dict): 视频 DiT 配置
            - action_cfg (dict): 动作 DiT 配置
            - full_cfg: 完整配置对象

    异常:
        ValueError: 配置中缺少必要字段或类型不正确
    """
    cfg = OmegaConf.load(str(path))
    if "video_dit_config" not in cfg or "action_dit_config" not in cfg:
        raise ValueError(
            f"`{path}` must contain both `video_dit_config` and `action_dit_config` at top level."
        )

    video_cfg = OmegaConf.to_container(cfg.video_dit_config, resolve=False)
    action_cfg = OmegaConf.to_container(cfg.action_dit_config, resolve=False)
    if not isinstance(video_cfg, dict) or not isinstance(action_cfg, dict):
        raise ValueError("`video_dit_config` and `action_dit_config` must resolve to dicts.")

    if _is_unresolved_interpolation(video_cfg.get("action_dim")):
        print("[WARN] `video_dit_config.action_dim` is unresolved; defaulting to 7 for preprocessing.")
        video_cfg["action_dim"] = 7

    if _is_unresolved_interpolation(action_cfg.get("action_dim")):
        print("[WARN] `action_dit_config.action_dim` is unresolved; defaulting to 7 for preprocessing.")
        action_cfg["action_dim"] = 7

    for key in ["num_heads", "attn_head_dim", "num_layers", "text_dim", "freq_dim"]:
        action_cfg[key] = _resolve_from_video_cfg(action_cfg.get(key), video_cfg)

    return video_cfg, action_cfg, cfg


def _require_int_config(cfg: dict[str, Any], key: str) -> int:
    """
    从配置字典中获取一个整数类型的配置项。

    如果配置项是未解析的插值表达式则抛出错误，确保在预处理时所有值都已解析。

    参数:
        cfg (dict): 配置字典
        key (str): 配置键名

    返回:
        int: 配置值

    异常:
        ValueError: 配置值为未解析的插值表达式
    """
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return int(value)


def _require_float_config(cfg: dict[str, Any], key: str) -> float:
    """
    从配置字典中获取一个浮点数类型的配置项。

    参数:
        cfg (dict): 配置字典
        key (str): 配置键名

    返回:
        float: 配置值

    异常:
        ValueError: 配置值为未解析的插值表达式
    """
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return float(value)


def main() -> None:
    """
    主处理函数：从 WanVideoDiT 提取骨干网络权重，插值适配后保存为 ActionDiT 初始权重。

    处理流程：
        1. 解析命令行参数
        2. 加载模型 YAML 配置文件，解析 video_dit_config 和 action_dit_config
        3. 加载 WanVideoDiT 预训练模型（通过 load_wan22_ti2v_5b_components）
        4. 创建空的 ActionDiT 模型
        5. 校验 ActionDiT 与 VideoDiT 的关键结构（num_heads, attn_head_dim, num_layers）一致
        6. 遍历 ActionDiT 骨干网络的所有权重键：
            a. 如果形状与视频 DiT 相同，直接拷贝
            b. 如果形状不同，执行 _resize_tensor_to_shape 进行多维线性插值
            c. 可选的 alpha 缩放：当最后一维尺寸变化时，
               用 alpha = sqrt(dim_src / dim_tgt) 缩放权重以保持方差稳定
        7. 组装包含策略描述、权重组和元信息的载荷字典
        8. 保存为 .pt 文件供训练时加载

    Alpha 缩放原理:
        当在权重插值中改变张量维度时，输出的方差会相应变化。
        alpha 因子在数学上等价于在保持权重初始化方差守恒的前提下调整缩放。
    """
    parser = argparse.ArgumentParser(
        description="Preprocess ActionDiT backbone weights from WanVideoDiT and save as .pt payload."
    )
    parser.add_argument("--model-config", required=True, help="Path to model yaml, e.g. configs/model/fastwam.yaml")
    parser.add_argument("--output", required=True, help="Output .pt path for preprocessed ActionDiT backbone.")
    parser.add_argument("--device", default="cpu", help="Device for loading model and preprocessing.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--apply-alpha-scaling",
        default="true",
        help="Whether to apply alpha=sqrt(dv/da) when the last dimension is resized (true/false). Default: true.",
    )
    args = parser.parse_args()

    model_config_path = Path(args.model_config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_alpha_scaling = _parse_bool(args.apply_alpha_scaling)

    video_cfg, action_cfg, cfg = _load_model_config(model_config_path)
    torch_dtype = _parse_dtype(args.dtype)
    redirect_common_files = _parse_bool(cfg.get("redirect_common_files", False))

    # 确保所有关键的数值型配置在预处理时都已解析（不能有未解析的 ${} 插值）
    int_fields = ["hidden_dim", "action_dim", "ffn_dim", "num_layers", "num_heads", "attn_head_dim", "text_dim", "freq_dim"]
    for key in int_fields:
        action_cfg[key] = _require_int_config(action_cfg, key)
    action_cfg["eps"] = _require_float_config(action_cfg, "eps")

    print(f"[INFO] Loaded model config from {model_config_path}. "
          f"Preprocessing ActionDiT backbone with dtype={torch_dtype} on device={args.device}, "
          f"apply_alpha_scaling={apply_alpha_scaling}.")
    load_text_encoder = _parse_bool(cfg.get("load_text_encoder", False))
    # 加载 Wan2.2 视频 DiT 预训练组件
    components = load_wan22_ti2v_5b_components(
        device=args.device,
        torch_dtype=torch_dtype,
        model_id=cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"),
        tokenizer_model_id=cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
        redirect_common_files=redirect_common_files,
        dit_config=video_cfg,
        load_text_encoder=load_text_encoder,
    )
    video_expert = components.dit

    action_expert = ActionDiT(**action_cfg).to(device=args.device, dtype=torch_dtype)

    # 校验 MoT 混合注意力所需的关键结构一致性
    if int(action_cfg["num_heads"]) != int(video_expert.num_heads):
        raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
    if int(action_cfg["attn_head_dim"]) != int(video_expert.attn_head_dim):
        raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
    if int(action_cfg["num_layers"]) != int(len(video_expert.blocks)):
        raise ValueError("ActionDiT `num_layers` must match video expert.")

    action_state = action_expert.state_dict()
    video_state = video_expert.state_dict()
    # 仅处理骨干网络权重（排除 action_embed 等特定层）
    backbone_keys = ActionDiT.backbone_key_set(action_state.keys())

    backbone_state_dict: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    for key in sorted(backbone_keys):
        if key not in video_state:
            raise ValueError(f"Key `{key}` not found in video expert state dict.")
        src = video_state[key]
        target = action_state[key]
        if tuple(src.shape) == tuple(target.shape):
            # 形状相同，直接复制视频 DiT 权重
            value = src
            copied += 1
        else:
            # 形状不同，执行多维插值适配
            value = _resize_tensor_to_shape(src, tuple(target.shape))
            if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target.shape[-1]:
                # alpha = sqrt(d_v / d_a)：当维度变化时保持权重方差
                alpha = (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
                value = value.to(torch.float32) * alpha
            interpolated += 1
        backbone_state_dict[key] = value.detach().to(dtype=target.dtype, device="cpu").contiguous()

    # 组装最终载荷：包含处理策略描述、权重数据和元信息
    payload = {
        "policy": {
            "skip_prefixes": list(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
        },
        "backbone_state_dict": backbone_state_dict,
        "meta": {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        },
    }
    torch.save(payload, str(output_path))

    skipped = len(action_state) - len(backbone_keys)
    print(
        "[INFO] Saved ActionDiT backbone payload to "
        f"{output_path} (copied={copied}, interpolated={interpolated}, skipped={skipped})."
    )


if __name__ == "__main__":
    main()
