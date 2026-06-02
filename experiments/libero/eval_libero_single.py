"""
FastWAM LIBERO 单任务评估模块 (eval_libero_single)。

该模块是 FastWAM 模型在 LIBERO 仿真环境中的单任务评估入口，核心功能：
1. 模型加载与初始化：加载 FastWAM 检查点、数据处理器和归一化器。
2. 观测处理：将 LIBERO 环境的观测（多摄像头图像、本体感知状态）
   转换为模型输入格式（图像拼接/裁剪、状态归一化）。
3. 动作推理与执行：调用扩散模型推理生成动作块，支持反归一化、
   gripper 动作翻转和动作集成 (ActionEnsembler)。
4. 逐 episode 评估：运行多个 episode，记录成功率/失败信息/重放视频。
5. 未来视频预测与 PSNR 评估：支持可视化预测的未来视频帧并计算 PSNR。

与 RoboTwin 策略相比，LIBERO 评估的不同点：
- 使用 LIBERO 的 benchmark API 获取任务和初始状态。
- 图像来自 agentview 和 wrist 两个摄像头，支持水平和垂直拼接。
- 动作空间包含 7 维控制（6-DoF 末端执行器位姿 + 夹爪）。

使用示例:
    python experiments/libero/eval_libero_single.py \
        ckpt=/path/to/ckpt.pt \
        EVALUATION.task_suite_name=libero_spatial \
        EVALUATION.task_id=0 \
        EVALUATION.num_trials=10 \
        EVALUATION.output_dir=/tmp/eval_results
"""

import json
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark
from action_ensembler import ActionEnsembler

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，支持 numpy 数据类型序列化。

    将 numpy 的整数、浮点数和数组类型自动转换为 Python 原生类型，
    避免 json.dump 时出现 TypeError。
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    """规范化混合精度配置字符串，验证合法性。

    支持的精度选项: "no" (全精度 fp32)、"fp16" (半精度)、"bf16" (bfloat16)。

    Args:
        mixed_precision: 混合精度配置字符串。

    Returns:
        str: 规范化后的精度标识。

    Raises:
        ValueError: 不支持的精度选项。
    """
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    """将混合精度配置转换为 PyTorch 数据类型。

    Args:
        mixed_precision: 混合精度配置字符串。

    Returns:
        torch.dtype: 对应的 PyTorch 数据类型。
    """
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    """解析评估设备（GPU/CPU）。

    优先使用配置中的 EVALUATION.device，
    未指定时自动检测：CUDA 可用则用 "cuda"，否则用 "cpu"。

    Args:
        cfg: Hydra 配置对象。

    Returns:
        str: 设备字符串 ("cuda" 或 "cpu")。
    """
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    """解析数据集统计文件 (dataset_stats.json) 的路径。

    搜索顺序:
        1. 配置中显式指定的 EVALUATION.dataset_stats_path。
        2. 检查点路径的父目录（向上最多 4 层）。

    Args:
        cfg: Hydra 配置对象。

    Returns:
        Path: 找到的 dataset_stats.json 路径。

    Raises:
        FileNotFoundError: 在所有候选位置都未找到时抛出。
    """
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    # 在检查点的父目录中搜索
    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    """加载模型检查点。

    优先使用模型的 load_checkpoint 方法（新格式），
    后续的旧格式加载代码仅为向后兼容保留（dead code）。

    Args:
        model: PyTorch 模型实例。
        ckpt: 检查点文件路径。
    """
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return

    # ======== 废弃的旧版检查点加载逻辑 (向后兼容) ========
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy checkpoint payload must be dict, got: {type(payload)}")

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """对图像进行中心裁剪并调整到目标尺寸。

    保持图像宽高比，先等比放大至覆盖目标尺寸，然后中心裁剪。
    这种方法可以在不严重扭曲图像的前提下将图像调整为任意尺寸。

    Args:
        image: 输入图像，形状为 [H, W, 3]。
        width: 目标宽度。
        height: 目标高度。

    Returns:
        np.ndarray: 裁剪并调整后的图像，形状为 [height, width, 3]。

    示例:
        >>> img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        >>> out = _center_crop_resize(img, 224, 224)
        >>> out.shape
        (224, 224, 3)
    """
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    """对机器人的本体感知状态（proprioception）进行归一化。

    使用数据集统计信息对末端执行器位姿和夹爪状态进行归一化，
    使其适合作为扩散模型的条件输入。

    Args:
        proprio: 原始状态向量，形状为 [D]（D=7: 6-DoF 位姿 + 1 夹爪）。
        processor: FastWAM 数据处理器，包含归一化器。

    Returns:
        torch.Tensor: 归一化后的状态张量，形状为 [1, D]。
    """
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    """将 LIBERO 环境观测转换为模型输入。

    处理流程:
        1. 从观测中提取 agentview 和 wrist 图像。
        2. 根据处理器配置的摄像头数量，对图像进行中心裁剪和 resize。
        3. 多摄像头图像拼接（支持水平/垂直拼接）。
        4. 转换为 CHW 格式张量，归一化到 [-1, 1]。
        5. 提取并归一化本体感知状态。

    Args:
        obs: LIBERO 环境观测字典。
        cfg: Hydra 配置。
        processor: FastWAM 数据处理器。
        width: 输入图像宽度。
        height: 输入图像高度。
        device: 张量所在设备。
        dtype: 张量数据类型。

    Returns:
        tuple: (image_tensor, proprio_tensor, raw_images)
            - image_tensor: 形状 [1, C, H, W] 的图像张量。
            - proprio_tensor: 形状 [1, D] 的归一化状态张量。
            - raw_images: 包含原始图像字典（用于视频保存）。

    Raises:
        ValueError: 摄像头数量或图像尺寸不匹配时抛出。
    """
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        """从 shape_meta 中提取图像尺寸 (H, W)。"""
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """从当前观测中提取仿真器状态，作为模型的本体感知输入。

    拼接末端执行器位置 (3)、旋转 (3, axis-angle)、夹爪开合 (1)，
    总共 7 维状态向量。

    Args:
        obs: LIBERO 环境观测字典。

    Returns:
        np.ndarray: 形状为 [7] 的状态向量，dtype=float32。

    输出示例:
        >>> state = _extract_sim_state(obs)
        >>> state.shape
        (7,)  # [eef_x, eef_y, eef_z, ax, ay, az, gripper_qpos]
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    """对模型输出的动作张量进行反归一化。

    将归一化的动作恢复到原始物理量纲，以便在仿真环境中执行。

    Args:
        action: 归一化动作张量，形状 [B, T, D] 或 [T, D]。
        processor: FastWAM 数据处理器，包含动作归一化器。

    Returns:
        np.ndarray: 反归一化后的动作数组，形状 [B, T, D]。

    Raises:
        ValueError: 动作张量维度不合法时抛出。
    """
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    """计算模型需要的视频帧数量。

    根据训练配置中的总帧数和动作视频采样频率比计算。

    Args:
        cfg: Hydra 配置。

    Returns:
        int: 视频帧数。
    """
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    """验证未来视频可视化配置的合法性。

    如果启用 visualize_future_video，要求模型配置中
    video_dit_config.action_conditioned 必须为 false。

    Args:
        cfg: Hydra 配置。

    Raises:
        ValueError: 配置冲突时抛出。
    """
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    """从模型预测的未来视频帧中选择需要保留的帧。

    根据 replan_steps 和 action_video_freq_ratio 确定要保留的帧数。

    Args:
        pred_video: 模型预测的未来视频帧列表。
        cfg: Hydra 配置。

    Returns:
        list[Image.Image]: 筛选后的帧列表。
    """
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    """计算需要捕获未来帧的仿真步数列表。

    Args:
        cfg: Hydra 配置。

    Returns:
        list[int]: 需要捕获未来帧的仿真步数索引列表。
    """
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    """将多种格式的帧数据统一转换为 RGB numpy 数组。

    支持:
        - dict: 多摄像头图像的水平拼接。
        - PIL Image: 直接转换为 numpy 数组。
        - 其他: 尝试直接转为 numpy 数组。

    Args:
        frame: 帧数据，可以是 dict、PIL Image 或 numpy 数组。

    Returns:
        np.ndarray: RGB 图像数组，形状 [H, W, 3]。
    """
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    """计算一组预测帧与真实帧之间的平均 PSNR（峰值信噪比）。

    PSNR 衡量预测帧的图像质量，值越高表示预测越接近真实。
    先计算每帧的 PSNR，再取平均。

    Args:
        gt_frames: 真实帧列表（来自仿真环境）。
        pred_frames: 预测帧列表（来自模型）。
        eps: 防止除零的小常数。

    Returns:
        Optional[float]: 平均 PSNR 值 (dB)，若输入为空则返回 None。
    """
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _load_cached_text_context(prompt: str, *, cache_dir: str, context_len: int = 128):
    """从预计算缓存加载 T5 文本嵌入（复用训练数据集相同的 hash 查找逻辑）。

    与 robot_video_dataset._get_cached_text_context 使用完全一致的
    缓存文件命名和格式约定，无需加载 T5 模型到 GPU。

    Args:
        prompt: 格式化后的完整提示词字符串。
        cache_dir: 文本嵌入缓存目录路径。
        context_len: token 序列长度，需与预编码时一致。

    Returns:
        tuple: (context [L, 4096], context_mask [L])
    """
    import hashlib
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    # enc_id 与训练预编码保持一致
    enc_id = "wan22ti2v5b"
    cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{context_len}.{enc_id}.pt")
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"Cached text embedding not found: {cache_path}. "
            f"Run: python scripts/precompute_libero_text_embeds.py --output-dir {cache_dir}"
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    context = payload["context"]
    context_mask = payload["mask"].bool()
    # Keep eval consistent with RobotVideoDataset: Wan-style text conditioning
    # zeroes padded token embeddings but keeps all text positions visible.
    context[~context_mask] = 0.0
    context_mask = torch.ones_like(context_mask)
    return context, context_mask


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    cached_context: Optional[torch.Tensor] = None,
    cached_context_mask: Optional[torch.Tensor] = None,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]]]:
    """执行单次完整推理，生成动作块 (action chunk)。

    处理流程:
        1. 从配置中获取推理步数等推理参数。
        2. 格式化语言指令提示词。
        3. 将观测转换为模型输入（图像 + 状态）。
        4. 调用模型推理（支持未来视频联合预测）。
        5. 反归一化并后处理动作（gripper 符号翻转）。

    Args:
        obs: LIBERO 环境观测。
        task_description: 自然语言任务描述。
        model: FastWAM 模型。
        processor: 数据处理器。
        cfg: Hydra 配置。
        action_horizon: 动作块长度。
        input_w: 输入图像宽度。
        input_h: 输入图像高度。
        model_device: 模型所在设备。

    Returns:
        tuple: (action_chunk, images, predicted_future_frames)
            - action_chunk: 形状 [T, D] 的反归一化动作数组。
            - images: 原始观测图像字典。
            - predicted_future_frames: 预测的未来帧列表（若启用可视化）或 None。
    """
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    # 优先使用缓存的 text embedding（无需 T5 在 GPU 上）
    if cached_context is not None and cached_context_mask is not None:
        infer_kwargs["prompt"] = None
        infer_kwargs["context"] = cached_context.unsqueeze(0).to(device=model_device)
        infer_kwargs["context_mask"] = cached_context_mask.unsqueeze(0).to(device=model_device)
    else:
        infer_kwargs["prompt"] = prompt
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    predicted_future_frames = None
    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    with torch.no_grad():
        if visualize_future_video:
            pred = model.infer_joint(**infer_kwargs)
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        else:
            pred = model.infer_action(**infer_kwargs)
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action, imgs, predicted_future_frames


def _get_max_steps(task_suite_name: str) -> int:
    """获取指定 LIBERO 任务套件允许的最大仿真步数。

    不同套件的任务复杂度不同，需要的最大步数也不同：
        - libero_spatial/object/goal: 400 步
        - libero_10/90: 700 步（更复杂的任务）

    Args:
        task_suite_name: LIBERO 任务套件名称。

    Returns:
        int: 最大仿真步数。

    Raises:
        ValueError: 未知的任务套件名称。
    """
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    cached_context: Optional[torch.Tensor] = None,
    cached_context_mask: Optional[torch.Tensor] = None,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float]]:
    """运行单个评估 episode。

    完整的 episode 执行流程:
        1. 重置环境并设置初始状态。
        2. 执行 num_steps_wait 步等待（dummy action），让仿真稳定。
        3. 循环执行 "推理-执行" 重规划策略:
           - 动作队列为空时: 调用模型推理生成动作块，填充队列
           - 从队列取出一个动作发送给环境
           - 可选: 使用 ActionEnsembler 进行动作集成
           - 可选: 捕获未来帧用于视频预测可视化
        4. 达到最大步数或任务完成时停止。
        5. 收集回放图像用于视频保存。

    Args:
        env: LIBERO 仿真环境。
        initial_state: 任务的初始状态（用于环境重置）。
        task_description: 自然语言任务描述。
        model: FastWAM 模型。
        processor: 数据处理器。
        cfg: Hydra 配置。
        episode_idx: 当前 episode 编号。
        action_horizon: 动作块长度。
        input_w: 输入图像宽度。
        input_h: 输入图像高度。
        model_device: 模型所在设备。

    Returns:
        tuple: (success, replay_images, predicted_future_video_clips, episode_mean_psnr)
            - success: 任务是否成功完成。
            - replay_images: 回放图像列表，用于保存评估视频。
            - predicted_future_video_clips: 未来帧预测片段列表（用于可视化）。
            - episode_mean_psnr: 未来帧预测的平均 PSNR，无预测时返回 None。
    """
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            action_chunk, imgs, predicted_future_frames = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                cached_context=cached_context,
                cached_context_mask=cached_context_mask,
            )
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    return bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    """运行单个 LIBERO 任务的所有 episode。

    依次运行 num_trials 个 episode，收集成功率数据和回放视频。
    如果启用了未来帧可视化，还会保存预测帧比较视频。

    Args:
        task: LIBERO 任务对象。
        initial_states: 任务的初始状态列表。
        model: FastWAM 模型。
        processor: 数据处理器。
        cfg: Hydra 配置。
        video_dir: 回放视频保存目录。
        predicted_video_dir: 预测帧比较视频保存目录。
        action_horizon: 动作块长度。
        input_w: 输入图像宽度。
        input_h: 输入图像高度。
        model_device: 模型所在设备。

    Returns:
        dict: 包含以下键的评估结果字典:
            - successes: 成功次数。
            - failure_episodes: 失败的 episode 编号列表。
            - success_episodes: 成功的 episode 编号列表。
            - task_description: 任务描述。
            - episode_future_video_psnr (可选): 每 episode 的 PSNR 列表。
            - future_video_psnr_mean (可选): 平均 PSNR。
    """
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))

    # 从预计算缓存加载 text embedding（与训练共用同一套缓存，无需加载 T5 到 GPU）
    cached_context = None
    cached_context_mask = None
    text_embedding_cache_dir = cfg.EVALUATION.get("text_embedding_cache_dir", None)
    if text_embedding_cache_dir:
        context_len = int(cfg.data.train.get("context_len", 128))
        prompt = DEFAULT_PROMPT.format(task=task_description)
        cached_context, cached_context_mask = _load_cached_text_context(
            prompt, cache_dir=text_embedding_cache_dir, context_len=context_len
        )
        logging.info("Loaded cached text context for task: %s", task_description)

    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        success, replay_images, predicted_future_video_clips, episode_mean_psnr = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            cached_context=cached_context,
            cached_context_mask=cached_context_mask,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)

        save_rollout_video(
            video_dir,
            replay_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
        )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    """LIBERO 单任务评估的 Hydra 入口函数。

    执行流程:
        1. 初始化：设置随机种子、验证配置、解析模型和设备。
        2. 加载模型：实例化 FastWAM 模型、加载检查点、设置为 eval 模式。
        3. 初始化处理器：加载数据处理器和数据集统计信息。
        4. 配置评估参数：action_horizon、图像尺寸、摄像头拼接方式。
        5. 创建输出目录：videos/ 和 predicted_videos/ 子目录。
        6. 获取任务：通过 LIBERO benchmark API 获取指定任务和初始状态。
        7. 执行评估：运行 run_single_task 进行多 episode 评估。
        8. 保存结果：写入 JSON 结果文件，打印成功率和耗时。

    Args:
        cfg: Hydra 配置，需包含:
            - ckpt: 检查点路径 (必需)
            - EVALUATION.task_suite_name: 套件名 (如 libero_spatial)
            - EVALUATION.task_id: 任务 ID
            - EVALUATION.num_trials: 每个任务运行的 episode 数
            - EVALUATION.output_dir: 输出目录
            - gpu_id: GPU 编号
            - 其他推理参数（可选）
    """
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
