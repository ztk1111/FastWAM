"""
FastWAM RoboTwin 策略部署模块 (WorldActionRobotWinPolicy)。

该模块是 FastWAM 模型在 RoboTwin 仿真环境中的策略部署核心实现，定义了以下核心功能：
1. WorldActionRobotWinPolicy 类：管理策略的生命周期，包括模型加载、图像预处理、
   动作推理、反归一化、动作队列管理和执行时序统计。
2. 辅助函数：处理配置解析、精度转换、路径解析等基础设施。
3. 与 RoboTwin 官方评估框架 (script/eval_policy.py) 的对接接口 (get_model/eval/reset_model)。

整体流程：
   接收观测 -> 构建图像张量 -> 状态归一化 -> 模型推理 -> 动作反归一化 ->
   填充动作队列 -> 按 re-plan 步长逐步执行 -> 时序统计

使用示例:
    policy = get_model(usr_args)
    obs = observation_dict
    policy.step(task_env, obs)    # 单步推理与执行
    policy.reset()                 # 重置策略状态（新 episode 开始）
"""

import logging
import os
import sys
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    """检查值是否为"空"类（None 或表示空值的字符串）。

    用于安全地判断用户传递的参数是否需要使用默认值。
    将 None、空字符串、"none"、"null" 等统一视为空。

    Args:
        value: 待检查的值。

    Returns:
        bool: 如果值为 None 或空字符串标记则返回 True。

    示例:
        >>> _is_none_like(None)
        True
        >>> _is_none_like("null")
        True
        >>> _is_none_like("bf16")
        False
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    """将多种格式的输入解析为布尔值。

    支持 bool 类型直接返回，以及字符串格式的 "true"/"false"/"yes"/"no"/"1"/"0"。

    Args:
        value: 待解析的值。

    Returns:
        bool: 解析后的布尔值。

    Raises:
        ValueError: 无法解析时抛出。

    示例:
        >>> _parse_bool("true")
        True
        >>> _parse_bool(0)
        ValueError
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    """解析可选的整数值，若为空则返回 None。

    Args:
        value: 待解析的值，可以是 None、字符串 "none" 或整数。

    Returns:
        Optional[int]: 解析后的整数，或 None。

    示例:
        >>> _parse_optional_int("8")
        8
        >>> _parse_optional_int(None)
        None
    """
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    """解析可选的浮点数值，若为空则返回 None。

    Args:
        value: 待解析的值。

    Returns:
        Optional[float]: 解析后的浮点数，或 None。

    示例:
        >>> _parse_optional_float("0.5")
        0.5
        >>> _parse_optional_float(None)
        None
    """
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    """规范化混合精度配置字符串，验证其合法性。

    支持的精度选项: "no" (全精度 fp32)、"fp16" (半精度)、"bf16" (bfloat16)。

    Args:
        mixed_precision: 混合精度配置字符串。

    Returns:
        str: 规范化后的精度标识 ("no" | "fp16" | "bf16")。

    Raises:
        ValueError: 传入不支持的精度选项时抛出。
    """
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    """将混合精度配置字符串转换为 PyTorch 数据类型。

    转换映射:
        "no"   -> torch.float32
        "fp16" -> torch.float16
        "bf16" -> torch.bfloat16

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


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    """解析仿真配置文件路径/名称。

    优先使用显式指定的 sim_cfg_path（必须位于 configs 目录下），
    否则回退到 sim_cfg_name，最后使用默认值 "sim_robotwin.yaml"。

    Args:
        sim_cfg_path: 可选的配置文件完整路径。
        sim_cfg_name: 可选的配置文件名称。

    Returns:
        str: 相对于 configs 目录的配置文件路径（POSIX 风格）。

    Raises:
        ValueError: 如果 sim_cfg_path 不在 configs 目录下则抛出。
    """
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    """使用 Hydra 组合仿真配置。

    先解析配置文件路径/名称，再通过 Hydra 的 compose API 加载配置，
    并可选的通过 overrides 覆盖 task 参数。

    Args:
        sim_cfg_path: 可选的配置文件完整路径。
        sim_cfg_name: 可选的配置文件名称。
        sim_task: 可选的仿真任务名称，会作为 Hydra override 传入。

    Returns:
        DictConfig: Hydra 组合后的配置对象。
    """
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    # 清理已有的 Hydra 实例，避免重复初始化冲突
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    """解析并验证数据集统计信息 JSON 文件的路径。

    Args:
        dataset_stats_path: 数据集统计信息文件路径。

    Returns:
        Path: 解析并验证后的路径对象。

    Raises:
        FileNotFoundError: 路径为空或文件不存在时抛出。
    """
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """将 RGB 图像调整到指定尺寸（双线性插值）。

    用于将不同分辨率的摄像头图像统一缩放到模型输入所需的尺寸。

    Args:
        image: 输入 RGB 图像，形状为 [H, W, 3]。
        size_wh: 目标尺寸 (width, height)。

    Returns:
        np.ndarray: 缩放后的图像，形状为 [height, width, 3]，dtype=uint8。

    示例:
        >>> img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        >>> resized = _resize_rgb(img, (320, 256))
        >>> resized.shape
        (256, 320, 3)
    """
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


class WorldActionRobotWinPolicy:
    """RoboTwin 仿真环境中的 FastWAM 策略类。

    该类封装了完整的策略部署生命周期：
    1. 初始化：加载模型检查点、处理器和数据集统计信息。
    2. 推理：接收仿真观测 -> 图像预处理和拼接 -> 状态归一化 -> 模型推理 -> 动作反归一化。
    3. 执行：将推理得到的动作块填充到动作队列，按 re-plan 步长逐步发送给仿真环境。
    4. 时序统计：记录推理时间和仿真执行时间，用于性能分析。
    5. 重置：清除动作队列，递增 episode 计数，重置计时器。

    核心流程示意:
        ┌─────────┐     ┌──────────┐     ┌─────────┐     ┌──────────────┐
        │ 观测输入 │────>│ 图像预处理 │────>│ 模型推理 │────>│ 动作队列管理 │
        └─────────┘     └──────────┘     └─────────┘     └──────┬───────┘
                                                               │
                                                        ┌──────▼───────┐
                                                        │ 仿真环境执行 │
                                                        └──────────────┘

    动作队列管理:
        - 每次推理输出 action_horizon 步的动作块 (action chunk)
        - 每次只执行 replan_steps 步，然后重新观测和推理
        - 这种"预测-重规划"机制在保证控制精度的同时降低推理频率
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
    ) -> None:
        """初始化策略实例。

        Args:
            model_cfg: 模型配置（Hydra DictConfig），包含模型架构参数。
            processor_cfg: 数据处理器配置，用于图像/状态预处理和归一化。
            checkpoint_path: 模型检查点文件路径。
            dataset_stats_path: 数据集统计信息 JSON 路径，用于动作/状态的归一化。
            device: 推理设备（如 "cuda:0" 或 "cpu"）。
            model_dtype: 模型推理精度（torch.float32 / float16 / bfloat16）。
            action_horizon: 每次推理输出的动作步数（动作块长度）。
            replan_steps: 每次执行后重新规划的动作步数（<= action_horizon）。
            num_inference_steps: 扩散模型推理步数。
            sigma_shift: 可选的 sigma 偏移参数，用于调整扩散噪声调度。
            seed: 随机种子，用于可重现推理。
            text_cfg_scale: 文本条件引导比例 (Classifier-Free Guidance scale)。
            negative_prompt: 负向提示词，用于 CFG 引导。
            rand_device: 随机数生成设备（"cpu" 或 "cuda"）。
            tiled: 是否使用分块 VAE 编码（减少大图像显存占用）。
            timing_enabled: 是否启用推理/执行时序统计。
            num_video_frames: 视频帧数（用于视频条件模型）。
        """
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        # 必须加载文本编码器以支持语言条件
        model_cfg_copy.load_text_encoder = True

        # 1. 实例化模型并加载检查点
        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        # 2. 初始化数据处理器和归一化器
        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        # 3. 配置推理参数
        self.action_horizon = int(action_horizon)
        # replan_steps 被限制在 [1, action_horizon] 范围内
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)

        # 4. 初始化运行时状态
        self.pending_actions: deque[np.ndarray] = deque()  # 待执行的动作队列
        self.episode_count = 0   # 已完成的 episode 计数
        self.step_count = 0      # 当前 episode 的步数计数
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}  # 推理/仿真时序统计

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        """对机器人状态向量进行归一化处理。

        使用数据集的统计信息（均值和标准差）对关节位置等状态进行归一化，
        使其分布接近标准正态分布，有利于模型推理。

        Args:
            state: 原始状态向量，如关节位置，形状为 [D]。

        Returns:
            torch.Tensor: 归一化后的状态张量，形状为 [1, D]。
        """
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        # 构造批次字典 -> 应用变换 -> 归一化
        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        """对模型输出的动作张量进行反归一化，恢复到原始物理量纲。

        模型输出的是归一化后的动作，需要通过数据集统计信息还原为实际
        的关节位置/速度值，才能发送给仿真环境执行。

        Args:
            action: 归一化后的动作张量，形状为 [B, T, D] 或 [T, D]。

        Returns:
            np.ndarray: 反归一化后的动作数组，形状为 [B, T, D]。

        Raises:
            ValueError: 如果 action 维度不为 2 或 3 则抛出。
        """
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        """构建 RoboTwin 输入图像张量。

        将三个摄像头（头部、左手、右手）的图像拼接成一个 384x320 的大图：
        - 头部图像 resize 到 320x256（宽x高）
        - 左右手图像 resize 到 160x128，水平拼接后放在下方
        - 上下拼接形成最终图像 [384, 320, 3]
        - 再转换为 CHW 格式，归一化到 [-1, 1] 范围

        Args:
            observation: RoboTwin 环境观测字典，包含多摄像头 RGB 图像。
                格式: {"observation": {"head_camera": {"rgb": ...},
                                       "left_camera": {"rgb": ...},
                                       "right_camera": {"rgb": ...}}}

        Returns:
            torch.Tensor: 拼接+归一化后的图像张量，形状为 [1, 3, 384, 320]。
        """
        obs_data = observation["observation"]
        # 头部摄像头 resize 到 320x256
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        # 左右手摄像头 resize 到 160x128 并水平拼接
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        # 上下拼接形成最终大图 [384, 320, 3]
        image = np.concatenate([head, bottom], axis=0)

        # HWC -> CHW, 添加 batch 维度, 归一化到 [-1, 1]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        """执行一次完整的模型推理，生成动作块 (action chunk)。

        步骤:
            1. 构建多摄像头拼接图像张量
            2. 提取并归一化关节状态作为本体感知输入
            3. 格式化语言指令提示词
            4. 调用模型推理，获得归一化的动作张量
            5. 反归一化得到原始物理量纲的动作

        Args:
            observation: RoboTwin 环境观测字典。
            instruction: 自然语言任务指令。

        Returns:
            np.ndarray: 反归一化后的动作块，形状为 [T, D]，其中 T=action_horizon, D=动作维度。

        输出示例:
            >>> action_chunk = policy._infer_action_chunk(obs, "pick up the block")
            >>> action_chunk.shape
            (16, 7)  # 16 步动作块, 7 维关节控制信号
        """
        # 1. 构建图像输入
        image_tensor = self._build_robotwin_image_tensor(observation)
        # 2. 提取并归一化本体感知状态
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        # 3. 格式化提示词并构建推理参数
        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        # 如果模型支持视频帧参数，则传入（兼容性处理）
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)

        # 4. 模型推理（关闭梯度计算以节省显存和加速）
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        # 5. 反归一化动作
        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        """执行推理并将生成的 replan_steps 个动作填入待执行队列。

        Args:
            observation: RoboTwin 环境观测字典。
            instruction: 自然语言任务指令。

        说明:
            推理得到 action_horizon 步的动作块，但只取前 replan_steps 步执行，
            剩余的丢弃。这实现了"预测-重规划"的控制策略。
        """
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        # 只取前 replan_steps 步填充队列
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        """判断当前是否需要请求新的环境观测。

        当动作队列为空时，表示需要在下一步进行重规划（re-plan），
        因此需要获取新的环境观测来触发推理。

        Returns:
            bool: 如果动作队列为空（需要重规划）则返回 True，否则 False。
        """
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        """执行一次策略步进：推理（如需）+ 发送动作到仿真环境。

        核心逻辑:
            1. 如果动作队列为空 -> 获取观测和指令 -> 推理填充动作队列
            2. 从队列头部取出一个动作
            3. 发送给仿真环境执行
            4. 记录执行耗时（如果启用时序统计）

        Args:
            task_env: RoboTwin 仿真环境对象，需支持 get_instruction() 和 take_action()。
            observation: 当前环境观测。在重规划步时必须提供，非重规划步可为 None。

        Raises:
            ValueError: 动作队列为空且未提供观测时抛出。
        """
        # 检查是否需要重规划（动作队列为空）
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            # 获取语言指令并执行推理填充队列
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        # 安全检查：如果推理失败导致队列仍为空则跳过
        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        # 从队列中取出一个动作并发送给仿真环境
        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        """重置当前 episode 的时序统计信息。"""
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        """获取当前 episode 的时序统计信息。

        Returns:
            Dict[str, float]: 包含推理时间 (infer_s) 和仿真执行时间 (sim_s) 的字典。
        """
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        """重置策略状态以开始新的 episode。

        执行:
            - 清空待执行的动作队列
            - episode 计数递增
            - 步数计数归零
            - 时序统计信息重置
        """
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """编码观测数据（当前为恒等映射，保留未来扩展接口）。

    该函数是 RoboTwin 官方评估框架要求的接口。目前直接返回原始观测，
    未来可以在此添加额外的观测预处理逻辑。

    Args:
        observation: 原始环境观测字典。

    Returns:
        Optional[Dict[str, Any]]: 编码后的观测（当前为恒等映射）。
    """
    return observation


def get_model(usr_args: Dict[str, Any]):
    """创建并初始化 WorldActionRobotWinPolicy 策略实例。

    该函数是 RoboTwin 官方评估框架的标准接口，从用户参数字典中提取
    所有配置，组装 Hydra 配置，然后实例化策略对象。

    主要步骤:
        1. 解析仿真配置（通过 Hydra compose）
        2. 解析检查点路径、设备、混合精度
        3. 定位数据集统计信息文件
        4. 处理 action_horizon / replan_steps / num_inference_steps 等参数
        5. 实例化 WorldActionRobotWinPolicy

    Args:
        usr_args: 用户参数字典，包含以下键（部分可选）:
            - sim_cfg_path: 仿真配置文件路径
            - sim_cfg_name: 仿真配置文件名称
            - sim_task: 仿真任务名
            - ckpt_setting: 模型检查点路径 (必需)
            - device: 推理设备
            - mixed_precision: 混合精度模式
            - dataset_stats_path: 数据集统计文件路径
            - action_horizon: 动作块长度
            - replan_steps: 重规划步长
            - num_inference_steps: 推理步数
            - sigma_shift: sigma 偏移
            - seed: 随机种子
            - text_cfg_scale: 文本 CFG 比例
            - negative_prompt: 负向提示词
            - rand_device: 随机数生成设备
            - tiled: 是否使用分块 VAE
            - timing_enabled: 是否启用时序统计

    Returns:
        WorldActionRobotWinPolicy: 配置完成且就绪的策略实例。

    使用示例:
        >>> usr_args = {"ckpt_setting": "/path/to/ckpt.pt",
        ...             "dataset_stats_path": "/path/to/stats.json"}
        >>> policy = get_model(usr_args)
        >>> policy.step(env, obs)
    """
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    """执行单步策略评估（RoboTwin 官方框架标准接口）。

    编码观测后调用模型的 step 方法执行一次推理+动作。

    Args:
        TASK_ENV: 仿真任务环境对象。
        model: WorldActionRobotWinPolicy 策略实例。
        observation: 可选的环境观测。重规划时必须提供。
    """
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    """重置策略状态以开始新的 episode（RoboTwin 官方框架标准接口）。

    Args:
        model: WorldActionRobotWinPolicy 策略实例。
    """
    model.reset()
