"""
FastWAM 训练器模块 —— Wan22Trainer 类定义。

该模块实现了 Wan22Trainer 训练器，负责管理视频-动作扩散模型的整个训练流程：
    - 使用 HuggingFace Accelerate 进行分布式训练和混合精度管理
    - 周期性执行评估（PSNR/SSIM 视频指标 + 动作 L1/L2 误差）
    - 检查点保存/恢复（权重检查点 + 加速器训练状态）
    - 学习率调度（Cosine 衰减 / 常数 + 预热）
    - 数据集长度一致性检查（跨所有分布式 rank）
    - Weights & Biases 日志记录

典型用法（通常在 run_training 内部调用）：
    trainer = Wan22Trainer(cfg, model, train_dataset, val_dataset)
    trainer.train()
"""

import logging
import json
import inspect
import os
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    """
    Wan22 模型的训练器，封装了完整的训练循环、评估和检查点管理。

    该训练器基于 HuggingFace Accelerate 实现，支持：
        - 单卡/多卡（DataParallel / DeepSpeed ZeRO）分布式训练
        - FP16/BF16 混合精度
        - 梯度累积和梯度裁剪
        - Cosine 学习率衰减 + 预热
        - 周期性验证评估（PSNR, SSIM, 动作 L1/L2）
        - Weights & Biases 实验跟踪
        - 训练中断恢复（加载加速器状态 + 数据加载器偏移）

    属性:
        model: 被训练的模型（经过 accelerator.prepare 包装）
        train_dataset: 训练数据集
        val_dataset: 验证数据集（可选）
        cfg: 完整配置
        global_step (int): 当前全局训练步数
        epoch (int): 当前 epoch 数
    """

    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        """
        初始化 Wan22Trainer 训练器。

        构造函数执行以下初始化流程：
            1. 解析配置参数（学习率、batch size、epoch 数等）
            2. 初始化 HuggingFace Accelerator（管理分布式和混合精度）
            3. 检查数据集长度在各 rank 间的一致性
            4. 冻结非 DiT 模块，仅训练 DiT + 可选的本体感知编码器
            5. 构建 AdamW 优化器、数据加载器和学习率调度器
            6. 创建输出目录结构（checkpoints/weights, checkpoints/state, eval）
            7. 通过 accelerator.prepare 包装模型/优化器/数据加载器
            8. 初始化 W&B（如果启用）并恢复检查点（如果指定）

        参数:
            model: 待训练的 PyTorch 模型
            train_dataset: 训练数据集（需实现 __len__）
            val_dataset: 验证数据集（可选，需实现 __len__）
            cfg: Hydra/OmegaConf 完整配置，包含所有训练超参数
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.persistent_workers = bool(cfg.get("persistent_workers", False)) and self.num_workers > 0
        prefetch_factor = cfg.get("prefetch_factor", None)
        self.prefetch_factor = None if prefetch_factor is None else int(prefetch_factor)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)

        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        # 初始化 Accelerator：管理分布式训练、混合精度、梯度累积
        # step_scheduler_with_optimizer=False 表示手动控制 scheduler.step()
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )

        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)

        # 检查数据集长度在分布式各 rank 间是否一致（防止数据切分错误）
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # 在优化器/DeepSpeed 初始化前冻结非训练模块。
        # 这确保 ZeRO 构建优化器状态时只包含 DiT、可选本体感知编码器和 goal adapter 的可训练参数。
        self._apply_dit_only_train_mode(self.model)
        trainable_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        video_goal_adapter = getattr(self.model, "video_goal_adapter", None)
        if video_goal_adapter is not None and any(p.requires_grad for p in video_goal_adapter.parameters()):
            trainable_params.extend(list(video_goal_adapter.parameters()))
        direction_embedding = getattr(self.model, "direction_embedding", None)
        if direction_embedding is not None and any(p.requires_grad for p in direction_embedding.parameters()):
            trainable_params.extend(list(direction_embedding.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        # 创建输出目录结构
        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        # 使用 accelerator.prepare 包装模型/优化器/数据加载器/调度器
        # 这会将模型移至正确设备、为分布式训练包装 DataLoader、配置 DeepSpeed 等
        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        """
        初始化 Weights & Biases 实验跟踪。

        仅在 wandb_enabled 为 True 且为主进程时初始化。
        如果 wandb 未安装则抛出 ImportError。
        """
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        """
        向 W&B 记录指标数据。

        参数:
            payload (dict): 要记录的键值对字典，将写入当前 global_step
        """
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        """结束 W&B 运行并清理资源。"""
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        """
        构建可恢复的数据加载器，使用 ResumableEpochSampler 支持训练中断续跑。

        参数:
            dataset: 训练数据集
            worker_init_fn: DataLoader 的 worker 初始化函数（用于可复现的随机种子）

        返回:
            DataLoader: 配置好的数据加载器
        """
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "sampler": self.train_sampler,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": worker_init_fn,
            "persistent_workers": self.persistent_workers,
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(dataset, **loader_kwargs)

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        """
        检查数据集长度在所有分布式 rank 间是否一致。

        通过 Accelerator 的 gather 操作收集各 rank 的数据集长度，
        如果长度不一致则抛出 RuntimeError。这是防止数据切分错误的重要校验。

        参数:
            dataset: 待检查的数据集
            dataset_name (str): 数据集名称（用于日志和错误消息）
        """
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        """
        估算总训练步数。

        如果配置中指定了 max_steps，则直接使用该值。
        否则根据数据集大小、分布式进程数、梯度累积步数和 epoch 数计算：
            global_batch_size = batch_size * num_processes
            micro_steps_per_epoch = ceil(len(dataset) / global_batch_size)
            opt_steps_per_epoch = ceil(micro_steps_per_epoch / grad_accum_steps)
            total_steps = opt_steps_per_epoch * num_epochs

        返回:
            int: 总训练步数

        异常:
            TypeError: 当 max_steps 为 None 且数据集未实现 __len__ 时
        """
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        """
        构建学习率调度器，支持余弦衰减或常数学习率，可选预热阶段。

        预热阶段使用 LinearLR 从极小的学习率线性增长到初始学习率，
        主阶段使用 CosineAnnealingLR（余弦衰减到初始学习率的 1%）或 ConstantLR。

        参数:
            scheduler_type (str): 调度器类型，"cosine" 或 "constant"
            total_train_steps (int): 总训练步数
            warmup_steps (int): 预热步数，默认 0（无预热）

        返回:
            _LRScheduler: 配置好的学习率调度器（可能是 SequentialLR 组合调度器）

        异常:
            ValueError: 不支持的调度器类型
        """
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        """
        估算训练完成所需剩余时间（ETA）。

        基于当前已花费的时间和已完成步数计算每秒步数，
        从而推算剩余步数所需时间。

        返回:
            tuple: (eta_str, steps_per_sec)
                - eta_str (str): 格式化的剩余时间 "HH:MM:SS"
                - steps_per_sec (float): 当前每秒训练步数
        """
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        """
        恢复或加载检查点。

        支持两种模式：
            1. 目录路径：加载完整的加速器训练状态（优化器、调度器、数据加载器偏移等）
            2. .pt 文件路径：仅加载模型权重（不恢复优化器/调度器状态）

        注意：在 DeepSpeed ZeRO-2 下，仅加载 .pt 权重将丢失优化器/调度器/步数状态。
        """
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        """
        将包装后的模型设置为"仅 DiT 训练模式"。

        解包 Accelerator 包装的模型后，调用 _apply_dit_only_train_mode。
        此方法用于 eval 结束后恢复训练状态。
        """
        # 匹配 DiffSynth 的 freeze_except("dit") 行为：仅 DiT 保持可训练/训练模式
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        """
        将模型设置为仅 DiT（+ 可选的本体感知编码器 / goal adapter）训练模式。

        将所有模块设为 eval 并冻结梯度，然后仅对 model.dit、
        model.proprio_encoder（如果存在）和可训练 goal adapter 启用训练模式和梯度计算。
        视频 VAE、文本编码器和其他辅助模块保持冻结。

        参数:
            model: 待修改的模型
        """
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
        video_goal_adapter = getattr(model, "video_goal_adapter", None)
        if video_goal_adapter is not None and bool(getattr(model, "train_video_goal_adapter", False)):
            video_goal_adapter.train()
            video_goal_adapter.requires_grad_(True)
        direction_embedding = getattr(model, "direction_embedding", None)
        if direction_embedding is not None:
            direction_embedding.train()
            direction_embedding.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        """
        将单个评估样本转换为 batch 格式，统一维度检查。

        输入样本预期包含 "video"（[3,T,H,W] 或 [B,3,T,H,W]）和 "prompt"（str 或 list[str]），
        可选包含 "action"（[B,T,a_dim]）、"proprio"（[B,T,d]）、"context"/"context_mask"。

        该方法确保所有张量至少有 batch 维度，并校验各维度一致性。

        输入维度示例:
            - video: [3, T=16, H=320, W=480] -> 扩展为 [1, 3, 16, 320, 480]
            - action: [T', a_dim=7] -> 扩展为 [1, T', 7]
            - proprio: [T'', d=12] -> 扩展为 [1, T'', 12]

        参数:
            sample (dict): 数据集返回的单样本字典

        返回:
            dict: 规范化后的 batch 字典，包含 video, prompt, action, proprio, context,
                  context_mask, action_horizon

        异常:
            TypeError: 类型不匹配时
            ValueError: 维度或 shape 不符合预期时
        """
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        """
        执行一次验证评估，计算多项指标并生成对比视频。

        评估流程：
            1. 从验证集中随机选择一个样本
            2. 计算验证损失（training_loss）
            3. 使用模型进行推理生成预测视频和动作
            4. 计算视频质量指标：PSNR（预测 vs GT, 预测 vs 重建, 重建 vs GT）和 SSIM
            5. 如果存在动作数据，通过数据集 processor 反归一化后计算动作 L1/L2 误差
            6. 计算 VAE 重建质量指标（编码-解码重建 vs GT）
            7. 拼接预测视频、VAE 重建视频和 GT 视频并保存为 MP4
            8. 通过 accelerate.gather_for_metrics 汇总所有 rank 的指标

        返回:
            dict | None: 包含以下键的字典（验证集为 None 时返回 None）:
                - val_loss: 验证损失
                - psnr_rg: 预测 vs GT 的 PSNR
                - ssim_rg: 预测 vs GT 的 SSIM
                - psnr_rd: 预测 vs 重建的 PSNR
                - ssim_rd: 预测 vs 重建的 SSIM
                - psnr_dg: 重建 vs GT 的 PSNR
                - ssim_dg: 重建 vs GT 的 SSIM
                - video_path: 拼接视频路径
                - action_l2: 动作 L2 误差（如果存在动作数据）
                - action_l1: 动作 L1 误差（如果存在动作数据）
        """
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # 使用全局步数 + 进程索引作为种子，确保各 rank 选取不同样本
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. 计算验证损失（使用模型的 training_loss 方法）
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()

        prompt = sample["prompt"][0]
        video0 = sample["video"][0]  # Tensor [3, T, H, W] 在 [-1, 1] 范围内
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None  # 从 [1, T, d] 取第 0 帧 [d]
        input_image = video0[:, 0].unsqueeze(0)  # 取视频第 0 帧作为条件输入图像
        _, num_frames, _, _ = video0.shape

        # 2. 推理生成视频，使用条件图像和可选的文本提示/上下文
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,  # 评估时使用无分类器引导的默认缩放
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,  # 评估时使用完整分辨率，不启用分块
        }
        if sample["context"] is not None:
            # 使用预计算的文本嵌入上下文（绕过文本编码器）
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )

        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. 计算推理视频相对于 GT 的指标
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()  # 从 [-1,1] 映射到 [0,1]

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        # 如果存在动作数据，计算动作预测误差
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)

            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            # 对预测动作和 GT 动作进行反归一化
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                # 反归一化流程：先解包动作状态融合，再反归一化，再重新打包
                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE 重建质量指标：将 GT 视频编码到隐空间再解码，评估 VAE 自身的信息损失
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        # 推理视频 vs VAE 重建（衡量扩散模型相对于 VAE 上限的额外退化）
        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        # 拼接三个视频（预测 | VAE 重建 | GT）以便可视对比
        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        """
        保存模型权重检查点（仅 .pt 文件，不含优化器状态）。

        参数:
            step_tag (str): 步数标签，如 "step_000042"

        返回:
            str: 检查点文件的完整路径
        """
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        """
        保存训练器状态元数据（JSON 文件），用于训练恢复。

        记录 global_step、epoch 和 batch_in_epoch，以便准确恢复数据加载器位置。

        参数:
            state_path (str): 状态保存目录路径
        """
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        """
        保存完整检查点，包含模型权重和加速器训练状态。

        检查点包含三部分：
            1. 模型权重 .pt 文件（仅主进程保存）
            2. 加速器完整状态（优化器、调度器、DeepSpeed ZeRO 状态等，所有进程保存）
            3. trainer_state.json 元数据（全局步数、epoch、batch 偏移）

        返回:
            dict: {"weights_path": .pt 路径, "state_path": 状态目录路径}
        """
        step_tag = f"step_{self.global_step:06d}"

        # 先由主进程保存权重，其他进程等待
        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        # 然后所有进程保存加速器完整状态（确保 DeepSpeed ZeRO 分片一致）
        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        """
        从目录加载完整训练状态（加速器状态 + 训练器元数据）。

        加速器状态包含优化器、调度器和 DeepSpeed ZeRO 分片状态。
        训练器状态（trainer_state.json）包含 global_step、epoch 和 batch_in_epoch，
        用于准确恢复数据加载器的数据读取位置。

        如果 trainer_state.json 缺失，则仅恢复加速器状态并尝试从目录名解析步数，
        数据加载器偏移恢复将被跳过。

        参数:
            state_dir (str): 状态目录路径，包含加速器子文件夹和可选的 trainer_state.json
        """
        # torch_npu can rebuild checkpoint tensors with requires_grad=True and
        # then call Tensor.set_ during torch.load, which autograd rejects unless
        # checkpoint loading is explicitly outside grad tracking.
        with torch.no_grad():
            self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                # 完整恢复：包含数据加载器进度
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                # 不完整的恢复：仅恢复优化器/调度器，从头开始遍历数据
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        # 无 trainer_state.json：尝试从目录名（如 step_000042）解析全局步数
        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        """
        执行主训练循环。

        训练循环以 while 循环方式运行，每次从数据加载器取一个 batch：
            1. 进入 accelerator.accumulate 上下文（自动处理梯度累积）
            2. 使用 autocast 进行混合精度前向传播
            3. 计算损失并反向传播
            4. 达到梯度累积步数后执行优化器步进和调度器步进
            5. 全局聚合损失和指标（跨所有 rank）
            6. 按 log_every 间隔输出日志和 W&B 记录
            7. 按 eval_every 间隔执行验证评估
            8. 按 save_every 间隔保存检查点

        当 global_step 达到 max_steps 时训练结束，自动保存最终检查点。

        关于数据加载器迭代：
            - 使用 iterator 方式（非 epoch 循环），自然地遍历数据
            - 捕获 StopIteration 时递增 epoch 计数器并重建 iterator
            - 支持通过 train_sampler 恢复训练时的 batch 偏移
        """
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                # 一个 epoch 结束，递增 epoch 并重新创建迭代器
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            # 梯度累积上下文：自动在多次微步后同步梯度
            with self.accelerator.accumulate(self.model):
                # 加速器包装的模型可能有 training_loss 方法，也可能需要解包
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    # 前向传播计算扩散损失
                    loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                # sync_gradients 在达到梯度累积步数时为 True，此时执行优化器步进
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    # 跨所有 rank 聚合损失值以得到全局平均
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    # 聚合各子损失项（如 video_loss, action_loss）
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    # 按 log_every 间隔记录训练日志到控制台和 W&B
                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    # 按 eval_every 间隔执行验证评估
                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    # 按 save_every 间隔保存检查点
                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    # 达到 max_steps 时保存最终检查点并退出
                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        # max_steps 耗尽后的最终保存（正常情况下不会执行到这里）
        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
