import numpy as np
import os
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Any, Optional, Sequence, Union

from .helpers.loader import load_wan22_ti2v_5b_components
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import WanVideoDiT


"""
Wan22Core 独立视频模型模块。

该文件定义了 Wan22Core 类，是 Wan2.2-TI2V-5B 模型的轻量封装，
提供了无需 pipeline API 的纯 PyTorch 训练和推理接口。

Wan22Core 是 FastWAM 体系中视频专家的基础类，FastWAM 的视频专家
（video_expert）即基于此能力构建。核心功能包括：
  - 视频扩散训练（training_loss）：对视频隐空间加噪，预测 Flow Matching 目标
  - 文本条件生成推理（infer）：从噪声逐步去噪为视频
  - VAE 编码/解码（像素 <-> 隐空间）
  - 文本提示编码
  - 模型检查点保存/加载
"""


class Wan22Core(torch.nn.Module):
    """Wan22Core 独立视频扩散模型类。

    作为 Wan2.2-TI2V-5B 的独立封装，支持视频生成任务的训练和推理。
    使用 Flow Matching 调度器进行噪声调度，结合文本提示条件生成视频。

    训练时：
      - 对视频 VAE 隐空间表示加噪，预测 Flow Matching 的噪声目标
      - 支持首帧 VAE 融合（fuse_vae_embedding_in_latents）以保持条件帧不变

    推理时：
      - 以首帧图像和文本提示为条件，从纯噪声逐步去噪为完整视频
      - 支持文本 CFG（Classifier-Free Guidance）增强生成质量
      - 支持动作条件（当 DiT 配置了 action_conditioned 时）

    使用 from_wan22_pretrained 工厂方法加载预训练权重。
    """

    def __init__(
        self,
        dit: WanVideoDiT,
        vae,
        text_encoder,
        tokenizer,
        device="cpu",
        torch_dtype=torch.float32,
        train_shift: float = 5.0,
        infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
    ):
        """初始化 Wan22Core 模型。

        参数:
            dit (WanVideoDiT): 视频 DiT 骨干网络，负责隐空间的噪声预测
            vae: VAE 模型，用于像素<->隐空间的编码/解码
            text_encoder: 文本编码器，将提示文本编码为条件向量
            tokenizer: 分词器，配合 text_encoder 使用
            device (str): 计算设备，默认 "cpu"
            torch_dtype (torch.dtype): 张量数据类型，默认 torch.float32
            train_shift (float): 训练时 Flow Matching 调度器的 shift 参数
            infer_shift (float): 推理时 Flow Matching 调度器的 shift 参数
            num_train_timesteps (int): 训练时间步数
        """
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.train_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=train_shift,
        )
        self.infer_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=infer_shift,
        )
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device="cuda",
        torch_dtype=torch.bfloat16,
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        redirect_common_files=True,
        dit_config: dict[str, Any] | None = None,
        train_shift: float = 5.0,
        infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
    ):
        """从预训练的 Wan2.2-TI2V-5B 检查点加载并构建 Wan22Core 模型。

        使用 load_wan22_ti2v_5b_components 辅助函数加载所有组件，
        包括 DiT、VAE、文本编码器和分词器。

        参数:
            device (str): 目标设备，默认 "cuda"
            torch_dtype (torch.dtype): 模型数据类型，默认 torch.bfloat16
            model_id (str): HuggingFace 上的模型 ID
            tokenizer_model_id (str): 分词器模型 ID
            tokenizer_max_len (int): 分词器最大长度
            redirect_common_files (bool): 是否重定向公共文件
            dit_config (dict, 可选): DiT 配置字典
            train_shift (float): 训练调度器 shift 参数
            infer_shift (float): 推理调度器 shift 参数
            num_train_timesteps (int): 训练时间步数

        返回:
            Wan22Core: 构建完成的模型实例
        """
        if dit_config is None:
            raise ValueError("`dit_config` is required for Wan22Core.from_wan22_pretrained().")
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=dit_config,
        )
        model = cls(
            dit=components.dit,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            device=device,
            torch_dtype=torch_dtype,
            train_shift=train_shift,
            infer_shift=infer_shift,
            num_train_timesteps=num_train_timesteps,
        )
        model.model_paths = {
            "dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
        }
        return model

    def to(self, *args, **kwargs):
        """将模型及其子模块移动到指定设备/数据类型。

        确保 dit、text_encoder 和 vae 同步移动。

        参数:
            *args, **kwargs: 传递给 torch.nn.Module.to() 的参数
        """
        super().to(*args, **kwargs)
        self.dit.to(*args, **kwargs)
        self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        """检查并调整视频维度至模型要求的对齐值。

        模型要求：H % 16 == 0, W % 16 == 0, T % 4 == 1。

        参数:
            height (int): 输入高度
            width (int): 输入宽度
            num_frames (int): 帧数

        返回:
            tuple[int, int, int]: 对齐后的 (height, width, num_frames)
        """
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        """将文本提示编码为条件向量序列。

        输入: prompt [batch_size] 或 [batch_size, ...]（字符串或字符串列表）
        输出: prompt_emb [B, L, D], mask [B, L]
              - B: batch_size
              - L: token 序列长度
              - D: text encoder 的特征维度
        """
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        return prompt_emb.to(device=self.device), mask

    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        """将视频像素张量编码为 VAE 隐空间表示。

        输入: video_tensor [B, 3, T, H, W]
        输出: z [B, C, T_lat, H_lat, W_lat]
              T_lat = (T-1) // temporal_downsample + 1
              H_lat = H // upsampling_factor, W_lat = W // upsampling_factor
        """
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        """将单张输入图像编码为 VAE 隐空间表示。

        用于推理时提供视频生成的首帧条件。

        输入: input_image [1, 3, H, W] 或 [3, H, W]
        输出: z [1, C, 1, H_lat, W_lat]
        """
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        """将 VAE 隐空间张量解码为 PIL 图像帧列表。

        输入: latents [1, C, T_lat, H_lat, W_lat]
        输出: list[Image] — 长度 = 视频帧数
              每帧为 PIL Image 格式 RGB 图像
        """
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def _model_fn(self,
                  latents: torch.Tensor, # [B, C, T, H, W]
                  timestep: torch.Tensor, # [B] or [1] (inference mode)
                  context: torch.Tensor, # [B, L, D]
                  context_mask: Optional[torch.Tensor] = None, # [B, L]
                  action: Optional[torch.Tensor] = None, # [B, T-1, a_dim]
                  fuse_vae_embedding_in_latents=False):
        """模型前向预测的统一入口。

        封装调用 self.dit，是推理和训练中多次使用的公共接口。

        输入:
            latents [B, C, T_lat, H_lat, W_lat]: 加噪的视频隐空间张量
            timestep [B]: 当前时间步
            context [B, L, D]: 文本条件上下文
            context_mask [B, L], 可选: 上下文 mask
            action [B, T, a_dim], 可选: 动作条件（用于 action-conditioned DiT）
            fuse_vae_embedding_in_latents (bool): 是否使用 VAE 首帧融合

        输出:
            pred [B, C, T_lat, H_lat, W_lat]: 预测的 Flow Matching 目标/噪声
        """
        return self.dit(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )

    def build_inputs(self, sample, tiled=False):
        """从数据样本构建模型训练所需的所有输入张量。

        处理步骤：
          1. 校验视频张量维度 [B, 3, T, H, W] 和空间/时间对齐要求
          2. 处理提示文本：将字符串或字符串列表编码为条件向量
          3. 将视频编码为 VAE 隐空间张量
          4. 检查首帧融合标志（fuse_vae_embedding_in_latents）
          5. 处理可选的动作条件张量

        输入 sample 的键:
          - "video" [B, 3, T, H, W]: 视频像素张量
          - "prompt" (str 或 list[str]): 文本提示
          - "action" [B, T, a_dim], 可选: 动作条件

        输出 dict 的键:
          - "context" [B, L, D]: 文本条件嵌入
          - "context_mask" [B, L]: 条件 mask
          - "input_latents" [B, C, T_lat, H_lat, W_lat]: VAE 隐空间张量
          - "first_frame_latents" [B, C, 1, H_lat, W_lat] 或 None: 首帧隐变量
          - "fuse_vae_embedding_in_latents" (bool): VAE 首帧融合标志
          - "action" [B, T, a_dim] 或 None: 动作条件
        """
        video = sample["video"]
        prompt = sample["prompt"]
        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"`sample['video']` must be a torch.Tensor with shape [B, 3, T, H, W], got {type(video)}"
            )
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        if isinstance(prompt, str):
            prompt_list = [prompt]
        elif isinstance(prompt, Sequence):
            prompt_list = list(prompt)
        else:
            raise TypeError(f"`sample['prompt']` must be str or list[str], got {type(prompt)}")

        batch_size, _, num_frames, height, width = video.shape
        if len(prompt_list) != batch_size:
            raise ValueError(
                f"Prompt batch mismatch: got len(prompt)={len(prompt_list)} and video batch={batch_size}"
            )
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")

        input_video = video.to(device=self.device, dtype=self.torch_dtype)
        input_latents = self._encode_video_latents(input_video, tiled=tiled) # [B, C, Latent_T, H', W']

        first_frame_latents = None
        fuse_flag = False
        # 如果 DiT 配置了融合 VAE 首帧嵌入，保留首帧隐变量用于后续条件约束
        if getattr(self.dit, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True
        context, context_mask = self.encode_prompt(prompt_list)

        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor with shape [B, T, a_dim], got {type(action)}"
                )
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] <= 0:
                raise ValueError(f"`sample['action']` temporal dimension must be positive, got {action.shape[1]}")
            if action.shape[1] % (num_frames - 1) != 0:
                raise ValueError(
                    "`sample['action']` temporal dimension must be divisible by video transitions "
                    f"({num_frames - 1}), got {action.shape[1]}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
        }

    def training_loss(self, sample, tiled=False):
        """计算视频扩散训练损失。

        训练流程：
          1. 调用 build_inputs 处理样本，得到隐空间张量和条件
          2. 对隐空间张量添加噪声（Flow Matching 调度器）
          3. 计算训练目标：噪声与干净隐空间张量之差（Flow Matching target）
          4. 通过 _model_fn（DiT）预测噪声
          5. 计算 MSE 损失，根据时间步加权平均

        输入: sample (dict) — 包含 video, prompt 等键值
              tiled (bool): VAE 是否使用分块处理

        输出: (loss_total, loss_dict)
              loss_total [Tensor]: 标量损失
              loss_dict (dict): {"loss_video": float}
        """
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        action = inputs["action"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]

        # 1. Continuous timestep sampling and noise injection.
        # 1. 连续时间步采样和噪声注入
        noise = torch.randn_like(input_latents)
        timestep = self.train_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_scheduler.add_noise(input_latents, noise, timestep)
        target = self.train_scheduler.training_target(input_latents, noise, timestep)

        # 2. fix first latent
        # 2. 固定首帧隐变量（保持条件帧不变，不参与噪声预测）
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0: 1] = inputs["first_frame_latents"]

        pred = self._model_fn(
            latents=latents, # [B, C, Latent_T, H', W']
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        # 如果首帧被融合，排除首帧的预测（首帧已知无损失）
        if inputs["first_frame_latents"] is not None:
            pred = pred[:, :, 1:]
            target = target[:, :, 1:]
        # 计算逐样本的 MSE 损失（在所有空间和时间维度上平均）
        loss_per_sample = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3, 4))
        # Flow Matching 训练权重：不同时间步的损失贡献不同
        sample_weight = self.train_scheduler.training_weight(timestep).to(
            loss_per_sample.device, dtype=loss_per_sample.dtype
        )
        loss_total = (loss_per_sample * sample_weight).mean()
        loss_dict = {
            "loss_video": float(loss_total.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def infer(
        self,
        prompt: str,
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        **kwargs
    ):
        """视频推理生成。

        从纯噪声开始，逐步去噪生成视频。支持文本 CFG 和动作 CFG 增强。

        推理流程：
          1. 校验输入图像和参数维度
          2. 编码首帧图像为 VAE 隐空间张量，替代噪声张量的首帧位置
          3. 编码文本提示为条件向量（正向），以及可选的负向提示
          4. 构建 Flow Matching 推理时间调度表
          5. 逐步去噪：每步通过 DiT 预测噪声，执行调度器步进
          6. 支持 CFG：文本 CFG（text_cfg_scale）和动作 CFG（action_cfg_scale）
          7. 解码隐空间为 PIL 帧列表

        参数:
            prompt (str): 正向文本提示
            input_image [1, 3, H, W] 或 [3, H, W]: 条件首帧图像
            num_frames (int): 要生成的视频帧数（需满足 T % 4 == 1）
            action [T, a_dim] 或 [1, T, a_dim], 可选: 动作条件
            negative_prompt (str, 可选): 负向文本提示
            text_cfg_scale (float): 文本 CFG 缩放系数 > 1 启用 CFG
            action_cfg_scale (float): 动作 CFG 缩放系数 > 1 且 action 非 None 时启用
            num_inference_steps (int): 去噪步数
            sigma_shift (float, 可选): 调度器 shift 覆盖值
            seed (int, 可选): 随机种子
            rand_device (str): 随机数生成设备
            tiled (bool): VAE 是否使用分块处理
            **kwargs: 额外的兼容性参数

        返回:
            dict: {"video": list[Image]} — 生成的视频帧列表
        """
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_frames:
            raise ValueError(
                f"`num_frames` must satisfy T % 4 == 1, got {num_frames}"
            )

        # 计算隐空间维度
        latent_t = (num_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        # 生成初始噪声
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if action is not None:
            action = action.to(device=self.device, dtype=latents.dtype)
            if action.ndim != 2:
                raise ValueError(f"`action` must be 2D [T, a_dim], got shape {tuple(action.shape)}")
            action = action.unsqueeze(0) # [1, T, a_dim]
            if action.shape[1] % (num_frames - 1) != 0:
                raise ValueError(
                    "`action` temporal dimension must be divisible by `num_frames - 1`, "
                    f"got {action.shape[1]} vs {num_frames - 1}"
                )
        if action_cfg_scale != 1.0 and action is None:
            raise ValueError("`action_cfg_scale` != 1.0 requires non-null `action` input.")

        # 编码首帧并替代噪声首帧
        z = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents[:, :, 0:1] = z
        first_frame_latents = z
        fuse_flag = True

        # 编码正向和负向文本提示
        context_posi, context_posi_mask = self.encode_prompt(prompt)
        context_nega = None
        context_nega_mask = None
        if text_cfg_scale != 1.0:
            context_nega, context_nega_mask = self.encode_prompt("" if negative_prompt is None else negative_prompt)
        # 构建动作 CFG 的负向条件：零动作
        action_nega = torch.zeros_like(action) if (action is not None and action_cfg_scale != 1.0) else None

        # 构建推理时间调度表
        infer_timesteps, infer_deltas = self.infer_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents.dtype,
            shift_override=sigma_shift,
        )
        # 逐步去噪循环
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep = step_t.unsqueeze(0).to(dtype=latents.dtype, device=self.device)
            noise_pred_posi = self._model_fn(
                latents=latents,
                timestep=timestep,
                context=context_posi,
                context_mask=context_posi_mask,
                action=action,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            noise_pred = noise_pred_posi
            # 文本 CFG：正向预测 + scale * (正向 - 负向)
            if context_nega is not None:
                noise_pred_text_nega = self._model_fn(
                    latents=latents,
                    timestep=timestep,
                    context=context_nega,
                    context_mask=context_nega_mask,
                    action=action,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                noise_pred = noise_pred + (text_cfg_scale - 1.0) * (noise_pred_posi - noise_pred_text_nega)
            # 动作 CFG：正向预测 + scale * (正向 - 零动作)
            if action_nega is not None:
                noise_pred_action_nega = self._model_fn(
                    latents=latents,
                    timestep=timestep,
                    context=context_posi,
                    context_mask=context_posi_mask,
                    action=action_nega,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                noise_pred = noise_pred + (action_cfg_scale - 1.0) * (noise_pred_posi - noise_pred_action_nega)
            latents = self.infer_scheduler.step(noise_pred, step_delta, latents)
            # 步进后恢复首帧为条件帧
            latents[:, :, 0:1] = first_frame_latents

        return {"video": self._decode_latents(latents, tiled=tiled)}

    def save_checkpoint(self, path, optimizer=None, step=None):
        """保存模型检查点到磁盘。

        参数:
            path (str): 保存路径
            optimizer: PyTorch 优化器实例（可选）
            step (int): 当前训练步数（可选）
        """
        payload = {
            "dit": self.dit.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        """从磁盘加载模型检查点。

        参数:
            path (str): 检查点路径
            optimizer: PyTorch 优化器实例（可选，将更新其状态）

        返回:
            payload (dict): 从检查点加载的原始字典
        """
        payload = torch.load(path, map_location="cpu")
        self.dit.load_state_dict(payload["dit"], strict=False)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        """前向传播。默认调用 training_loss。

        参数:
            *args, **kwargs: 传递给 training_loss 的参数
        """
        return self.training_loss(*args, **kwargs)
