from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import (
    apply_video_backbone_preset,
    load_wan_video_components,
    resolve_video_backbone_type,
    sync_action_dit_config_with_video_backbone,
)
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)

"""
FastWAM（Fast World Action Model）主模型模块。

该文件定义了 FastWAM 类，是 FastWAM 项目的核心模型。FastWAM 通过 MoT（Mixture-of-Transformers）
混合注意力机制，将视频生成专家（基于 Wan2.2-TI2V）与动作生成专家（ActionDiT）深度融合，
实现视频-动作的联合扩散建模与推理。

主要功能：
  - 联合视频-动作训练（training_loss），支持 padding mask 和逐样本权重
  - 联合视频-动作推理（infer_joint），同步去噪生成视频和动作序列
  - 纯动作推理（infer_action），利用视频首帧预填充 KV 缓存加速
  - VAE 编码/解码，支持 tiled 分块处理
  - 文本提示编码及可选的 Proprioception（本体感知）编码融合
  - 模型检查点保存/加载（save_checkpoint / load_checkpoint）
"""


class FastWAM(torch.nn.Module):
    """FastWAM 主模型类。

    继承自 torch.nn.Module。
    通过 MoT 层混合视频专家（video_expert）和动作专家（action_expert）的注意力，
    实现视频-动作联合扩散。

    训练时（training_loss）：
      对视频和动作分别使用 Flow Matching 调度器采样时间步、加噪，
      各自经过 pre_dit 编码为 token 序列，在 MoT 层中进行混合注意力交互，
      最后通过 post_dit 重建预测噪声，计算 MSE 损失。

    推理时支持三种模式：
      - infer_joint：视频与动作同步去噪，每个去噪步联合预测两种模态的噪声
      - infer_action：仅动作去噪，视频侧使用预填充 KV 缓存以加速迭代
      - infer：infer_joint 的简单别名

    使用 from_wan22_pretrained 工厂方法从预训练检查点加载模型组件。
    """

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        video_latent_spatial_downsample_factor: int = 1,
        apply_video_latent_downsample_to_action_branch: bool = False,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        goal_token_config: Optional[dict] = None,
        subgoal_latent_config: Optional[dict] = None,
        bidirectional_config: Optional[dict] = None,
    ):
        """初始化 FastWAM 模型。

        参数:
            video_expert: 视频专家 DiT 模型（WanVideoDiT 实例），负责视频隐空间的去噪
            action_expert (ActionDiT): 动作专家 DiT 模型，负责动作序列的去噪
            mot (MoT): Mixture-of-Transformers 混合注意力层，管理视频与动作专家的注意力路由
            vae: VAE 模型，用于视频像素空间与隐空间之间的编码/解码
            text_encoder: 文本编码器（可选），将文本提示编码为条件向量
            tokenizer: 分词器（可选），配合 text_encoder 使用
            text_dim (int, 可选): 文本/条件向量的特征维度。未指定时从 text_encoder 自动获取
            proprio_dim (int, 可选): 本体感知向量的维度，非 None 时创建线性投影层
            device (str): 计算设备，默认 "cpu"
            torch_dtype (torch.dtype): 模型张量的数据类型，默认 torch.float32
            video_train_shift (float): 视频训练 Flow Matching 调度器的 shift 参数
            video_infer_shift (float): 视频推理 Flow Matching 调度器的 shift 参数
            video_num_train_timesteps (int): 视频训练时间步数
            action_train_shift (float): 动作训练 Flow Matching 调度器的 shift 参数
            action_infer_shift (float): 动作推理 Flow Matching 调度器的 shift 参数
            action_num_train_timesteps (int): 动作训练时间步数
            loss_lambda_video (float): 视频损失项的权重系数
            loss_lambda_action (float): 动作损失项的权重系数
        """
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        # 保留训练器兼容性：优化器和冻结逻辑使用 `model.dit` 属性指向 MoT
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        # 可选别名，与 Wan22Core 的命名保持一致
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.video_latent_spatial_downsample_factor = int(video_latent_spatial_downsample_factor)
        if self.video_latent_spatial_downsample_factor < 1:
            raise ValueError(
                "`video_latent_spatial_downsample_factor` must be >= 1, "
                f"got {self.video_latent_spatial_downsample_factor}."
            )
        self.apply_video_latent_downsample_to_action_branch = bool(
            apply_video_latent_downsample_to_action_branch
        )
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.subgoal_latent_config = dict(subgoal_latent_config or {})
        self.subgoal_latent_enabled = bool(self.subgoal_latent_config.get("enabled", False))

        # Goal token bank (optional, for IDM Stage 1 video/subgoal latent generation)
        self.goal_token_encoder: Optional[nn.Module] = None
        self.video_goal_adapter: Optional[nn.Linear] = None
        self.train_video_goal_adapter = False
        if goal_token_config is not None and bool(goal_token_config.get("enabled", True)):
            self._init_goal_token(goal_token_config)

        self.to(self.device)

    def _init_goal_token(self, goal_token_config: dict):
        """Initialise goal token encoder and video-goal adapter from config.

        Loads weights from an alignment checkpoint and validates config consistency.
        The encoder is frozen by default; the adapter trainability follows the config.
        """
        from .goal_token_bank import GoalTokenBank

        cfg = dict(goal_token_config)
        checkpoint_path = cfg["checkpoint_path"]
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_cfg = ckpt.get("config", {})
        # Validate critical dims match between user config and checkpoint
        for key in ("goal_dim", "num_goal_tokens", "num_heads"):
            if int(cfg.get(key, 0)) != int(ckpt_cfg.get(key, 0)):
                raise ValueError(
                    f"goal_token config mismatch for '{key}': "
                    f"config={cfg.get(key)}, checkpoint={ckpt_cfg.get(key)}"
                )
        goal_dim = int(cfg["goal_dim"])
        num_heads = int(cfg["num_heads"])
        num_goal_tokens = int(cfg["num_goal_tokens"])
        hidden_dim = int(ckpt_cfg.get("hidden_dim", cfg.get("hidden_dim", 2048)))

        self.goal_token_encoder = GoalTokenBank(
            text_dim=4096,
            goal_dim=goal_dim,
            num_goal_tokens=num_goal_tokens,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=float(ckpt_cfg.get("dropout", 0.0)),
        ).to(device=self.device, dtype=self.torch_dtype)
        # Load alignment weights (strict=False — goal_token_bank has extra image-path params)
        missing, unexpected = self.goal_token_encoder.load_state_dict(ckpt["model"], strict=False)
        if missing:
            logger.warning("GoalTokenBank missing keys: %s", missing)
        if unexpected:
            logger.warning("GoalTokenBank unexpected keys: %s", unexpected)

        if cfg.get("freeze_encoder", True):
            for p in self.goal_token_encoder.parameters():
                p.requires_grad = False
            self.goal_token_encoder.eval()

        self.video_goal_adapter = nn.Linear(goal_dim, self.video_expert.hidden_dim).to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        train_adapter = bool(cfg.get("train_video_adapter", True))
        self.train_video_goal_adapter = train_adapter
        self.video_goal_adapter.requires_grad_(train_adapter)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        video_backbone_type: str = "wan2_2_ti2v",
        video_backbone_name: str | None = None,
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        video_latent_spatial_downsample_factor: int = 1,
        apply_video_latent_downsample_to_action_branch: bool = False,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        goal_token_config: Optional[dict] = None,
        subgoal_latent_config: Optional[dict] = None,
        bidirectional_config: Optional[dict] = None,
    ):
        """从预训练的 Wan2.2-TI2V-5B 检查点加载并构建 FastWAM 模型。

        该工厂方法执行以下步骤：
          1. 调用 load_wan22_ti2v_5b_components 加载视频专家（DiT）、VAE、文本编码器等组件
          2. 从动作 DiT 配置和预训练路径加载 ActionDiT 专家
          3. 校验视频专家与动作专家的 num_heads、attn_head_dim、num_layers 一致
          4. 用两个专家构建 MoT 混合注意力层
          5. 组装 FastWAM 实例并记录所有组件的模型路径

        参数:
            device (str): 目标设备，默认 "cuda"
            torch_dtype (torch.dtype): 模型数据类型，默认 torch.bfloat16
            model_id (str): HuggingFace 上的 Wan2.2-TI2V-5B 模型 ID
            tokenizer_model_id (str): 分词器模型 ID
            tokenizer_max_len (int): 分词器最大长度
            load_text_encoder (bool): 是否加载文本编码器
            proprio_dim (int, 可选): 本体感知维度
            redirect_common_files (bool): 是否重定向公共文件
            video_dit_config (dict, 可选): 视频 DiT 配置字典
            action_dit_config (dict, 可选): 动作 DiT 配置字典
            action_dit_pretrained_path (str, 可选): 动作 DiT 预训练权重路径
            skip_dit_load_from_pretrain (bool): 是否跳过从预训练权重加载 DiT
            mot_checkpoint_mixed_attn (bool): MoT 是否启用混合注意力梯度检查点
            video_train_shift, video_infer_shift, video_num_train_timesteps: 视频调度器参数
            action_train_shift, action_infer_shift, action_num_train_timesteps: 动作调度器参数
            loss_lambda_video (float): 视频损失权重
            loss_lambda_action (float): 动作损失权重

        返回:
            FastWAM: 构建完成的 FastWAM 模型实例
        """
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")
        resolved_video_backbone_type = resolve_video_backbone_type(video_backbone_type)
        video_dit_config = apply_video_backbone_preset(
            dict(video_dit_config),
            resolved_video_backbone_type,
        )
        action_dit_config = {} if action_dit_config is None else dict(action_dit_config)
        action_dit_config = sync_action_dit_config_with_video_backbone(
            action_dit_config=action_dit_config,
            video_dit_config=video_dit_config,
        )

        components = load_wan_video_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            video_backbone_type=resolved_video_backbone_type,
            video_backbone_name=video_backbone_name,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model_kwargs = {}
        if bidirectional_config is not None:
            model_kwargs["bidirectional_config"] = bidirectional_config

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            video_latent_spatial_downsample_factor=video_latent_spatial_downsample_factor,
            apply_video_latent_downsample_to_action_branch=apply_video_latent_downsample_to_action_branch,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            goal_token_config=goal_token_config,
            subgoal_latent_config=subgoal_latent_config,
            **model_kwargs,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        """将模型及其子模块移动到指定设备/数据类型。

        重载父类的 to 方法，确保 mot、text_encoder 和 vae 也同步移动。

        参数:
            *args, **kwargs: 传递给 torch.nn.Module.to() 的参数
        """
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        """检查并调整视频的空间和时间维度至模型要求的对齐值。

        模型要求：
          - 高度和宽度为 16 的倍数
          - 帧数满足 T % 4 == 1

        参数:
            height (int): 输入图像高度
            width (int): 输入图像宽度
            num_frames (int): 期望的视频帧数

        返回:
            tuple[int, int, int]: 调整后的 (height, width, num_frames)
        """
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        """将文本提示编码为条件向量序列。

        使用内部的分词器（tokenizer）和文本编码器（text_encoder）将原始文本转换为
        密集向量表示。对 padding 位置进行清零处理以避免在 cross-attention 中产生伪影。

        输入: prompt [batch_size] 或 [batch_size, ...]（字符串序列）
        输出: prompt_emb [B, L, D], mask [B, L]
              - B: batch_size
              - L: token 序列长度
              - D: text_dim 特征维度
        """
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        # FIXME: 原始实现的零填充在 cross-attention 中可见
        # 对 padding 位置显式清零，消除可能的信息泄露
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """将本体感知（Proprioception）编码为 token 并追加到条件上下文序列末尾。

        输入: context [B, L, D], context_mask [B, L], proprio [B, D]
        输出: new_context [B, L+1, D], new_mask [B, L+1]
              - 在 context 末尾追加一个 proprio token，mask 相应扩展
        """
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        # 通过线性投影层将 proprio 映射到文本特征空间
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        # 构建对应的 mask，proprio token 始终可见（非 padding）
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _ensure_vae_device(self):
        # Some distributed wrappers only move trainable modules; keep frozen VAE aligned explicitly.
        vae_param = next(self.vae.parameters(), None)
        if vae_param is not None and vae_param.device != self.device:
            self.vae.to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        """将视频像素张量编码为 VAE 隐空间表示。

        输入: video_tensor [B, 3, T, H, W]（像素值，归一化到 [-1, 1]）
        输出: z [B, C, T_lat, H_lat, W_lat]
              - C: VAE 隐空间通道数
              - T_lat = (T - 1) // temporal_downsample_factor + 1
              - H_lat = H // upsampling_factor, W_lat = W // upsampling_factor
        """
        self._ensure_vae_device()
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        """将单张条件输入图像编码为 VAE 隐空间表示。

        用于推理时提供视频生成的首帧条件。

        输入: input_image [1, 3, H, W] 或 [3, H, W]（像素值）
        输出: z [1, C, 1, H_lat, W_lat]（单帧隐空间表示）
        """
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        self._ensure_vae_device()
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        """将 VAE 隐空间张量解码为 PIL 图像帧列表。

        输入: latents [1, C, T_lat, H_lat, W_lat]
        输出: list[Image] — 长度 = T_lat（解码后逐帧的 PIL 图像列表）
              每张图像尺寸: (H, W)，三通道 RGB
        """
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def _maybe_downsample_video_latents_for_backbone(
        self,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[dict[str, Any]]]:
        """Optionally compress full-video latent grids before the video backbone.

        This is intended for Wan2.1-T2V style backbones where the latent grid is much
        denser than Wan2.2-TI2V. Default factor=1 keeps the original FastWAM path
        exactly unchanged.
        """
        factor = int(self.video_latent_spatial_downsample_factor)
        if factor == 1:
            return latents, None
        if latents.ndim != 5:
            raise ValueError(f"`latents` must be [B, C, T, H, W], got shape {tuple(latents.shape)}")
        height = int(latents.shape[-2])
        width = int(latents.shape[-1])
        if height % factor != 0 or width % factor != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by "
                f"`video_latent_spatial_downsample_factor={factor}`, "
                f"got HxW=({height}, {width})."
            )
        latents_down = F.avg_pool3d(
            latents,
            kernel_size=(1, factor, factor),
            stride=(1, factor, factor),
        )
        return latents_down, {
            "original_spatial_shape": (height, width),
            "downsample_factor": factor,
        }

    def _restore_video_prediction_spatial_resolution(
        self,
        pred_video: torch.Tensor,
        compression_meta: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        if compression_meta is None:
            return pred_video
        if pred_video.ndim != 5:
            raise ValueError(f"`pred_video` must be [B, C, T, H, W], got shape {tuple(pred_video.shape)}")
        original_spatial_shape = compression_meta.get("original_spatial_shape")
        if original_spatial_shape is None or len(original_spatial_shape) != 2:
            raise ValueError("`compression_meta['original_spatial_shape']` must be a 2-tuple.")
        original_height = int(original_spatial_shape[0])
        original_width = int(original_spatial_shape[1])
        if pred_video.shape[-2:] == (original_height, original_width):
            return pred_video

        batch_size, channels, num_frames, _, _ = pred_video.shape
        pred_video_btc = pred_video.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames,
            channels,
            pred_video.shape[-2],
            pred_video.shape[-1],
        )
        pred_video_btc = F.interpolate(
            pred_video_btc,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )
        return pred_video_btc.reshape(
            batch_size,
            num_frames,
            channels,
            original_height,
            original_width,
        ).permute(0, 2, 1, 3, 4).contiguous()

    def _build_video_pre(
        self,
        latents_video: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        action: Optional[torch.Tensor] = None,
        apply_spatial_downsample: bool = True,
        extra_context_emb: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        compression_meta = None
        latents_for_backbone = latents_video
        if apply_spatial_downsample:
            latents_for_backbone, compression_meta = self._maybe_downsample_video_latents_for_backbone(latents_video)
        video_pre = self.video_expert.pre_dit(
            x=latents_for_backbone,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            extra_context_emb=extra_context_emb,
        )
        return video_pre, compression_meta

    def _use_lowres_video_training_objective(self) -> bool:
        return int(self.video_latent_spatial_downsample_factor) > 1

    def _prepare_video_training_targets(
        self,
        video_supervision_latents: torch.Tensor,
        timestep_video: torch.Tensor,
        first_frame_latents: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        video_supervision_latents_model = video_supervision_latents
        first_frame_latents_model = first_frame_latents
        apply_spatial_downsample = True
        restore_spatial_resolution = True

        if self._use_lowres_video_training_objective():
            video_supervision_latents_model, _ = self._maybe_downsample_video_latents_for_backbone(
                video_supervision_latents
            )
            if first_frame_latents is not None:
                first_frame_latents_model, _ = self._maybe_downsample_video_latents_for_backbone(
                    first_frame_latents
                )
            apply_spatial_downsample = False
            restore_spatial_resolution = False

        noise_video = torch.randn_like(video_supervision_latents_model)
        latents_video = self.train_video_scheduler.add_noise(
            video_supervision_latents_model,
            noise_video,
            timestep_video,
        )
        target_video = self.train_video_scheduler.training_target(
            video_supervision_latents_model,
            noise_video,
            timestep_video,
        )
        if first_frame_latents_model is not None:
            latents_video[:, :, 0:1] = first_frame_latents_model

        return {
            "video_supervision_latents_model": video_supervision_latents_model,
            "first_frame_latents_model": first_frame_latents_model,
            "latents_video": latents_video,
            "target_video": target_video,
            "apply_spatial_downsample": apply_spatial_downsample,
            "restore_spatial_resolution": restore_spatial_resolution,
        }

    def _decode_video_tokens(
        self,
        video_tokens: torch.Tensor,
        video_pre: dict[str, Any],
        compression_meta: Optional[dict[str, Any]],
        restore_spatial_resolution: bool = True,
    ) -> torch.Tensor:
        pred_video = self.video_expert.post_dit(video_tokens, video_pre)
        if not restore_spatial_resolution:
            return pred_video
        return self._restore_video_prediction_spatial_resolution(pred_video, compression_meta)

    def build_inputs(self, sample, tiled: bool = False):
        """从数据样本构建模型训练/推理所需的所有输入张量。

        该函数执行以下步骤：
          1. 校验视频张量的维度（必须为 [B, 3, T, H, W]）
          2. 校验视频帧数满足 T % 4 == 1，空间维度满足 16 的倍数
          3. 校验动作张量维度为 [B, T_a, D_a]
          4. 将视频像素编码为 VAE 隐空间张量
          5. 处理条件上下文（context）和可选的本体感知（proprio）编码
          6. 检查动作 padding mask（action_is_pad）和图像 padding mask（image_is_pad）的维度

        输入 sample 的键:
          - "video" [B, 3, T, H, W]: 视频像素张量
          - "context" [B, L, D]: 条件上下文嵌入
          - "context_mask" [B, L]: 上下文 mask
          - "action" [B, T_a, a_dim]: 动作序列（T_a 应为 (T-1) 的整数倍）
          - "proprio" [B, T, d], 可选: 本体感知序列
          - "action_is_pad" [B, T_a], 可选: 动作的 padding 标记
          - "image_is_pad" [B, T], 可选: 视频帧的 padding 标记

        输出 dict 的键:
          - "context" [B, L', D]: 处理后的条件上下文（可能含 proprio token）
          - "context_mask" [B, L']: 对应的 mask
          - "input_latents" [B, C, T_lat, H_lat, W_lat]: VAE 编码后的视频隐空间张量
          - "first_frame_latents" [B, C, 1, H_lat, W_lat]: 首帧隐空间表示（用于 fuse）
          - "fuse_vae_embedding_in_latents" (bool): 是否在隐空间中融合 VAE 首帧嵌入
          - "action" [B, T_a, a_dim]: 动作张量
          - "action_is_pad" [B, T_a], 可选
          - "image_is_pad" [B, T], 可选
        """
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )

        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        # 如果 DiT 配置了融合 VAE 首帧嵌入，则保留首帧隐变量
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            # 取序列中第一个 proprio 作为全局条件
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """构建 MoT 混合注意力的注意力掩码矩阵。

        控制视频 token 和动作 token 之间的注意力可见性：
          - video -> video: 使用视频专家的视频到视频注意力掩码（通常是 causal 模式）
          - action -> action: 完全可见（全 1）
          - action -> video: 仅能关注视频序列的首帧 token（前 video_tokens_per_frame 个）

        参数:
            video_seq_len (int): 视频 token 总数
            action_seq_len (int): 动作 token 总数
            video_tokens_per_frame (int): 每帧对应的视频 token 数量
            device: 目标设备

        返回:
            mask [total_seq_len, total_seq_len] 布尔张量，True 表示允许注意力
              total_seq_len = video_seq_len + action_seq_len
        """
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        # 动作 token 仅能关注视频的首帧 token（前 tokens_per_frame 个）
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        """逐样本计算视频损失，支持基于 padding 的 masked reduction。

        对每个样本，计算所有空间位置的 MSE 后按帧平均，然后根据 image_is_pad
        排除 padding 帧，最终在有效帧上取平均。

        输入:
            pred_video [B, C, T_lat, H_lat, W_lat]: 预测的视频噪声/目标
            target_video [B, C, T_lat, H_lat, W_lat]: 真实视频噪声/目标
            image_is_pad [B, T] 或 None: 输入视频帧的 padding 标记
            include_initial_video_step (bool): 是否将视频首帧计入损失

        返回:
            loss [B]: 每个样本的视频损失标量
        """
        # 计算逐 token 的 MSE 损失，然后在通道和空间维度平均
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        # 根据 VAE 的时间下采样因子，将帧级 padding 映射到隐空间时间步
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        # 排除首帧，将剩余帧分组为隐空间时间步
        tail_is_pad = image_is_pad[:, 1:]
        # 若一个隐时间步对应的所有帧都是 padding，则该时间步为 padding
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        # 在有效（非 padding）帧上计算平均 loss
        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample, tiled: bool = False):
        """计算单个 batch 的联合训练损失（视频 + 动作）。

        训练流程：
          1. 调用 build_inputs 处理输入样本，得到隐空间表示、条件和动作张量
          2. 对视频隐空间张量加噪（Flow Matching 调度器），得到加噪视频和训练目标
          3. 对动作张量加噪（Flow Matching 调度器），得到加噪动作和训练目标
          4. 视频和动作分别通过各自的 pre_dit 编码为 token 序列
          5. 构建 MoT 混合注意力掩码，通过 MoT 层进行跨模态注意力交互
          6. 分别通过 post_dit 得到预测的噪声/目标
          7. 计算视频损失（支持 image_is_pad masked reduction）和动作损失
          8. 合并为总损失，返回 loss_dict 包含各分量

        输入: sample (dict) — 包含 video, context, context_mask, action 等键值
              tiled (bool): VAE 编码是否使用分块处理

        输出: (loss_total, loss_dict)
              loss_total (Tensor): 标量总损失
              loss_dict (dict): {"loss_video": float, "loss_action": float}
        """
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        video_train_targets = self._prepare_video_training_targets(
            video_supervision_latents=input_latents,
            timestep_video=timestep_video,
            first_frame_latents=inputs["first_frame_latents"],
        )
        latents = video_train_targets["latents_video"]
        target_video = video_train_targets["target_video"]

        # --- 动作分支加噪 ---
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # --- 视频和动作分别经过 pre_dit ---
        video_pre, compression_meta = self._build_video_pre(
            latents_video=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            apply_spatial_downsample=video_train_targets["apply_spatial_downsample"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        # --- 构建混合注意力掩码，执行 MoT 交互 ---
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        # --- post_dit 重建 ---
        pred_video = self._decode_video_tokens(
            tokens_out["video"],
            video_pre,
            compression_meta,
            restore_spatial_resolution=video_train_targets["restore_spatial_resolution"],
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        # --- 视频损失计算 ---
        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            # 如果首帧被 fuse 了，则排除首帧的预测（首帧已知无损失）
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        # Flow Matching 训练权重（根据时间步调整不同噪声水平的贡献）
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        # --- 动作损失计算 ---
        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        # --- 合并总损失 ---
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """推理时联合预测视频和动作的噪声（Flow Matching 预测）。

        输入:
            latents_video [1, C, T_lat, H_lat, W_lat]: 当前视频隐空间张量
            latents_action [1, T, a_dim]: 当前动作噪声张量
            timestep_video [1]: 视频当前时间步
            timestep_action [1]: 动作当前时间步
            context [1, L, D]: 文本/条件上下文
            context_mask [1, L]: 上下文 mask
            fuse_vae_embedding_in_latents (bool): 是否融合 VAE 首帧嵌入
            gt_action [1, T, a_dim], 可选: 用于视频专家交叉注意力的真实动作（训练时）或 None

        输出:
            pred_video [1, C, T_lat, H_lat, W_lat]: 视频专家预测的噪声
            pred_action [1, T, a_dim]: 动作专家预测的噪声
        """
        video_pre, compression_meta = self._build_video_pre(
            latents_video=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self._decode_video_tokens(tokens_out["video"], video_pre, compression_meta)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        """推理时仅预测动作噪声（视频侧使用首帧条件，视频时间步设为 0）。

        视频专家以首帧隐变量为输入，时间步为 0，action 为 None，
        这意味着视频专家在此处仅作为视觉条件编码器，不进行去噪。
        动作专家正常进行去噪预测。

        输入:
            first_frame_latents [1, C, 1, H_lat, W_lat]: 首帧隐空间张量（已知，无噪声）
            latents_action [1, T, a_dim]: 当前加噪的动作张量
            timestep_action [1]: 动作当前时间步
            context [1, L, D]: 文本/条件上下文
            context_mask [1, L]: 上下文 mask
            fuse_vae_embedding_in_latents (bool): 是否融合 VAE 首帧嵌入

        输出:
            pred_action [1, T, a_dim]: 动作专家预测的噪声
        """
        # 视频侧时间步设为 0（无噪声），仅提供视觉条件
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre, _ = self._build_video_pre(
            latents_video=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            apply_spatial_downsample=self.apply_video_latent_downsample_to_action_branch,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """使用视频 KV 缓存高效地预测动作噪声。

        在动作推理场景中，视频侧的 token 处理只需进行一次（预填充 KV 缓存），
        后续动作去噪迭代可直接复用缓存的视频 K/V，避免重复计算视频 self-attention。

        输入:
            latents_action [1, T, a_dim]: 当前加噪的动作张量
            timestep_action [1]: 动作当前时间步
            context [1, L, D]: 条件上下文
            context_mask [1, L]: 上下文 mask
            video_kv_cache (list[dict]): 预填充的视频 KV 缓存，与 MoT 的层一一对应
            attention_mask [total_seq_len, total_seq_len]: 完整注意力掩码
            video_seq_len (int): 视频 token 总数

        输出:
            pred_action [1, T, a_dim]: 动作专家预测的噪声
        """
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        # 通过 MoT 的 forward_action_with_video_cache 复用视频 KV 缓存
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        """联合视频-动作推理（同步去噪）。

        推理流程：
          1. （可选）若 test_action_with_infer_action=True，先调用 infer_action 得到纯动作推理结果用于对比
          2. 校验输入图像维度，检查尺寸对齐要求
          3. 编码输入图像首帧为 VAE 隐空间张量
          4. 生成初始随机噪声：视频 [1,C,T_lat,H_lat,W_lat] 和动作 [1,T,a_dim]
          5. 构建条件上下文（从 prompt 编码或从预计算 context 获取）
          6. 构建视频和动作的推理时间表（Flow Matching 调度器）
          7. 逐步去噪：每一步联合预测视频和动作噪声，执行调度器步进
          8. 保持首帧隐变量在整个去噪过程中不变（条件帧约束）
          9. 解码视频隐空间为 PIL 帧序列，返回动作结果

        参数:
            prompt (str, 可选): 文本提示（与 context/context_mask 互斥）
            input_image [1, 3, H, W] 或 [3, H, W]: 条件首帧图像
            num_video_frames (int): 要生成的视频总帧数（需满足 T % 4 == 1）
            action_horizon (int): 动作序列长度
            action [1, T, a_dim], 可选: 用于视频条件化的参考动作（非动作专家输入）
            proprio [1, D] 或 [D], 可选: 本体感知编码
            context [1, L, D], 可选: 预编码的条件上下文（与 prompt 互斥）
            context_mask [1, L], 可选: 上下文 mask
            negative_prompt (str, 可选): 负向提示（当前未使用）
            text_cfg_scale (float): 文本 CFG 缩放（当前未使用）
            num_inference_steps (int): 推理去噪步数
            sigma_shift (float, 可选): 调度器 sigma shift 覆盖值
            seed (int, 可选): 随机种子
            rand_device (str): 生成随机数的设备
            tiled (bool): VAE 是否使用分块处理
            test_action_with_infer_action (bool): 是否与纯动作推理结果对比校验

        返回:
            dict: {"video": list[Image], "action": Tensor [T, a_dim]}
        """
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # 计算隐空间维度
        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        # 生成初始随机噪声
        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        # 编码首帧并替换视频噪声的首帧位置
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # 处理条件上下文
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        # 构建视频和动作的推理时间调度表
        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        # 逐步去噪循环：视频和动作同步步进
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            # 每次步进后恢复首帧为条件帧（防止首帧被噪声污染）
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """仅推理动作序列（利用视频 KV 缓存加速）。

        此方法将视频侧作为固定条件（首帧），通过视频 KV 缓存破解（prefill）实现一次计算、
        多次复用的高效推理。动作侧的每个去噪步只需计算动作 token 的自注意力和
        对视频缓存的交叉注意力，无需重新计算视频 token。

        推理流程：
          1. 编码首帧图像为 VAE 隐变量
          2. 视频专家以首帧和时间步 0 为输入，经 pre_dit 编码
          3. 预填充视频 KV 缓存（prefill_video_cache）
          4. 构建注意力掩码（动作可关注首帧视频）
          5. 动作去噪循环：每步通过 _predict_action_noise_with_cache 预测噪声
          6. 调度器步进更新动作张量

        参数:
            prompt (str, 可选): 文本提示（与 context/context_mask 互斥）
            input_image [1, 3, H, W] 或 [3, H, W]: 条件首帧图像
            action_horizon (int): 动作序列长度
            proprio [1, D] 或 [D], 可选: 本体感知编码
            context [1, L, D], 可选: 预编码的上下文（与 prompt 互斥）
            context_mask [1, L], 可选: 上下文 mask
            negative_prompt (str, 可选): 负向提示（当前未使用）
            text_cfg_scale (float): 文本 CFG 缩放（当前未使用）
            num_inference_steps (int): 推理去噪步数
            sigma_shift (float, 可选): 调度器 shift 覆盖
            seed (int, 可选): 随机种子
            rand_device (str): 生成随机数的设备
            tiled (bool): VAE 是否使用分块处理

        返回:
            dict: {"action": Tensor [T, a_dim]}
        """
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # 生成动作初始噪声
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # 条件上下文处理
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        # 视频预填充：编码首帧并预计算 KV 缓存
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre, _ = self._build_video_pre(
            latents_video=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            apply_spatial_downsample=self.apply_video_latent_downsample_to_action_branch,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        # 预填充视频 KV 缓存，后续动作去噪步可直接复用
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        # 构建动作推理时间调度表
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        # 动作去噪循环
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        """FastWAM 推理入口，封装调用 infer_joint。

        作为 infer_joint 的简便别名，兼容 Wan22Core 的 infer 接口签名。

        参数:
            prompt (str, 可选): 文本提示
            input_image [1, 3, H, W]: 条件首帧图像
            num_frames (int): 要生成的视频总帧数
            action [1, T, a_dim], 可选: 视频条件动作
            action_horizon (int, 可选): 动作序列长度
            proprio [D], 可选: 本体感知
            context/context_mask: 可选的条件上下文
            negative_prompt (str, 可选): 负向提示
            text_cfg_scale (float): 文本 CFG 缩放
            action_cfg_scale (float): 动作 CFG 缩放（当前未使用）
            num_inference_steps (int): 推理步数
            sigma_shift (float, 可选): 调度器 shift 覆盖
            seed (int, 可选): 随机种子
            rand_device (str): 随机数生成设备
            tiled (bool): VAE 分块处理

        返回:
            dict: {"video": list[Image], "action": Tensor [T, a_dim]}
        """
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        """保存模型检查点到磁盘。

        保存内容包括：
          - MoT（包括 video_expert 和 action_expert）的状态字典
          - 当前训练步数
          - 模型数据类型
          - 优化器状态（可选）
          - proprio_encoder 权重（如启用）

        参数:
            path (str): 保存路径
            optimizer: PyTorch 优化器（可选）
            step (int): 当前训练步数（可选）
        """
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.goal_token_encoder is not None:
            payload["goal_token_encoder"] = self.goal_token_encoder.state_dict()
        if self.video_goal_adapter is not None:
            payload["video_goal_adapter"] = self.video_goal_adapter.state_dict()
        direction_embedding = getattr(self, "direction_embedding", None)
        if direction_embedding is not None:
            payload["direction_embedding"] = direction_embedding.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        """从磁盘加载模型检查点。

        支持向后兼容：优先加载 "mot" 键（MoT 整体），
        如不存在则尝试加载 "dit" 键（仅视频专家，旧版格式）。

        参数:
            path (str): 检查点路径
            optimizer: PyTorch 优化器实例（可选，将更新其状态）

        返回:
            payload (dict): 从检查点加载的原始字典
        """
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if self.goal_token_encoder is not None:
            if "goal_token_encoder" in payload:
                self.goal_token_encoder.load_state_dict(payload["goal_token_encoder"], strict=False)
            else:
                logger.warning("Checkpoint has no `goal_token_encoder` weights; keeping current encoder params.")
        if self.video_goal_adapter is not None:
            if "video_goal_adapter" in payload:
                self.video_goal_adapter.load_state_dict(payload["video_goal_adapter"], strict=True)
            else:
                logger.warning("Checkpoint has no `video_goal_adapter` weights; keeping current adapter params.")

        direction_embedding = getattr(self, "direction_embedding", None)
        if direction_embedding is not None:
            if "direction_embedding" in payload:
                direction_embedding.load_state_dict(payload["direction_embedding"], strict=True)
            else:
                logger.warning("Checkpoint has no `direction_embedding` weights; keeping current direction params.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        """前向传播。默认调用 training_loss。

        参数:
            *args, **kwargs: 传递给 training_loss 的参数
        """
        return self.training_loss(*args, **kwargs)
