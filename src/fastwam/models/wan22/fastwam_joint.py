from typing import Any, Optional

import torch

from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM

logger = get_logger(__name__)

"""
FastWAMJoint 变体模型模块。

该文件定义了 FastWAMJoint 类，继承自 FastWAM。与基础 FastWAM 的关键区别在于
注意力掩码策略：FastWAMJoint 允许动作 token 关注完整的视频 token 序列（而非仅首帧），
从而实现更充分的跨模态信息交互。

适用场景：
  - 当需要动作 token 访问所有视频帧的上下文信息时
  - 视频序列较短、计算开销可接受的情况下
"""


class FastWAMJoint(FastWAM):
    """FastWAMJoint 变体模型类。

    继承 FastWAM，重写了 _build_mot_attention_mask 和 infer 相关方法。

    与 FastWAM 的核心区别：
      - 注意力掩码中，动作 token 可以关注所有视频 token（而非仅首帧）
      - infer_action 需要完整的 num_video_frames 参数（因为动作需要完整视频上下文）
      - infer_joint 忽略 test_action_with_infer_action 参数（因为两种推理的注意力模式不同）
      - 视频专家的 action_conditioned 必须为 False（FastWAMJoint 以视觉条件为主）
    """

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        """从预训练检查点加载 FastWAMJoint 模型。

        校验视频 DiT 配置要求：
          - action_conditioned 必须为 False（FastWAMJoint 不使用动作条件化视频专家）

        参数:
            **kwargs: 传递给父类 from_wan22_pretrained 的参数
        """
        video_dit_config = kwargs.get("video_dit_config", None)
        if not isinstance(video_dit_config, dict):
            raise ValueError(
                "`video_dit_config` must be provided as dict for FastWAMJoint."
            )
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "FastWAMJoint requires `video_dit_config['action_conditioned']=false`."
            )
        return super().from_wan22_pretrained(**kwargs)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """构建 FastWAMJoint 的混合注意力掩码。

        与父类的区别：动作 token 允许关注全部视频 token（而非仅首帧）。

        注意力可见性：
          - video -> video: 使用视频专家的视频到视频掩码
          - action -> action: 完全可见
          - action -> video: 完全可见（与 FastWAM 的关键区别）

        参数:
            video_seq_len (int): 视频 token 总数
            action_seq_len (int): 动作 token 总数
            video_tokens_per_frame (int): 每帧对应的 token 数（此处不使用）
            device: 目标设备

        返回:
            mask [total_seq_len, total_seq_len] 布尔张量
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
        # action -> full video
        # FastWAMJoint 的核心不同：动作 token 可关注所有视频 token
        mask[video_seq_len:, :video_seq_len] = True
        return mask

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
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
        """FastWAMJoint 的联合视频-动作推理。

        与父类的区别：忽略 test_action_with_infer_action 参数，
        因为 FastWAMJoint 的注意力模式与父类不同，不能直接对比两种推理结果。

        参数:
            同父类 FastWAM.infer_joint，但 test_action_with_infer_action 始终视为 False。
        """
        if test_action_with_infer_action:
            logger.warning(
                "`FastWAMJoint.infer_joint` ignores `test_action_with_infer_action=True` "
                "and always runs with `test_action_with_infer_action=False`."
            )
        return super().infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
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
            test_action_with_infer_action=False,
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
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
        """FastWAMJoint 的纯动作推理。

        与父类不同：FastWAMJoint 需要完整的视频去噪过程来为动作提供完整的视觉上下文
        （因为动作可以关注所有视频帧），因此这里实际执行的是联合推理并返回动作结果。

        不同于 FastWAM.infer_action（使用视频 KV 缓存），此方法：
          - 需要 num_video_frames 参数（用于生成完整视频）
          - 完整执行视频+动作的同步去噪
          - 仅返回动作结果，视频结果被丢弃

        参数:
            prompt (str, 可选): 文本提示
            input_image [1, 3, H, W]: 条件首帧
            action_horizon (int): 动作序列长度
            num_video_frames (int): 视频总帧数（需要满足 T % 4 == 1）
            proprio [1, D], 可选: 本体感知
            context [1, L, D], 可选: 预编码上下文
            context_mask [1, L], 可选: 上下文 mask
            negative_prompt (str, 可选): 负向提示（未使用）
            text_cfg_scale (float): 文本 CFG（未使用）
            num_inference_steps (int): 推理步数
            sigma_shift (float, 可选): 调度器 shift 覆盖
            seed (int, 可选): 随机种子
            rand_device (str): 随机数生成设备
            tiled (bool): VAE 分块处理

        返回:
            dict: {"action": Tensor [T, a_dim]}
        """
        self.eval()

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

        # 生成视频和动作的初始噪声
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

        # 编码首帧并替代噪声首帧位置
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
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

        # 构建视频和动作的推理调度表
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
        # 同步去噪循环
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
                gt_action=None,
            )

            latents_video = self.infer_video_scheduler.step(pred_video_posi, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action_posi, step_delta_action, latents_action)
            # 步进后恢复首帧条件帧
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
