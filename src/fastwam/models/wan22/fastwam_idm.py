from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam_joint import FastWAMJoint

logger = get_logger(__name__)

"""
FastWAMIDM（Inverse Dynamics Model）变体模型模块。

该文件定义了 FastWAMIDM 类，继承自 FastWAMJoint。IDM（逆动力学模型）的核心思想是：
给定视频观测序列，预测产生这些观测的潜在动作。

FastWAMIDM 的关键特点：
  - 训练时采用 Teacher-Forcing 策略：条件视频分支使用原始（或轻微加噪）的视频，
    噪声视频分支用于视频去噪学习，动作分支以条件视频为上下文进行去噪
  - 推理时采用两阶段流程：第一阶段独立去噪生成视频，第二阶段以生成的视频为条件去噪动作
  - 条件视频以一定概率（video_cond_noise_prob）被加噪，增强对噪声的鲁棒性
"""


class FastWAMIDM(FastWAMJoint):
    """FastWAMIDM（逆动力学模型）变体类。

    继承自 FastWAMJoint。IDM 模式将视频视为已知条件，专注于从视频中推断动作，
    即建模 P(action | video, context) 的逆动力学。

    训练流程：
      1. 三个分支并行：噪声视频（A）、加噪动作（B）、条件视频（C）
      2. 视频专家处理 A + C（拼接作为视频序列），动作专家处理 B
      3. 构建 Teacher-Forcing 注意力掩码：动作可关注条件视频（C）而非噪声视频（A）
      4. 仅噪声视频分支参与视频损失，动作分支正常计算动作损失

    推理流程：
      第一阶段：独立去噪视频（使用 video_expert 直接推理，无动作条件）
      第二阶段：以去噪后的视频为条件，通过 KV 缓存高效去噪动作
    """

    # Hardcoded probability: during training, cond-video is noised with this chance.
    # 训练时条件视频被加噪的概率。用于增强对噪声视频条件的鲁棒性
    video_cond_noise_prob = 0.5

    def __init__(self, *args, bidirectional_config: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bidirectional_config = dict(bidirectional_config or {})
        self.bidirectional_enabled = bool(self.bidirectional_config.get("enabled", False))
        self.bidirectional_direction_token = self.bidirectional_enabled and bool(
            self.bidirectional_config.get("direction_token", True)
        )
        if self.bidirectional_direction_token:
            self.direction_embedding = nn.Embedding(2, self.text_dim).to(device=self.device, dtype=self.torch_dtype)
        else:
            self.direction_embedding = None

    def _direction_id(self, direction: str) -> int:
        if direction == "forward":
            return 0
        if direction == "backward":
            return 1
        raise ValueError(f"Unknown IDM direction: {direction}")

    def _append_direction_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        direction: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.direction_embedding is None:
            return context, context_mask
        direction_ids = torch.full(
            (context.shape[0],),
            self._direction_id(direction),
            device=context.device,
            dtype=torch.long,
        )
        direction_emb = self.direction_embedding(direction_ids).to(dtype=context.dtype).unsqueeze(1)
        direction_mask = torch.ones((context.shape[0], 1), device=context_mask.device, dtype=torch.bool)
        return torch.cat([direction_emb, context], dim=1), torch.cat([direction_mask, context_mask], dim=1)

    def _bidirectional_delta_action_mask(self, device: torch.device, action_dim: int) -> Optional[torch.Tensor]:
        mask = self.bidirectional_config.get("delta_action_dim_mask")
        if mask is None:
            return None
        if isinstance(mask, dict):
            mask = mask.get("default")
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
        if mask_tensor.numel() != action_dim:
            raise ValueError(
                f"bidirectional.delta_action_dim_mask length must be {action_dim}, got {mask_tensor.numel()}"
            )
        return mask_tensor

    def _reverse_action(self, action: torch.Tensor) -> torch.Tensor:
        action_rev = action.flip(dims=[1]).clone()
        delta_mask = self._bidirectional_delta_action_mask(action_rev.device, action_rev.shape[-1])
        if delta_mask is not None:
            action_rev[..., delta_mask] = -action_rev[..., delta_mask]
        return action_rev

    def _make_backward_sample(self, sample: dict) -> dict:
        sample_bwd = dict(sample)
        sample_bwd["video"] = sample["video"].flip(dims=[2])
        if sample.get("image_is_pad") is not None:
            sample_bwd["image_is_pad"] = sample["image_is_pad"].flip(dims=[1])
        sample_bwd["action"] = self._reverse_action(sample["action"])
        if sample.get("action_is_pad") is not None:
            sample_bwd["action_is_pad"] = sample["action_is_pad"].flip(dims=[1])
        if sample.get("proprio") is not None:
            sample_bwd["proprio"] = sample["proprio"].flip(dims=[1])
        if sample.get("proprio_is_pad") is not None:
            sample_bwd["proprio_is_pad"] = sample["proprio_is_pad"].flip(dims=[1])
        return sample_bwd

    @staticmethod
    def _merge_bidirectional_loss_dict(dict_fwd: dict, dict_bwd: dict, lambda_backward: float) -> dict:
        loss_dict = {}
        for key, value in dict_fwd.items():
            loss_dict[f"{key}_forward"] = value
        for key, value in dict_bwd.items():
            loss_dict[f"{key}_backward"] = value
        if "loss_video" in dict_fwd and "loss_video" in dict_bwd:
            loss_dict["loss_video"] = 0.5 * (dict_fwd["loss_video"] + dict_bwd["loss_video"])
        if "loss_action" in dict_fwd and "loss_action" in dict_bwd:
            loss_dict["loss_action"] = 0.5 * (dict_fwd["loss_action"] + dict_bwd["loss_action"])
        loss_dict["loss_bidirectional_total"] = dict_fwd.get("loss_total", 0.0) + lambda_backward * dict_bwd.get("loss_total", 0.0)
        return loss_dict

    def _encode_goal_tokens(self, context: torch.Tensor, context_mask: torch.Tensor) -> Optional[torch.Tensor]:
        # Goal-token conditioning is intentionally disabled for the bidirectional IDM version.
        # Keep this stub so old checkpoints/configs remain loadable while the training path is text-only.
        return None

    @torch.no_grad()
    def _build_teacher_forcing_attention_mask(
        self,
        noisy_video_seq_len: int,
        cond_video_seq_len: int,
        action_seq_len: int,
        noisy_video_tokens_per_frame: int,
        cond_video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """构建 Teacher-Forcing 训练模式的注意力掩码。

        在 IDM 训练中，视频专家处理两个分支的拼接：[噪声视频 | 条件视频]。
        动作专家处理动作分支。注意力规则：
          - 噪声视频 -> 噪声视频: 使用视频专家掩码（通常是 causal）
          - 条件视频 -> 条件视频: 使用视频专家掩码
          - 动作 -> 动作: 完全可见
          - 动作 -> 条件视频: 完全可见（动作观察条件视频来推断动作）
          - 噪声视频与条件视频之间无注意力交互（隔离）

        参数:
            noisy_video_seq_len (int): 噪声视频分支的 token 数
            cond_video_seq_len (int): 条件视频分支的 token 数
            action_seq_len (int): 动作分支的 token 数
            noisy_video_tokens_per_frame (int): 噪声视频每帧 token 数
            cond_video_tokens_per_frame (int): 条件视频每帧 token 数
            device: 目标设备

        返回:
            mask [total_seq_len, total_seq_len] 布尔张量
              序列布局: [noisy_video | cond_video | action]
        """
        if noisy_video_tokens_per_frame != cond_video_tokens_per_frame:
            raise ValueError(
                "Teacher-forcing requires identical `tokens_per_frame` for noisy and cond video branches, "
                f"got {noisy_video_tokens_per_frame} and {cond_video_tokens_per_frame}."
            )

        noisy_end = noisy_video_seq_len
        cond_end = noisy_video_seq_len + cond_video_seq_len
        total_seq_len = cond_end + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # noisy_video -> noisy_video
        mask[:noisy_end, :noisy_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=noisy_video_seq_len,
            video_tokens_per_frame=noisy_video_tokens_per_frame,
            device=device,
        )
        # cond_video -> cond_video
        mask[noisy_end:cond_end, noisy_end:cond_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=cond_video_seq_len,
            video_tokens_per_frame=cond_video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[cond_end:, cond_end:] = True
        # action -> cond_video only（不关注噪声视频）
        mask[cond_end:, noisy_end:cond_end] = True
        return mask

    def training_loss(self, sample, tiled: bool = False):
        """FastWAMIDM training loss with optional paired backward dynamics training."""
        if not self.bidirectional_enabled:
            return self._training_loss_single_direction(sample, direction="forward", tiled=tiled)

        sample_bwd = self._make_backward_sample(sample)
        loss_fwd, dict_fwd = self._training_loss_single_direction(sample, direction="forward", tiled=tiled)
        loss_bwd, dict_bwd = self._training_loss_single_direction(sample_bwd, direction="backward", tiled=tiled)
        lambda_backward = float(self.bidirectional_config.get("lambda_backward", 1.0))
        loss_total = loss_fwd + lambda_backward * loss_bwd
        loss_dict = self._merge_bidirectional_loss_dict(dict_fwd, dict_bwd, lambda_backward)
        return loss_total, loss_dict

    def _training_loss_single_direction(self, sample, direction: str, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        context, context_mask = self._append_direction_to_context(context, context_mask, direction)
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]

        first_frame_latents = inputs["first_frame_latents"]

        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        video_train_targets = self._prepare_video_training_targets(
            video_supervision_latents=input_latents,
            timestep_video=timestep_video,
            first_frame_latents=first_frame_latents,
        )
        video_supervision_latents_model = video_train_targets["video_supervision_latents_model"]
        first_frame_latents_model = video_train_targets["first_frame_latents_model"]
        latents_noisy = video_train_targets["latents_video"]
        target_video = video_train_targets["target_video"]

        # Branch B: noisy action.
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # Branch C: teacher-forcing condition uses GT video latents.
        cond_noise_mask = torch.rand((batch_size,), device=self.device) < float(self.video_cond_noise_prob)
        timestep_video_cond = torch.zeros_like(timestep_video, dtype=input_latents.dtype, device=self.device)
        latents_cond = video_supervision_latents_model
        if bool(cond_noise_mask.any()):
            timestep_video_cond_sampled = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=video_supervision_latents_model.dtype,
            )
            timestep_video_cond = torch.where(cond_noise_mask, timestep_video_cond_sampled, timestep_video_cond)
            noise_video_cond = torch.randn_like(video_supervision_latents_model)
            latents_cond_noisy = self.train_video_scheduler.add_noise(
                video_supervision_latents_model, noise_video_cond, timestep_video_cond_sampled
            )
            cond_noise_selector = cond_noise_mask.view(batch_size, 1, 1, 1, 1)
            latents_cond = torch.where(cond_noise_selector, latents_cond_noisy, video_supervision_latents_model)
        if first_frame_latents_model is not None:
            latents_cond = latents_cond.clone()
            latents_cond[:, :, 0:1] = first_frame_latents_model

        video_pre_noisy, compression_meta_noisy = self._build_video_pre(
            latents_video=latents_noisy,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            apply_spatial_downsample=video_train_targets["apply_spatial_downsample"],
            extra_context_emb=None,
        )
        video_pre_cond, _ = self._build_video_pre(
            latents_video=latents_cond,
            timestep_video=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            apply_spatial_downsample=video_train_targets["apply_spatial_downsample"],
            extra_context_emb=None,
        )
        if video_pre_noisy["t_mod"].ndim != 4 or video_pre_cond["t_mod"].ndim != 4:
            raise ValueError(
                "Teacher-forcing requires token-wise `t_mod`; "
                "ensure `seperated_timestep=true` and `fuse_vae_embedding_in_latents=true`."
            )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        noisy_video_seq_len = int(video_pre_noisy["tokens"].shape[1])
        cond_video_seq_len = int(video_pre_cond["tokens"].shape[1])
        noisy_video_tokens_per_frame = int(video_pre_noisy["meta"]["tokens_per_frame"])
        cond_video_tokens_per_frame = int(video_pre_cond["meta"]["tokens_per_frame"])
        #满足npu算子
        video_pre_noisy["freqs"] = video_pre_noisy["freqs"].to(torch.complex64)
        video_pre_cond["freqs"] = video_pre_cond["freqs"].to(torch.complex64)
        merged_video_tokens = torch.cat([video_pre_noisy["tokens"], video_pre_cond["tokens"]], dim=1)
        merged_video_freqs = torch.cat([video_pre_noisy["freqs"], video_pre_cond["freqs"]], dim=0)
        merged_video_t_mod = torch.cat([video_pre_noisy["t_mod"], video_pre_cond["t_mod"]], dim=1)
        merged_video_context_mask = torch.cat([video_pre_noisy["context_mask"], video_pre_cond["context_mask"]], dim=1)

        attention_mask = self._build_teacher_forcing_attention_mask(
            noisy_video_seq_len=noisy_video_seq_len,
            cond_video_seq_len=cond_video_seq_len,
            action_seq_len=action_pre["tokens"].shape[1],
            noisy_video_tokens_per_frame=noisy_video_tokens_per_frame,
            cond_video_tokens_per_frame=cond_video_tokens_per_frame,
            device=merged_video_tokens.device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": merged_video_tokens,
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": merged_video_freqs,
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre_noisy["context"],
                    "mask": merged_video_context_mask,
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": merged_video_t_mod,
                "action": action_pre["t_mod"],
            },
        )

        pred_video_tokens = tokens_out["video"][:, :noisy_video_seq_len]
        pred_video = self._decode_video_tokens(
            pred_video_tokens,
            video_pre_noisy,
            compression_meta_noisy,
            restore_spatial_resolution=video_train_targets["restore_spatial_resolution"],
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = first_frame_latents_model is None
        if first_frame_latents_model is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
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

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_total": float(loss_total.detach().item()),
        }
        return loss_total, loss_dict

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
        """FastWAMIDM 的纯动作推理。

        IDM 模式的动作推理复用 infer_joint 的两阶段流程，但仅返回动作结果。

        参数:
            同父类 FastWAMJoint.infer_action。
        """
        # Reuse infer_joint pipeline and keep infer_action output contract.
        # 复用 infer_joint 流程，仅返回动作输出以保持 infer_action 接口合约
        out = self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=None,
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
        return {"action": out["action"]}

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
        """FastWAMIDM 的两阶段联合推理。

        阶段 1：视频去噪
          使用 video_expert 独立执行视频去噪（不依赖动作条件）。
          这是标准的视频扩散过程，从纯噪声逐步恢复为视频。

        阶段 2：动作去噪
          以阶段 1 生成的完整视频为条件（teacher-forcing 模式），通过 KV 缓存
          高效地去噪动作序列。视频侧的时间步设为 0（视为已知条件）。

        与父类的区别：
          - 第一阶段完全独立去噪视频，使用 video_expert 直接从噪声到视频
          - 第二阶段以完整的去噪视频作为条件（而非仅首帧）
          - 使用 _build_mot_attention_mask 构建动作对完整视频的注意力

        参数:
            同父类 FastWAMJoint.infer_joint

        返回:
            dict: {"video": list[Image], "action": Tensor [T, a_dim]}
        """
        del negative_prompt, text_cfg_scale, test_action_with_infer_action
        self.eval()

        if action is not None:
            logger.warning(
                "`FastWAMIDM.infer_joint` ignores `action` input; "
                "video is denoised in a standalone first stage."
            )

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

        # 计算完整视频隐空间维度。IDM 推理只使用正向完整 latent chunk。
        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        # 生成初始噪声
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

        # 编码首帧并固定为视频 latent anchor。
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
        context, context_mask = self._append_direction_to_context(context, context_mask, "forward")
        video_goal_hidden = None
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        # Stage 1: denoise video only.
        # 第一阶段：仅去噪视频。使用 video_expert 独立进行视频扩散
        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video in zip(infer_timesteps_video, infer_deltas_video):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            video_pre, compression_meta = self._build_video_pre(
                latents_video=latents_video,
                timestep_video=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
                extra_context_emb=None,
            )
            video_tokens = self.video_expert.forward_backbone(video_pre)
            pred_video = self._decode_video_tokens(video_tokens, video_pre, compression_meta)
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        # Stage 2: freeze denoised video as cond and denoise action via video K/V cache.
        # 第二阶段：冻结去噪后的视频作为条件，通过 KV 缓存高效去噪动作
        timestep_video_cond = torch.zeros(
            (latents_video.shape[0],), dtype=latents_video.dtype, device=self.device
        )
        video_pre_cond, _ = self._build_video_pre(
            latents_video=latents_video,
            timestep_video=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            extra_context_emb=None,
        )
        video_seq_len = int(video_pre_cond["tokens"].shape[1])
        # 构建动作可关注完整视频的注意力掩码
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre_cond["meta"]["tokens_per_frame"]),
            device=video_pre_cond["tokens"].device,
        )
        # 预填充视频 KV 缓存
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre_cond["tokens"],
            video_freqs=video_pre_cond["freqs"],
            video_t_mod=video_pre_cond["t_mod"],
            video_context_payload={
                "context": video_pre_cond["context"],
                "mask": video_pre_cond["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        # 动作去噪循环
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
