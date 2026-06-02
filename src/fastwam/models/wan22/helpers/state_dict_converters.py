"""
FastWAM 项目中 Wan2.2 模型状态字典（state_dict）的键名转换工具模块。

该模块负责在不同模型格式之间转换权重键名，使得 FastWAM 能够加载和复用
来自不同框架（DiffSynth、Diffusers）或不同版本的预训练权重。

主要转换器：
  1. wan_video_vae_state_dict_converter: DiffSynth VAE 格式 -> FastWAM 格式
  2. wan_video_dit_from_diffusers: Diffusers DiT 格式 -> FastWAM 格式
  3. wan_video_dit_state_dict_converter: 通用 FastWAM 格式清理（移除前缀和无关键）
"""


def wan_video_vae_state_dict_converter(state_dict):
    """
    DiffSynth 格式的 VAE 状态字典键名转换器。

    将 DiffSynth 导出的 VAE 权重键名映射到 FastWAM 期望的格式：
      - 如果状态字典被包裹在 "model_state" 键下，先自动解包。
      - 为所有键添加 "model." 前缀。

    DiffSynth 原始格式:
        {"model_state": {"encoder.conv_in.weight": ..., ...}}
    FastWAM 目标格式:
        {"model.encoder.conv_in.weight": ..., ...}

    Args:
        state_dict: DiffSynth 格式的 VAE 状态字典。

    Returns:
        转换后的 FastWAM 格式状态字典。
    """
    converted = {}
    # 处理可能的包裹层 "model_state"
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    for name, value in state_dict.items():
        converted[f"model.{name}"] = value
    return converted


def wan_video_dit_from_diffusers(state_dict):
    """
    Diffusers 格式的 DiT 状态字典键名转换器。

    将 HuggingFace Diffusers 导出的 DiT 权重键名映射到 FastWAM 期望的命名格式。
    主要映射关系包括：
      - attn1 -> self_attn（自注意力）
      - attn2 -> cross_attn（交叉注意力）
      - ffn.net.0/2 -> ffn.0/2（前馈网络）
      - condition_embedder.* -> 扁平的 text_embedding/time_embedding 等

    支持动态块索引：通过 probe_name 机制，将对 block.0 的定义映射推广到任意块索引。

    Args:
        state_dict: Diffusers 格式的 DiT 状态字典。
            例如: {"blocks.0.attn1.to_q.weight": ..., "condition_embedder.time_embedder.linear_1.weight": ...}

    Returns:
        转换后的 FastWAM 格式状态字典。
            例如: {"blocks.0.self_attn.q.weight": ..., "time_embedding.0.weight": ...}
    """
    # 重命名映射表：左侧为 Diffusers 格式键名，右侧为 FastWAM 格式键名
    # 以 blocks.0 为模板，其他块索引通过动态探测自动转换
    rename_dict = {
        "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
        "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
        "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
        "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
        "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
        "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
        "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
        "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
        "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
        "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
        "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
        "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
        "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
        "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
        "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
        "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
        "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
        "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
        "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
        "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
        "blocks.0.attn2.add_k_proj.bias": "blocks.0.cross_attn.k_img.bias",
        "blocks.0.attn2.add_k_proj.weight": "blocks.0.cross_attn.k_img.weight",
        "blocks.0.attn2.add_v_proj.bias": "blocks.0.cross_attn.v_img.bias",
        "blocks.0.attn2.add_v_proj.weight": "blocks.0.cross_attn.v_img.weight",
        "blocks.0.attn2.norm_added_k.weight": "blocks.0.cross_attn.norm_k_img.weight",
        "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
        "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
        "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
        "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
        "blocks.0.norm2.bias": "blocks.0.norm3.bias",
        "blocks.0.norm2.weight": "blocks.0.norm3.weight",
        "blocks.0.scale_shift_table": "blocks.0.modulation",
        "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
        "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
        "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
        "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
        "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
        "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
        "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
        "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
        "condition_embedder.time_proj.bias": "time_projection.1.bias",
        "condition_embedder.time_proj.weight": "time_projection.1.weight",
        "condition_embedder.image_embedder.ff.net.0.proj.bias": "img_emb.proj.1.bias",
        "condition_embedder.image_embedder.ff.net.0.proj.weight": "img_emb.proj.1.weight",
        "condition_embedder.image_embedder.ff.net.2.bias": "img_emb.proj.3.bias",
        "condition_embedder.image_embedder.ff.net.2.weight": "img_emb.proj.3.weight",
        "condition_embedder.image_embedder.norm1.bias": "img_emb.proj.0.bias",
        "condition_embedder.image_embedder.norm1.weight": "img_emb.proj.0.weight",
        "condition_embedder.image_embedder.norm2.bias": "img_emb.proj.4.bias",
        "condition_embedder.image_embedder.norm2.weight": "img_emb.proj.4.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "patch_embedding.weight": "patch_embedding.weight",
        "scale_shift_table": "head.modulation",
        "proj_out.bias": "head.head.bias",
        "proj_out.weight": "head.head.weight",
    }
    converted = {}
    for name, value in state_dict.items():
        if name in rename_dict:
            # 直接命中的键（非块级别，如 condition_embedder 等）
            converted[rename_dict[name]] = value
        else:
            # 动态块索引探测：将给定的键名中的块索引替换为 "0" 后查表
            # 例如 "blocks.5.attn1.to_q.weight" -> probe_name = "blocks.0.attn1.to_q.weight"
            probe_name = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
            if probe_name in rename_dict:
                # 从 rename_dict 中获取映射目标，再将块索引替换回原始索引
                mapped = rename_dict[probe_name]
                mapped = ".".join(mapped.split(".")[:1] + [name.split(".")[1]] + mapped.split(".")[2:])
                converted[mapped] = value
    return converted


def wan_video_dit_state_dict_converter(state_dict):
    """
    FastWAM 格式的 DiT 状态字典通用转换器。

    执行以下清理操作：
      1. 移除 "vace" 前缀相关的键（Vision-Audio-Context Embedding）。
      2. 移除姿态嵌入、人脸适配器、人脸编码器、运动编码器等可选模块的键。
      3. 如果键以 "model." 开头，则剥离此前缀。

    此转换用于 DiffSynth 格式到 FastWAM 格式的最终适配。

    Args:
        state_dict: DiffSynth / FastWAM 混合格式的状态字典。
            例如: {"model.blocks.0.self_attn.q.weight": ..., "vace.xxx": ..., "pose_patch_embedding.weight": ...}

    Returns:
        清理后的状态字典。
            例如: {"blocks.0.self_attn.q.weight": ...}
    """
    converted = {}
    for name, value in state_dict.items():
        # 过滤掉 VACE 相关的键
        if name.startswith("vace"):
            continue
        # 过滤掉可选的辅助模块（姿态、人脸、运动）
        if name.split(".")[0] in ["pose_patch_embedding", "face_adapter", "face_encoder", "motion_encoder"]:
            continue
        # 剥离 "model." 前缀（通常由 wan_video_vae_state_dict_converter 添加）
        key = name[6:] if name.startswith("model.") else name
        converted[key] = value
    return converted
