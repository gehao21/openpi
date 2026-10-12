from typing import Literal

import torch
from torch import nn
from transformers import GemmaForCausalLM # Gemma因果语言模型
from transformers import PaliGemmaForConditionalGeneration # PailGemma视觉语言模型
from transformers.models.auto import CONFIG_MAPPING # Hugging Face 的配置注册表 自动配置映射表
from transformers.models.gemma import modeling_gemma # Gemma模型的建模模块 底层实现模块


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        """
        Initialize the PaliGemmaWithExpertModel.

        Args:
            vlm_config: The configuration for the VLM.
            action_expert_config: The configuration for the action expert.
            use_adarms: A list of booleans indicating whether to use AdaRMS for each model. 自适应归一化
            precision: The precision to use for the model parameters.
        """
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]() # PaliGemma 配置
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001 模型词表
        vlm_config_hf.image_token_index = 257152 # 图像token的索引
        vlm_config_hf.text_config.hidden_size = vlm_config.width # 文本隐藏层大小
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim # 文本中间层大小
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads # 文本注意力头数
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim # 文本注意力头维度
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth # 文本隐藏层数
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads # 文本键值头数
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh" # 文本隐藏层激活函数
        vlm_config_hf.text_config.torch_dtype = "float32" # 文本数据类型
        vlm_config_hf.text_config.vocab_size = 257152 # 文本词表大小
        vlm_config_hf.text_config.use_adarms = use_adarms[0] # 是否使用自适应归一化
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None # 自适应归一化条件维度
        vlm_config_hf.vision_config.intermediate_size = 4304 # 视觉中间层大小
        vlm_config_hf.vision_config.projection_dim = 2048 # 视觉投影维度
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast" # 视觉投影隐藏层激活函数
        vlm_config_hf.vision_config.torch_dtype = "float32" # 视觉数据类型

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim, # 动作专家注意力头维度
            hidden_size=action_expert_config.width, # 动作专家隐藏层大小
            intermediate_size=action_expert_config.mlp_dim, # 动作专家中间层大小
            num_attention_heads=action_expert_config.num_heads, # 动作专家注意力头数
            num_hidden_layers=action_expert_config.depth, # 动作专家隐藏层数
            num_key_value_heads=action_expert_config.num_kv_heads, # 动作专家键值头数
            vocab_size=257152, # 动作专家词表大小
            hidden_activation="gelu_pytorch_tanh", # 动作专家隐藏层激活函数
            torch_dtype="float32", # 动作专家数据类型
            use_adarms=use_adarms[1], # 是否使用自适应归一化
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None, # 自适应归一化条件维度
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf) # PaliGemma视觉语言模型 VLM
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf) # Gemma因果语言模型 Action Expert
        self.gemma_expert.model.embed_tokens = None # Remove the embedding layer from the Gemma expert model to avoid redundancy

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        """
        Convert the model parameters to bfloat16 for selected parameters, while keeping others in float32
        """
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        """
        Embed an image using the PaliGemma model.
        """
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        """
        Embed language tokens using the PaliGemma model.
        """
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        """
        Forward pass for the PaliGemmaWithExpertModel.
        Args:
            attention_mask: Attention mask for the input tokens.
            position_ids: Position IDs for the input tokens.
            past_key_values: Past key values for the transformer layers.
            inputs_embeds: List of input embeddings for the two models.
            use_cache: Whether to use caching for the transformer layers.
            adarms_cond: List of AdaRMS conditions for the two models.
        Returns:
            A tuple containing the outputs of the two models and the past key values for the prefix model.
        """
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            # Only the prefix model is used 仅仅使用VLM部分
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            # Only the suffix model is used 仅仅使用Action Expert部分
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            # Both models are used, and we will process them layer by layer.  两个模型都使用，我们将逐层处理它们
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            # gradient checkpointing is a technique to save memory during training by not storing intermediate activations and recomputing them during the backward pass. This is useful for large models that may not fit in memory otherwise.
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                """每一层分别计算QKV然后联合拼接计算注意力，然后再分别计算每一层的输出"""
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx] 
                    # (B, seq_len, embed_dim)
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1] # (B, seq_len)
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim) # (B, seq_len, num_heads, head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (B, num_heads, seq_len, head_dim)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state) # 
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2) # 联合拼接 (B, num_heads, seq_len1 + seq_len2, head_dim)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                ) # (B, seq_len1 + seq_len2, head_dim)
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids) # 计算旋转嵌入的余弦和正弦 (B, seq_len1 + seq_len2, head_dim)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim) # Reshape to (B, seq_len1 + seq_len2, embed_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1] # 计算每个模型的输出序列长度

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos]) # (B, seq_len, embed_dim)

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
