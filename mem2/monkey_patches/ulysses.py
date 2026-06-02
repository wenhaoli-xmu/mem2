import torch
import types
from typing import Optional
from torch.utils.checkpoint import checkpoint

from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache

try:
    from transformers.models.qwen3.modeling_qwen3 import FlashAttentionKwargs, Unpack, TransformersKwargs
except ImportError:
    from transformers.models.llama.modeling_llama import FlashAttentionKwargs, Unpack, TransformersKwargs

from .ulysses_utils import gather_seq_dist_head, gather_head_dist_seq
import torch.distributed as dist


def apply_rotary_pos_emb(x, cos, sin, head_dim=2):
    cos = cos.unsqueeze(head_dim)
    sin = sin.unsqueeze(head_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(query, key, position_embeddings, head_first=True):
    cos, sin = position_embeddings
    head_dim = 1 if head_first else 2
    seq_dim = 2 if head_first else 1
    query_length = query.shape[seq_dim]
    key_length = key.shape[seq_dim]
    cos = cos[:, :key_length]
    sin = sin[:, :key_length]
    if query_length == key_length:
        query = apply_rotary_pos_emb(query, cos, sin, head_dim=head_dim)
    else:
        query = apply_rotary_pos_emb(
            query, 
            cos[:, -query_length:], 
            sin[:, -query_length:], 
            head_dim=head_dim)
    key = apply_rotary_pos_emb(key, cos, sin, head_dim=head_dim)
    return query, key


def flash_attention_ulysses(layer_idx, query, key, value, config):
    B, Q_len, H, D = query.shape
    K_len = key.shape[1]
    KH = key.shape[2]
    group_size = H // KH

    Q_trans = query.transpose(1, 2)
    K_trans = key.transpose(1, 2)
    V_trans = value.transpose(1, 2)

    if group_size > 1:
        K_trans = K_trans.repeat_interleave(group_size, dim=1)
        V_trans = V_trans.repeat_interleave(group_size, dim=1)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        Q_trans, K_trans, V_trans, is_causal=True)

    return attn_output.transpose(1, 2)


def attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    query_states = self.q_norm(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim))
    key_states = self.k_norm(self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim))
    value_states = self.v_proj(hidden_states).view(*input_shape, -1, self.head_dim)

    query_states, key_states = apply_rope(query_states, key_states, position_embeddings, head_first=False)

    if dist.is_initialized() and dist.get_world_size() > 1:
        query_states = gather_seq_dist_head(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_dist_head(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_dist_head(value_states, seq_dim=1, head_dim=2)

    attn_output = flash_attention_ulysses(
        self.layer_idx, query_states, key_states, value_states, config=self.train_config)

    if dist.is_initialized() and dist.get_world_size() > 1:
        attn_output = gather_head_dist_seq(attn_output, seq_dim=1, head_dim=2)

    attn_output = attn_output.flatten(2).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None, None


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    
    hidden_states, _, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    mlp_outputs = self.mlp(hidden_states)
    hidden_states = mlp_outputs[0] if isinstance(mlp_outputs, tuple) else mlp_outputs
    hidden_states = residual + hidden_states
    return hidden_states, None


def model_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for decoder_layer in self.layers:
        hidden_states, _ = checkpoint(
            decoder_layer,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            cache_position,
            position_embeddings,
            use_reentrant=False,
            **kwargs
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states)


def causal_forward(self, input_ids=None, labels=None, **kwargs) -> CausalLMOutputWithPast:
    outputs = self.model(input_ids=input_ids, **kwargs)
    logits = self.lm_head(outputs.last_hidden_state)
    
    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return CausalLMOutputWithPast(loss=loss, logits=logits)


def monkey_patch(model, config):
    model.forward = types.MethodType(causal_forward, model)
    model.model.forward = types.MethodType(model_forward, model.model)
    
    for layer in model.model.layers:
        attn = layer.self_attn

        if not hasattr(attn, 'q_norm'):
            attn.q_norm = torch.nn.Identity()
            attn.k_norm = torch.nn.Identity()

        layer.forward = types.MethodType(layer_forward, layer)
        attn.forward = types.MethodType(attention_forward, attn)
        attn.train_config = config

    return model
