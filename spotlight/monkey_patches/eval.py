import torch
import types
from typing import Optional, Union
from .hash_utils import LRUCache
from flash_attn import flash_attn_func

from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

try:
    from transformers.models.qwen3.modeling_qwen3 import (
        FlashAttentionKwargs,
        Unpack,
        TransformersKwargs,
    )
except ImportError:
    from transformers.models.llama.modeling_llama import (
        FlashAttentionKwargs,
        Unpack,
        TransformersKwargs,
    )


def aggregate_topk(x, k):
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    _, x_topk = x.topk(k=k, dim=-1)
    return x_topk


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, position_embeddings):
    cos = position_embeddings[0].unsqueeze(2)
    sin = position_embeddings[1].unsqueeze(2)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


@torch.no_grad()
def causal_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[LRUCache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> CausalLMOutputWithPast:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

    >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,)

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def get_kv_length(past_key_values):
    if past_key_values is None:
        return 0

    if hasattr(past_key_values, "get_seq_length"):
        kv_length = past_key_values.get_seq_length()
    elif isinstance(past_key_values, list) and len(past_key_values) > 0 and hasattr(past_key_values[0], "get_seq_length"):
        kv_length = past_key_values[0].get_seq_length()
    elif hasattr(past_key_values, "__len__") and len(past_key_values) > 0 and hasattr(past_key_values[0], "get_seq_length"):
        kv_length = past_key_values[0].get_seq_length()
    else:
        kv_length = past_key_values[0][0].shape[-2] if isinstance(past_key_values[0], tuple) else past_key_values[0].shape[-2]

    return kv_length


def model_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[LRUCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    hidden_states = inputs_embeds

    # Important for chunked prefill: when feeding multiple tokens while a cache already exists,
    # we must offset absolute positions by the current KV length (RoPE).
    if position_ids is None:
        kv_length = get_kv_length(past_key_values)
        seq_len = input_ids.shape[-1]
        position_ids = torch.arange(
            kv_length, kv_length + seq_len, dtype=torch.long, device=inputs_embeds.device
        ).unsqueeze(0)
    self.position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        hidden_states = decoder_layer(
            hidden_states,
            past_key_value=past_key_values[layer_idx],
            position_embeddings=self.position_embeddings,
            **kwargs)

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None)


def attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[LRUCache] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

    assert getattr(self, 'sliding_window', None) is None, f"we do not support sliding window currently."

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    if hasattr(self, 'q_norm'):
        query_states = self.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, position_embeddings)

    if hasattr(past_key_value, 'update_query'):
        past_key_value.update_query(query_states)

    key_states = self.k_proj(hidden_states).view(hidden_shape)
    if hasattr(self, 'k_norm'):
        key_states = self.k_norm(key_states)
    value_states = self.v_proj(hidden_states).view(hidden_shape)
    key_states = apply_rotary_pos_emb(key_states, position_embeddings)

    key_states, value_states = past_key_value.update(key_states, value_states)

    attn_output = flash_attn_func(
        query_states,
        key_states,
        value_states,
        causal=True)

    attn_output = attn_output.flatten(2).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[LRUCache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    
    hidden_states = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    mlp_out = self.mlp(hidden_states)
    if isinstance(mlp_out, tuple):
        mlp_out = mlp_out[0]
    hidden_states = residual + mlp_out
    return hidden_states


def monkey_patch(model, config):
    model.forward = types.MethodType(causal_forward, model)
    model.model.forward = types.MethodType(model_forward, model.model)
    for layer_idx, layer in enumerate(model.model.layers):
        layer.forward = types.MethodType(layer_forward, layer)
        layer.self_attn.forward = types.MethodType(attention_forward, layer.self_attn)
    return model
