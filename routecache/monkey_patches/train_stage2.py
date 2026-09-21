import torch
import types
import wandb
from typing import Optional
from torch.utils.checkpoint import checkpoint

from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache

try:
    from transformers.models.qwen3.modeling_qwen3 import FlashAttentionKwargs, Unpack, TransformersKwargs
except ImportError:
    from transformers.models.llama.modeling_llama import FlashAttentionKwargs, Unpack, TransformersKwargs

from .hash_utils import HashModule, DynamicGain
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


def compute_topk(score, num_kv_heads, top_k, causal_mask):
    assert score.ndim == 4    
    score = score.masked_fill(causal_mask[None, None], float('-inf'))
    return torch.topk(score, min(top_k, score.shape[-1]), dim=-1).indices


def flash_attention_vgate(layer_idx, query, key, q_hash, k_hash, value, config, enable_topk=False):
    B, Q_len, H, D = query.shape
    K_len = key.shape[1]
    KH = key.shape[2]
    group_size = H // KH

    Q_trans = query.transpose(1, 2)
    K_trans = key.transpose(1, 2).repeat_interleave(group_size, dim=1)
    V_trans = value.transpose(1, 2).repeat_interleave(group_size, dim=1)
    k_hash = k_hash.repeat_interleave(group_size, dim=2)

    qh_trans = q_hash.transpose(1, 2)
    kh_trans = k_hash.transpose(1, 2)

    def compute_chunk_for_topk(q_chunk, kh_trans, Q_trans_chunk, K_trans, V_trans, q_start, q_end):
        causal_mask_chunk = torch.arange(q_start, q_end, device=q_chunk.device)[:, None] < torch.arange(K_len, device=q_chunk.device)[None, :]

        assert torch.is_grad_enabled() == False
        if group_size > 1:
            q_chunk_grouped = q_chunk.unflatten(1, (KH, group_size)).sum(2)
            kh_trans_grouped = kh_trans.unflatten(1, (KH, group_size))[:, :, 0]
            hash_score_chunk_grouped = torch.matmul(q_chunk_grouped, kh_trans_grouped.transpose(-1, -2))
            
            top_indices_grouped = compute_topk(hash_score_chunk_grouped, KH, config.get('top_k'), causal_mask_chunk)
            mask_grouped = torch.full_like(hash_score_chunk_grouped, float('-inf'))
            mask_grouped.scatter_(3, top_indices_grouped, 0.0)
            mask_grouped.masked_fill_(causal_mask_chunk[None, None], float('-inf'))
            mask = mask_grouped.repeat_interleave(group_size, dim=1)
        else:
            hash_score_chunk = torch.matmul(q_chunk, kh_trans.transpose(-1, -2))
            top_indices = compute_topk(hash_score_chunk, KH, config.get('top_k'), causal_mask_chunk)
            mask = torch.full_like(hash_score_chunk, float('-inf'))
            mask.scatter_(3, top_indices, 0.0)
            mask.masked_fill_(causal_mask_chunk[None, None], float('-inf'))

        return torch.nn.functional.scaled_dot_product_attention(
            Q_trans_chunk, 
            K_trans, 
            V_trans, 
            attn_mask=mask)

    if enable_topk:
        chunk_size = config.get('chunk_size')
        attn_outputs = []

        for i in range(0, Q_len, chunk_size):
            q_start = i
            q_end = min(i + chunk_size, Q_len)
            
            q_chunk = qh_trans[..., q_start:q_end, :]
            Q_trans_chunk = Q_trans[..., q_start:q_end, :]

            chunk_out = compute_chunk_for_topk(
                q_chunk,
                kh_trans,
                Q_trans_chunk,
                K_trans,
                V_trans,
                q_start,
                q_end)
            attn_outputs.append(chunk_out)

        attn_output = torch.cat(attn_outputs, dim=-2)

    else:
        assert Q_trans.shape[-1] == qh_trans.shape[-1]

        if group_size > 1:
            qh_trans = (
                qh_trans.unflatten(1, (KH, group_size))
                .mean(dim=2, keepdim=True)
                .repeat_interleave(group_size, dim=2)
                .flatten(1, 2))


        Q_cat = torch.cat([Q_trans, qh_trans], dim=-1)
        K_cat = torch.cat([K_trans, kh_trans], dim=-1)

        scalar = 1.0 / (D ** 0.25)
        Q_cat = Q_cat * scalar
        K_cat = K_cat * scalar

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            Q_cat, K_cat, V_trans, 
            is_causal=True,
            scale=1.0)

    return attn_output.transpose(1,2)


def attention_vgate(layer_idx, query, key, q_hash, k_hash, value, config, enable_topk=False):
    B, Q_len, H, D = query.shape
    K_len = key.shape[1]
    KH = key.shape[2]
    group_size = H // KH

    Q_trans = query.transpose(1, 2)
    K_trans = key.transpose(1, 2).repeat_interleave(group_size, dim=1)
    V_trans = value.transpose(1, 2).repeat_interleave(group_size, dim=1)
    k_hash = k_hash.repeat_interleave(group_size, dim=2)

    qh_trans = q_hash.transpose(1, 2)
    kh_trans = k_hash.transpose(1, 2)

    def compute_chunk(q_chunk, kh_trans, Q_trans_chunk, K_trans, V_trans, q_start, q_end):
        causal_mask_chunk = torch.arange(q_start, q_end, device=q_chunk.device)[:, None] < torch.arange(K_len, device=q_chunk.device)[None, :]
        hash_score_chunk = torch.matmul(q_chunk, kh_trans.transpose(-1, -2))

        if group_size > 1:
            hash_score_chunk = hash_score_chunk.unflatten(1, (KH, group_size)).mean(2).repeat_interleave(group_size, dim=1)

        if enable_topk:
            top_indices = compute_topk(hash_score_chunk, KH, config.get('top_k'), causal_mask_chunk)
            mask = torch.full_like(hash_score_chunk, float('-inf'))
            mask.scatter_(3, top_indices, 0.0)
            mask.masked_fill_(causal_mask_chunk[None, None], float('-inf'))
            attn_output_chunk = torch.nn.functional.scaled_dot_product_attention(Q_trans_chunk, K_trans, V_trans, attn_mask=mask)
        else:   
            additive_gate = hash_score_chunk.float() / (q_hash.shape[-1] ** 0.5)
            mask = additive_gate
            mask.masked_fill_(causal_mask_chunk[None, None], float('-inf'))
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                attn_output_chunk = torch.nn.functional.scaled_dot_product_attention(Q_trans_chunk, K_trans, V_trans, attn_mask=mask)

        return attn_output_chunk

    chunk_size = config.get('chunk_size')
    attn_outputs = []

    for i in range(0, Q_len, chunk_size):
        q_start = i
        q_end = min(i + chunk_size, Q_len)
        
        q_chunk = qh_trans[..., q_start:q_end, :]
        Q_trans_chunk = Q_trans[..., q_start:q_end, :]

        chunk_out = checkpoint(
            compute_chunk,
            q_chunk,
            kh_trans,
            Q_trans_chunk,
            K_trans,
            V_trans,
            q_start,
            q_end,
            use_reentrant=False
        )
        attn_outputs.append(chunk_out)

    attn_output = torch.cat(attn_outputs, dim=-2)

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
    q_hash = self.query_hash(query_states)
    k_hash = self.key_hash(key_states)

    if getattr(self, "enable_topk", False):
        q_hash = ((q_hash > 0).to(q_hash.dtype) * 2.0 - 1.0)
        k_hash = ((k_hash > 0).to(k_hash.dtype) * 2.0 - 1.0)
    else:
        q_hash = torch.tanh(q_hash)
        k_hash = torch.tanh(k_hash)

    if dist.is_initialized() and dist.get_world_size() > 1:
        query_states = gather_seq_dist_head(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_dist_head(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_dist_head(value_states, seq_dim=1, head_dim=2)
        q_hash = gather_seq_dist_head(q_hash, seq_dim=1, head_dim=2)
        k_hash = gather_seq_dist_head(k_hash, seq_dim=1, head_dim=2)

    attn_output = flash_attention_vgate(
        self.layer_idx, query_states, key_states, q_hash, k_hash, value_states, config=self.train_config, enable_topk=self.enable_topk)

    if dist.is_initialized() and dist.get_world_size() > 1:
        attn_output = gather_head_dist_seq(attn_output, seq_dim=1, head_dim=2)

    attn_output = attn_output.flatten(2).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output


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
    
    hidden_states = self.self_attn(
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
    return hidden_states


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
        hidden_states = checkpoint(
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
    
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype

    for layer_idx, layer in enumerate(model.model.layers):

        attn = layer.self_attn
        num_q_heads = attn.config.num_attention_heads
        num_kv_heads = attn.config.num_key_value_heads
        attn.key_hash = HashModule(num_kv_heads, dims=config['hash_dims'], dtype=dtype, device=device)
        attn.query_hash = HashModule(num_q_heads, dims=config['hash_dims'], dtype=dtype, device=device)

        if layer_idx not in config.get("skip_layers", []):
            if not hasattr(attn, 'q_norm'):
                attn.q_norm = torch.nn.Identity()
                attn.k_norm = torch.nn.Identity()

            layer.forward = types.MethodType(layer_forward, layer)
            attn.forward = types.MethodType(attention_forward, attn)
            attn.enable_topk = config.get("enable_topk", False)
            attn.train_config = config

    return model