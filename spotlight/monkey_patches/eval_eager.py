import torch
import types
from typing import Optional, Union, Any
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from spotlight.kernel import (
    hash_packbits,
    hash_packbits_hamming,
)

from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache

try:
    from transformers.models.qwen3.modeling_qwen3 import (
        FlashAttentionKwargs,
        Unpack,
        TransformersKwargs,
        create_causal_mask,
        create_sliding_window_causal_mask,
    )
except ImportError:
    from transformers.models.llama.modeling_llama import (
        FlashAttentionKwargs,
        Unpack,
        TransformersKwargs,
    )
    create_causal_mask = None
    create_sliding_window_causal_mask = None

from .hash_utils import HashModule
from dataclasses import dataclass
import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict


def _lru_worker(args):
    b, h, topk_data, cache_size, S = args
    
    mask_slice = np.zeros((S, S), dtype=np.bool_)
    hit_rates_slice = np.zeros(S, dtype=np.float32)
    
    lru_budget = cache_size - 1
    lru_cache = OrderedDict()
    
    for q in range(S):
        requests = topk_data[q]
        
        hits = 0
        total_valid = 0
        
        for idx in requests:
            if idx < 0:
                continue
            idx = int(idx)
            if idx >= q:
                continue
            total_valid += 1
            
            if idx in lru_cache:
                lru_cache.move_to_end(idx)
                hits += 1
            else:
                while len(lru_cache) >= lru_budget:
                    lru_cache.popitem(last=False)
                lru_cache[idx] = True
        
        if total_valid > 0:
            hit_rates_slice[q] = hits / total_valid
        
        for idx in lru_cache.keys():
            mask_slice[q, idx] = True
        mask_slice[q, q] = True
    
    return b, h, mask_slice, hit_rates_slice


def simulate_lru_and_build_mask(topk_indices, cache_size, seq_len, num_workers=None):
    B, H, S, K = topk_indices.shape
    device = topk_indices.device
    
    topk_cpu = topk_indices.cpu().numpy()
    
    mask_cpu = np.zeros((B, H, S, S), dtype=np.bool_)
    hit_rates_cpu = np.zeros((B, H, S), dtype=np.float32)
    
    total_tasks = B * H
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, total_tasks)
    
    use_parallel = num_workers > 1 and total_tasks > 1
    
    if use_parallel:
        tasks = []
        for b in range(B):
            for h in range(H):
                tasks.append((b, h, topk_cpu[b, h], cache_size, S))
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_lru_worker, tasks))
        
        for b, h, mask_slice, hit_rates_slice in results:
            mask_cpu[b, h] = mask_slice
            hit_rates_cpu[b, h] = hit_rates_slice
    else:
        for b in range(B):
            for h in range(H):
                _, _, mask_slice, hit_rates_slice = _lru_worker(
                    (b, h, topk_cpu[b, h], cache_size, S)
                )
                mask_cpu[b, h] = mask_slice
                hit_rates_cpu[b, h] = hit_rates_slice
    
    mask = torch.from_numpy(mask_cpu).to(device)
    hit_rates = torch.from_numpy(hit_rates_cpu).to(device)
    
    return mask, hit_rates


def aggregate_topk(x, k):
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    _, x_topk = x.topk(k=k, dim=-1)
    return x_topk


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, head_dim=2):
    cos = cos.unsqueeze(head_dim)
    sin = sin.unsqueeze(head_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


@dataclass
class CausalLMOutputWithMetrics(CausalLMOutputWithPast):
    metrics: Optional[tuple[dict]] = None


@torch.no_grad()
def causal_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
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
    outputs, metrics = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    return CausalLMOutputWithMetrics(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        metrics=metrics
    )


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
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    past_key_values = None

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        if create_causal_mask is not None:
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if getattr(self, 'has_sliding_layers', False):
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            causal_mask_mapping = {
                "full_attention": None,
            }

    hidden_states = inputs_embeds

    if not hasattr(self, 'position_embeddings'):
        position_ids = torch.arange(self.config.max_position_embeddings, dtype=torch.long, device='cuda')
        position_ids = position_ids.unsqueeze(0)
        self.position_embeddings = self.rotary_emb(hidden_states, position_ids)
    position_ids = None
    metrics = []

    for decoder_layer in tqdm.tqdm(self.layers[: self.config.num_hidden_layers]):
        hidden_states, metric = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping.get(getattr(decoder_layer, 'attention_type', 'full_attention'), None),
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=self.position_embeddings,
            **kwargs)
        metrics.append(metric)


    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None), metrics


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


def attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

    assert getattr(self, 'sliding_window', None) is None, f"we do not support sliding window currently."

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    if hasattr(self, 'q_norm'):
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    query_states, key_states = apply_rope(query_states, key_states, position_embeddings, False)

    metric = dict[Any, Any]()

    if hasattr(self, 'query_hash'):
        # Use hash_packbits kernel for keys (fused hash + packbits)
        k_proj0, k_proj1 = self.key_hash.get_proj_weights()
        k_bins = hash_packbits.hash_packbits(key_states, k_proj0, k_proj1)  # [B, T, KH, D/32]

        # Use hash_packbits_hamming kernel for queries (one query at a time)
        q_proj0, q_proj1 = self.query_hash.get_proj_weights()
        seq_len = query_states.shape[1]
        kv_length = torch.tensor(seq_len, device=query_states.device, dtype=torch.int32)
        
        hamming_list = []
        for q_idx in range(seq_len):
            query_single = query_states[:, q_idx:q_idx+1, :, :]  # [B, 1, QH, D]
            hamming_single = hash_packbits_hamming.hash_packbits_hamming(
                query_single, q_proj0, q_proj1, k_bins, kv_length)  # [B, KH, seq_len]
            hamming_list.append(hamming_single.unsqueeze(2))  # [B, KH, 1, seq_len]
        
        hamming = torch.cat(hamming_list, dim=2)  # [B, KH, S, S]
        
        seq_len = hamming.shape[-1]
        seq_range = torch.arange(seq_len, device=hamming.device)
        diagonal_mask = seq_range[:, None] == seq_range[None, :]
        diagonal_mask = diagonal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = seq_range[:, None] < seq_range[None, :]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        hamming = hamming * ~causal_mask

        lru_budget = self.hash_config.get('lru_budget', 0.05)
        lru_size = int(lru_budget * seq_len)

        top_budget = int(self.hash_config.get('top_budget', 0.025) * seq_len)
        pred_topk_indices = hamming.topk(k=top_budget, dim=-1).indices  # [batch, num_kv_heads, seq, top_budget]
        
        if lru_size > 0:
            mask, cache_hit_rates = simulate_lru_and_build_mask(
                pred_topk_indices, lru_size, seq_len)
        else:
            mask = torch.zeros_like(hamming, dtype=torch.bool)
            mask.scatter_(dim=-1, index=pred_topk_indices, value=True)
            mask.masked_fill_(causal_mask, value=False)
            mask.masked_fill_(diagonal_mask, value=True)
            cache_hit_rates = None

        if self.hash_config['visualize_head0']:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(mask[0,0].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
            plt.colorbar()
            plt.savefig(f'visualize/layer-{self.layer_idx}-estimated.jpg')
            plt.close()

        attn_output_heads = []
        q_heads = query_states.transpose(1,2)  # [batch, num_q_heads, seq, head_dim]
        k_heads = key_states.transpose(1,2)    # [batch, num_kv_heads, seq, head_dim]
        v_heads = value_states.transpose(1,2)
        group_size = q_heads.shape[1] // k_heads.shape[1]
        batch_size = q_heads.shape[0]
        num_kv_heads = k_heads.shape[1]

        if lru_size > 0 and lru_size < seq_len:
            num_tokens_after_lru = seq_len - lru_size
            iou_per_token = torch.zeros(batch_size, num_kv_heads, num_tokens_after_lru, device=hamming.device)
            
            chunk_size = 256
            
            for chunk_start in range(lru_size, seq_len, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seq_len)
                chunk_len = chunk_end - chunk_start
                out_start = chunk_start - lru_size
                out_end = chunk_end - lru_size
                
                pred_chunk = pred_topk_indices[:, :, chunk_start:chunk_end, :]  # [batch, heads, chunk, top_budget]
                
                for kv_head_idx in range(num_kv_heads):
                    q_group_start = kv_head_idx * group_size
                    q_group_end = q_group_start + group_size
                    q_chunk = q_heads[:, q_group_start:q_group_end, chunk_start:chunk_end, :].mean(dim=1, keepdim=True)  # [batch, 1, chunk, head_dim]
                    k_head = k_heads[:, kv_head_idx:kv_head_idx+1, :, :]  # [batch, 1, seq, head_dim]
                    
                    # Compute attention scores [batch, 1, chunk, seq]
                    attn_scores = torch.matmul(q_chunk, k_head.transpose(-1, -2)) / (self.head_dim ** 0.5)
                    
                    # Apply causal mask for this chunk
                    chunk_query_pos = torch.arange(chunk_start, chunk_end, device=attn_scores.device).view(1, 1, -1, 1)
                    key_pos = torch.arange(seq_len, device=attn_scores.device).view(1, 1, 1, -1)
                    chunk_causal = key_pos > chunk_query_pos
                    attn_scores = attn_scores.masked_fill(chunk_causal, float('-inf'))
                    
                    # Get oracle topk indices [batch, 1, chunk, top_budget]
                    oracle_chunk = attn_scores.topk(k=top_budget, dim=-1).indices
                    
                    # Get predicted topk for this head
                    pred_head_chunk = pred_chunk[:, kv_head_idx:kv_head_idx+1, :, :]  # [batch, 1, chunk, top_budget]
                    
                    # Compute intersection using scatter-gather approach (memory efficient)
                    # Create indicator for oracle positions
                    indicator = torch.zeros(batch_size, 1, chunk_len, seq_len, device=hamming.device, dtype=torch.bool)
                    indicator.scatter_(dim=-1, index=oracle_chunk, value=True)
                    
                    # Check how many predicted positions hit oracle
                    hits = indicator.gather(dim=-1, index=pred_head_chunk)  # [batch, 1, chunk, top_budget]
                    intersection = hits.sum(dim=-1).float()  # [batch, 1, chunk]
                    
                    # Union = |pred| + |oracle| - |intersection| = 2 * top_budget - intersection
                    union = 2 * top_budget - intersection
                    iou_chunk = intersection / union.clamp(min=1)
                    
                    iou_per_token[:, kv_head_idx:kv_head_idx+1, out_start:out_end] = iou_chunk
                    
                    # Clean up to free memory
                    del attn_scores, oracle_chunk, indicator, hits
            
            # Return per-token metrics (averaged over heads for simplicity, but keep full tensor)
            metric['iou'] = iou_per_token.mean(dim=1).squeeze(0)  # [num_tokens_after_lru]
        else:
            metric['iou'] = torch.tensor([], device=hamming.device)
        
        # 使用真正的LRU缓存命中率（来自模拟）
        if cache_hit_rates is not None:
            # cache_hit_rates: [batch, num_kv_heads, seq]
            # 只取lru_size之后的token，与iou对齐
            if lru_size > 0 and lru_size < seq_len:
                metric['cache_hit_rate'] = cache_hit_rates[:, :, lru_size:].mean(dim=1).squeeze(0)
            else:
                metric['cache_hit_rate'] = cache_hit_rates.mean(dim=1).squeeze(0)
        else:
            metric['cache_hit_rate'] = torch.tensor([], device=hamming.device)

        for head_idx in range(q_heads.shape[1]):
            kv_head_idx = head_idx // group_size
            q_head = q_heads[:, head_idx: head_idx + 1, :]
            k_head = k_heads[:, kv_head_idx: kv_head_idx + 1, :]
            v_head = v_heads[:, kv_head_idx: kv_head_idx + 1, :]
            attn_output_head = torch.nn.functional.scaled_dot_product_attention(
                q_head,
                k_head,
                v_head,
                attn_mask=mask[:, kv_head_idx:kv_head_idx + 1, :, :],
                is_causal=False)
            attn_output_heads.append(attn_output_head)

        attn_output = torch.cat(attn_output_heads, dim=1)
        attn_output = attn_output.transpose(1,2).flatten(2).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, metric

    else:

        query_states = query_states.transpose(1,2)
        key_states = key_states.transpose(1,2)
        value_states = value_states.transpose(1,2)
            
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
            enable_gqa=True)

    attn_output = attn_output.transpose(1,2).flatten(2).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, metric


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    
    hidden_states, metrics = self.self_attn(
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
    return hidden_states, metrics


def monkey_patch(model, config):
    model.forward = types.MethodType(causal_forward, model)
    model.model.forward = types.MethodType(model_forward, model.model)

    dtype = next(model.parameters()).dtype
    for layer_idx, layer in enumerate(model.model.layers):
        layer.forward = types.MethodType(layer_forward, layer)
        layer.self_attn.forward = types.MethodType(attention_forward, layer.self_attn)

        if layer_idx not in config['freeze_layers']:
            layer.self_attn.hash_config = config
            
            layer.self_attn.key_hash = HashModule(
                layer.self_attn.config.num_key_value_heads,
                dims=config['hash_dims'],
                dtype=dtype,
                device='cuda')

            layer.self_attn.query_hash = HashModule(
                layer.self_attn.config.num_attention_heads,
                dims=config['hash_dims'],
                dtype=dtype,
                device='cuda')

    return model
