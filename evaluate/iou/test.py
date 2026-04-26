import torch
import argparse
import os
import json
import numpy as np
import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from spotlight.monkey_patches import get_monkey_patch
from spotlight.monkey_patches.hash_utils import LRUCache, FullCache
from spotlight.misc import get_env_conf
from spotlight.data import get_corpus
from torch.utils.data import DataLoader
from itertools import chain
from pygments.console import colorize

# Global storage for IOU ratios captured during attention_forward
# Format: {layer_idx: [head0_ratios, head1_ratios, ...]}
LAYER_RATIOS = {}

def get_color(value):
    if value > 75:
        return colorize("green", f"{value:>4d}")
    elif value > 50:
        return colorize("yellow", f"{value:>4d}")
    else:
        return colorize("red", f"{value:>4d}")

def reset_ratios():
    global LAYER_RATIOS
    LAYER_RATIOS = {}

def compute_iou(pred_indices, oracle_indices):
    """
    pred_indices: [B, H, 1, K]
    oracle_indices: [B, H, 1, K]
    """
    B, H, _, K = pred_indices.shape
    ious = []
    for b in range(B):
        h_ious = []
        for h in range(H):
            p = set(pred_indices[b, h, 0].tolist())
            o = set(oracle_indices[b, h, 0].tolist())
            intersection = len(p.intersection(o))
            union = len(p.union(o))
            h_ious.append(intersection / union if union > 0 else 0)
        ious.append(h_ious)
    return np.array(ious) # [B, H]

def create_lru_caches(model, args, skip_layers):
    num_layers = model.config.num_hidden_layers
    caches = []
    for layer_idx in range(num_layers):
        if layer_idx in skip_layers:
            cache = FullCache()
        else:
            cache = LRUCache(
                checkpoint_dir=args.checkpoint_dir, 
                layer_idx=layer_idx,
                batch_size=1,
                max_position_embeddings=args.max_position_embeddings, 
                num_attention_heads=model.config.num_attention_heads, 
                num_key_value_heads=model.config.num_key_value_heads, 
                hash_module_dims=args.dims, 
                lru_budget=args.lru_budget, 
                top_budget=args.top_budget, 
                device='cuda')
        caches.append(cache)
    return caches

@torch.inference_mode()
def test_iou_with_caches(model, tokenizer, caches, task_name, num_instance, truncation, 
                         prefill_chunk_size=2048, prefill_length=None):
    task = get_corpus(task_name)
    loader = iter(DataLoader(task, batch_size=1, shuffle=False))
    
    if prefill_length is None:
        # Default to lru_size if using LRUCache, otherwise 0
        prefill_length = getattr(caches[0], 'lru_size', 0)
    
    # We will hook into the model to record IOU during the "Token-by-token" phase
    # where the HashModule is active and we can compare with Oracle.
    
    def instrument_attention(model, top_budget):
        import types
        from flash_attn import flash_attn_func
        
        def apply_rotary_pos_emb(x, position_embeddings):
            cos = position_embeddings[0].unsqueeze(2)
            sin = position_embeddings[1].unsqueeze(2)
            x_embed = (x * cos) + (rotate_half(x) * sin)
            return x_embed

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def iou_attention_forward(self, hidden_states, position_embeddings, past_key_value=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            
            q = self.q_proj(hidden_states).view(hidden_shape)
            if hasattr(self, 'q_norm'): q = self.q_norm(q)
            q = apply_rotary_pos_emb(q, position_embeddings)
            
            # Predict top-k using hash
            if isinstance(past_key_value, LRUCache):
                past_key_value.update_query(q)
                
                # Manual IOU calculation for monitoring
                # We need the key_bins and query projections
                q_proj0, q_proj1 = past_key_value.query_hash.get_proj_weights()
                from spotlight.kernel import hash_packbits_hamming
                
                kv_len = torch.tensor(past_key_value.num_tokens, device=q.device, dtype=torch.int32)
                # Note: hash_packbits_hamming computes counts of matching bits
                hamming = hash_packbits_hamming.hash_packbits_hamming(
                    q.bfloat16(), q_proj0, q_proj1, past_key_value.key_bins, kv_len) # [B, KH, 1, T]
                
                pred_indices = hamming.topk(k=top_budget, dim=-1).indices # [B, KH, 1, K]
                
                # Oracle top-k
                k_full = past_key_value.key_cache[:, :kv_len]
                # Average Q heads per KV group for IOU matching if GQA
                group_size = model.config.num_attention_heads // model.config.num_key_value_heads
                q_avg = q.view(q.shape[0], q.shape[1], -1, group_size, q.shape[-1]).mean(dim=-2) # [B, 1, KH, D]
                q_avg = q_avg.transpose(1, 2) # [B, KH, 1, D]
                k_full = k_full.transpose(1, 2) # [B, KH, T, D]
                
                scores = torch.matmul(q_avg, k_full.transpose(-1, -2)) / (self.head_dim ** 0.5)
                oracle_indices = scores.topk(k=top_budget, dim=-1).indices
                
                ious = compute_iou(pred_indices, oracle_indices) # [B, H]
                
                if self.layer_idx not in LAYER_RATIOS:
                    LAYER_RATIOS[self.layer_idx] = [[] for _ in range(model.config.num_key_value_heads)]
                for h in range(model.config.num_key_value_heads):
                    LAYER_RATIOS[self.layer_idx][h].append(ious[0, h])

            k = self.k_proj(hidden_states).view(hidden_shape)
            if hasattr(self, 'k_norm'): k = self.k_norm(k)
            v = self.v_proj(hidden_states).view(hidden_shape)
            k = apply_rotary_pos_emb(k, position_embeddings)
            
            k, v = past_key_value.update(k, v)
            
            attn_output = flash_attn_func(q, k, v, causal=True)
            attn_output = attn_output.flatten(2).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output

        for idx, layer in enumerate(model.model.layers):
            layer.self_attn.layer_idx = idx
            layer.self_attn.forward = types.MethodType(iou_attention_forward, layer.self_attn)

    instrument_attention(model, args.top_budget)

    for instance_idx in range(num_instance):
        data = next(loader)
        text = data['text'][0] if isinstance(data['text'], list) else data['text']
        if isinstance(text, str):
            tokens = tokenizer(text, truncation=False, return_tensors='pt').input_ids
        else:
            tokens = text if text.ndim == 2 else text[None, :]
        input_ids = tokens[:, :truncation].cuda()
        seq_len = input_ids.shape[1]
        
        for cache in caches: cache.reset()
        reset_ratios()
        
        # Phase 1: Prefill
        actual_prefill_len = min(prefill_length, seq_len)
        print(f"[Instance {instance_idx + 1}/{num_instance}] Prefill {actual_prefill_len} tokens...")
        for start in range(0, actual_prefill_len, prefill_chunk_size):
            end = min(start + prefill_chunk_size, actual_prefill_len)
            model(input_ids=input_ids[:, start:end], past_key_values=caches, use_cache=True)
        
        # Phase 2: Token-by-token (IOU tracking happens here)
        remaining = seq_len - actual_prefill_len
        if remaining > 0:
            print(f"[Instance {instance_idx + 1}/{num_instance}] Decoding {remaining} tokens with IOU tracking...")
            for pos in tqdm.tqdm(range(actual_prefill_len, seq_len), desc="IOU Tracking"):
                model(input_ids=input_ids[:, pos:pos+1], past_key_values=caches, use_cache=True)
            
            # Print IOU Table
            print_iou_table(model.config.num_key_value_heads)

def print_iou_table(num_heads):
    print("\nIOU Table (Hit Rate %):")
    print("      ", end='')
    for head_id in range(num_heads):
        print(f'#{head_id:>4d}', end=' ')
    print(f"avg")

    sorted_layers = sorted(LAYER_RATIOS.keys())
    mean_ratios = [[] for _ in range(num_heads + 1)]

    for layer_idx in sorted_layers:
        print(f"{layer_idx:>4d}:", end=' ')
        layer_head_ious = []
        for h in range(num_heads):
            val = int(np.mean(LAYER_RATIOS[layer_idx][h]) * 100)
            print(get_color(val), end=' ')
            layer_head_ious.append(val)
            mean_ratios[h].append(val)
        
        avg_val = int(np.mean(layer_head_ious))
        print(get_color(avg_val))
        mean_ratios[-1].append(avg_val)

    print(f"AVG :", end=' ')
    for h_list in mean_ratios:
        if h_list:
            avg = sum(h_list) // len(h_list)
            print(get_color(avg), end=' ')
    print("\n")

def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--max-position-embeddings", type=int, default=40960)
    parser.add_argument("--lru-budget", type=int, default=2048)
    parser.add_argument("--top-budget", type=int, default=1024)
    parser.add_argument("--dims", type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument("--skip-layers", type=int, nargs='+', default=[0, 1])
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)
    parser.add_argument("--prefill-length", type=int, default=None)
    
    args = parser.parse_args()

    test_conf = get_env_conf("evaluate/iou/test.json")
    
    monkey_patch = get_monkey_patch('hash-gen')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='cuda',
        torch_dtype=torch.bfloat16)
    model = monkey_patch(model)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    caches = create_lru_caches(model, args, set(args.skip_layers))

    for task in test_conf:
        print(f"\nTesting IOU on: {task['task_name']}")
        test_iou_with_caches(
            model=model,
            tokenizer=tokenizer,
            caches=caches,
            task_name=task['task_name'],
            num_instance=task['num_instance'],
            truncation=task['truncation'],
            prefill_chunk_size=args.prefill_chunk_size,
            prefill_length=args.prefill_length
        )

if __name__ == '__main__':
    main()
