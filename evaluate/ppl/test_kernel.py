import torch
import argparse
import os
import json
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from spotlight import get_monkey_patch, load_checkpoint, create_lru_caches

from spotlight.misc import get_env_conf
from spotlight.data import get_corpus
from torch.utils.data import DataLoader
from itertools import chain
import tqdm
import math
import heapq

from spotlight.eval import perplexity
from utils import plot_ppl_curve


def reset_caches(caches):
    for cache in caches:
        cache.reset()


@torch.inference_mode()
def test_ppl_with_caches(model, tokenizer, caches, task_name, task_type, num_instance, truncation, 
                          prefill_chunk_size=1024, prefill_length=None, return_raw=False):
    """
    Test PPL on a task using:
    1. Chunked prefill for initial context (up to prefill_length or lru_size)
    2. Token-by-token simulation for the rest (to properly trigger LRU updates)
    
    Args:
        prefill_length: Number of tokens to prefill before token-by-token simulation.
                       If None, uses lru_size from the first cache.
    """
    task = get_corpus(task_name)
    loader = iter(DataLoader(task, batch_size=1, shuffle=False))
    all_logprobs = []
    
    # Use lru_size as default prefill length (before LRU kicks in)
    if prefill_length is None:
        prefill_length = caches[0].lru_size
    
    for instance_idx in range(num_instance):
        data = next(loader)
        text = data['text'][0] if isinstance(data['text'], list) else data['text']
        
        # Tokenize
        if isinstance(text, str):
            tokens = tokenizer(text, truncation=False, return_tensors='pt').input_ids
        else:
            tokens = text if text.ndim == 2 else text[None, :]

        input_ids = tokens[:, :truncation].cuda()
        seq_len = input_ids.shape[1]
        
        # Reset caches for new instance
        reset_caches(caches)
        
        all_logits = []
        
        # Phase 1: Chunked prefill (up to prefill_length)
        actual_prefill_len = min(prefill_length, seq_len)
        print(f"[Instance {instance_idx + 1}/{num_instance}] Prefill {actual_prefill_len} tokens...", flush=True)
        
        for start in range(0, actual_prefill_len, prefill_chunk_size):
            end = min(start + prefill_chunk_size, actual_prefill_len)
            input_chunk = input_ids[:, start:end]
            
            outputs = model(
                input_ids=input_chunk,
                past_key_values=caches,
                logits_to_keep=input_chunk.shape[1])
            
            all_logits.append(outputs.logits.cpu())
        
        # Phase 2: Token-by-token simulation (for the rest)
        remaining_tokens = seq_len - actual_prefill_len
        if remaining_tokens > 0:
            print(f"[Instance {instance_idx + 1}/{num_instance}] Token-by-token decode {remaining_tokens} tokens...", flush=True)
            
            for pos in tqdm.tqdm(range(actual_prefill_len, seq_len), desc="Decoding", leave=False):
                # Single token input
                input_token = input_ids[:, pos:pos+1]
                
                outputs = model(
                    input_ids=input_token,
                    past_key_values=caches,
                    logits_to_keep=1)
                
                all_logits.append(outputs.logits.cpu())
        
        # Concatenate all logits
        full_logits = torch.cat(all_logits, dim=1)

        # Compute log probabilities
        log_probs = full_logits.log_softmax(dim=-1)
        gold_indices = input_ids[:, 1:].cpu()  # Shift for next token prediction
        logprobs = torch.gather(log_probs[:, :-1], -1, gold_indices.unsqueeze(-1))
        logprobs = logprobs.squeeze(-1).squeeze(0).tolist()
        
        torch.cuda.empty_cache()
        
        if task_type in ["top_ppl", "top-ppl"]:
            top_budget = max(int(len(logprobs) * 0.01), 1)
            logprobs = heapq.nsmallest(top_budget, logprobs)

        all_logprobs.append(logprobs)
        
        # Compute per-instance PPL
        instance_ppl = perplexity(logprobs)
        print(f"[Instance {instance_idx + 1}/{num_instance}] PPL: {instance_ppl:.4f}", flush=True)
    
    # Compute overall perplexity
    flat_logprobs = list(chain.from_iterable(all_logprobs))
    ppl = perplexity(flat_logprobs)
    
    result = {
        "task_name": task_name,
        "task_type": task_type,
        "num_instance": num_instance,
        "truncation": truncation,
        "prefill_length": prefill_length,
        "ppl": float(ppl),
        "num_tokens": len(flat_logprobs)
    }

    if return_raw:
        result["raw_logprobs"] = flat_logprobs

    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--checkpoint-path-or-dir", type=str, default=None,
                        help="Directory containing hash module weights")
    
    # Cache parameters
    parser.add_argument("--max-position-embeddings", type=int, default=40960)
    parser.add_argument("--lru-budget", type=int, default=2048)
    parser.add_argument("--top-budget", type=int, default=1024)
    parser.add_argument("--dims", type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument("--skip-layers", type=int, nargs='+', default=[])
    parser.add_argument("--double-layers", type=int, nargs='+', default=[])
    
    # Prefill parameters
    parser.add_argument("--prefill-chunk-size", type=int, default=2048,
                        help="Chunk size for prefill phase")
    parser.add_argument("--prefill-length", type=int, default=2048,
                        help="Number of tokens to prefill before token-by-token. Default: lru_budget")

    # Plot results
    parser.add_argument("--plot-results", action='store_true')
    
    args = parser.parse_args()

    # Load test config
    test_conf = get_env_conf("evaluate/ppl/perplexity_tasks.json")
    print('config loaded ✅', flush=True)

    # Load model with hash-gen monkey patch
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16)
    model = get_monkey_patch(method='eval')(model)
    model.eval()
    print('model loaded ✅', flush=True)

    caches = create_lru_caches(
        model,
        checkpoint_path_or_dir=args.checkpoint_path_or_dir,
        batch_size=1,
        max_length=args.max_position_embeddings,
        lru_budget=args.lru_budget,
        top_budget=args.top_budget,
        skip_layers=args.skip_layers,
        double_layers=args.double_layers,
        hash_dims=args.dims,
        device='cuda:0')
    print(f'LRUCache created ✅ (skipping hash weights for layers: {args.skip_layers})', flush=True)
    print(f'  - lru_budget: {args.lru_budget}', flush=True)
    print(f'  - top_budget: {args.top_budget}', flush=True)
    print(f'  - prefill_length: {args.prefill_length}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Run PPL tests
    model_name = os.path.basename(os.path.normpath(args.model_name_or_path)).lower()
    vis_dir = f"evaluate/ppl/visualize/{model_name}"

    all_results = []
    for task in test_conf:
        task_name = task['task_name']
        task_type = task.get('task_type', 'ppl')
        filename_base = f"{task_name.split('.')[0]}-kernel"
        cache_path = os.path.join(vis_dir, f"{filename_base}.json")

        # Check cache first
        if args.plot_results and os.path.exists(cache_path):
            print(f"\nPlotting {task_name} (from cache)...", flush=True)
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            plot_ppl_curve(cached["neg_logprobs"], task_name, vis_dir, filename_base)
            print(json.dumps({k: v for k, v in cached.items() if k != "neg_logprobs"}, indent=4), flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"Testing: {task_name}", flush=True)
        print(f"{'='*60}", flush=True)
        
        result = test_ppl_with_caches(
            model=model,
            tokenizer=tokenizer,
            caches=caches,
            task_name=task_name,
            task_type=task_type,
            num_instance=task['num_instance'],
            truncation=task['truncation'],
            prefill_chunk_size=args.prefill_chunk_size,
            prefill_length=args.prefill_length,
            return_raw=args.plot_results)
        
        if args.plot_results:
            raw_logprobs = result.pop("raw_logprobs")
            neg_logprobs = [-x for x in raw_logprobs]

            # Cache to JSON
            os.makedirs(vis_dir, exist_ok=True)
            cache_data = dict(result)
            cache_data["neg_logprobs"] = neg_logprobs
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            print(f'  Cached to {cache_path}', flush=True)

            plot_ppl_curve(neg_logprobs, task_name, vis_dir, filename_base)

        print(f"\nResult:", flush=True)
        print(json.dumps(result, indent=4), flush=True)
        all_results.append(result)


    # Cleanup caches
    del caches
    print("\nDone ✅")


if __name__ == '__main__':
    main()
