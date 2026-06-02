from mem2 import get_monkey_patch
import argparse
import torch

from transformers import AutoModelForCausalLM, StaticCache, AutoConfig, AutoTokenizer
from mem2.monkey_patches.hash_utils import LRUCache, TopkCache
from profiler import WallTime
from torch.profiler import profile, ProfilerActivity

import os


NUM_LAYERS = 1


def load_pg19_prompt(pg19_path, tokenizer, prefill_tokens):
    with open(pg19_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    prompt = text
    while True:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        if input_ids.shape[1] >= prefill_tokens:
            input_ids = input_ids[:, :prefill_tokens]
            break
        prompt = prompt + "\n" + text
    
    return input_ids


def test_ours(args, input_ids, config):
    monkey_patch = get_monkey_patch(args.method)
    model = AutoModelForCausalLM.from_config(
        config,
        dtype=torch.bfloat16).cuda()
    model = monkey_patch(model)
    model.eval()

    caches = [
        LRUCache(
            checkpoint_path_or_dir=args.checkpoint_dir, 
            layer_idx=layer_idx,
            batch_size=args.batch_size, 
            max_position_embeddings=args.prefill_tokens + args.decode_tokens, 
            num_attention_heads=model.config.num_attention_heads, 
            num_key_value_heads=model.config.num_key_value_heads, 
            hash_module_dims=args.dims, 
            lru_budget=args.lru_budget, 
            top_budget=args.top_budget, 
            device='cuda:0')
        for layer_idx in range(NUM_LAYERS)]

    prefill_seq_len = input_ids.shape[1]
    prefill_chunk_size = min(1024, prefill_seq_len)
    for start in range(0, prefill_seq_len, prefill_chunk_size):
        end = min(start + prefill_chunk_size, prefill_seq_len)
        input_chunk = input_ids[:, start:end]
        cache_position = torch.arange(start, end, device=input_chunk.device, dtype=torch.long)
        position_ids = cache_position.unsqueeze(0)
        outputs = model(
            input_ids=input_chunk,
            past_key_values=caches,
            position_ids=position_ids,
            cache_position=cache_position,
            logits_to_keep=1)

    input_ids = outputs.logits.argmax(dim=-1)

    for _ in range(args.decode_tokens - 1):
        input_ids = model(
            input_ids=input_ids,
            past_key_values=caches).logits.argmax(dim=-1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        input_ids = model(
            input_ids=input_ids,
            past_key_values=caches).logits.argmax(dim=-1)

    prof.export_chrome_trace("evaluate/throughput/trace.json")


def test_topk_baseline(args, input_ids, config):
    monkey_patch = get_monkey_patch(args.method)
    model = AutoModelForCausalLM.from_config(
        config,
        dtype=torch.bfloat16).cuda()
    model = monkey_patch(model)
    model.eval()

    caches = [
        TopkCache(
            checkpoint_path_or_dir=args.checkpoint_dir,
            layer_idx=layer_idx,
            batch_size=args.batch_size,
            max_position_embeddings=args.prefill_tokens + args.decode_tokens,
            num_attention_heads=model.config.num_attention_heads,
            num_key_value_heads=model.config.num_key_value_heads,
            hash_module_dims=args.dims,
            top_budget=args.top_budget,
            device='cuda:0')
        for layer_idx in range(NUM_LAYERS)]

    prefill_seq_len = input_ids.shape[1]
    prefill_chunk_size = min(1024, prefill_seq_len)
    for start in range(0, prefill_seq_len, prefill_chunk_size):
        end = min(start + prefill_chunk_size, prefill_seq_len)
        input_chunk = input_ids[:, start:end]
        cache_position = torch.arange(start, end, device=input_chunk.device, dtype=torch.long)
        position_ids = cache_position.unsqueeze(0)
        outputs = model(
            input_ids=input_chunk,
            past_key_values=caches,
            position_ids=position_ids,
            cache_position=cache_position,
            logits_to_keep=1)

    input_ids = outputs.logits.argmax(dim=-1)

    for _ in range(args.decode_tokens - 1):
        input_ids = model(
            input_ids=input_ids,
            past_key_values=caches).logits.argmax(dim=-1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        input_ids = model(
            input_ids=input_ids,
            past_key_values=caches).logits.argmax(dim=-1)

    prof.export_chrome_trace("evaluate/throughput/trace_topk.json")


def test_baseline(args, input_ids, config):
    model = AutoModelForCausalLM.from_config(
        config,
        dtype=torch.bfloat16,
        attn_implementation='flash_attention_2').cuda()
    model.eval()

    past_key_values = StaticCache(
        config=config,
        batch_size=args.batch_size,
        max_cache_len=config.max_position_embeddings,
        device='cuda:0',
        dtype=torch.bfloat16)

    prefill_seq_len = input_ids.shape[1]
    prefill_chunk_size = min(1024, prefill_seq_len)
    for start in range(0, prefill_seq_len, prefill_chunk_size):
        end = min(start + prefill_chunk_size, prefill_seq_len)
        input_chunk = input_ids[:, start:end]
        cache_position = torch.arange(start, end, device=input_chunk.device, dtype=torch.long)
        position_ids = cache_position.unsqueeze(0)
        outputs = model(
            input_ids=input_chunk,
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=cache_position,
            logits_to_keep=1)

    input_ids = outputs.logits.argmax(dim=-1)

    for _ in range(args.decode_tokens - 1):
        input_ids = model(
            input_ids=input_ids,
            past_key_values=past_key_values).logits.argmax(dim=-1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        input_ids = model(
            input_ids=input_ids,
            past_key_values=past_key_values).logits.argmax(dim=-1)

    prof.export_chrome_trace("evaluate/throughput/trace_baseline.json")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--method", type=str, default='eval')
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--prefill-tokens", type=int, default=None)
    parser.add_argument("--decode-tokens", type=int, default=1024)

    parser.add_argument("--lru-budget", type=int, default=None)
    parser.add_argument("--top-budget", type=int, default=None)
    parser.add_argument("--dims", type=int, nargs='+', default=[128,128,128])

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    pg19_path = os.environ['SPOTLIGHT_PG19_PATH']
    input_ids = load_pg19_prompt(pg19_path, tokenizer, args.prefill_tokens)
    input_ids = input_ids.expand(args.batch_size, -1).cuda()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.max_position_embeddings = args.prefill_tokens + args.decode_tokens
    config.num_hidden_layers = NUM_LAYERS
    config.max_window_layers = NUM_LAYERS
    config.layer_types = ['full_attention'] * NUM_LAYERS

    # test_ours(args, input_ids, config)
    # test_topk_baseline(args, input_ids, config)
    test_baseline(args, input_ids, config)


if __name__ == '__main__':
    main()
