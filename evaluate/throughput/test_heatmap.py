"""
Throughput heatmap: batch_size (Y) x context_length (X).
Measures decode latency (ms/token) with first 3 layers.
Tests ours (LRUCache) fully, then baseline (FA2 + StaticCache) fully.
Uses true next-token (argmax) for autoregressive decode.
"""
from spotlight import get_monkey_patch
from spotlight.monkey_patches.hash_utils import LRUCache, CacheList
import argparse, csv, torch, time, os, gc, copy
import numpy as np
from transformers import AutoModelForCausalLM, StaticCache, DynamicCache, AutoConfig, AutoTokenizer


BATCH_SIZES = [4, 8, 16, 32]
CONTEXT_LENGTHS = [32768, 65536, 131072, 262144]
CONTEXT_LABELS = ['32K', '64K', '128K', '256K']
NUM_LAYERS = 1
WARMUP_STEPS = 5
MEASURE_STEPS = 20
CHUNK_SIZE = 1024


def load_pg19_tokens(pg19_path, tokenizer, max_tokens):
    cache_path = pg19_path + f".tok_{max_tokens}.pt"
    if os.path.exists(cache_path):
        print(f"Loading cached tokens from {cache_path}")
        return torch.load(cache_path)
    with open(pg19_path, 'r', encoding='utf-8') as f:
        text = f.read()
    input_ids = tokenizer.encode(text, return_tensors='pt')
    if input_ids.shape[1] < max_tokens:
        repeats = (max_tokens // input_ids.shape[1]) + 1
        input_ids = input_ids.repeat(1, repeats)
    input_ids = input_ids[:, :max_tokens]
    torch.save(input_ids, cache_path)
    return input_ids


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


# ── Ours (monkey-patched + LRUCache) ────────────────────────────────────────

def make_ours_cache(args, config, bs, max_pos):
    caches = []
    for layer_idx in range(NUM_LAYERS):
        caches.append(LRUCache(
            checkpoint_path_or_dir=args.checkpoint_dir,
            layer_idx=layer_idx,
            batch_size=bs,
            max_position_embeddings=max_pos,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hash_module_dims=args.dims,
            lru_budget=args.lru_budget,
            top_budget=args.top_budget,
            device='cuda'))
    return CacheList(caches)


def measure_ours(model, args, config, raw_ids, bs, ctx_len):
    max_pos = ctx_len + WARMUP_STEPS + MEASURE_STEPS + 16
    input_ids = raw_ids[:, :ctx_len + 1].expand(bs, -1).cuda()
    cache = make_ours_cache(args, config, bs, max_pos)

    # Chunked prefill
    for start in range(0, ctx_len, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, ctx_len)
        model(input_ids=input_ids[:, start:end], past_key_values=cache, logits_to_keep=1)

    # First decode token (true next token)
    out = model(input_ids=input_ids[:, ctx_len:ctx_len + 1], past_key_values=cache, logits_to_keep=1)
    next_token = out.logits[:, -1:].argmax(dim=-1)
    del input_ids

    # Warmup
    for _ in range(WARMUP_STEPS):
        out = model(input_ids=next_token, past_key_values=cache, logits_to_keep=1)
        next_token = out.logits[:, -1:].argmax(dim=-1)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        out = model(input_ids=next_token, past_key_values=cache, logits_to_keep=1)
        next_token = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    del cache
    return (t1 - t0) / MEASURE_STEPS / bs * 1000.0


# ── Baseline (FA2 + StaticCache) ────────────────────────────────────────────

def measure_baseline(model, config, raw_ids, bs, ctx_len):
    max_pos = ctx_len + WARMUP_STEPS + MEASURE_STEPS + 16
    input_ids = raw_ids[:, :ctx_len + 1].expand(bs, -1).cuda()

    cache_config = copy.deepcopy(config)
    cache_config.num_hidden_layers = NUM_LAYERS
    cache_config.max_window_layers = NUM_LAYERS
    cache_config.layer_types = ['full_attention'] * NUM_LAYERS
    cache = StaticCache(config=cache_config, batch_size=bs, max_cache_len=max_pos,
                        device='cuda', dtype=torch.bfloat16)

    # Chunked prefill
    for start in range(0, ctx_len, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, ctx_len)
        cache_position = torch.arange(start, end, device='cuda')
        model(input_ids=input_ids[:, start:end], cache_position=cache_position,
              past_key_values=cache, logits_to_keep=1)

    # First decode token
    pos = ctx_len
    cache_position = torch.tensor([pos], device='cuda')
    out = model(input_ids=input_ids[:, pos:pos + 1], cache_position=cache_position,
                past_key_values=cache, logits_to_keep=1)
    next_token = out.logits[:, -1:].argmax(dim=-1)
    pos += 1
    del input_ids

    # Warmup
    for _ in range(WARMUP_STEPS):
        cache_position = torch.tensor([pos], device='cuda')
        out = model(input_ids=next_token, cache_position=cache_position,
                    past_key_values=cache, logits_to_keep=1)
        next_token = out.logits[:, -1:].argmax(dim=-1)
        pos += 1

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        cache_position = torch.tensor([pos], device='cuda')
        out = model(input_ids=next_token, cache_position=cache_position,
                    past_key_values=cache, logits_to_keep=1)
        next_token = out.logits[:, -1:].argmax(dim=-1)
        pos += 1
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    del cache
    return (t1 - t0) / MEASURE_STEPS / bs * 1000.0


# ── Dynamic Cache (FA2 + DynamicCache) ─────────────────────────────────────

def measure_dynamic(model, raw_ids, bs, ctx_len):
    input_ids = raw_ids[:, :ctx_len + 1].expand(bs, -1).cuda()
    cache = DynamicCache()

    # Chunked prefill
    for start in range(0, ctx_len, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, ctx_len)
        model(input_ids=input_ids[:, start:end], past_key_values=cache, logits_to_keep=1)

    # First decode token
    out = model(input_ids=input_ids[:, ctx_len:ctx_len + 1], past_key_values=cache, logits_to_keep=1)
    next_token = out.logits[:, -1:].argmax(dim=-1)
    del input_ids

    # Warmup
    for _ in range(WARMUP_STEPS):
        out = model(input_ids=next_token, past_key_values=cache, logits_to_keep=1)
        next_token = out.logits[:, -1:].argmax(dim=-1)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        out = model(input_ids=next_token, past_key_values=cache, logits_to_keep=1)
        next_token = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    del cache
    return (t1 - t0) / MEASURE_STEPS / bs * 1000.0


# ── Sweep + output ──────────────────────────────────────────────────────────

def sweep(name, measure_fn, results):
    for ci, ctx_len in enumerate(CONTEXT_LENGTHS):
        for bi, bs in enumerate(BATCH_SIZES):
            print(f"[{name}] bs={bs}, ctx={CONTEXT_LABELS[ci]}", end="  ", flush=True)
            try:
                lat = measure_fn(bs, ctx_len)
                results[bi, ci] = lat
                print(f"{lat:.2f} ms/tok", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"FAILED: {e}", flush=True)
            cleanup()


def print_table(results, label):
    print(f"\n=== {label} (ms/token) ===")
    header = f"{'BS':>6}" + "".join(f"{l:>10}" for l in CONTEXT_LABELS)
    print(header)
    for bi, bs in enumerate(BATCH_SIZES):
        row = f"{bs:>6}"
        for ci in range(len(CONTEXT_LENGTHS)):
            v = results[bi, ci]
            row += f"{v:>10.2f}" if not np.isnan(v) else f"{'OOM':>10}"
        print(row)


def save_csv(results_ours, results_baseline, results_dynamic):
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heatmap.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["method", "batch_size", "context_length", "ms_per_token"])
        for label, results in [("ours", results_ours), ("baseline", results_baseline),
                                ("dynamic", results_dynamic)]:
            if np.all(np.isnan(results)):
                continue
            for bi, bs in enumerate(BATCH_SIZES):
                for ci, ctx in enumerate(CONTEXT_LENGTHS):
                    v = results[bi, ci]
                    writer.writerow([label, bs, ctx, f"{v:.4f}" if not np.isnan(v) else "OOM"])
    print(f"CSV saved to {csv_path}")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--method", type=str, default='eval')
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--lru-budget", type=int, default=4096)
    parser.add_argument("--top-budget", type=int, default=2048)
    parser.add_argument("--dims", type=int, nargs='+', default=[128, 128, 128])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    # Shrink to NUM_LAYERS
    config.num_hidden_layers = NUM_LAYERS
    config.max_window_layers = NUM_LAYERS
    config.layer_types = ['full_attention'] * NUM_LAYERS

    # Load tokens
    pg19_path = os.environ['SPOTLIGHT_PG19_PATH']
    max_total = max(CONTEXT_LENGTHS) + WARMUP_STEPS + MEASURE_STEPS + 16
    raw_ids = load_pg19_tokens(pg19_path, tokenizer, max_total)

    results_ours = np.full((len(BATCH_SIZES), len(CONTEXT_LENGTHS)), np.nan)
    results_baseline = np.full((len(BATCH_SIZES), len(CONTEXT_LENGTHS)), np.nan)
    results_dynamic = np.full((len(BATCH_SIZES), len(CONTEXT_LENGTHS)), np.nan)

    # ── Test ours ──
    print("Creating monkey-patched model...", flush=True)
    monkey_patch = get_monkey_patch(args.method)
    ours_model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).cuda().eval()
    ours_model = monkey_patch(ours_model)
    print("Model ready.", flush=True)

    sweep("Ours", lambda bs, ctx: measure_ours(ours_model, args, config, raw_ids, bs, ctx), results_ours)
    del ours_model
    cleanup()

    # # ── Test baseline (FA2 + StaticCache) ──
    # print("Creating baseline model (FA2)...", flush=True)
    # baseline_model = AutoModelForCausalLM.from_config(
    #     config, dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval()
    # print("Model ready.", flush=True)

    # sweep("Base", lambda bs, ctx: measure_baseline(baseline_model, config, raw_ids, bs, ctx), results_baseline)
    # del baseline_model
    # cleanup()

    # # ── Test dynamic (FA2 + DynamicCache) ──
    # print("Creating dynamic cache model (FA2)...", flush=True)
    # dynamic_model = AutoModelForCausalLM.from_config(
    #     config, dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval()
    # print("Model ready.", flush=True)

    # sweep("Dynamic", lambda bs, ctx: measure_dynamic(dynamic_model, raw_ids, bs, ctx), results_dynamic)
    # del dynamic_model
    # cleanup()

    # ── Results ──
    print_table(results_ours, "Ours (LRUCache)")
    print_table(results_baseline, "Baseline (FA2 + StaticCache)")
    print_table(results_dynamic, "Dynamic (FA2 + DynamicCache)")
    save_csv(results_ours, results_baseline, results_dynamic)


if __name__ == '__main__':
    main()