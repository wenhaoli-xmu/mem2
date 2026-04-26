"""AIME evaluation — kernel (with spotlight monkey-patch and LRU caches).

Loads AIME JSONL data, runs model.generate() with custom hash KV caches,
strips <think>...</think> tags, computes exact-match accuracy,
and reports think-tag length statistics.

Usage:
    python evaluate/aime/test_kernel.py \
        --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-4B-Thinking-2507 \
        --checkpoint-path-or-dir train_results/qwen3-4b-thinking-2507/stage2.safetensors \
        --data-path ${GDRIVE_LOCAL}/data/aime-2024/aime-2024.jsonl \
        --apply-chat-template \
        --save-dir evaluate/aime/results
"""

import os
import re
import json
import argparse
import torch
import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from spotlight import get_monkey_patch, create_lru_caches


# ── Think tag utilities ──────────────────────────────────────────────────
THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)


def extract_think_lengths(raw_text):
    """Return list of character lengths of all <think>...</think> blocks."""
    return [len(m.group(1)) for m in THINK_PATTERN.finditer(raw_text)]


def strip_think_tags(text):
    """Remove all <think>...</think> blocks from text."""
    return THINK_PATTERN.sub('', text).strip()


# ── JSONL I/O ────────────────────────────────────────────────────────────
def read_jsonl(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: skipped invalid JSON line in {path}")
    return lines


# ── Cache helpers ────────────────────────────────────────────────────────
def reset_caches(caches):
    for cache in caches:
        cache.reset()


# ── Generation with caches ───────────────────────────────────────────────
@torch.inference_mode()
def generate_with_caches(messages, model, tokenizer, caches,
                         max_new_tokens=32768, apply_chat_template=False):
    """Run model.generate() using custom hash KV caches."""
    if apply_chat_template:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(m["content"] for m in messages)

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')

    pad_token_id = tokenizer.eos_token_id
    if hasattr(tokenizer, 'generation_kwargs') and tokenizer.generation_kwargs:
        pad_token_id = tokenizer.generation_kwargs.get('pad_token_id', pad_token_id)

    try:
        outputs = model.generate(
            input_ids,
            past_key_values=caches,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=False)
        reset_caches(caches)

        generated_ids = outputs[0, input_ids.shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)
    except Exception as e:
        print(f"Error during generation: {e}")
        reset_caches(caches)
        return ''


# ── Metric ───────────────────────────────────────────────────────────────
def exact_match(pred, label):
    """Check if the label string appears in the prediction."""
    return label.strip().lower() in pred.strip().lower()


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AIME evaluation with LRUCache")
    parser.add_argument("--model-name-or-path", "-m", type=str, required=True)
    parser.add_argument("--checkpoint-path-or-dir", type=str, default=None,
                        help="Directory containing hash module weights")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to AIME JSONL file")

    # Cache parameters
    parser.add_argument("--max-position-embeddings", type=int, default=262144)
    parser.add_argument("--lru-budget", type=int, default=4096)
    parser.add_argument("--top-budget", type=int, default=2048)
    parser.add_argument("--dims", type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument("--skip-layers", type=int, nargs='+', default=[])
    parser.add_argument("--double-layers", type=int, nargs='+', default=[])

    # Generation args
    parser.add_argument("--save-dir", "-s", type=str, default="evaluate/aime/results")
    parser.add_argument("--apply-chat-template", action="store_true",
                        help="Apply chat template to prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--postfix", type=str, default='')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    model_shortname = os.path.basename(os.path.normpath(args.model_name_or_path))

    # Load data
    samples = read_jsonl(args.data_path)
    print(f"Loaded {len(samples)} AIME problems", flush=True)

    # Load model with monkey-patch
    print("Loading model with monkey patch...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='cuda',
        torch_dtype=torch.bfloat16)
    model = get_monkey_patch(method='eval')(model)
    model.eval()
    print("Model loaded", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create caches
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
        device='cuda')
    print(f"LRUCache created (skipping layers: {args.skip_layers})", flush=True)
    print(f"  - lru_budget: {args.lru_budget}", flush=True)
    print(f"  - top_budget: {args.top_budget}", flush=True)

    # Resume support
    pred_file = os.path.join(args.save_dir, f"{model_shortname}_kernel{args.postfix}.jsonl")
    existing = {}
    if os.path.exists(pred_file):
        for item in read_jsonl(pred_file):
            existing[item['index']] = item

    remaining = [(i, s) for i, s in enumerate(samples) if i not in existing]
    print(f"  {len(samples)} total, {len(existing)} done, {len(remaining)} remaining", flush=True)

    # Generate
    all_think_lengths = []

    if remaining:
        fout = open(pred_file, 'a', encoding='utf-8')
        for idx, sample in tqdm.tqdm(remaining, desc="Generating"):
            messages = sample['prompt']
            raw_response = generate_with_caches(
                messages, model, tokenizer, caches,
                max_new_tokens=args.max_new_tokens,
                apply_chat_template=args.apply_chat_template)

            think_lens = extract_think_lengths(raw_response)
            cleaned = strip_think_tags(raw_response)

            result = {
                'index': idx,
                'prompt': messages,
                'label': sample['label'],
                'raw_pred': raw_response,
                'pred': cleaned,
                'think_lengths': think_lens,
            }
            existing[idx] = result
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            fout.flush()
            torch.cuda.empty_cache()
        fout.close()

    # Collect all think lengths and compute metrics
    correct = 0
    total = 0
    for i in range(len(samples)):
        item = existing.get(i)
        if item is None:
            continue
        total += 1
        all_think_lengths.extend(item.get('think_lengths', []))
        if exact_match(item['pred'], item['label']):
            correct += 1

    accuracy = correct / total * 100 if total > 0 else 0

    # Print results
    print(f"\n{'='*60}")
    print(f"AIME Results: {model_shortname} (kernel{args.postfix})")
    print(f"{'='*60}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")

    # Think tag statistics
    print(f"\n{'='*60}")
    print(f"Think Tag Statistics")
    print(f"{'='*60}")
    if all_think_lengths:
        avg_len = sum(all_think_lengths) / len(all_think_lengths)
        max_len = max(all_think_lengths)
        min_len = min(all_think_lengths)
        print(f"  Total think blocks: {len(all_think_lengths)}")
        print(f"  Average length: {avg_len:.1f} chars")
        print(f"  Max length:     {max_len} chars")
        print(f"  Min length:     {min_len} chars")
    else:
        print("  No <think> tags found in outputs.")

    # Save summary
    summary = {
        'model': model_shortname,
        'method': 'kernel',
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'think_stats': {
            'count': len(all_think_lengths),
            'avg': sum(all_think_lengths) / len(all_think_lengths) if all_think_lengths else 0,
            'max': max(all_think_lengths) if all_think_lengths else 0,
            'min': min(all_think_lengths) if all_think_lengths else 0,
        }
    }
    summary_file = os.path.join(args.save_dir, f"{model_shortname}_kernel{args.postfix}_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    del caches
    print("Done")


if __name__ == '__main__':
    main()
