"""AIME evaluation — vanilla baseline (no monkey-patch, no custom caches).

Loads AIME JSONL data, runs model.generate() with chat template,
strips <think>...</think> tags, computes exact-match accuracy,
and reports think-tag length statistics.

Usage:
    python evaluate/aime/test_vanilla.py \
        --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-4B-Thinking-2507 \
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


# ── Generation ───────────────────────────────────────────────────────────
@torch.inference_mode()
def generate_response(messages, model, tokenizer, max_new_tokens=32768,
                      apply_chat_template=False):
    """Run model.generate() and return raw response text (before think-tag stripping)."""
    if apply_chat_template:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback: concatenate message contents
        prompt = "\n".join(m["content"] for m in messages)

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')

    pad_token_id = tokenizer.eos_token_id
    if hasattr(tokenizer, 'generation_kwargs') and tokenizer.generation_kwargs:
        pad_token_id = tokenizer.generation_kwargs.get('pad_token_id', pad_token_id)

    try:
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=False)

        generated_ids = outputs[0, input_ids.shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)
    except Exception as e:
        print(f"Error during generation: {e}")
        return ''


# ── Metric ───────────────────────────────────────────────────────────────
def exact_match(pred, label):
    """Check if the label string appears in the prediction."""
    return label.strip().lower() in pred.strip().lower()


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AIME evaluation (vanilla baseline)")
    parser.add_argument("--model-name-or-path", "-m", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to AIME JSONL file")
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

    # Load model
    print(f"Loading model {args.model_name_or_path}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='cuda',
        torch_dtype=torch.bfloat16)
    model.eval()
    print("Model loaded", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Resume support
    pred_file = os.path.join(args.save_dir, f"{model_shortname}_vanilla{args.postfix}.jsonl")
    existing = {}
    if os.path.exists(pred_file):
        for item in read_jsonl(pred_file):
            existing[item['index']] = item

    remaining = [(i, s) for i, s in enumerate(samples) if i not in existing]
    print(f"  {len(samples)} total, {len(existing)} done, {len(remaining)} remaining", flush=True)

    # Generate
    all_think_lengths = []  # lengths of each <think> block across all samples

    if remaining:
        fout = open(pred_file, 'a', encoding='utf-8')
        for idx, sample in tqdm.tqdm(remaining, desc="Generating"):
            messages = sample['prompt']
            raw_response = generate_response(
                messages, model, tokenizer,
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
    print(f"AIME Results: {model_shortname} (vanilla{args.postfix})")
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
        'method': 'vanilla',
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
    summary_file = os.path.join(args.save_dir, f"{model_shortname}_vanilla{args.postfix}_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    print("Done")


if __name__ == '__main__':
    main()
