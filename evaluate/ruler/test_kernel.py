import os
import sys
import json
import argparse
import torch
import tqdm
import yaml
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from spotlight import get_monkey_patch, create_lru_caches


# ── RULER metric functions ────────────────────────────────────────────────
def string_match_part(preds, refs):
    score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref])
                 for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

def string_match_all(preds, refs):
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
                 for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

TASK_METRICS = {
    'niah': string_match_all,
    'variable_tracking': string_match_all,
    'common_words_extraction': string_match_all,
    'freq_words_extraction': string_match_all,
    'qa': string_match_part,
}

# ── Task config ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RULER_SCRIPTS = os.path.join(SCRIPT_DIR, 'RULER', 'scripts')

def load_task_configs():
    yaml_path = os.path.join(RULER_SCRIPTS, 'synthetic.yaml')
    with open(yaml_path, 'r') as f:
        tasks_customized = yaml.safe_load(f)

    sys.path.insert(0, os.path.join(RULER_SCRIPTS, 'data'))
    from synthetic.constants import TASKS as tasks_base
    sys.path.pop(0)

    for name, config in tasks_customized.items():
        config.update(tasks_base[config['task']])
    return tasks_customized


TASKS_ORDER = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]


# ── Cache helpers ─────────────────────────────────────────────────────────
def reset_caches(caches):
    for cache in caches:
        cache.reset()


# ── JSONL I/O ─────────────────────────────────────────────────────────────
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


# ── Generation with caches ────────────────────────────────────────────────
@torch.inference_mode()
def generate_with_caches(prompt, model, tokenizer, caches, max_new_tokens=128, apply_chat_template=False, debug=False):
    """Run model.generate() using custom hash KV caches."""
    if apply_chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')

    pad_token_id = tokenizer.eos_token_id
    if hasattr(tokenizer, 'generation_kwargs') and tokenizer.generation_kwargs:
        pad_token_id = tokenizer.generation_kwargs.get('pad_token_id', pad_token_id)

    outputs = model.generate(
        input_ids,
        past_key_values=caches,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        do_sample=False,
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        return_dict_in_generate=debug,
        output_scores=debug)
    reset_caches(caches)

    if debug:
        sequences = outputs.sequences
        scores = outputs.scores
    else:
        sequences = outputs
        scores = None

    # ── Debug: check NaN and first-token distribution ────────────
    if debug and scores:
        seq_len = input_ids.shape[1]
        nan_steps = []
        for step_idx, score in enumerate(scores):
            if torch.isnan(score).any():
                nan_count = torch.isnan(score).sum().item()
                nan_steps.append((step_idx, nan_count))

        with open('debug.log', 'a', encoding='utf-8') as f:
            if nan_steps:
                f.write(f"[NaN] prompt_len={seq_len}, nan_steps={nan_steps}\n")
            else:
                f.write(f"[OK] prompt_len={seq_len}, no NaN in {len(scores)} steps\n")

            # First token distribution
            first_logits = scores[0][0]  # (vocab_size,)
            probs = torch.softmax(first_logits.float(), dim=-1)

            eos_id = tokenizer.eos_token_id
            eos_prob = probs[eos_id].item()

            top_k = 20
            top_probs, top_ids = torch.topk(probs, k=top_k)
            generated_token_id = sequences[0, seq_len].item()

            f.write(f"  first_token_generated: id={generated_token_id}, "
                    f"repr={repr(tokenizer.decode([generated_token_id]))}\n")
            f.write(f"  eos_token: id={eos_id}, prob={eos_prob:.6f}, "
                    f"is_top1={'YES' if top_ids[0].item() == eos_id else 'NO'}\n")
            f.write(f"  top-{top_k} distribution:\n")
            for rank in range(top_k):
                tid = top_ids[rank].item()
                p = top_probs[rank].item()
                tok_str = repr(tokenizer.decode([tid]))
                marker = " <-- EOS" if tid == eos_id else ""
                f.write(f"    rank={rank+1:>2}  id={tid:>6}  prob={p:.6f}  token={tok_str}{marker}\n")

            entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
            f.write(f"  entropy={entropy:.4f}\n\n")

    generated_ids = sequences[0, input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Per-task evaluation ───────────────────────────────────────────────────
@torch.inference_mode()
def evaluate_task(task_name, task_config, data_dir, pred_dir, model, tokenizer, caches, args):
    """Evaluate a single task: generate predictions and compute metrics."""
    task_file = os.path.join(data_dir, task_name, 'validation.jsonl')
    if not os.path.exists(task_file):
        print(f"  ⚠ Data file not found: {task_file}")
        return None

    samples = read_jsonl(task_file)
    if len(samples) == 0:
        return None

    tokens_to_generate = task_config['tokens_to_generate']
    if args.max_new_tokens is not None:
        tokens_to_generate = args.max_new_tokens

    # Resume support
    pred_file = os.path.join(pred_dir, f'{task_name}.jsonl')
    existing = {}
    if os.path.exists(pred_file):
        for item in read_jsonl(pred_file):
            existing[item['index']] = item

    remaining = [s for s in samples if s['index'] not in existing]
    print(f"  {task_name}: {len(samples)} total, {len(existing)} done, {len(remaining)} remaining")

    if remaining:
        fout = open(pred_file, 'a', encoding='utf-8')
        for sample in tqdm.tqdm(remaining, desc=f"  {task_name}"):
            prompt = sample['input']
            response = generate_with_caches(
                prompt, model, tokenizer, caches,
                max_new_tokens=tokens_to_generate,
                apply_chat_template=args.apply_chat_template,
                debug=args.debug)

            result = {
                'index': sample['index'],
                'input': prompt,
                'outputs': sample['outputs'],
                'pred': response.strip(),
                'others': sample.get('others', {}),
            }
            existing[sample['index']] = result
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            fout.flush()
            torch.cuda.empty_cache()
        fout.close()

    # Compute metrics
    all_preds = []
    all_refs = []
    for sample in samples:
        item = existing.get(sample['index'])
        if item is None:
            continue
        all_preds.append(item['pred'])
        all_refs.append(item['outputs'])

    if len(all_preds) == 0:
        return None

    base_task = task_config['task']
    metric_fn = TASK_METRICS[base_task]
    score = metric_fn(all_preds, all_refs)
    nulls = sum(1 for p in all_preds if len(p) == 0)
    print(f"  {task_name}: score={score:.2f}, nulls={nulls}/{len(all_preds)}")
    return score


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RULER evaluation with LRUCache")
    parser.add_argument("--model-name-or-path", "-m", type=str, required=True)
    parser.add_argument("--checkpoint-path-or-dir", type=str, default=None,
                        help="Directory containing hash module weights")

    # Cache parameters
    parser.add_argument("--max-position-embeddings", type=int, default=40960)
    parser.add_argument("--lru-budget", type=int, default=2048)
    parser.add_argument("--top-budget", type=int, default=1024)
    parser.add_argument("--dims", type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument("--skip-layers", type=int, nargs='+', default=[])
    parser.add_argument("--double-layers", type=int, nargs='+', default=[])

    # RULER args
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--save-dir", "-s", type=str, default="evaluate/ruler/results")
    parser.add_argument("--seq-lengths", type=int, nargs='+',
                        default=[4096, 8192, 16384, 32768, 65536, 131072])
    parser.add_argument("--tasks", type=str, nargs='+', default=None)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--postfix", type=str, default='')
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging (NaN check, first-token distribution) to debug.log")

    args = parser.parse_args()
    if args.debug:
        open('debug.log', 'w').close()  # clear previous debug log
    os.makedirs(args.save_dir, exist_ok=True)
    model_shortname = os.path.basename(os.path.normpath(args.model_name_or_path))
    data_root = args.data_root or os.path.join(SCRIPT_DIR, 'benchmark_root')

    # Load task configs
    task_configs = load_task_configs()
    tasks = args.tasks or TASKS_ORDER

    # Load model with monkey-patch
    print("Loading model with monkey patch...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='cuda',
        torch_dtype=torch.bfloat16)
    model = get_monkey_patch(method='eval')(model)
    model.eval()
    print("Model loaded ✅", flush=True)

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
    print(f"LRUCache created ✅ (skipping layers: {args.skip_layers})", flush=True)
    print(f"  - lru_budget: {args.lru_budget}", flush=True)
    print(f"  - top_budget: {args.top_budget}", flush=True)

    # Evaluate per sequence length
    summary = {}
    for seq_len in args.seq_lengths:
        data_dir = os.path.join(data_root, model_shortname, 'synthetic', str(seq_len), 'data')
        pred_dir = os.path.join(args.save_dir, model_shortname + '_kernel' + args.postfix, str(seq_len))
        os.makedirs(pred_dir, exist_ok=True)

        if not os.path.exists(data_dir):
            print(f"\n⚠ Data not found for seq_len={seq_len}: {data_dir}")
            print(f"  Run prepare_data.sh first.")
            continue

        print(f"\n{'='*60}")
        print(f"Sequence length: {seq_len}")
        print(f"{'='*60}")

        summary[seq_len] = {}
        for task_name in tasks:
            if task_name not in task_configs:
                print(f"  ⚠ Unknown task: {task_name}")
                continue
            score = evaluate_task(
                task_name, task_configs[task_name],
                data_dir, pred_dir, model, tokenizer, caches, args)
            if score is not None:
                summary[seq_len][task_name] = score

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Summary: {model_shortname} (kernel)")
    print(f"{'='*60}")
    header = f"{'Task':<25}" + "".join(f"{sl:>8}" for sl in args.seq_lengths)
    print(header)
    print("-" * len(header))
    for task_name in tasks:
        row = f"{task_name:<25}"
        for sl in args.seq_lengths:
            score = summary.get(sl, {}).get(task_name)
            row += f"{score:>8.2f}" if score is not None else f"{'N/A':>8}"
        print(row)

    row = f"{'AVERAGE':<25}"
    for sl in args.seq_lengths:
        scores = list(summary.get(sl, {}).values())
        avg = sum(scores) / len(scores) if scores else 0
        row += f"{avg:>8.2f}"
    print("-" * len(header))
    print(row)

    # Save summary JSON
    summary_file = os.path.join(args.save_dir, f"{model_shortname}_kernel{args.postfix}_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    del caches
    print("\nDone ✅")


if __name__ == '__main__':
    main()
