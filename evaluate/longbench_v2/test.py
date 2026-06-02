import os
import json
import re
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mem2 import get_monkey_patch, load_checkpoint, create_lru_caches

# ── prompt templates (loaded relative to this script) ──────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
template_0shot     = open(os.path.join(SCRIPT_DIR, 'prompts/0shot.txt'), encoding='utf-8').read()
template_0shot_cot = open(os.path.join(SCRIPT_DIR, 'prompts/0shot_cot.txt'), encoding='utf-8').read()
template_0shot_cot_ans = open(os.path.join(SCRIPT_DIR, 'prompts/0shot_cot_ans.txt'), encoding='utf-8').read()


# ── prompt construction ────────────────────────────────────────────────────
def build_prompt(item, cot=False):
    """Fill the prompt template with the item's fields."""
    template = template_0shot_cot if cot else template_0shot
    return (template
            .replace('$DOC$', item['context'].strip())
            .replace('$Q$', item['question'].strip())
            .replace('$C_A$', item['choice_A'].strip())
            .replace('$C_B$', item['choice_B'].strip())
            .replace('$C_C$', item['choice_C'].strip())
            .replace('$C_D$', item['choice_D'].strip()))

def build_cot_answer_prompt(item, cot_response):
    """Build a second-pass prompt to extract the answer from CoT reasoning."""
    return (template_0shot_cot_ans
            .replace('$Q$', item['question'].strip())
            .replace('$C_A$', item['choice_A'].strip())
            .replace('$C_B$', item['choice_B'].strip())
            .replace('$C_C$', item['choice_C'].strip())
            .replace('$C_D$', item['choice_D'].strip())
            .replace('$COT$', cot_response))

def extract_answer(response):
    """Extract single letter answer (A-D) from model response."""
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    match = re.search(r'The correct answer is ([A-D])', response)
    if match:
        return match.group(1)
    return None

@torch.inference_mode()
def generate_sync(prompt: str, model, tokenizer, caches, args):
    device = 'cuda:0'
    
    # Constrain pre-template prompt to half of max_len
    prompt_ids = tokenizer.encode(prompt)
    half_max = args.max_len // 2
    if len(prompt_ids) > half_max:
        half = half_max // 2
        prompt_ids = prompt_ids[:half] + prompt_ids[-half:]
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        
    if args.enable_chat_template:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors='pt'
        ).to(device)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    pad_token_id = tokenizer.eos_token_id
    if hasattr(tokenizer, 'generation_kwargs') and tokenizer.generation_kwargs:
        pad_token_id = tokenizer.generation_kwargs.get('pad_token_id', pad_token_id)
        
    actual_max_new_tokens = min(args.max_new_tokens, args.max_len - input_ids.shape[1])
    kwargs = {
        "max_new_tokens": actual_max_new_tokens,
        "do_sample": False,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "return_dict_in_generate": False
    }

    if args.enable_efficient and caches is not None:
        kwargs["past_key_values"] = caches

    outputs = model.generate(input_ids, **kwargs)
    
    if args.enable_efficient and caches is not None:
        for cache in caches:
            cache.reset()

    generated_ids = outputs[0, input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def get_pred(data, args, fout, model, tokenizer, caches):
    for item in tqdm(data):
        # Phase 1: generate
        prompt = build_prompt(item, cot=args.cot)

        output = generate_sync(prompt, model, tokenizer, caches, args)

        if output == '':
            continue

        # Phase 2: CoT extraction
        if args.cot:
            cot_response = output.strip()
            item['response_cot'] = cot_response
            prompt2 = build_cot_answer_prompt(item, cot_response)
            output = generate_sync(prompt2, model, tokenizer, caches, args)
            if output == '':
                continue

        response = output.strip()

        # Strip <think> content if present
        if '</think>' in response:
            idx = response.index('</think>')
            response = response[idx:].replace('</think>', '').lstrip()

        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = item['context'][:1000]

        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    parser = argparse.ArgumentParser(description="LongBench V2 evaluation (Local)")
    parser.add_argument("--model-name-or-path", "-m", type=str, required=True, help="Used for output naming")
    parser.add_argument("--save-dir", "-s", type=str, default="evaluate/longbench_v2/results")
    parser.add_argument("--cot", action='store_true', help="Use chain-of-thought prompting")
    parser.add_argument("--enable-efficient", action='store_true', help="Use patched HuggingFace model")
    
    # Cache parameters (HashGen)
    parser.add_argument("--checkpoint-path-or-dir", type=str, default=None, help="Directory containing hash module weights")
    parser.add_argument("--lru-budget", type=int, default=2048)
    parser.add_argument("--top-budget", type=int, default=1024)
    parser.add_argument("--dims", type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument("--skip-layers", type=int, nargs='+', default=[])
    parser.add_argument("--double-layers", type=int, nargs='+', default=[])
    # Generation bounds
    parser.add_argument("--max-len", type=int, default=32768, help="Max context length")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max new tokens to generate")
    parser.add_argument("--enable-chat-template", action='store_true')
    parser.add_argument("--postfix", type=str, default="", help="Custom postfix for the output file name")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading LongBench v2 dataset...", flush=True)
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    data_all = [{
        "_id": item["_id"], "domain": item["domain"],
        "sub_domain": item["sub_domain"], "difficulty": item["difficulty"],
        "length": item["length"], "question": item["question"],
        "choice_A": item["choice_A"], "choice_B": item["choice_B"],
        "choice_C": item["choice_C"], "choice_D": item["choice_D"],
        "answer": item["answer"], "context": item["context"]
    } for item in dataset]
    print(f"Dataset loaded ✅  ({len(data_all)} items)", flush=True)

    model_shortname = os.path.basename(os.path.normpath(args.model_name_or_path))
    cot_suffix = "_cot" if args.cot else ""
    model_suffix = "_kernel" if args.enable_efficient else "_vanilla"
    postfix_suffix = f"{args.postfix}" if args.postfix else ""
    out_file = os.path.join(args.save_dir, f"{model_shortname}{model_suffix}{cot_suffix}{postfix_suffix}.jsonl")

    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}

    data = [item for item in data_all if item["_id"] not in has_data]
    print(f"Skipping {len(has_data)} existing items. Remaining: {len(data)}", flush=True)

    if len(data) == 0:
        print("All records already processed. Exiting.")
        return

    # ---- Model Initialization ----
    device = "cuda:0"
    print(f"GPUs SHARED: Model loading onto {device}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device,
        torch_dtype=torch.bfloat16)

    if args.enable_efficient:
        print("Evaluate our method...")
        model = get_monkey_patch(method='eval')(model)
        
        caches = create_lru_caches(
            model,
            checkpoint_path_or_dir=args.checkpoint_path_or_dir,
            batch_size=1,
            max_length=args.max_len,
            lru_budget=args.lru_budget,
            top_budget=args.top_budget,
            skip_layers=args.skip_layers,
            double_layers=args.double_layers,
            hash_dims=args.dims,
            device=device)
    else:
        print("Evaluate vanilla model...")
        caches = None

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # ---- Generate Predictions ----
    fout = open(out_file, 'a', encoding='utf-8')
    print(f"Starting prediction looping...")
    get_pred(data, args, fout, model, tokenizer, caches)
    
    fout.close()
    print("\nDone ✅")

if __name__ == '__main__':
    main()
