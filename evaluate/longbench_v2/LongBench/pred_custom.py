import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
import torch
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
from mem2.monkey_patches import get_monkey_patch
from mem2.monkey_patches.hash_utils import LRUCache, DynamicCache

# LongBench length configuration mapped from the original logic
model_maxlen = {
    "qwen3-8b": 32768, 
    "qwen3-14b": 32768, 
    "qwen3-30b": 32768, 
    "llama3.1-8b": 131072,
    "default": 32768 # Default fallback
}

template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()


def create_lru_caches(model, args, skip_layers):
    """
    Create LRUCache for each layer.
    For skipped layers, checkpoint_dir is set to None (no hash weights loaded).
    """
    num_layers = model.config.num_hidden_layers
    caches = []
    
    for layer_idx in range(num_layers):
        if layer_idx in skip_layers:
            cache = DynamicCache(
                batch_size=1,
                num_key_value_heads=model.config.num_key_value_heads,
                head_dim=model.config.head_dim,
                dtype=torch.bfloat16,
                device='cuda')
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
                dtype=torch.bfloat16, 
                device='cuda')
        caches.append(cache)
    
    return caches


def reset_caches(caches):
    """Reset all caches to prepare for next instance."""
    for cache in caches:
        cache.reset()
        

@torch.inference_mode()
def query_llm(prompt, model, tokenizer, caches, max_len=32768, temperature=0.5, max_new_tokens=128):
    """
    Runs HF model.generate using the custom hash KV caches.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
    
    # Truncate if the encoded prompt exceeds max context length allowed (subtracting max_new_tokens)
    if input_ids.shape[1] > max_len - max_new_tokens:
        available_len = max_len - max_new_tokens
        half_len = available_len // 2
        input_ids = torch.cat([input_ids[:, :half_len], input_ids[:, -half_len:]], dim=1)
        
    reset_caches(caches)
    
    try:
        if kwargs := getattr(tokenizer, "generation_kwargs", None):
            pad_token_id = kwargs.get("pad_token_id", tokenizer.eos_token_id)
        else:
            pad_token_id = tokenizer.eos_token_id

        # Since batch_size is 1, temperature > 0 implies do_sample=True, but temperature=0 means greedy
        do_sample = temperature > 0.0
        
        # When temperature is specifically set to 0.1 in the original file, it implies some sampling
        # We will use do_sample=do_sample but explicitly bound the temperature to avoid 0 if set to greedy
        actual_temp = temperature if temperature > 0.0 else 1.0

        outputs = model.generate(
            input_ids,
            past_key_values=caches,
            max_new_tokens=max_new_tokens,
            temperature=actual_temp,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=False,
        )
        
        # We only want the generated token IDs
        generated_ids = outputs[0, input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response
    
    except Exception as e:
        print(f"Error Occurs during generation: {str(e)}")
        # Reset cache on error defensively
        reset_caches(caches)
        return ''


def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None


def get_pred(data, args, fout, model, tokenizer, caches, max_len):
    """
    Runs evaluation linearly using HF local model instead of calling API.
    """
    for item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
            
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        
        if args.cot:
            output = query_llm(prompt, model, tokenizer, caches, max_len=max_len, temperature=0.1, max_new_tokens=1024)
        else:
            output = query_llm(prompt, model, tokenizer, caches, max_len=max_len, temperature=0.1, max_new_tokens=128)
            
        if output == '':
            continue
            
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            output = query_llm(prompt, model, tokenizer, caches, max_len=max_len, temperature=0.1, max_new_tokens=128)
            if output == '':
                continue
                
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()
        
        torch.cuda.empty_cache()


def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print("Arguments:", args)
    
    model_shortname = args.model_name_or_path.split("/")[-1]
    
    if args.rag > 0:
        out_file_suffix = f"_rag_{str(args.rag)}.jsonl"
    elif args.no_context:
        out_file_suffix = "_no_context.jsonl"
    elif args.cot:
        out_file_suffix = "_cot.jsonl"
    else:
        out_file_suffix = ".jsonl"
        
    out_file = os.path.join(args.save_dir, f"{model_shortname}_{args.method}{out_file_suffix}")

    print("Loading data...")
    dataset = load_dataset('THUDM/LongBench-v2', split='train') 
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # Handle previously processed data cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
            
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    print(f"Skipping {len(has_data)} existing items. Total remaining: {len(data)}")

    if len(data) == 0:
        print("All records processed. Exiting.")
        fout.close()
        return

    # Load HF Model & Tokenizer
    print("Loading local Hugging Face model and initializing test caches...")
    monkey_patch = get_monkey_patch(args.method)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='cuda',
        torch_dtype=torch.bfloat16)
        
    if monkey_patch:
        model = monkey_patch(model)
        
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # Establish inference max_len config
    max_len = model_maxlen.get(model_shortname.lower(), model_maxlen["default"])
    
    if args.method == "hash-gen":
        skip_layers = set(args.skip_layers)
        caches = create_lru_caches(model, args, skip_layers)
        print(f"Successfully configured hash-gen LRUCache:")
        print(f"  - lru_budget: {args.lru_budget}")
        print(f"  - top_budget: {args.top_budget}")
        print(f"  - skip_layers: {args.skip_layers}")
    else:
        # standard generation without our custom hash mechanic if method == none
        print(f"Using default model caches (method={args.method})")
        caches = None
        
    # Run evaluation linearly since model is massive and running locally
    get_pred(data, args, fout, model, tokenizer, caches, max_len)
    
    fout.close()
    
    if caches is not None:
        del caches
    print("Done Testing LongBench!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # LongBench args
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--cot", "-cot", action='store_true') 
    parser.add_argument("--no_context", "-nc", action='store_true') 
    parser.add_argument("--rag", "-rag", type=int, default=0) 
    
    # HF / Spotlight Model config args
    parser.add_argument("--model-name-or-path", "-m", type=str, required=True)
    parser.add_argument("--method", type=str, default="hash-gen", choices=["none", "hash-gen", "hash-eval"])
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory containing hash module weights")
    
    # Cache parameters args mapping to Spotlight implementation
    parser.add_argument("--max-position-embeddings", type=int, default=131072)
    parser.add_argument("--lru-budget", type=int, default=2048)
    parser.add_argument("--top-budget", type=int, default=1024)
    parser.add_argument("--dims", type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument("--skip-layers", type=int, nargs='+', default=[0, 1])

    args = parser.parse_args()
    main()
