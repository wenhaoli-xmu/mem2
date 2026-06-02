from mem2 import get_monkey_patch, load_checkpoint
from mem2.misc import Evaluator, get_env_conf
from mem2.eval import test_on_task
import argparse
import torch
import os
import json
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import plot_ppl_curve


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=None)

    parser.add_argument("--check-results", action='store_true')
    parser.add_argument("--task", type=str, default="evaluate/ppl/perplexity_tasks.json")

    parser.add_argument("--checkpoint-path-or-dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=1024)
    parser.add_argument("--hash_dims", type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--skip-layers", type=int, nargs='+', default=[])

    args = parser.parse_args()

    test_conf = get_env_conf(args.task)
    print('config loaded ✅', flush=True)

    model_name = os.path.basename(os.path.normpath(args.model_name_or_path)).lower()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map={'': 0},
        torch_dtype=torch.bfloat16)
    
    model = get_monkey_patch(
        method='train-stage2',
        top_k=args.top_k,
        hash_dims=args.hash_dims,
        chunk_size=args.chunk_size,
        skip_layers=args.skip_layers,
        enable_topk=True)(model)

    load_checkpoint(model, args.checkpoint_path_or_dir)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    evaluator = Evaluator(model, tokenizer, eval=None, tasks=test_conf)
    result = evaluator.evaluate(return_raw=args.check_results)

    if args.check_results:
        import IPython
        IPython.embed(header='check results')
