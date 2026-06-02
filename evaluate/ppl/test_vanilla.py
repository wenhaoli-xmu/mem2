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
    parser.add_argument("--task", type=str, default="evaluate/ppl/perplexity_tasks.json")

    args = parser.parse_args()

    test_conf = get_env_conf(args.task)
    print('config loaded ✅', flush=True)

    model_name = os.path.basename(os.path.normpath(args.model_name_or_path)).lower()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map={'': 0},
        torch_dtype=torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    evaluator = Evaluator(model, tokenizer, eval=None, tasks=test_conf)
    evaluator.evaluate()
