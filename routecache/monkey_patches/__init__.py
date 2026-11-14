import os
import json
from functools import partial


def get_config(method):
    config_file = os.path.join("config", f"{method}.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        return {}


def get_monkey_patch(method, **kwargs):
    if method is None:
        raise ValueError("method must be provided (e.g. 'origin', 'hash-eval', ...)")

    # Be forgiving about shell scripts / CSV-like inputs accidentally including whitespace or commas.
    method = str(method).strip().rstrip(",")

    if method == "ulysses":
        from .ulysses import monkey_patch
    
    elif method == 'eval-eager':
        from .eval_eager import monkey_patch

    elif method == 'eval':
        from .eval import monkey_patch

    elif method == 'train-stage1':
        from .train_stage1 import monkey_patch

    elif method == 'train-stage2':
        from .train_stage2 import monkey_patch

    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            "Available: origin, origin-tp, topk-eval, topk-gen, topk-block-eval, hash-sft, hash-rl, hash-eval"
        )

    config = get_config(method)
    config.update(kwargs)

    return partial(monkey_patch, config=config)