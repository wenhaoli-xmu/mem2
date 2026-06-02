import torch
import gc
import os
import torch.distributed as dist

from functools import partial
from mem2.misc import adjust_lr

from transformers import AutoConfig
from safetensors.torch import save_file

import json
import random
import wandb


class TrainingData(torch.utils.data.Dataset):
    def __init__(self, args):
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.max_tokens = args.max_tokens
        self.vocab_size = config.vocab_size
        self.train_iters = args.train_iters

        self.data = []
        if args.train_data is not None:
            for data in args.train_data:
                with open(data, 'r') as f:
                    for line in f:
                        try:
                            self.data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping malformed JSON line: {e}")
                            continue

        assert len(self.data) >= self.train_iters, f"Not enough data to train for {self.train_iters} iterations."
    
    def __getitem__(self, idx):
        if self.data is None:
            return {
                "input_ids": [random.randint(0, self.vocab_size-1) for _ in range(self.max_tokens)]
            }
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


def compute_iou(draft_attn_masked, top_indices, top_cnt):
    with torch.no_grad():
        ious = []
        # Support batch size > 1 if needed, but current training uses batch size 1
        # For simplicity and consistency with existing code, we compute for head 0 of sample 0
        head0_pred = torch.topk(draft_attn_masked[0, 0], k=top_cnt, dim=-1).indices.detach().cpu().tolist()
        head0_label = top_indices[0, 0].detach().cpu().tolist()
        for q_pred, q_label in zip(head0_pred, head0_label):
            intersection = len(set(q_pred) & set(q_label))
            union = len(set(q_pred) | set(q_label))
            ious.append(intersection / union if union > 0 else 0.0)
    return sum(ious) / len(ious)


def compute_attn_supervise_loss(
        draft_attn, 
        true_attn, 
        query_index, 
        max_top, 
        max_oth, 
        top_k):

    criterion = torch.nn.BCEWithLogitsLoss()

    num_kv = true_attn.shape[-1]
    oth_cnt = num_kv - top_k

    if query_index is not None:
        j_indices = torch.arange(num_kv, device=true_attn.device)
        causal_mask = (j_indices[None, :] > query_index[:, None])[None, None, :, :]
    else:
        causal_mask = torch.triu(torch.ones((num_kv, num_kv), dtype=torch.bool, device=true_attn.device), diagonal=1)[None, None, :, :]

    true_attn_masked = true_attn.masked_fill(causal_mask, value=torch.finfo(true_attn.dtype).min)
    draft_attn_masked = draft_attn.masked_fill(causal_mask, value=torch.finfo(draft_attn.dtype).min)

    _, top_indices = torch.topk(true_attn_masked, k=top_k, dim=-1, largest=True, sorted=False)

    iou = compute_iou(draft_attn_masked, top_indices, top_k)

    _, oth_indices = torch.topk(true_attn_masked, k=oth_cnt, dim=-1, largest=False, sorted=False)

    if max_top is not None:
        top_rnd_indices = torch.randperm(top_k, dtype=torch.int64, device=top_indices.device)[:max_top]
        top_indices = top_indices[..., top_rnd_indices]
    if max_oth is not None:
        oth_rnd_indices = torch.randperm(oth_cnt, dtype=torch.int64, device=oth_indices.device)[:max_oth]
        oth_indices = oth_indices[..., oth_rnd_indices]

    if query_index is not None:
        top_mask = (top_indices > query_index[None, None, :, None])[..., :, None]
        oth_mask = (oth_indices > query_index[None, None, :, None])[..., None, :]
    else:
        num_q = true_attn.shape[-2]
        q_indices = torch.arange(num_q, device=true_attn.device)
        top_mask = (top_indices > q_indices[None, None, :, None])[..., :, None]
        oth_mask = (oth_indices > q_indices[None, None, :, None])[..., None, :]

    top_draft_attn = torch.gather(draft_attn_masked, dim=-1, index=top_indices)[..., :, None]
    oth_draft_attn = torch.gather(draft_attn_masked, dim=-1, index=oth_indices)[..., None, :]

    residual = top_draft_attn - oth_draft_attn
    residual_mask = (top_mask | oth_mask).expand_as(residual).flatten(-3)

    logits = residual.flatten(-3)[~residual_mask]
    labels = torch.ones_like(logits)
    loss = criterion(logits, labels).cpu()

    del true_attn_masked, draft_attn_masked, top_indices, oth_indices
    del top_draft_attn, oth_draft_attn, residual, residual_mask

    return iou, loss


def get_optimizer_and_lr_adjuster(max_lr, train_iters, warmup, weight_decay, beta1, beta2, params, **kwargs):
    optim = torch.optim.AdamW(params, lr=max_lr, betas=[beta1, beta2], weight_decay=weight_decay)
    lr_adjuster = partial(adjust_lr, optim=optim, total=train_iters, max_lr=max_lr, min_lr=kwargs.get('min_lr', 0), restart=1, warmup=warmup, plateau=0)
    return optim, lr_adjuster


def clear_cache():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def collate_fn(batch, pad_token_id, max_tokens):
    if pad_token_id is None:
        pad_token_id = 0

    input_ids = [x.get('input_ids') for x in batch]
    input_len = [len(x) for x in input_ids]

    assert all([length <= max_tokens for length in input_len]), f"Input length exceed `max_tokens`, please enlarge `max_tokens` or shrink truncation length to fix the problem."

    # padding
    input_ids = [x + [pad_token_id] * (max_tokens - len(x)) for x in input_ids]

    # to tensor
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    input_len = torch.tensor(input_len, dtype=torch.int64)

    return {"input_ids": input_ids.cuda(), "input_len": input_len}


def reset_buffer_dir(buffer, disable):
    if not disable:
        if dist.get_rank() == 0:
            for file in os.listdir(buffer):
                os.remove(os.path.join(buffer, file))
        dist.barrier()


class WandbLogger:
    def __init__(self, args):
        self.args = args
        if dist.get_rank() == 0:
            wandb.init(
                project="spotlight-v2",
                name=args.run_name,
            )
            wandb.define_metric("train_step")
            wandb.define_metric("layer/*", step_metric="train_step")

    def log(self, metrics_dict, step, out_cycle_idx=None):
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()

        if out_cycle_idx is not None:
            # Stage 1 logic: gather metrics from all ranks (each rank corresponds to a layer)
            local_metrics = torch.tensor(
                [metrics_dict.get("loss", 0.0), metrics_dict.get("iou", 0.0)],
                dtype=torch.float32,
                device='cuda',
            )
            gathered_metrics = [torch.zeros_like(local_metrics) for _ in range(world_size)]
            dist.all_gather(gathered_metrics, local_metrics)

            if local_rank == 0:
                log_dict = {}
                for r, metrics in enumerate(gathered_metrics):
                    global_layer_id = out_cycle_idx * world_size + r
                    log_dict[f"layer/{global_layer_id}/loss"] = metrics[0].item()
                    log_dict[f"layer/{global_layer_id}/iou"] = metrics[1].item()

                log_dict["train_step"] = step
                wandb.log(log_dict, step=step)
        else:
            # Stage 2 logic: log aggregated metrics directly from rank 0
            if local_rank == 0:
                metrics_dict["train_step"] = step
                wandb.log(metrics_dict, step=step)