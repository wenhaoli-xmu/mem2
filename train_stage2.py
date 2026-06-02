import torch
import gc
from safetensors.torch import save_file, load_file
import os

from torch.utils.data import DataLoader

from functools import partial
import tqdm
import argparse

from mem2 import get_monkey_patch, load_checkpoint
from mem2.monkey_patches import get_monkey_patch
from mem2.misc import adjust_lr
from mem2.monkey_patches.ulysses_utils import dist_seq

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from train_utils import (
    TrainingData,
    get_optimizer_and_lr_adjuster,
    clear_cache,
    collate_fn,
    WandbLogger
)

import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, 
    get_optimizer_state_dict,
    set_model_state_dict, 
    set_optimizer_state_dict,
    StateDictOptions)



def enable_topk(model):
    for layer in model.model.layers:
        layer.self_attn.enable_topk = True


def disable_topk(model):
    for layer in model.model.layers:
        layer.self_attn.enable_topk = False


def load_ref_model(args):

    # 1. 先加载模型
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.enable_bf16 else torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True)
    
    # 2. 再应用monkey-patch
    ref_model = get_monkey_patch('ulysses' if dist.get_world_size() > 1 else 'vanilla')(ref_model)
    ref_model.eval()

    # 3. 再应用fsdp2     
    for layer in ref_model.model.layers:
        fully_shard(layer)
    fully_shard(ref_model)

    # 4. 转移到local rank上
    ref_model = ref_model.to(dist.get_rank())
    ref_model.eval()

    return ref_model


def load_model(args):

    # 1. 先加载model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.enable_bf16 else torch.float32,
        device_map='cpu',
        low_cpu_mem_usage=True)
    
    # 2. 再应用monkey-patch
    model = get_monkey_patch(
        method='train-stage2', 
        hash_dims=[128, 128, 128], 
        chunk_size=args.chunk_size,
        top_k=args.top_k,  
        enable_topk=False
    )(model)

    # 3. 再加载checkpoint
    if args.checkpoint_dir is not None:
        load_checkpoint(model, args.checkpoint_dir)

    # 4. 最后一步应用fsdp2
    for layer in model.model.layers:
        fully_shard(layer)
    fully_shard(model)

    # 5. 转移到local rank
    model = model.to(dist.get_rank())
    model.train()

    # 6. 然后汇总需要被优化的参数，设置他们的requires grad属性
    params = []
    for name, p in model.named_parameters():
        if 'hash' in name:
            p.requires_grad = True
            params.append(p)
        else:
            p.requires_grad = False

    return model, params


def auto_resume(args, model, optim):
    save_path = args.model_name_or_path.split('/')[-1].lower() 
    ckpt_path = f"train_results/{save_path}/resume{args.postfix}.pt"

    resume_step = 0
    if args.auto_resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        
        set_model_state_dict(model, model_state_dict=ckpt['model'], options=StateDictOptions(full_state_dict=True, strict=False))
        set_optimizer_state_dict(model, optim, optim_state_dict=ckpt['optim'], options=StateDictOptions(full_state_dict=True))

        resume_step = ckpt['step']
        if dist.get_rank() == 0: 
            print(f"-> Resumed from step {resume_step}", flush=True)

    return resume_step


def auto_save(args, model, optim, step):
    save_path = args.model_name_or_path.split('/')[-1].lower() 
    ckpt_path = f"train_results/{save_path}/resume{args.postfix}.pt"

    if args.save_interval > 0 and step % args.save_interval == 0:
        
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_sd = get_model_state_dict(model, options=options)
        optim_sd = get_optimizer_state_dict(model, optim, options=options)

        if dist.get_rank() == 0:
            hash_weights = {}
            for name, tensor in model_sd.items():
                if 'hash' in name:
                    hash_weights[name] = tensor.cpu().contiguous()
            torch.save({'model': hash_weights, 'optim': optim_sd, 'step': step}, ckpt_path)
        
        if dist.get_world_size() > 1:
            dist.barrier()  


def final_save(args, model):
    save_path = args.model_name_or_path.split('/')[-1].lower() 
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state_dict = get_model_state_dict(model, options=options)

    if dist.get_rank() == 0:        
        hash_weights = {}
        for name, tensor in state_dict.items():
            if "hash" in name:
                hash_weights[name] = tensor.cpu().contiguous()
        save_file(hash_weights, f"train_results/{save_path}/stage2{args.postfix}.safetensors")

    if dist.get_world_size() > 1:
        dist.barrier()


def calc_grad_norm(params):
    grad_abs_mean = 0.0
    total_grad_sum = sum(p.grad.abs().sum() for p in params if p.grad is not None)
    total_grad_num = sum(p.grad.numel() for p in params if p.grad is not None)
    if total_grad_num > 0: 
        grad_abs_mean = (total_grad_sum / total_grad_num).item()
    return grad_abs_mean


def calc_inputs(batch):
    input_ids = batch['input_ids']
    if dist.get_world_size() > 1:
        dist.broadcast(input_ids, src=0)
        position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        input_ids = dist_seq(input_ids, seq_dim=1)
        position_ids = dist_seq(position_ids, seq_dim=1)
    else:
        position_ids = None

    return {
        "input_ids": input_ids,
        "position_ids": position_ids}


def calc_mean_metric(**kwargs):
    x_tensor = torch.tensor(
        [x.item() if isinstance(x, torch.Tensor) else x for x in kwargs.values()], 
        device=dist.get_rank(), 
        dtype=torch.float32)

    if dist.get_world_size() > 1:
        dist.all_reduce(x_tensor, op=dist.ReduceOp.AVG)

    return {key: x_tensor[i].item() for i, key in enumerate(kwargs.keys())}


def train(args):

    torch.manual_seed(0)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    else:
        torch.cuda.set_device(0)

    logger = WandbLogger(args)
    
    # 建立工作空间
    save_path = args.model_name_or_path.split('/')[-1].lower() 
    os.makedirs(f"train_results/{save_path}", exist_ok=True)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    pad_token_id = tokenizer.pad_token_id

    # 加载模型
    model, params = load_model(args)
    ref_model = load_ref_model(args)

    # 加载优化器
    optim, lr_adjust = get_optimizer_and_lr_adjuster(
        args.max_lr, args.train_iters, args.warmup, args.weight_decay,
        args.beta1, args.beta2, params=params, min_lr=args.min_lr)

    # 加载数据集
    corpus = TrainingData(args)
    partial_collate_fn = partial(
        collate_fn,
        pad_token_id=pad_token_id, 
        max_tokens=args.max_tokens)
    
    # 加载迭代器
    loader = DataLoader(
        corpus, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=partial_collate_fn)

    # debug
    # import debugpy
    # debugpy.listen(('0.0.0.0', 5678 + torch.distributed.get_rank()))
    # print("✅",flush=True)
    # debugpy.wait_for_client()
    
    # 检查是否要恢复训练
    resume_step = auto_resume(args, model, optim)
    step = 0

    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')

    for batch in tqdm.tqdm(loader, desc="Training", total=args.train_iters, disable=(local_rank > 0)):

        # 仅仅在[resume_step, train_iters]区间内进行训练
        if step < resume_step:
            step += 1
            continue
        elif step >= args.train_iters:
            break

        # 学习率调整
        lr_adjust(step=step)
        
        # 得到输入
        inputs = calc_inputs(batch)

        # 得到labels
        labels = torch.full_like(inputs['input_ids'], -100)
        labels[:, :-1] = inputs['input_ids'][:, 1:]

        # 前向传播，得到logits
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 计算lm loss
        lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        del outputs, logits

        # 反向传播
        lm_loss.backward()
        
        # 计算reference和top-k的loss
        with torch.no_grad():
            ref_outputs = ref_model(**inputs)
            ref_logits = ref_outputs.logits
            ref_loss = loss_fct(ref_logits.view(-1, ref_logits.size(-1)), labels.view(-1))
            del ref_outputs, ref_logits

            enable_topk(model)
            topk_outputs = model(**inputs)
            topk_logits = topk_outputs.logits
            topk_loss = loss_fct(topk_logits.view(-1, topk_logits.size(-1)), labels.view(-1))
            disable_topk(model)
            del topk_outputs, topk_logits
        
        # 计算一些监控指标，并上传到wandb
        metrics = {
            "lm_loss": lm_loss,
            "topk_loss": topk_loss,
            "ref_loss": ref_loss,
            "grad_abs_mean": calc_grad_norm(params)}
        metrics = calc_mean_metric(**metrics)
        metrics["residual_loss"] = metrics["topk_loss"] - metrics["ref_loss"]
        logger.log(metrics, step)

        # 本地打印这些指标
        if local_rank == 0:
            print(metrics, flush=True)
        
        # 优化器优化
        optim.step()
        optim.zero_grad()
        
        # 迭代step + 1
        step += 1

        # 保存用于resume train的检查点
        auto_save(args, model, optim, step)
        
    # 最终保存检查点
    final_save(args, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--hash_dims", type=int, default=32)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Path to the SFT checkpoint directory containing weight-*.safetensors")
    
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=4096)
    
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    
    parser.add_argument("--save-interval", type=int, default=0, help="Steps between checkpoints")
    parser.add_argument("--auto-resume", action="store_true")
    
    parser.add_argument("--train-data", type=str, action='append')
    parser.add_argument("--train-iters", type=int, default=1000)
    parser.add_argument("--max-lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--enable-bf16", action='store_true')
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--postfix", type=str, default="")
    
    args = parser.parse_args()
    train(args)
