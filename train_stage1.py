import torch
import gc
from safetensors.torch import save_file, load_file
import os

from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from functools import partial
import tqdm
import argparse

from concurrent.futures import ThreadPoolExecutor

from mem2 import get_monkey_patch
from mem2.misc import adjust_lr

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


from train_utils import (
    TrainingData,
    compute_attn_supervise_loss,
    get_optimizer_and_lr_adjuster,
    clear_cache,
    collate_fn,
    reset_buffer_dir,
    WandbLogger,
    compute_iou
)


def train(args):
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group('nccl', rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    logger = WandbLogger(args)

    num_gpus = dist.get_world_size()
    os.makedirs(args.buffer, exist_ok=True)
    reset_buffer_dir(args.buffer, False)

    assert args.train_iters % args.instance_per_cycle == 0
    assert args.instance_per_cycle % num_gpus == 0

    num_inn_cycle = args.train_iters // args.instance_per_cycle
    num_out_cycle = (args.num_layers + num_gpus - 1) // num_gpus

    async_executor = ThreadPoolExecutor(max_workers=1)
    monkey_patch = get_monkey_patch(
        'train-stage1', 
        hash_dims=[128,128,128])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    pad_token_id = tokenizer.pad_token_id
    del tokenizer

    # start training
    save_path = args.model_name_or_path.split('/')[-1].lower() + args.postfix
    os.makedirs(f"train_results/{save_path}/stage1", exist_ok=True)

    # Skip entirely if all requested layers already exist
    all_exist = all([os.path.exists(f"train_results/{save_path}/stage1/weight-{layer_idx}.safetensors") for layer_idx in range(args.num_layers)])
    if all_exist:
        print("All layers already trained. Exiting.")
        dist.destroy_process_group()
        return

    # Scaffold the target layers on CPU beforehand so we don't incur disk loading inside loops
    print(f"RANK-{local_rank} Preparing layer scaffolds on CPU...")
    temp_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda')
    temp_model = monkey_patch(temp_model)

    pos_ids = torch.arange(args.max_tokens, dtype=torch.long, device='cuda').unsqueeze(0)
    with torch.no_grad():
        fake_tensor = torch.empty(1, dtype=torch.float, device='cuda')
        pos_emb = temp_model.model.rotary_emb(fake_tensor, pos_ids)
        del pos_ids, fake_tensor
    
    # load modules to be trained
    target_layers = {}
    all_attn_modules = temp_model.dump_as_attn_modules()
    for out_cycle_idx in range(num_out_cycle):
        layer_idx = out_cycle_idx * num_gpus + local_rank
        if layer_idx >= args.num_layers:
            continue
        layer = all_attn_modules[layer_idx]
        layer.train()
        target_layers[layer_idx] = layer
    del all_attn_modules, temp_model
    clear_cache()
    print(f"RANK-{local_rank} Scaffolds prepared.")
    dist.barrier()

    # initialize model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda')
    model = monkey_patch(model)
    model.eval()
    
    # initialize dataloader
    corpus = TrainingData(args)
    partial_collate_fn = partial(
        collate_fn, 
        pad_token_id=pad_token_id, 
        max_tokens=args.max_tokens)
    sampler = DistributedSampler(
        corpus, 
        num_replicas=num_gpus, 
        rank=local_rank, 
        shuffle=True)
    loader = DataLoader(
        corpus, 
        batch_size=1, 
        sampler=sampler,
        collate_fn=partial_collate_fn)
    data_iter = iter(loader)
    sampler.set_epoch(0)

    for inn_cycle_idx in range(num_inn_cycle):

        # Phase 1: Prepare Hidden States
        save_future = None
        for idx in tqdm.tqdm(range(0, args.instance_per_cycle, num_gpus), desc=f"Data Prep Cycle {inn_cycle_idx}"):
            
            # prepare inputs
            inputs = next(data_iter)
            inputs.update({"return_hidden_states": True})
            inputs.pop("input_len")

            # forward pass     
            with torch.no_grad():
                outputs = model(**inputs)

            def save_batch(tensors, base_idx, rank):
                for layer_idx, tensor in enumerate(tensors):
                    save_file(
                        {"hidden_states": tensor.cpu()}, 
                        f"{args.buffer}/layer-{layer_idx}-sample-{base_idx + rank}.safetensors")

            # save asynchronizely
            if save_future is not None: 
                save_future.result()
            save_future = async_executor.submit(save_batch, outputs.hidden_states, idx, local_rank)

            del inputs, outputs

        # wait for last save
        if save_future is not None: 
            save_future.result()
        
        # recycle resources
        clear_cache()
        dist.barrier()
             
        # Phase 2: Sequential Training per Layer
        compute_loss = partial(
            compute_attn_supervise_loss,
            top_k=args.top_k if args.top_k is not None else 128,
            max_top=args.max_top,
            max_oth=args.max_oth)

        for out_cycle_idx in range(num_out_cycle):
            layer_idx = out_cycle_idx * num_gpus + local_rank
            is_active = layer_idx < args.num_layers
            
            if is_active:
                layer = target_layers[layer_idx]

                params = list(layer.query_hash.parameters()) 
                params += list(layer.key_hash.parameters())

                for param in layer.parameters(): param.requires_grad_(False)
                for param in params: param.requires_grad_(True)
                
                optim, lr_adjust = get_optimizer_and_lr_adjuster(
                    args.max_lr, args.train_iters, args.warmup, args.weight_decay,
                    args.beta1, args.beta2, params=params, min_lr=args.min_lr)

                step = inn_cycle_idx * args.instance_per_cycle

                temp_dir = f"train_results/{save_path}/stage1/temp"
                os.makedirs(temp_dir, exist_ok=True)
                temp_optim_file = f"{temp_dir}/optim-{layer_idx}.pth"
                
                # load past training states
                if inn_cycle_idx > 0:
                    optim.load_state_dict(torch.load(temp_optim_file))

                # load the first sample
                hidden_states = load_file(f"{args.buffer}/layer-{layer_idx}-sample-0.safetensors")["hidden_states"]

            for sample_idx in range(args.instance_per_cycle):
                if is_active:
                    # preload next sample
                    next_sample_idx = (sample_idx + 1) % args.instance_per_cycle
                    next_path = f"{args.buffer}/layer-{layer_idx}-sample-{next_sample_idx}.safetensors"
                    future_layer = async_executor.submit(
                        lambda p=next_path: load_file(p)["hidden_states"])
                    
                    lr_adjust(step=step)

                    # select query
                    random_query_index = None
                    if args.max_que is not None:
                        random_query_index = torch.randperm(
                            hidden_states.shape[-2], 
                            dtype=torch.int64)[:args.max_que].sort().values.cuda()

                    # forward
                    _, extra_rets = layer(
                        hidden_states=hidden_states.cuda(),
                        position_embeddings=pos_emb,
                        random_query_index=random_query_index)

                    # compute loss
                    draft_attn = extra_rets['hash_score']
                    true_attn = extra_rets['attn_score']
                    iou, loss = compute_loss(draft_attn, true_attn, random_query_index)
                    loss_val = loss.item()
                    
                    loss /= args.gradient_accumulation
                    loss.backward()
                    
                    metrics = {"loss": loss_val, "iou": iou}
                else:
                    metrics = {"loss": 0.0, "iou": 0.0}

                logger.log(
                    metrics_dict=metrics,
                    step=inn_cycle_idx * args.instance_per_cycle + sample_idx,
                    out_cycle_idx=out_cycle_idx
                )

                if is_active:
                    print(f"Cycle: {inn_cycle_idx+1}/{num_inn_cycle} | layer: {layer_idx:>5d} | "
                        f"step: {step:>5d} | "
                        f"loss: {loss_val:>05.3f} | "
                        f"iou: {iou:>05.3f}", flush=True)

                    if (step + 1) % args.gradient_accumulation == 0:
                        if args.gradient_clipping is not None:
                            torch.nn.utils.clip_grad_norm_(params, max_norm=args.gradient_clipping)
                        optim.step()
                        optim.zero_grad()

                    step += 1
                    del draft_attn, true_attn, hidden_states, loss
                    hidden_states = future_layer.result()
            
            if is_active:
                torch.save(optim.state_dict(), temp_optim_file)
                if inn_cycle_idx == num_inn_cycle - 1:
                    save_file({f"param_{i}": p.data.cpu() for i, p in enumerate(params)}, f"train_results/{save_path}/stage1/weight-{layer_idx}.safetensors")
                del optim, params
                clear_cache()
            
        dist.barrier()
        if local_rank == 0:
            reset_buffer_dir(args.buffer, disable=False)
        dist.barrier()
        
    print(f"RANK-{local_rank} training done !")
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model related parameters (NOTE: need to change according to the base model)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)

    # model non-related parameters
    parser.add_argument("--buffer", type=str, default='train_buffer')

    # resource controling related parameters
    parser.add_argument("--instance_per_cycle", type=int, default=1000)
    parser.add_argument("--max_que", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_top", type=int, default=None)
    parser.add_argument("--max_oth", type=int, default=None)

    # training data
    parser.add_argument("--train-data", type=str, action='append', help="will be random if not given.")
    parser.add_argument("--train-iters", type=int)

    # adamw configuration
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=0)
    parser.add_argument("--warmup", type=float, default=0.)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--gradient-clipping", type=float, default=None)
    parser.add_argument("--postfix", type=str, default="")
    args = parser.parse_args()
    train(args)
