import torch
import torch.distributed as dist

class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, scatter_idx, gather_idx):
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        if group is None:
            group = dist.group.WORLD

        seq_world_size = dist.get_world_size(group)
        if seq_world_size == 1:
            return input

        input_list = [t.contiguous() for t in torch.tensor_split(input, seq_world_size, dim=scatter_idx)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
        
        dist.all_to_all(output_list, input_list, group=group)
        return torch.cat(output_list, dim=gather_idx).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        return (None, _SeqAllToAll.apply(ctx.group, grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)

def gather_seq_dist_head(input, seq_dim=1, head_dim=2, group=None):
    """
    Given an input tensor partitioned across the sequence dimension, 
    gathers the full sequence and instead partitions across the head dimension.
    """
    return _SeqAllToAll.apply(group, input, head_dim, seq_dim)

def gather_head_dist_seq(input, seq_dim=1, head_dim=2, group=None):
    """
    Given an input tensor partitioned across the head dimension, 
    gathers all heads and instead partitions across the sequence dimension.
    """
    return _SeqAllToAll.apply(group, input, seq_dim, head_dim)

def dist_seq(input, seq_dim=1, group=None):
    """
    Partitions the sequence dimension of a tensor to the local rank.
    """
    if group is None:
        group = dist.group.WORLD
        
    if not dist.is_initialized():
        return input

    seq_world_size = dist.get_world_size(group)
    if seq_world_size == 1:
        return input

    rank = dist.get_rank(group)
    return torch.tensor_split(input, seq_world_size, dim=seq_dim)[rank].contiguous()

def gather_seq(input, seq_dim=1, group=None):
    """
    Gathers partitioned sequence dimension from all ranks.
    """
    if group is None:
        group = dist.group.WORLD
        
    if not dist.is_initialized():
        return input

    seq_world_size = dist.get_world_size(group)
    if seq_world_size == 1:
        return input

    output_list = [torch.empty_like(input) for _ in range(seq_world_size)]
    dist.all_gather(output_list, input.contiguous(), group=group)
    return torch.cat(output_list, dim=seq_dim).contiguous()
