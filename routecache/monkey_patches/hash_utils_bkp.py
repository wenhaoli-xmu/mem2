import torch
import os
import torch.distributed as dist
import threading
from queue import SimpleQueue

from itertools import chain
from safetensors.torch import load_file

from transformers.cache_utils import Cache


# ---- Pure PyTorch replacements for custom CUDA kernels ----

def _popcount_int32(x):
    """Bit-parallel popcount for int32 tensors (Hamming weight)."""
    v = x.to(torch.int64) & 0xFFFFFFFF
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    v = v + (v >> 8)
    v = v + (v >> 16)
    return (v & 0x3F).to(torch.int32)


def _packbits(x):
    """Pack sign bits (x > 0) into int32.  x: [..., D] with D % 32 == 0.  Returns [..., D//32] int32."""
    D = x.shape[-1]
    sign = (x > 0).to(torch.int32).view(*x.shape[:-1], D // 32, 32)
    powers = (1 << torch.arange(32, device=x.device, dtype=torch.int32))
    return (sign * powers).sum(dim=-1)


def _hash_forward_packbits(x, proj0, proj1):
    """2-layer hash network (silu+residual, linear+residual) followed by packbits.
    Replaces hash_packbits.hash_packbits kernel."""
    h = torch.nn.functional.silu(torch.einsum('bnhd,hde->bnhe', x, proj0)) + x
    out = torch.einsum('bnhd,hde->bnhe', h, proj1) + h
    return _packbits(out)


def _qhash_project_pack(query, q_proj0, q_proj1, KH, G):
    """Project query through hash network, pack bits, group by key heads.
    query: [B, 1, QH, D] -> returns [B, KH, G*num_int32] int32."""
    h = torch.nn.functional.silu(torch.einsum('bnhd,hde->bnhe', query, q_proj0)) + query
    out = torch.einsum('bnhd,hde->bnhe', h, q_proj1) + h
    packed = _packbits(out)  # [B, 1, QH, D//32]
    B, _, QH, num_int32 = packed.shape
    return packed.squeeze(1).view(B, KH, G, num_int32).reshape(B, KH, G * num_int32)


def _hamming_topk(q_packed, key_bins, num_tokens, top_budget, KH, G):
    """Compute hamming distance between packed query and key hashes, return top-k closest.
    q_packed: [B, KH, G*num_int32], key_bins: [B, MaxT, KH, num_int32]
    Returns: [B, KH, top_budget] int64 indices."""
    B = q_packed.shape[0]
    num_int32 = key_bins.shape[-1]
    T = int(num_tokens) if isinstance(num_tokens, int) else num_tokens.item()

    kb = key_bins[:, :T].permute(0, 2, 1, 3)          # [B, KH, T, num_int32]
    qp = q_packed.view(B, KH, G, num_int32)            # [B, KH, G, num_int32]

    xor = qp.unsqueeze(3) ^ kb.unsqueeze(2)            # [B, KH, G, T, num_int32]
    hamming_dist = _popcount_int32(xor).sum(dim=-1).sum(dim=2)  # [B, KH, T]

    k = min(top_budget, T)
    _, topk_indices = hamming_dist.topk(k, dim=-1, largest=False)

    if k < top_budget:
        pad = topk_indices[:, :, :1].expand(B, KH, top_budget - k)
        topk_indices = torch.cat([topk_indices, pad], dim=-1)

    return topk_indices.to(torch.int64)


def _gather_topk(topk_out, key_cache, value_cache):
    """Gather top-k keys and values from the full cache using topk indices.
    topk_out: [B, KH, top_budget] int64 indices
    key_cache: [B, MaxT, KH, D]
    Returns: gathered_keys [B, top_budget, KH, D], gathered_values [B, top_budget, KH, D]"""
    D = key_cache.shape[-1]
    # topk_out: [B, KH, top_budget] -> [B, top_budget, KH] -> [B, top_budget, KH, D]
    idx = topk_out.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, D)
    gathered_k = key_cache.gather(1, idx)
    gathered_v = value_cache.gather(1, idx)
    return gathered_k, gathered_v


# ---- Original utility classes (unchanged) ----

class CacheList(Cache):
    def __new__(cls, caches=None):
        instance = object.__new__(cls)
        return instance

    def __init__(self, caches=None):
        self.caches = caches or []
        self.layers = []

    @property
    def is_compileable(self):
        return False

    def __getitem__(self, layer_idx):
        return self.caches[layer_idx]

    def __iter__(self):
        return iter(self.caches)

    def __len__(self):
        return len(self.caches)

    def __bool__(self):
        return bool(self.caches)

    def get_seq_length(self, layer_idx=0, cache_position=None):
        if len(self.caches) == 0:
            return 0
        return self.caches[layer_idx].get_seq_length()

    def get_mask_sizes(self, cache_position, layer_idx=0):
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    def reset(self):
        for cache in self.caches:
            cache.reset()


class FullCache:
    def __init__(self):
        self.key_cache = None
        self.value_cache = None
        self.num_tokens = 0

    def get_seq_length(self):
        return self.num_tokens

    def reset(self):
        self.key_cache = None
        self.value_cache = None
        self.num_tokens = 0

    def update_query(self, query):
        pass

    @torch.inference_mode()
    def update(self, keys, values):
        T = keys.shape[1]
        self.num_tokens += T
        if self.key_cache is None:
            self.key_cache = keys
            self.value_cache = values
        else:
            self.key_cache = torch.cat([self.key_cache, keys], dim=1)
            self.value_cache = torch.cat([self.value_cache, values], dim=1)
        return self.key_cache, self.value_cache


def compute_lsh_score(q_hash, k_hash, random_query_index):
    group_size = q_hash.shape[2] // k_hash.shape[2]

    if random_query_index is not None:
        q_hash = q_hash[:, random_query_index]

    q_hash = q_hash.transpose(1, 2)
    k_hash = k_hash.transpose(1, 2)
    k_hash = k_hash.unsqueeze(2).expand(-1, -1, group_size, -1, -1).flatten(1,2)

    q_bin = (q_hash > 0).float() * 2 - 1
    k_bin = (k_hash > 0).float() * 2 - 1

    q_soft = torch.tanh(q_hash)
    k_soft = torch.tanh(k_hash)

    q_final = (q_bin - q_soft).detach() + q_soft
    k_final = (k_bin - k_soft).detach() + k_soft

    sim = q_final @ k_final.transpose(-1, -2) / q_final.shape[-1] ** 0.5

    return sim


def compute_attn_score(q, k, random_query_index):
    score = []
    num_heads = q.shape[2]
    group_size = num_heads // k.shape[2]

    if random_query_index is not None:
        q = q[:, random_query_index]

    rng = torch.arange(k.shape[1], device=q.device)
    msk = random_query_index[:, None] < rng[None, :]
    msk = msk[None, :, :]

    for head_idx in range(num_heads):
        q_head = q[..., head_idx, :]
        k_head = k[..., head_idx // group_size , :]
        head_score = q_head @ k_head.transpose(-1,-2) / q_head.shape[-1] ** 0.5
        head_score.masked_fill_(msk, value=torch.finfo(head_score.dtype).min)
        head_score = head_score.softmax(dim=-1, dtype=torch.float).type(q.dtype)
        score.append(head_score)

    return torch.stack(score, dim=1)



class HashLayer(torch.nn.Module):
    def __init__(self, num_heads, dim_inp, dim_out, silu, dtype=torch.bfloat16, device='cpu'):
        super().__init__()

        self.proj = torch.nn.Parameter(
            torch.zeros((num_heads,dim_inp,dim_out), dtype=dtype, device=device),
            requires_grad=True)

        std = (2.0 / dim_out) ** 0.5
        torch.nn.init.normal_(self.proj, mean=0.0, std=std)

        self.silu = torch.nn.SiLU(inplace=True) if silu else None
        self.enable_residual = dim_inp == dim_out

    def forward(self, x):
        out = torch.einsum('bnhd,hde->bnhe', x, self.proj)
        if self.silu is not None:
            out = self.silu(out)
        if self.enable_residual:
            out = out + x
        return out



class DynamicGain(torch.nn.Module):
    def __init__(self, init=1.0, dtype=torch.bfloat16, device='cpu'):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.full((1, 1, 1, 1), init, dtype=dtype, device=device))

    def forward(self, x):
        return self.gain * x


class HashModule(torch.nn.Module):
    def __init__(self, num_heads, dims, dtype=torch.bfloat16, device='cpu'):
        super().__init__()
        n_layers = len(dims) - 1
        self.dims = dims
        self.num_heads = num_heads

        self.mlp = torch.nn.Sequential(*[
            HashLayer(
                num_heads,
                dims[i],
                dims[i+1],
                i < n_layers - 1,
                dtype=dtype,
                device=device)
            for i in range(n_layers)])

    def forward(self, x):
        embed = self.mlp(x)
        return embed

    def fused_packbits(self, x):
        assert len(self.dims) == 3, "fused_packbits only supports 2-layer networks"
        assert self.dims[0] == self.dims[1] == self.dims[2], "all dims must be equal"
        assert self.dims[0] % 32 == 0, "dims must be divisible by 32"

        proj0 = self.mlp[0].proj  # [H, D, D]
        proj1 = self.mlp[1].proj  # [H, D, D]

        return _hash_forward_packbits(x, proj0, proj1)

    def get_proj_weights(self):
        assert len(self.dims) == 3, "get_proj_weights only supports 2-layer networks"
        return self.mlp[0].proj, self.mlp[1].proj


class LRUCache:
    def __init__(self,
            checkpoint_path_or_dir,
            layer_idx,
            batch_size,
            max_position_embeddings,
            num_attention_heads,
            num_key_value_heads,
            hash_module_dims,
            top_budget=1024,
            device='cuda',
            **kwargs):

        dtype = torch.bfloat16
        self.B = batch_size
        self.MaxT = max_position_embeddings
        self.QH = num_attention_heads
        self.KH = num_key_value_heads
        self.dims = hash_module_dims
        self.device = device
        self.layer_idx = layer_idx

        self.query_hash = HashModule(self.QH, self.dims, dtype=dtype, device=device)
        self.key_hash = HashModule(self.KH, self.dims, dtype=dtype, device=device)

        self.key_cache = torch.zeros((self.B, self.MaxT, self.KH, self.dims[0]), dtype=dtype, device=device)
        self.value_cache = torch.zeros((self.B, self.MaxT, self.KH, self.dims[0]), dtype=dtype, device=device)

        assert self.dims[-1] % 32 == 0, "hash_module_dims[-1] must be divisible by 32"
        num_int32 = self.dims[-1] // 32
        self.key_bins = torch.zeros((self.B, self.MaxT, self.KH, num_int32), dtype=torch.int32, device=device)

        self.num_tokens = 0
        self.top_budget = int(top_budget)

        # Stores the latest retrieve result, written by async worker, read by update()
        self.topk_out = torch.zeros((self.B, self.KH, self.top_budget), dtype=torch.int64, device=device)

        self.load_checkpoint(checkpoint_path_or_dir, layer_idx)

        self.q_proj0, self.q_proj1 = self.query_hash.get_proj_weights()
        self.G = self.QH // self.KH

        self.async_retrieve_stream = torch.cuda.Stream()
        self.async_update_stream = torch.cuda.Stream()
        self.current_stream = torch.cuda.current_stream()

        self._task_queue = SimpleQueue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()


    def __del__(self):
        pass


    def get_seq_length(self):
        return self.num_tokens


    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.key_bins.zero_()
        self.topk_out.zero_()
        self.num_tokens = 0


    def load_checkpoint(self, checkpoint_path_or_dir, layer_idx, template="weight-{layer_idx}.safetensors"):
        if checkpoint_path_or_dir is None:
            return

        if os.path.isdir(checkpoint_path_or_dir):
            weight_path = os.path.join(checkpoint_path_or_dir, template.format(layer_idx=layer_idx))
            if not os.path.exists(weight_path):
                return
            weights = load_file(weight_path, device='cuda')
        elif os.path.isfile(checkpoint_path_or_dir):
            weight_path = checkpoint_path_or_dir
            weights = load_file(weight_path, device='cuda')
        else:
            return

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Try flat format (param_i) first
        if 'param_0' in weights:
            params = chain.from_iterable((
                self.query_hash.parameters(),
                self.key_hash.parameters()))
            for i, p in enumerate(params):
                param_key = f'param_{i}'
                if param_key in weights:
                    w = weights[param_key]
                    if world_size > 1 and w.shape[0] > p.shape[0]:
                        shard_size = p.shape[0]
                        w = w[rank * shard_size : (rank + 1) * shard_size]
                    p.data.copy_(w.to(p.dtype))
        else:
            # Try full name format
            for prefix, module in [
                (f"model.layers.{layer_idx}.self_attn.query_hash.", self.query_hash),
                (f"model.layers.{layer_idx}.self_attn.key_hash.", self.key_hash)
            ]:
                for name, p in module.named_parameters():
                    key = prefix + name
                    if key in weights:
                        w = weights[key]
                        if world_size > 1 and w.shape[0] > p.shape[0]:
                            shard_size = p.shape[0]
                            w = w[rank * shard_size : (rank + 1) * shard_size]
                        p.data.copy_(w.to(p.dtype))


    @torch.inference_mode()
    def _async_retrieve_worker(self, query, current_num_tokens):
        q_packed = _qhash_project_pack(
            query, self.q_proj0, self.q_proj1, self.KH, self.G)
        self.topk_out.copy_(_hamming_topk(
            q_packed, self.key_bins, current_num_tokens,
            self.top_budget, self.KH, self.G))


    @torch.inference_mode()
    def _async_update_worker(self, key, value, start):
        self.key_cache[:, start:start+1] = key
        self.value_cache[:, start:start+1] = value
        k_bins = self.key_hash.fused_packbits(key)
        self.key_bins[:, start:start+1] = k_bins


    def _worker_loop(self):
        while True:
            fn, args = self._task_queue.get()
            fn(*args)

    @torch.inference_mode()
    def _launch_retrieve(self, query, num_tokens):
        self.async_retrieve_stream.wait_stream(self.current_stream)
        with torch.cuda.stream(self.async_retrieve_stream):
            self._async_retrieve_worker(query, num_tokens)


    @torch.inference_mode()
    def _launch_update(self, keys, values, start):
        self.async_update_stream.wait_stream(self.current_stream)
        with torch.cuda.stream(self.async_update_stream):
            self._async_update_worker(keys, values, start)


    @torch.inference_mode()
    def update_query(self, query):
        if query.shape[1] == 1:
            self._task_queue.put((self._launch_retrieve, (query, self.num_tokens)))


    @torch.inference_mode()
    def update(self, keys, values):
        T = keys.shape[1]
        start = self.num_tokens
        end = start + T
        self.num_tokens = end
        if self.num_tokens > self.MaxT:
            raise ValueError(f"Cached KV length {self.num_tokens} is greater than max position embeddings {self.MaxT}.")

        if T == 1 and self.num_tokens > self.top_budget:
            # decoding: gather top-k from full cache + append current token
            self._task_queue.put((self._launch_update, (keys, values, start)))

            self.current_stream.wait_stream(self.async_retrieve_stream)
            gathered_k, gathered_v = _gather_topk(self.topk_out, self.key_cache, self.value_cache)
            return torch.cat([gathered_k, keys], dim=1), torch.cat([gathered_v, values], dim=1)
        else:
            # prefilling
            self.key_cache[:, start:end].copy_(keys, non_blocking=True)
            self.value_cache[:, start:end].copy_(values, non_blocking=True)

            key_bins = self.key_hash.fused_packbits(keys)
            self.key_bins[:, start:end].copy_(key_bins, non_blocking=True)

            return self.key_cache[:, :end], self.value_cache[:, :end]
