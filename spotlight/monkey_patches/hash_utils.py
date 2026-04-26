import torch
import os
import torch.distributed as dist
import threading
from queue import SimpleQueue

from itertools import chain
from safetensors.torch import load_file

from spotlight.kernel import (
    lru_cache_update_v2 as lru_cache_update,
    hash_packbits,
    hamming_topk_v3,
    qhash_fused_v2,
)

from transformers.cache_utils import Cache


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

        return hash_packbits.hash_packbits(x, proj0, proj1)

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
            lru_budget=2048,
            top_budget=1024,
            warmup_steps=32,
            device='cuda'):

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
        self.key_bins = torch.zeros((self.B, self.KH, self.MaxT, num_int32), dtype=torch.int32, device=device)

        self.num_tokens = 0
        self.is_prefilled = False
        self.saved_queries = None
        self.warmup_steps = warmup_steps

        self.lru_size = int(lru_budget)
        self.lru_budget = self.lru_size - 1
        self.top_budget = int(top_budget)

        self.key_lru = torch.zeros((batch_size, self.lru_size, self.KH, self.dims[0]), dtype=dtype, device=device)
        self.value_lru = torch.zeros((batch_size, self.lru_size, self.KH, self.dims[0]), dtype=dtype, device=device)

        self.lru_indices = torch.full((batch_size, self.KH, self.lru_size), -1, dtype=torch.int32, device=device)
        self.lru_timestamps = torch.zeros((batch_size, num_key_value_heads, self.lru_size), dtype=torch.int32, device=device)
        self.current_time = torch.zeros((batch_size, num_key_value_heads), dtype=torch.int32, device=device)

        self.load_checkpoint(checkpoint_path_or_dir, layer_idx)

        self.q_proj0, self.q_proj1 = self.query_hash.get_proj_weights()
        self.G = self.QH // self.KH

        # Preallocated workspace for hamming_topk_v3
        n_blocks = (self.MaxT + 128 - 1) // 128
        self.block_packed = torch.empty((batch_size, self.KH, n_blocks, 128), dtype=torch.uint32, device=device)
        self.topk_out = torch.empty((batch_size, self.KH, self.top_budget), dtype=torch.int64, device=device)

        self.async_retrieve_stream = torch.cuda.Stream()
        self.async_update_stream = torch.cuda.Stream()
        self.current_stream = torch.cuda.current_stream()

        self._task_queue = SimpleQueue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self._init_retrieve_graph(dtype, device)
        self._init_update_graph(dtype, device)


    @torch.inference_mode()
    def _init_retrieve_graph(self, _dtype, device):
        self.static_query = torch.zeros((self.B, 1, self.QH, self.dims[0]), dtype=torch.bfloat16, device=device)
        self.static_num_tokens = torch.tensor(self.MaxT, dtype=torch.int32, device=device)
        self.graph = torch.cuda.CUDAGraph()
        self._async_retrieve_worker(self.static_query, self.static_num_tokens)
        with torch.cuda.graph(self.graph, stream=self.async_retrieve_stream):
            self._async_retrieve_worker(self.static_query, self.static_num_tokens)


    @torch.inference_mode()
    def _init_update_graph(self, dtype, device):
        self.static_key = torch.zeros((self.B, 1, self.KH, self.dims[0]), dtype=dtype, device=device)
        self.static_value = torch.zeros((self.B, 1, self.KH, self.dims[0]), dtype=dtype, device=device)
        self.static_start = torch.zeros((1,), dtype=torch.long, device=device)
        self.update_graph = torch.cuda.CUDAGraph()
        self._async_update_worker(self.static_key, self.static_value, self.static_start)
        with torch.cuda.graph(self.update_graph, stream=self.async_update_stream):
            self._async_update_worker(self.static_key, self.static_value, self.static_start)


    def get_seq_length(self):
        return self.num_tokens


    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.key_bins.zero_()

        self.num_tokens = 0
        self.is_prefilled = False
        self.key_lru.zero_()
        self.value_lru.zero_()

        self.lru_indices.fill_(-1)
        self.lru_timestamps.zero_()
        self.current_time.zero_()


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
        q_packed = qhash_fused_v2.qhash_project_pack_bf16(
            query,
            self.q_proj0,
            self.q_proj1,
            self.KH,
            self.G)
        hamming_topk_v3.hamming_topk_from_qpacked_cfg_out(
            q_packed,
            self.key_bins,
            current_num_tokens,
            self.block_packed,
            self.topk_out,
            self.top_budget,
            128, 256)
        lru_cache_update.update(
            self.topk_out,
            self.lru_indices,
            self.lru_timestamps,
            self.current_time,
            self.key_cache,
            self.value_cache,
            self.key_lru,
            self.value_lru,
            self.lru_budget,
            self.top_budget)


    @torch.inference_mode()
    def _async_update_worker(self, key, value, start):
        self.key_cache.index_copy_(1, start, key)
        self.value_cache.index_copy_(1, start, value)
        k_bins = self.key_hash.fused_packbits(key)
        k_bins = k_bins.permute(0, 2, 1, 3)
        self.key_bins.index_copy_(2, start, k_bins)


    @torch.inference_mode()
    def _prefill_lru_cache(self):
        s = self.num_tokens - self.lru_size
        self.key_lru.copy_(self.key_cache[:, s:self.num_tokens])
        self.value_lru.copy_(self.value_cache[:, s:self.num_tokens])
        indices = torch.arange(s, self.num_tokens, dtype=torch.int32, device=self.device)
        self.lru_indices.copy_(indices[None, None, :].expand(self.B, self.KH, -1))

        if self.saved_queries is not None:
            num_tokens_t = torch.tensor(self.num_tokens, dtype=torch.int32, device=self.device)
            for i in range(self.saved_queries.shape[1]):
                query = self.saved_queries[:, i:i+1].contiguous()
                self._async_retrieve_worker(query, num_tokens_t)
            self.saved_queries = None


    def _worker_loop(self):
        while True:
            fn, args = self._task_queue.get()
            fn(*args)

    @torch.inference_mode()
    def _launch_retrieve(self, query, num_tokens):
        self.async_retrieve_stream.wait_stream(self.current_stream)
        with torch.cuda.stream(self.async_retrieve_stream):
            self.static_query.copy_(query, non_blocking=True)
            self.static_num_tokens.fill_(num_tokens)
            self.graph.replay()


    @torch.inference_mode()
    def _launch_update(self, keys, values, start):
        self.async_update_stream.wait_stream(self.current_stream)
        with torch.cuda.stream(self.async_update_stream):
            self.static_key.copy_(keys, non_blocking=True)
            self.static_value.copy_(values, non_blocking=True)
            self.static_start.fill_(start)
            self.update_graph.replay()


    @torch.inference_mode()
    def update_query(self, query):
        if query.shape[1] == 1:
            self._task_queue.put((self._launch_retrieve, (query, self.num_tokens)))
        else:
            lastn = min(self.warmup_steps, query.shape[1])
            self.saved_queries = query[:, -lastn:].clone()


    @torch.inference_mode()
    def update(self, keys, values):
        T = keys.shape[1]
        start = self.num_tokens
        end = start + T
        self.num_tokens = end
        if self.num_tokens > self.MaxT:
            raise ValueError(f"Cached KV length {self.num_tokens} is greater than max position embeddings {self.MaxT}.")

        if T == 1 and self.num_tokens > self.lru_size:
            # decoding
            self.key_lru[:, -1:].copy_(keys)
            self.value_lru[:, -1:].copy_(values)
            self.lru_indices[:, :, -1].fill_(start)

            self._task_queue.put((self._launch_update, (keys, values, start)))

            self.current_stream.wait_stream(self.async_retrieve_stream)
            return self.key_lru, self.value_lru
        else:
            # prefilling
            self.key_cache[:, start:end].copy_(keys, non_blocking=True)
            self.value_cache[:, start:end].copy_(values, non_blocking=True)

            key_bins = self.key_hash.fused_packbits(keys)            # [B, T, KH, D_PACK]
            key_bins = key_bins.permute(0, 2, 1, 3).contiguous()     # [B, KH, T, D_PACK]
            self.key_bins[:, :, start:end].copy_(key_bins, non_blocking=True)

            if not self.is_prefilled and self.num_tokens >= self.lru_size:
                self._prefill_lru_cache()
                self.is_prefilled = True

            return self.key_cache[:, :end], self.value_cache[:, :end]


class TopkCache:
    def __init__(self,
            checkpoint_path_or_dir,
            layer_idx,
            batch_size,
            max_position_embeddings,
            num_attention_heads,
            num_key_value_heads,
            hash_module_dims,
            top_budget=1024,
            warmup_steps=32,
            device='cuda'):

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
        self.key_bins = torch.zeros((self.B, self.KH, self.MaxT, num_int32), dtype=torch.int32, device=device)

        self.num_tokens = 0
        self.is_prefilled = False
        self.saved_queries = None
        self.warmup_steps = warmup_steps
        self.top_budget = int(top_budget)

        self.sparse_keys = torch.zeros((batch_size, self.top_budget, self.KH, self.dims[0]), dtype=dtype, device=device)
        self.sparse_values = torch.zeros((batch_size, self.top_budget, self.KH, self.dims[0]), dtype=dtype, device=device)

        self.load_checkpoint(checkpoint_path_or_dir, layer_idx)

        self.q_proj0, self.q_proj1 = self.query_hash.get_proj_weights()
        self.G = self.QH // self.KH

        n_blocks = (self.MaxT + 128 - 1) // 128
        self.block_packed = torch.empty((batch_size, self.KH, n_blocks, 128), dtype=torch.uint32, device=device)
        self.topk_out = torch.empty((batch_size, self.KH, self.top_budget), dtype=torch.int64, device=device)

        # Preallocated index buffer for torch.gather (kept contiguous for graph capture).
        self.gather_idx = torch.empty(
            (batch_size, self.top_budget, self.KH, self.dims[0]),
            dtype=torch.int64, device=device)

        self.async_retrieve_stream = torch.cuda.Stream()
        self.async_update_stream = torch.cuda.Stream()
        self.current_stream = torch.cuda.current_stream()

        self._task_queue = SimpleQueue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self._init_retrieve_graph(dtype, device)
        self._init_update_graph(dtype, device)


    @torch.inference_mode()
    def _init_retrieve_graph(self, _dtype, device):
        self.static_query = torch.zeros((self.B, 1, self.QH, self.dims[0]), dtype=torch.bfloat16, device=device)
        self.static_num_tokens = torch.tensor(self.MaxT, dtype=torch.int32, device=device)
        self.graph = torch.cuda.CUDAGraph()
        self._async_retrieve_worker(self.static_query, self.static_num_tokens)
        with torch.cuda.graph(self.graph, stream=self.async_retrieve_stream):
            self._async_retrieve_worker(self.static_query, self.static_num_tokens)


    @torch.inference_mode()
    def _init_update_graph(self, dtype, device):
        self.static_key = torch.zeros((self.B, 1, self.KH, self.dims[0]), dtype=dtype, device=device)
        self.static_value = torch.zeros((self.B, 1, self.KH, self.dims[0]), dtype=dtype, device=device)
        self.static_start = torch.zeros((1,), dtype=torch.long, device=device)
        self.update_graph = torch.cuda.CUDAGraph()
        self._async_update_worker(self.static_key, self.static_value, self.static_start)
        with torch.cuda.graph(self.update_graph, stream=self.async_update_stream):
            self._async_update_worker(self.static_key, self.static_value, self.static_start)


    def get_seq_length(self):
        return self.num_tokens


    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.key_bins.zero_()
        self.sparse_keys.zero_()
        self.sparse_values.zero_()
        self.num_tokens = 0
        self.is_prefilled = False


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
        q_packed = qhash_fused_v2.qhash_project_pack_bf16(
            query,
            self.q_proj0,
            self.q_proj1,
            self.KH,
            self.G)
        hamming_topk_v3.hamming_topk_from_qpacked_cfg_out(
            q_packed,
            self.key_bins,
            current_num_tokens,
            self.block_packed,
            self.topk_out,
            self.top_budget,
            128, 256)
        # topk_out: [B, KH, top_budget] int64 -> broadcast to [B, top_budget, KH, D]
        D = self.dims[0]
        idx_view = self.topk_out.permute(0, 2, 1).unsqueeze(-1).expand(self.B, self.top_budget, self.KH, D)
        self.gather_idx.copy_(idx_view)
        torch.gather(self.key_cache, dim=1, index=self.gather_idx, out=self.sparse_keys)
        torch.gather(self.value_cache, dim=1, index=self.gather_idx, out=self.sparse_values)


    @torch.inference_mode()
    def _async_update_worker(self, key, value, start):
        self.key_cache.index_copy_(1, start, key)
        self.value_cache.index_copy_(1, start, value)
        k_bins = self.key_hash.fused_packbits(key)
        k_bins = k_bins.permute(0, 2, 1, 3)
        self.key_bins.index_copy_(2, start, k_bins)


    @torch.inference_mode()
    def _prefill_topk_warmup(self):
        if self.saved_queries is not None:
            num_tokens_t = torch.tensor(self.num_tokens, dtype=torch.int32, device=self.device)
            for i in range(self.saved_queries.shape[1]):
                query = self.saved_queries[:, i:i+1].contiguous()
                self._async_retrieve_worker(query, num_tokens_t)
            self.saved_queries = None


    def _worker_loop(self):
        while True:
            fn, args = self._task_queue.get()
            fn(*args)


    @torch.inference_mode()
    def _launch_retrieve(self, query, num_tokens):
        self.async_retrieve_stream.wait_stream(self.current_stream)
        with torch.cuda.stream(self.async_retrieve_stream):
            self.static_query.copy_(query, non_blocking=True)
            self.static_num_tokens.fill_(num_tokens)
            self.graph.replay()


    @torch.inference_mode()
    def _launch_update(self, keys, values, start):
        self.async_update_stream.wait_stream(self.current_stream)
        with torch.cuda.stream(self.async_update_stream):
            self.static_key.copy_(keys, non_blocking=True)
            self.static_value.copy_(values, non_blocking=True)
            self.static_start.fill_(start)
            self.update_graph.replay()


    @torch.inference_mode()
    def update_query(self, query):
        if query.shape[1] == 1:
            self._task_queue.put((self._launch_retrieve, (query, self.num_tokens)))
        else:
            lastn = min(self.warmup_steps, query.shape[1])
            self.saved_queries = query[:, -lastn:].clone()


    @torch.inference_mode()
    def update(self, keys, values):
        T = keys.shape[1]
        start = self.num_tokens
        end = start + T
        self.num_tokens = end
        if self.num_tokens > self.MaxT:
            raise ValueError(f"Cached KV length {self.num_tokens} is greater than max position embeddings {self.MaxT}.")

        if T == 1 and self.num_tokens > self.top_budget:
            # decoding: write new K/V into the full cache (async); the in-flight
            # retrieve has already populated sparse_keys/sparse_values via gather.
            self._task_queue.put((self._launch_update, (keys, values, start)))
            self.current_stream.wait_stream(self.async_retrieve_stream)
            return self.sparse_keys, self.sparse_values
        else:
            # prefilling
            self.key_cache[:, start:end].copy_(keys, non_blocking=True)
            self.value_cache[:, start:end].copy_(values, non_blocking=True)

            key_bins = self.key_hash.fused_packbits(keys)
            key_bins = key_bins.permute(0, 2, 1, 3).contiguous()
            self.key_bins[:, :, start:end].copy_(key_bins, non_blocking=True)

            if not self.is_prefilled and self.num_tokens >= self.top_budget:
                self._prefill_topk_warmup()
                self.is_prefilled = True

            return self.key_cache[:, :end], self.value_cache[:, :end]
