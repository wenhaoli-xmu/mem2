import torch
import os
import torch.distributed as dist

from itertools import chain
from safetensors.torch import load_file

from spotlight.kernel import (
    fifo_cache_update,
    lru_cache_update,
    hash_packbits,
    hash_packbits_hamming
)

import queue
import threading

from transformers.cache_utils import DynamicCache, Cache


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
    def __init__(self, num_heads, dim_inp, dim_out, silu, dtype=torch.float, device='cpu'):
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
    def __init__(self, init=1.0, dtype=torch.float, device='cpu'):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.full((1, 1, 1, 1), init, dtype=dtype, device=device))

    def forward(self, x):
        return self.gain * x


class HashModule(torch.nn.Module):
    def __init__(self, num_heads, dims, dtype=torch.float, device='cpu'):
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



class FIFOCache:
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
            dtype=torch.bfloat16,
            device='cuda',
            async_retrieve=True,
            async_update=True):
        
        self.B = batch_size
        self.MaxT = max_position_embeddings
        self.QH = num_attention_heads
        self.KH = num_key_value_heads
        self.dims = hash_module_dims
        self.device = device
        self.layer_idx = layer_idx
        self.async_retrieve = async_retrieve
        self.async_update = async_update
        
        self.query_hash = HashModule(self.QH, self.dims, dtype=dtype, device=device)
        self.key_hash = HashModule(self.KH, self.dims, dtype=dtype, device=device)

        self.key_cache = torch.zeros((self.B, self.MaxT, self.KH, self.dims[0]), dtype=dtype, device=device)
        self.value_cache = torch.zeros((self.B, self.MaxT, self.KH, self.dims[0]), dtype=dtype, device=device)
        
        assert self.dims[-1] % 32 == 0, "hash_module_dims[-1] must be divisible by 32"
        num_int32 = self.dims[-1] // 32
        self.key_bins = torch.zeros((self.B, self.MaxT, self.KH, num_int32), dtype=torch.int32, device=device)
        
        self.num_tokens = 0
        self.is_prefilled = False

        self.lru_size = int(lru_budget)
        self.lru_budget = self.lru_size - 1
        self.top_budget = int(top_budget)
        
        self.key_lru = torch.zeros((batch_size, self.lru_size, self.KH, self.dims[0]), dtype=dtype, device=device)
        self.value_lru = torch.zeros((batch_size, self.lru_size, self.KH, self.dims[0]), dtype=dtype, device=device)
        
        self.lru_ptr = torch.zeros((batch_size, self.KH), dtype=torch.int32, device=device)
        self.lru_indices = torch.full((batch_size, self.KH, self.lru_size), -1, dtype=torch.int32, device=device)

        self.load_checkpoint(checkpoint_path_or_dir, layer_idx)

        # async streams
        self.async_retrieve_stream = torch.cuda.Stream() 
        self.async_update_stream = torch.cuda.Stream() 

        # cuda graph
        with torch.inference_mode():
            self.static_query = torch.zeros((self.B, 1, self.QH, self.dims[0]), dtype=dtype, device=device)
            self.static_num_tokens = torch.tensor(self.MaxT, dtype=torch.int32, device=device)
            self.graph = torch.cuda.CUDAGraph()
            self._async_retrieve_worker(self.static_query, self.static_num_tokens)
            with torch.cuda.graph(self.graph, stream=self.async_retrieve_stream):
                self._async_retrieve_worker(self.static_query, self.static_num_tokens)
                
            self.static_key = torch.zeros((self.B, 1, self.KH, self.dims[0]), dtype=dtype, device=device)
            self.static_value = torch.zeros((self.B, 1, self.KH, self.dims[0]), dtype=dtype, device=device)
            self.static_start = torch.zeros((1,), dtype=torch.long, device=device)
            self.update_graph = torch.cuda.CUDAGraph()
            self._async_update_worker(self.static_key, self.static_value, self.static_start)
            with torch.cuda.graph(self.update_graph, stream=self.async_update_stream):
                self._async_update_worker(self.static_key, self.static_value, self.static_start)


    def __del__(self):
        pass


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
        self.lru_ptr.zero_()
        self.lru_indices.fill_(-1)


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
        proj0, proj1 = self.query_hash.get_proj_weights()
        mask = hash_packbits_hamming.hash_packbits_hamming(
            query, proj0, proj1, self.key_bins, current_num_tokens)
        topk_indices = torch.topk(mask, k=self.top_budget).indices
        fifo_cache_update.update(
            topk_indices,
            self.lru_indices,
            self.lru_ptr,
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
        self.key_bins.view(torch.int32).index_copy_(1, start, k_bins.view(torch.int32))


    @torch.inference_mode()
    def _prefill_cache(self):
        start = self.num_tokens - self.lru_size
        end = self.num_tokens

        # copy key value states to lru cache
        self.key_lru.copy_(self.key_cache[:, start: end], non_blocking=True)
        self.value_lru.copy_(self.value_cache[:, start: end], non_blocking=True)

        # record lru indices
        indices = torch.arange(start, end, dtype=torch.int32, device=self.device)
        indices = indices[None, None, :].expand(self.B, self.KH, -1)
        self.lru_indices.copy_(indices, non_blocking=True)

        self.lru_ptr.zero_()


    def update_query(self, query):
        if query.shape[1] == 1:
            if self.async_retrieve:
                self.async_retrieve_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.async_retrieve_stream):
                    self.static_query.copy_(query, non_blocking=True)
                    self.static_num_tokens.fill_(self.num_tokens)
                    self.graph.replay()
            else:
                self.static_num_tokens.fill_(self.num_tokens)
                self._async_retrieve_worker(query, self.static_num_tokens)


    @torch.inference_mode()
    def update(self, keys, values):
        T = keys.shape[1]
        start = self.num_tokens
        end = start + T
        self.num_tokens = end
        
        if T == 1 and self.is_prefilled:
            
            self.key_lru[:, -1:].copy_(keys)
            self.value_lru[:, -1:].copy_(values)
            self.lru_indices[:, :, -1] = start
            
            if self.async_update:
                self.async_update_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.async_update_stream):
                    self.static_key.copy_(keys, non_blocking=True)
                    self.static_value.copy_(values, non_blocking=True)
                    self.static_start.fill_(start)
                    self.update_graph.replay()
            else:
                self.static_start.fill_(start)
                self._async_update_worker(keys, values, self.static_start)

            torch.cuda.current_stream().wait_stream(self.async_retrieve_stream)
            return self.key_lru, self.value_lru
        else:
            self.key_cache[:, start:end].copy_(keys, non_blocking=True)
            self.value_cache[:, start:end].copy_(values, non_blocking=True)

            key_bins = self.key_hash.fused_packbits(keys)
            self.key_bins[:, start:end].copy_(key_bins, non_blocking=True)

            if not self.is_prefilled and self.num_tokens >= self.lru_size:
                self._prefill_cache()
                self.is_prefilled = True
            ret_keys = self.key_cache[:, :self.num_tokens].contiguous()
            ret_values = self.value_cache[:, :self.num_tokens].contiguous()

            return ret_keys, ret_values


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
            dtype=torch.bfloat16,
            device='cuda',
            async_retrieve=True,
            async_update=False):
        
        self.B = batch_size
        self.MaxT = max_position_embeddings
        self.QH = num_attention_heads
        self.KH = num_key_value_heads
        self.dims = hash_module_dims
        self.device = device
        self.layer_idx = layer_idx
        self.async_retrieve = async_retrieve
        self.async_update = async_update
        
        self.query_hash = HashModule(self.QH, self.dims, dtype=dtype, device=device)
        self.key_hash = HashModule(self.KH, self.dims, dtype=dtype, device=device)

        self.key_cache = torch.zeros((self.B, self.MaxT, self.KH, self.dims[0]), dtype=dtype, device=device)
        self.value_cache = torch.zeros((self.B, self.MaxT, self.KH, self.dims[0]), dtype=dtype, device=device)
        
        assert self.dims[-1] % 32 == 0, "hash_module_dims[-1] must be divisible by 32"
        num_int32 = self.dims[-1] // 32
        self.key_bins = torch.zeros((self.B, self.MaxT, self.KH, num_int32), dtype=torch.int32, device=device)
        
        self.num_tokens = 0
        self.is_prefilled = False

        self.lru_size = int(lru_budget)
        self.lru_budget = self.lru_size - 1
        self.top_budget = int(top_budget)
        
        self.key_lru = torch.zeros((batch_size, self.lru_size, self.KH, self.dims[0]), dtype=dtype, device=device)
        self.value_lru = torch.zeros((batch_size, self.lru_size, self.KH, self.dims[0]), dtype=dtype, device=device)
        
        self.lru_indices = torch.full((batch_size, self.KH, self.lru_size), -1, dtype=torch.int32, device=device)
        self.lru_timestamps = torch.zeros((batch_size, num_key_value_heads, self.lru_size), dtype=torch.int32, device=device)
        self.current_time = torch.zeros((batch_size, num_key_value_heads), dtype=torch.int32, device=device)

        self.load_checkpoint(checkpoint_path_or_dir, layer_idx)

        # async streams
        self.async_retrieve_stream = torch.cuda.Stream() 
        self.async_update_stream = torch.cuda.Stream() 

        # cuda graph
        with torch.inference_mode():
            self.static_query = torch.zeros((self.B, 1, self.QH, self.dims[0]), dtype=dtype, device=device)
            self.static_num_tokens = torch.tensor(self.MaxT, dtype=torch.int32, device=device)
            self.graph = torch.cuda.CUDAGraph()
            self._async_retrieve_worker(self.static_query, self.static_num_tokens)
            with torch.cuda.graph(self.graph, stream=self.async_retrieve_stream):
                self._async_retrieve_worker(self.static_query, self.static_num_tokens)
                
            self.static_key = torch.zeros((self.B, 1, self.KH, self.dims[0]), dtype=dtype, device=device)
            self.static_value = torch.zeros((self.B, 1, self.KH, self.dims[0]), dtype=dtype, device=device)
            self.static_start = torch.zeros((1,), dtype=torch.long, device=device)
            self.update_graph = torch.cuda.CUDAGraph()
            self._async_update_worker(self.static_key, self.static_value, self.static_start)
            with torch.cuda.graph(self.update_graph, stream=self.async_update_stream):
                self._async_update_worker(self.static_key, self.static_value, self.static_start)


    def __del__(self):
        pass


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
        proj0, proj1 = self.query_hash.get_proj_weights()
        mask = hash_packbits_hamming.hash_packbits_hamming(
            query, proj0, proj1, self.key_bins, current_num_tokens)
        topk_indices = torch.topk(mask, k=self.top_budget).indices
        lru_cache_update.update(
            topk_indices,
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
        self.key_bins.index_copy_(1, start, k_bins)


    def update_query(self, query):
        if query.shape[1] == 1:
            if self.async_retrieve:
                self.async_retrieve_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.async_retrieve_stream):
                    self.static_query.copy_(query, non_blocking=True)
                    self.static_num_tokens.fill_(self.num_tokens)
                    self.graph.replay()
            else:
                self.static_num_tokens.fill_(self.num_tokens)
                self._async_retrieve_worker(query, self.static_num_tokens)


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
            self.lru_indices[:, :, -1] = start

            if self.async_update:
                self.async_update_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.async_update_stream):
                    self.static_key.copy_(keys, non_blocking=True)
                    self.static_value.copy_(values, non_blocking=True)
                    self.static_start.fill_(start)
                    self.update_graph.replay()
            else:
                self.static_start.fill_(start)
                self._async_update_worker(keys, values, self.static_start)

            torch.cuda.current_stream().wait_stream(self.async_retrieve_stream)
            return self.key_lru, self.value_lru
        else:
            # prefilling
            self.key_cache[:, start:end].copy_(keys, non_blocking=True)
            self.value_cache[:, start:end].copy_(values, non_blocking=True)

            key_bins = self.key_hash.fused_packbits(keys)
            self.key_bins[:, start:end].copy_(key_bins, non_blocking=True)

            ret_keys = self.key_cache[:, :end].contiguous()
            ret_values = self.value_cache[:, :end].contiguous()

            return ret_keys, ret_values
