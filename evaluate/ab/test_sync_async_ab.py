"""
A/B test: Compare LRU cache behavior between
  A) Synchronous mode  (async_retrieve=False, async_update=False)
  B) Asynchronous mode  (async_retrieve=True,  async_update=True)

Both LRUCache instances use identical hash weights, key/value data,
and the same query sequence. The test mimics the real inference loop
in attention_forward (hash_gen.py):
    1. update_query(query)    ← kicks off retrieval
    2. [compute K/V on main stream — simulated with a small sleep/compute]
    3. update(keys, values)   ← inserts into cache, waits for retrieve

If there is a CUDA stream race condition between async_retrieve_stream
and async_update_stream, the async version will produce:
  - Different lru_indices / key_lru / value_lru compared to sync
  - NaN values in key_lru or value_lru (garbage data → NaN in attention)
"""

import torch
import time
import numpy as np
from mem2.monkey_patches.hash_utils import LRUCache


def copy_hash_weights(src: LRUCache, dst: LRUCache):
    """Copy hash module weights from src to dst so both caches are identical."""
    for sp, dp in zip(src.query_hash.parameters(), dst.query_hash.parameters()):
        dp.data.copy_(sp.data)
    for sp, dp in zip(src.key_hash.parameters(), dst.key_hash.parameters()):
        dp.data.copy_(sp.data)


def create_cache(async_retrieve, async_update, B, MaxT, QH, KH, D, lru_budget, top_budget, dtype, device):
    """Create an LRUCache with no checkpoint (random hash weights)."""
    return LRUCache(
        checkpoint_dir=None,
        layer_idx=0,
        batch_size=B,
        max_position_embeddings=MaxT,
        num_attention_heads=QH,
        num_key_value_heads=KH,
        hash_module_dims=[D, D, D],
        lru_budget=lru_budget,
        top_budget=top_budget,
        dtype=dtype,
        device=device,
        async_retrieve=async_retrieve,
        async_update=async_update,
    )


@torch.inference_mode()
def run_decoding(cache: LRUCache, prefill_keys, prefill_values, 
                 decode_queries, decode_keys, decode_values, num_decode_steps):
    """
    Run prefill + decode through a cache, mimicking attention_forward's call pattern.
    Returns lists of (key_lru_snapshot, value_lru_snapshot, lru_indices_snapshot) per decode step.
    """
    snapshots = []

    # --- Prefill phase ---
    cache.update_query(prefill_keys)  # no-op for prefill (T > 1)
    cache.update(prefill_keys, prefill_values)
    torch.cuda.synchronize()

    # --- Decode phase ---
    for step in range(num_decode_steps):
        q = decode_queries[:, step:step+1]  # [B, 1, QH, D]
        k = decode_keys[:, step:step+1]     # [B, 1, KH, D]
        v = decode_values[:, step:step+1]   # [B, 1, KH, D]

        # Step 1: update_query (starts retrieve — async or sync)
        cache.update_query(q)

        # Step 2: simulate K/V projection compute on main stream
        # (In real inference, this is q_proj/k_proj/v_proj + RoPE)
        # A tiny matmul to occupy the default stream briefly
        _dummy = torch.randn(128, 128, device='cuda') @ torch.randn(128, 128, device='cuda')

        # Step 3: update (inserts K/V, waits for retrieve)
        ret_k, ret_v = cache.update(k, v)
        torch.cuda.synchronize()

        # Snapshot the current state
        snapshots.append({
            'key_lru': cache.key_lru.clone(),
            'value_lru': cache.value_lru.clone(),
            'lru_indices': cache.lru_indices.clone(),
            'lru_timestamps': cache.lru_timestamps.clone(),
            'ret_k': ret_k.clone(),
            'ret_v': ret_v.clone(),
        })

    return snapshots


def test_sync_async_ab():
    # ---- Parameters ----
    B = 1
    QH = 4          # num query heads
    KH = 2          # num kv heads
    D = 128         # head dim
    MaxT = 512
    lru_budget = 32
    top_budget = 8
    prefill_len = 64    # must be >= lru_budget to trigger is_prefilled
    num_decode_steps = 200
    dtype = torch.bfloat16
    device = 'cuda'

    torch.manual_seed(42)

    # ---- Create reference (fully sync) cache ----
    print("Creating reference sync cache...")
    ref_cache = create_cache(
        async_retrieve=False, async_update=False,
        B=B, MaxT=MaxT, QH=QH, KH=KH, D=D,
        lru_budget=lru_budget, top_budget=top_budget,
        dtype=dtype, device=device)

    # ---- Generate data (before creating other caches, so RNG is deterministic) ----
    prefill_keys = torch.randn(B, prefill_len, KH, D, dtype=dtype, device=device)
    prefill_values = torch.randn(B, prefill_len, KH, D, dtype=dtype, device=device)
    decode_queries = torch.randn(B, num_decode_steps, QH, D, dtype=dtype, device=device)
    decode_keys = torch.randn(B, num_decode_steps, KH, D, dtype=dtype, device=device)
    decode_values = torch.randn(B, num_decode_steps, KH, D, dtype=dtype, device=device)

    # ---- Test all 4 combinations ----
    configs = [
        ("sync_retrieve + sync_update",  False, False),
        ("ASYNC_retrieve + sync_update", True,  False),
        ("sync_retrieve + ASYNC_update", False, True),
        ("ASYNC_retrieve + ASYNC_update", True,  True),
    ]

    # Run reference first
    print(f"\nRunning reference (fully sync)...")
    import time
    t0 = time.time()
    ref_snapshots = run_decoding(
        ref_cache, prefill_keys.clone(), prefill_values.clone(),
        decode_queries, decode_keys, decode_values, num_decode_steps)
    ref_time = time.time() - t0
    print(f"  Time: {ref_time:.3f}s")

    for config_name, ar, au in configs[1:]:  # skip (False, False), it's the reference
        print(f"\nCreating cache: {config_name}...")
        test_cache = create_cache(
            async_retrieve=ar, async_update=au,
            B=B, MaxT=MaxT, QH=QH, KH=KH, D=D,
            lru_budget=lru_budget, top_budget=top_budget,
            dtype=dtype, device=device)
        copy_hash_weights(ref_cache, test_cache)
        torch.cuda.synchronize()

        t0 = time.time()
        test_snapshots = run_decoding(
            test_cache, prefill_keys.clone(), prefill_values.clone(),
            decode_queries, decode_keys, decode_values, num_decode_steps)
        test_time = time.time() - t0

        # Compare
        nan_count = 0
        idx_mismatch = 0
        ts_mismatch = 0
        kv_mismatch = 0
        mismatch_steps = []

        for step in range(num_decode_steps):
            rs = ref_snapshots[step]
            ts = test_snapshots[step]

            has_nan = (torch.isnan(ts['ret_k']).any().item() or 
                       torch.isnan(ts['ret_v']).any().item())
            if has_nan:
                nan_count += 1

            idx_diff = not torch.equal(rs['lru_indices'], ts['lru_indices'])
            ts_diff = not torch.equal(rs['lru_timestamps'], ts['lru_timestamps'])
            kv_diff = not torch.allclose(rs['key_lru'].float(), ts['key_lru'].float(), atol=1e-2)

            if idx_diff:
                idx_mismatch += 1
            if ts_diff:
                ts_mismatch += 1
            if kv_diff:
                kv_mismatch += 1
            if idx_diff or ts_diff or kv_diff or has_nan:
                mismatch_steps.append(step)

        print(f"  Time: {test_time:.3f}s (ref: {ref_time:.3f}s, speedup: {ref_time/test_time:.1f}x)")
        print(f"  NaN steps:       {nan_count}")
        print(f"  Index mismatch:  {idx_mismatch}")
        print(f"  TS mismatch:     {ts_mismatch}")
        print(f"  K/V mismatch:    {kv_mismatch}")
        if mismatch_steps:
            print(f"  Mismatch at steps: {mismatch_steps[:20]}{'...' if len(mismatch_steps) > 20 else ''}")
            # Print detail for first mismatch
            step = mismatch_steps[0]
            rs = ref_snapshots[step]
            ts = test_snapshots[step]
            for h in range(KH):
                ri = sorted(int(x) for x in rs['lru_indices'][0, h].cpu() if x >= 0)
                ti = sorted(int(x) for x in ts['lru_indices'][0, h].cpu() if x >= 0)
                if ri != ti:
                    print(f"    step={step}, h={h}: ref={ri[:10]}... test={ti[:10]}...")
                    break
        else:
            print(f"  ✅ PASS")

        # Clean up
        del test_cache, test_snapshots

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    test_sync_async_ab()

