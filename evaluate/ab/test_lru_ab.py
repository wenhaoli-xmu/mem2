"""
A/B test: Compare LRU cache behavior between
  A) Python simulation that faithfully reproduces the CUDA kernel logic
  B) CUDA kernel (lru_cache_update.update from hash_utils.py)

Uses random topk indices on a single layer to verify both produce
identical cache index sets and hit rates at every decoding step.

Key: The CUDA kernel processes all topk candidates in PARALLEL against
the pre-existing cache state. Hits update timestamps; misses evict
the oldest entries. Topk indices are deduplicated to avoid ambiguity
from non-deterministic parallel duplicate handling.
"""

import torch
import numpy as np
from mem2.kernel import lru_cache_update


class PythonLRUCache:
    """
    Python simulation that faithfully mirrors the CUDA kernel behavior.
    
    State:
        indices: list of length cache_size, each entry is a token index or -1
        timestamps: list of length cache_size, each entry is a timestamp
        current_time: integer, monotonically increasing time counter
    """
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.indices = [-1] * cache_size
        self.timestamps = [0] * cache_size
        self.current_time = 0

    def update(self, topk: list[int]):
        """
        Process a batch of topk candidates, exactly mirroring the CUDA kernel:
        
        Phase 1: Build lookup table from existing indices
        Phase 2: Classify each candidate as hit or miss (against pre-existing state)
        Phase 3: For hits, assign new timestamps; for misses, find victims  
        Phase 4: Replace victims with miss indices + timestamps
        
        Returns:
            hit_rate: float
        """
        # Phase 1: Build lookup (index -> position in cache)
        lookup = {}
        for pos, idx in enumerate(self.indices):
            if idx >= 0:
                lookup[idx] = pos

        # Phase 2: Classify hits and misses (parallel in CUDA, sequential here)
        hits = []      # list of cache_pos
        misses = []    # list of token_idx

        for candidate in topk:
            candidate = int(candidate)
            if candidate < 0:
                continue
            if candidate in lookup:
                hits.append(lookup[candidate])
            else:
                misses.append(candidate)

        # Hit rate
        total = len(hits) + len(misses)
        hit_rate = len(hits) / total if total > 0 else 0.0

        # Phase 3a: Update timestamps for hits
        if hits:
            base_time = self.current_time
            self.current_time += len(hits)
            for rank, cache_pos in enumerate(hits):
                self.timestamps[cache_pos] = base_time + rank + 1

        # Phase 3b: Find victims for misses
        if misses:
            insert_base_time = self.current_time + 1
            self.current_time += len(misses)

            # Build effective timestamp list (AFTER hit updates)
            ts_cache = []
            for pos in range(self.cache_size):
                idx = self.indices[pos]
                ts = self.timestamps[pos]
                if idx < 0:
                    ts = -1  # prioritize empty slots
                ts_cache.append(ts)

            victim_positions = []
            for m in range(len(misses)):
                min_ts = 0x7FFFFFFF
                min_pos = -1
                for pos in range(self.cache_size):
                    if ts_cache[pos] != 0x7FFFFFFF:
                        if ts_cache[pos] < min_ts:
                            min_ts = ts_cache[pos]
                            min_pos = pos
                victim_positions.append(min_pos)
                if min_pos >= 0:
                    ts_cache[min_pos] = 0x7FFFFFFF

            # Phase 4: Apply updates
            for i, victim_pos in enumerate(victim_positions):
                if victim_pos >= 0:
                    self.indices[victim_pos] = misses[i]
                    self.timestamps[victim_pos] = insert_base_time + i

        return hit_rate

    def get_index_set(self):
        return set(idx for idx in self.indices if idx >= 0)


def generate_unique_topk(q, B, KH, top_budget):
    """Generate topk indices with no duplicates within each (b, h, q) slice."""
    result = np.full((B, KH, top_budget), -1, dtype=np.int64)
    if q == 0:
        return result
    for b in range(B):
        for h in range(KH):
            if q <= top_budget:
                # Not enough unique candidates, sample all available + pad
                chosen = np.arange(q)
                padded = np.full(top_budget, -1, dtype=np.int64)
                padded[:len(chosen)] = chosen
                np.random.shuffle(padded[:len(chosen)])
                result[b, h] = padded
            else:
                result[b, h] = np.random.choice(q, size=top_budget, replace=False)
    return result


def test_lru_ab():
    # ---- Parameters ----
    B = 1           # batch size
    KH = 2          # num kv heads
    S = 200         # sequence length (num query positions to simulate)
    top_budget = 10
    lru_size = 20   # total cache slots
    D = 128         # head dimension (for key/value data)
    MaxT = S        # max position embeddings
    dtype = torch.bfloat16
    device = 'cuda'

    lru_budget = lru_size - 1  # one slot reserved for current token

    torch.manual_seed(42)
    np.random.seed(42)

    # ---- Generate unique random topk indices [B, KH, S, top_budget] ----
    topk_all = np.full((B, KH, S, top_budget), -1, dtype=np.int64)
    for q in range(1, S):
        topk_all[:, :, q, :] = generate_unique_topk(q, B, KH, top_budget)

    # ---- Generate random key/value cache data ----
    key_cache = torch.randn(B, MaxT, KH, D, dtype=dtype, device=device)
    value_cache = torch.randn(B, MaxT, KH, D, dtype=dtype, device=device)

    # ============================================================
    # Method A: Python LRU simulation (matching CUDA kernel logic)
    # ============================================================
    py_caches = [[PythonLRUCache(lru_budget) for _ in range(KH)] for _ in range(B)]
    py_index_sets = []
    py_hit_rates = np.zeros((B, KH, S), dtype=np.float32)

    for q in range(S):
        step_sets = [[set() for _ in range(KH)] for _ in range(B)]
        for b in range(B):
            for h in range(KH):
                topk_step = topk_all[b, h, q].copy()
                topk_step[topk_step >= q] = -1
                hr = py_caches[b][h].update(topk_step.tolist())
                py_hit_rates[b, h, q] = hr
                step_sets[b][h] = py_caches[b][h].get_index_set()
        py_index_sets.append(step_sets)

    # ============================================================
    # Method B: CUDA kernel LRU
    # ============================================================
    cuda_lru_indices = torch.full((B, KH, lru_size), -1, dtype=torch.int32, device=device)
    cuda_lru_timestamps = torch.zeros((B, KH, lru_size), dtype=torch.int32, device=device)
    cuda_current_time = torch.zeros((B, KH), dtype=torch.int32, device=device)
    cuda_key_lru = torch.zeros((B, lru_size, KH, D), dtype=dtype, device=device)
    cuda_value_lru = torch.zeros((B, lru_size, KH, D), dtype=dtype, device=device)

    cuda_index_sets = []
    cuda_hit_rates = np.zeros((B, KH, S), dtype=np.float32)

    for q in range(S):
        topk_step = torch.from_numpy(topk_all[:, :, q, :].copy()).to(device=device, dtype=torch.int64)
        topk_step[topk_step >= q] = -1

        hit_rates_tensor = lru_cache_update.update(
            topk_step,
            cuda_lru_indices,
            cuda_lru_timestamps,
            cuda_current_time,
            key_cache,
            value_cache,
            cuda_key_lru,
            cuda_value_lru,
            lru_budget,
            top_budget
        )
        torch.cuda.synchronize()

        cuda_hit_rates[:, :, q] = hit_rates_tensor.cpu().numpy()

        step_sets = [[set() for _ in range(KH)] for _ in range(B)]
        lru_idx_cpu = cuda_lru_indices.cpu().numpy()
        for b in range(B):
            for h in range(KH):
                step_sets[b][h] = set(int(x) for x in lru_idx_cpu[b, h] if x >= 0)
        cuda_index_sets.append(step_sets)

    # ============================================================
    # Compare Results
    # ============================================================
    index_mismatches = 0
    hit_rate_mismatches = 0
    total_checks = 0

    for q in range(S):
        for b in range(B):
            for h in range(KH):
                total_checks += 1
                py_set = py_index_sets[q][b][h]
                cuda_set = cuda_index_sets[q][b][h]

                if py_set != cuda_set:
                    index_mismatches += 1
                    if index_mismatches <= 5:
                        print(f"  [INDEX MISMATCH] q={q}, b={b}, h={h}")
                        print(f"    Python ({len(py_set)}):  {sorted(py_set)}")
                        print(f"    CUDA   ({len(cuda_set)}):  {sorted(cuda_set)}")
                        only_py = py_set - cuda_set
                        only_cuda = cuda_set - py_set
                        if only_py:
                            print(f"    Only in Python: {sorted(only_py)}")
                        if only_cuda:
                            print(f"    Only in CUDA:   {sorted(only_cuda)}")

                py_hr = py_hit_rates[b, h, q]
                cuda_hr = cuda_hit_rates[b, h, q]
                if abs(py_hr - cuda_hr) > 1e-5:
                    hit_rate_mismatches += 1
                    if hit_rate_mismatches <= 5:
                        print(f"  [HIT RATE MISMATCH] q={q}, b={b}, h={h}: "
                              f"Python={py_hr:.4f}, CUDA={cuda_hr:.4f}")

    print("=" * 60)
    print(f"Total checks: {total_checks}")
    print(f"Index set mismatches:  {index_mismatches}")
    print(f"Hit rate mismatches:   {hit_rate_mismatches}")
    if index_mismatches == 0 and hit_rate_mismatches == 0:
        print("✅ PASS: CUDA kernel and Python simulation are fully consistent!")
    else:
        print("❌ FAIL: Inconsistencies detected.")
    print("=" * 60)

    # K/V data consistency check
    print("\nVerifying K/V data consistency (spot check)...")
    cuda_key_lru_cpu = cuda_key_lru.float().cpu()
    key_cache_cpu = key_cache.float().cpu()
    kv_mismatches = 0
    kv_checks = 0
    lru_idx_cpu = cuda_lru_indices.cpu().numpy()
    for b in range(B):
        for h in range(KH):
            for slot in range(lru_budget):
                token_idx = int(lru_idx_cpu[b, h, slot])
                if token_idx >= 0:
                    kv_checks += 1
                    cached_k = cuda_key_lru_cpu[b, slot, h]
                    original_k = key_cache_cpu[b, token_idx, h]
                    if not torch.allclose(cached_k, original_k, atol=1e-3):
                        kv_mismatches += 1
                        if kv_mismatches <= 3:
                            print(f"  [K/V MISMATCH] b={b}, h={h}, slot={slot}, "
                                  f"token={token_idx}")
    print(f"K/V spot checks: {kv_checks}, mismatches: {kv_mismatches}")
    if kv_mismatches == 0:
        print("✅ K/V data is consistent!")
    else:
        print("❌ K/V data has mismatches!")


if __name__ == "__main__":
    test_lru_ab()
