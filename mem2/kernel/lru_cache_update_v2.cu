#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

// ============================================================
// lru_cache_update_v2: Optimized over v1
//   1. Phase 3: Bitonic sort O(cache * log²cache) replaces
//      serial O(num_misses * cache) victim finding
//   2. Phase 4: Parallel metadata update (all threads, not tid==0)
// ============================================================

#define BF16X8_WRITE(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define BF16X8_READ(pointer)  (reinterpret_cast<const float4*>(&(pointer))[0])

constexpr int EMPTY_SLOT    = -1;
constexpr int INVALID_TIME  = 0x7FFFFFFF;

__device__ __forceinline__ int hash_func(int key, int mask) {
    return ((unsigned int)key * 2654435761u) & mask;
}

// -----------------------------------------------------------------
// Bitonic compare-and-swap (ascending sort by key)
// -----------------------------------------------------------------
__device__ __forceinline__ void bitonic_cas(
    int* keys, int* vals, int i, int j, bool ascending
) {
    int ki = keys[i], kj = keys[j];
    int vi = vals[i], vj = vals[j];
    bool swap = ascending ? (ki > kj) : (ki < kj);
    if (swap) {
        keys[i] = kj; keys[j] = ki;
        vals[i] = vj; vals[j] = vi;
    }
}

template <int HASH_TABLE_SIZE>
__global__ void optimized_lru_update_v2_kernel(
    const int64_t*      __restrict__ topk_indices,   // [B, KH, top_budget]
    int*                __restrict__ lru_indices,     // [B, KH, allocated_size]
    int*                __restrict__ lru_timestamps,  // [B, KH, allocated_size]
    int*                __restrict__ current_time,    // [B, KH]
    const __nv_bfloat16* __restrict__ global_k,
    const __nv_bfloat16* __restrict__ global_v,
    __nv_bfloat16*       __restrict__ cache_k,
    __nv_bfloat16*       __restrict__ cache_v,
    float*               __restrict__ hit_rates,
    const int B, const int KH,
    const int cache_size,        // active LRU slots
    const int allocated_size,    // total allocated (>= cache_size)
    const int top_budget,
    const int D, const int MaxT
) {
    // -----------------------------------------------------------
    // 0. Setup
    // -----------------------------------------------------------
    constexpr int HASH_MASK = HASH_TABLE_SIZE - 1;
    int b        = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid      = threadIdx.x;
    int batch_head_idx = b * KH + head_idx;

    extern __shared__ char shared_mem[];

    // ---- Shared memory layout (Phase 1-2) ----
    // [hash_table: HTS] [hash_values: HTS] [miss_indices: top_budget]
    //  [miss_count: 1] [hit_count: 1] [global_base_time: 1]
    int* hash_table  = reinterpret_cast<int*>(shared_mem);
    int* hash_values = hash_table + HASH_TABLE_SIZE;
    int* s_miss_indices = hash_values + HASH_TABLE_SIZE;
    int* s_miss_count      = s_miss_indices + top_budget;
    int* s_hit_count       = s_miss_count + 1;
    int* s_global_base_time = s_hit_count + 1;

    // Initialize counters
    if (tid == 0) {
        *s_miss_count = 0;
        *s_hit_count  = 0;
        *s_global_base_time = 0;
    }

    // Clear hash table
    for (int i = tid; i < HASH_TABLE_SIZE; i += blockDim.x) {
        hash_table[i]  = EMPTY_SLOT;
        hash_values[i] = EMPTY_SLOT;
    }
    __syncthreads();

    int* g_lru_idx = lru_indices    + (batch_head_idx * allocated_size);
    int* g_lru_ts  = lru_timestamps + (batch_head_idx * allocated_size);

    // -----------------------------------------------------------
    // Phase 1: Build hash table from current LRU cache
    // -----------------------------------------------------------
    for (int i = tid; i < cache_size; i += blockDim.x) {
        int token_idx = g_lru_idx[i];
        if (token_idx >= 0) {
            int h = hash_func(token_idx, HASH_MASK);
            while (true) {
                int old = atomicCAS(&hash_values[h], EMPTY_SLOT, token_idx);
                if (old == EMPTY_SLOT || old == token_idx) {
                    if (old == EMPTY_SLOT) hash_table[h] = i;
                    break;
                }
                h = (h + 1) & HASH_MASK;
            }
        }
    }
    __syncthreads();

    // -----------------------------------------------------------
    // Phase 2: Classify topk as hit / miss
    // -----------------------------------------------------------
    const int64_t* my_topk = topk_indices + (batch_head_idx * top_budget);

    // Per-thread hit buffer (register)
    int my_hit_pos[32];
    int my_hit_rank[32];
    int my_n_hits = 0;

    for (int i = tid; i < top_budget; i += blockDim.x) {
        int cand = static_cast<int>(my_topk[i]);
        if (cand < 0) continue;

        int h = hash_func(cand, HASH_MASK);
        int cache_pos = -1;
        while (hash_values[h] != EMPTY_SLOT) {
            if (hash_values[h] == cand) { cache_pos = hash_table[h]; break; }
            h = (h + 1) & HASH_MASK;
        }

        if (cache_pos >= 0) {
            if (my_n_hits < 32) {
                my_hit_pos[my_n_hits]  = cache_pos;
                my_hit_rank[my_n_hits] = atomicAdd(s_hit_count, 1);
                my_n_hits++;
            }
        } else {
            int mi = atomicAdd(s_miss_count, 1);
            s_miss_indices[mi] = cand;
        }
    }
    __syncthreads();

    int num_misses = *s_miss_count;
    int num_hits   = *s_hit_count;

    // Report hit rate & allocate timestamps for hits
    if (tid == 0) {
        int total = num_hits + num_misses;
        hit_rates[batch_head_idx] = (total > 0) ? (float)num_hits / total : 0.0f;
        if (num_hits > 0) {
            *s_global_base_time = atomicAdd(&current_time[batch_head_idx], num_hits);
        }
    }
    __syncthreads();

    // Write hit timestamps to global memory (parallel)
    int base_time = *s_global_base_time;
    for (int k = 0; k < my_n_hits; k++) {
        g_lru_ts[my_hit_pos[k]] = base_time + my_hit_rank[k] + 1;
    }
    __syncthreads();

    if (num_misses == 0) return;

    // -----------------------------------------------------------
    // Phase 3: Batch victim finding via bitonic sort
    //   Sort (timestamp, position) pairs ascending by timestamp.
    //   The first num_misses entries are victims.
    //   Reuse hash_table / hash_values memory for sort arrays.
    // -----------------------------------------------------------
    __shared__ int s_insert_base_time;
    if (tid == 0) {
        s_insert_base_time = atomicAdd(&current_time[batch_head_idx], num_misses) + 1;
    }

    // Compute sort_N = smallest power-of-2 >= cache_size
    int sort_N = 1;
    while (sort_N < cache_size) sort_N <<= 1;

    // Reuse hash area (2 * HASH_TABLE_SIZE ints) for sort buffers
    // sort_ts and sort_pos each need sort_N ints
    // 2*sort_N <= 2*cache_size <= HASH_TABLE_SIZE <= 2*HASH_TABLE_SIZE  ✓
    int* sort_ts  = hash_table;            // reuse
    int* sort_pos = hash_table + sort_N;   // right after sort_ts

    // Load timestamps (with updated hit timestamps visible after sync)
    for (int i = tid; i < cache_size; i += blockDim.x) {
        int idx = g_lru_idx[i];
        int ts  = g_lru_ts[i];
        sort_ts[i]  = (idx < 0) ? INVALID_TIME : ts;
        sort_pos[i] = i;
    }
    // Pad to sort_N with sentinel values
    for (int i = cache_size + tid; i < sort_N; i += blockDim.x) {
        sort_ts[i]  = INVALID_TIME;
        sort_pos[i] = -1;
    }
    __syncthreads();

    // Bitonic sort — ascending by timestamp
    for (int size = 2; size <= sort_N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < sort_N; i += blockDim.x) {
                int j = i ^ stride;
                if (j > i) {
                    bool ascending = ((i & size) == 0);
                    bitonic_cas(sort_ts, sort_pos, i, j, ascending);
                }
            }
            __syncthreads();
        }
    }

    // sort_pos[0 .. num_misses-1] are the victim cache positions
    // (smallest timestamps = least recently used)

    // -----------------------------------------------------------
    // Phase 4: Update metadata — PARALLEL (all threads)
    // -----------------------------------------------------------
    __syncthreads();
    int ins_base = s_insert_base_time;

    for (int i = tid; i < num_misses; i += blockDim.x) {
        int victim_pos    = sort_pos[i];
        int new_token_idx = s_miss_indices[i];
        if (victim_pos >= 0) {
            g_lru_idx[victim_pos] = new_token_idx;
            g_lru_ts[victim_pos]  = ins_base + i;
        }
    }
    __syncthreads();

    // -----------------------------------------------------------
    // Phase 5: Vectorized K/V copy (BF16 × 8 per iteration)
    // -----------------------------------------------------------
    int D_vec = D / 8;
    int total_tasks = num_misses * D_vec;

    for (int task_id = tid; task_id < total_tasks; task_id += blockDim.x) {
        int miss_idx  = task_id / D_vec;
        int d_vec_idx = task_id % D_vec;
        int d_offset  = d_vec_idx * 8;

        int token_global = s_miss_indices[miss_idx];
        int write_pos    = sort_pos[miss_idx];   // read directly from sort result

        if (write_pos >= 0) {
            long src_off = ((long)b * MaxT * KH * D)
                         + ((long)token_global * KH * D)
                         + ((long)head_idx * D)
                         + d_offset;

            long dst_off = ((long)b * allocated_size * KH * D)
                         + ((long)write_pos * KH * D)
                         + ((long)head_idx * D)
                         + d_offset;

            float4 kv;
            kv = BF16X8_READ(global_k[src_off]);
            BF16X8_WRITE(cache_k[dst_off]) = kv;

            kv = BF16X8_READ(global_v[src_off]);
            BF16X8_WRITE(cache_v[dst_off]) = kv;
        }
    }
}

// -----------------------------------------------------------------
// Host wrapper
// -----------------------------------------------------------------
torch::Tensor lru_update_v2_cuda(
    torch::Tensor topk_indices,
    torch::Tensor lru_indices,
    torch::Tensor lru_timestamps,
    torch::Tensor current_time,
    torch::Tensor global_k,
    torch::Tensor global_v,
    torch::Tensor cache_k,
    torch::Tensor cache_v,
    int cache_size,
    int top_budget
) {
    TORCH_CHECK(global_k.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(cache_k.scalar_type()  == torch::kBFloat16);
    TORCH_CHECK(lru_timestamps.scalar_type() == torch::kInt32);

    const int B   = topk_indices.size(0);
    const int KH  = topk_indices.size(1);
    const int MaxT = global_k.size(1);
    const int D    = global_k.size(3);
    const int allocated_size = cache_k.size(1);

    // Choose hash table size (power of 2, >= 2 * cache_size)
    int hash_size = 1024;
    if      (cache_size * 2 > 8192) hash_size = 16384;
    else if (cache_size * 2 > 4096) hash_size = 8192;
    else if (cache_size * 2 > 2048) hash_size = 4096;
    else if (cache_size * 2 > 1024) hash_size = 2048;

    TORCH_CHECK(cache_size <= hash_size / 2, "cache_size too large for compiled hash sizes");

    auto hit_rates = torch::empty({B, KH}, topk_indices.options().dtype(torch::kFloat32));

    dim3 grid(B, KH);
    int threads = 256;

    // Shared memory:
    //   hash area  : 2 * hash_size * sizeof(int)   (reused for sort in Phase 3)
    //   miss array : top_budget * sizeof(int)
    //   counters   : 3 * sizeof(int)
    size_t smem_size = (hash_size * 2 * sizeof(int))
                     + (top_budget * sizeof(int))
                     + (128 * sizeof(int));   // counters + padding

    auto stream = at::cuda::getCurrentCUDAStream();

    if (smem_size > 48 * 1024) {
        int device;
        cudaGetDevice(&device);
        int max_smem;
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

        #define SET_SMEM(SIZE) \
            cudaFuncSetAttribute(optimized_lru_update_v2_kernel<SIZE>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        if ((size_t)max_smem >= smem_size) {
            switch(hash_size) {
                case 1024:  SET_SMEM(1024);  break;
                case 2048:  SET_SMEM(2048);  break;
                case 4096:  SET_SMEM(4096);  break;
                case 8192:  SET_SMEM(8192);  break;
                case 16384: SET_SMEM(16384); break;
            }
        }
        #undef SET_SMEM
    }

    #define LAUNCH(SIZE) \
        optimized_lru_update_v2_kernel<SIZE><<<grid, threads, smem_size, stream>>>( \
            topk_indices.data_ptr<int64_t>(), \
            lru_indices.data_ptr<int>(), \
            lru_timestamps.data_ptr<int>(), \
            current_time.data_ptr<int>(), \
            reinterpret_cast<const __nv_bfloat16*>(global_k.data_ptr<at::BFloat16>()), \
            reinterpret_cast<const __nv_bfloat16*>(global_v.data_ptr<at::BFloat16>()), \
            reinterpret_cast<__nv_bfloat16*>(cache_k.data_ptr<at::BFloat16>()), \
            reinterpret_cast<__nv_bfloat16*>(cache_v.data_ptr<at::BFloat16>()), \
            hit_rates.data_ptr<float>(), \
            B, KH, cache_size, allocated_size, top_budget, D, MaxT \
        );

    switch (hash_size) {
        case 1024:  LAUNCH(1024);  break;
        case 2048:  LAUNCH(2048);  break;
        case 4096:  LAUNCH(4096);  break;
        case 8192:  LAUNCH(8192);  break;
        case 16384: LAUNCH(16384); break;
        default: TORCH_CHECK(false, "Unsupported hash size");
    }
    #undef LAUNCH

    return hit_rates;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("update", &lru_update_v2_cuda, "LRU Cache Update v2 (bitonic sort victim finding)");
}
