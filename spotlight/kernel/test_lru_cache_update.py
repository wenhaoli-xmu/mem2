import torch
from spotlight.kernel import lru_cache_update

def test_lru_kernel():
    B = 1
    KH = 1
    D = 128
    MaxT = 8192
    
    top_budget = 511
    lru_size = 512
    lru_budget = lru_size - 1 # 511
    
    # 模拟输入参数
    # 当前序列总长度8192，step=8192，意味着对前8191个token使用topk (indices 0 to 8190)
    # 我们随机生成topk_indices (在 0 到 8190 之间选取 511 个互不相同的 indices)
    topk_indices = torch.randperm(8191)[:top_budget].unsqueeze(0).unsqueeze(0).cuda() # [1, 1, 511]
    
    # 当前lru cache里面的indices都是-1（全空）
    # lru_indices的shape是 [B, KH, lru_size] -> [1, 1, 512]
    lru_indices = torch.full((B, KH, lru_size), -1, dtype=torch.int32, device='cuda')
    
    # 根据模型逻辑，最后一个token可能是人工替换进去的（取决于外部逻辑），但在这里我们全保持-1或者把最后一个设为8191
    # 题目要求检查：除掉lru cache的最后一个token，其余511个token是否都在topk中
    lru_indices[:, :, -1] = 8191 # 当前step是8192（index=8191的token放最后一个位置）
    
    lru_timestamps = torch.zeros((B, KH, lru_size), dtype=torch.int32, device='cuda')
    current_time = torch.zeros((B, KH), dtype=torch.int32, device='cuda')
    
    global_k = torch.randn((B, MaxT, KH, D), dtype=torch.bfloat16, device='cuda')
    global_v = torch.randn((B, MaxT, KH, D), dtype=torch.bfloat16, device='cuda')
    
    cache_k = torch.zeros((B, lru_size, KH, D), dtype=torch.bfloat16, device='cuda')
    cache_v = torch.zeros((B, lru_size, KH, D), dtype=torch.bfloat16, device='cuda')
    
    # 执行kernel
    lru_cache_update.update(
        topk_indices,     # [B, KH, top_budget]
        lru_indices,      # [B, KH, lru_size]
        lru_timestamps,   # [B, KH, lru_size]
        current_time,     # [B, KH]
        global_k,
        global_v,
        cache_k,
        cache_v,
        lru_budget,       # cache_size 参数为 lru_budget=511
        top_budget        # top_budget 参数为 511
    )
    
    # 验证
    retrieved_indices = topk_indices[0, 0].cpu().numpy().tolist()
    lru_indices_list = lru_indices[0, 0].cpu().numpy().tolist()
    
    # 确保除掉lru cache的最后一个token，其余511个token都在对前8191个token使用topk retrieve出来的token集合里。
    lru_subset = lru_indices_list[:-1] # 前 511 个位置
    
    retrieved_set = set(retrieved_indices)
    lru_subset_set = set(lru_subset)
    
    print("Retrieved topk size:", len(retrieved_set))
    print("LRU subset size (excluding last):", len(lru_subset_set))
    print("Is LRU subset exactly the topk retrieved set?", retrieved_set == lru_subset_set)
    
    missing_in_lru = retrieved_set - lru_subset_set
    extra_in_lru = lru_subset_set - retrieved_set
    
    if missing_in_lru:
        print("Tokens in topk but not in LRU:", missing_in_lru)
    if extra_in_lru:
        print("Tokens in LRU but not in topk:", extra_in_lru)
        
    assert retrieved_set == lru_subset_set, "Mismatch between retrieved top-k and LRU cache!"
    print("Test passed successfully!")

if __name__ == "__main__":
    test_lru_kernel()
