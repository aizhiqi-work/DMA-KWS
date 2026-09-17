import torch
import numpy as np
from numba import njit
from KWStreamingSearch.base_search import KWSBaseSearch

# 定义常量
PH = -1e35 

# ================== 1. 改良版灵活内核 (支持 Inplace 与 全记录切换) ==================
@njit(cache=True)
def _numba_inplace_flexible_kernel(post, full_tgt_ids, blank, prune_threshold, ph_val, U_phi, need_full_matrix):
    T, V = post.shape
    
    # --- 基础状态变量 (原地更新所需) ---
    prev_log_alpha = np.full(U_phi, -1e35, dtype=np.float32)
    prev_start_alpha = np.full(U_phi, -1, dtype=np.int64)
    prev_total_alpha = np.full(U_phi, 0, dtype=np.int64)
    
    curr_log_alpha = np.empty(U_phi, dtype=np.float32)
    curr_start_alpha = np.empty(U_phi, dtype=np.int64)
    curr_total_alpha = np.empty(U_phi, dtype=np.int64)
    
    log_alpha_each_t = np.full(T, ph_val, dtype=np.float32)
    start_alpha_each_t = np.full(T, -1, dtype=np.int64)
    total_alpha_each_t = np.full(T, 0, dtype=np.int64)

    # --- 条件分配全矩阵 (仅在可视化时使用) ---
    if need_full_matrix:
        full_log_alpha_mat = np.full((T, U_phi), -1e35, dtype=np.float32)
    else:
        full_log_alpha_mat = np.zeros((1, 1), dtype=np.float32)

    # 初始化
    prev_log_alpha[0:2] = 0.0
    prev_start_alpha[0:2] = 0
    prev_total_alpha[0:2] = 1
    if need_full_matrix:
        full_log_alpha_mat[0, 0:2] = 0.0

    for t in range(T):
        # PSD 策略：跳过高 Blank 帧
        if np.exp(post[t, blank]) > prune_threshold:
            if t > 0:
                log_alpha_each_t[t] = log_alpha_each_t[t-1]
                start_alpha_each_t[t] = start_alpha_each_t[t-1]
                total_alpha_each_t[t] = total_alpha_each_t[t-1]
                if need_full_matrix:
                    full_log_alpha_mat[t] = full_log_alpha_mat[t-1]
            continue

        # 核心逻辑：遍历状态节点 U
        for u in range(U_phi):
            # 来源 0: 自环 (u)
            v0, s0, t0 = prev_log_alpha[u], prev_start_alpha[u], prev_total_alpha[u]
            
            # 来源 1: 跳转 (u-1)
            v1 = prev_log_alpha[u-1] if u >= 1 else -1e35
            s1 = prev_start_alpha[u-1] if u >= 1 else -1
            t1 = prev_total_alpha[u-1] if u >= 1 else 0
            
            # 基础 2 路比较
            if v0 >= v1:
                res_v, res_s, res_t = v0, s0, t0
            else:
                res_v, res_s, res_t = v1, s1, t1
            
            # 来源 2: 跨越跳转 (u-2) - 仅限 Vocab 节点 (奇数索引)
            if u % 2 != 0 and u >= 2:
                v2 = prev_log_alpha[u-2]
                if v2 > res_v:
                    res_v, res_s, res_t = v2, prev_start_alpha[u-2], prev_total_alpha[u-2]
            
            # 加上发射概率并存储
            curr_log_alpha[u] = res_v + post[t, full_tgt_ids[u]]
            curr_start_alpha[u] = res_s
            curr_total_alpha[u] = res_t + 1

        # 边界重置：每一帧都可以是潜在起点
        curr_log_alpha[0:2] = 0.0
        curr_start_alpha[0:2] = t
        curr_total_alpha[0:2] = 1

        # 原地拷贝到 prev 状态 (为下一帧准备)
        for i in range(U_phi):
            prev_log_alpha[i] = curr_log_alpha[i]
            prev_start_alpha[i] = curr_start_alpha[i]
            prev_total_alpha[i] = curr_total_alpha[i]

        # 如果需要可视化，存入大矩阵
        if need_full_matrix:
            full_log_alpha_mat[t] = curr_log_alpha

        # 记录每帧最佳输出结果
        idx = U_phi - 1 if curr_log_alpha[U_phi-1] >= curr_log_alpha[U_phi-2] else U_phi - 2
        log_alpha_each_t[t] = curr_log_alpha[idx]
        start_alpha_each_t[t] = curr_start_alpha[idx]
        total_alpha_each_t[t] = curr_total_alpha[idx]

    return log_alpha_each_t, start_alpha_each_t, total_alpha_each_t, full_log_alpha_mat

# ================== 2. 加速版类定义 ==================
class CTCFsdStreamingSearch(KWSBaseSearch):
    def __init__(self, blank: int = 0, max_keep_blank_threshold=1.0, need_full_matrix=False):
        super().__init__(blank)
        self.prune_threshold = max_keep_blank_threshold
        self.blank = blank
        self.need_full_matrix = need_full_matrix

    def forward(self, log_posteriors, targets, logits_lens, target_lens):
        # 执行内核计算
        log_alpha_matrix, normed_logscores, logalpha_tlist, start_tlist, total_tlist = \
            self.streaming_search(log_posteriors, targets, logits_lens, target_lens)
        
        return log_alpha_matrix, normed_logscores, logalpha_tlist, start_tlist, total_tlist

    def streaming_search(self, posteriors, targets, logits_lens, target_lens):
        B, T, V = posteriors.shape
        U = int(max(target_lens).item())
        U_phi = 2 * U + 1
        
        # 数据准备
        tgt_np = targets.detach().view(-1).cpu().numpy().astype(np.int32)
        full_tgt_ids = np.zeros(U_phi, dtype=np.int32)
        full_tgt_ids[0::2] = self.blank
        full_tgt_ids[1::2] = tgt_np
        
        post_np = posteriors.detach().squeeze(0).cpu().numpy().astype(np.float32)
        
        # 调用核心内核
        log_a_t, start_a_t, total_a_t, full_mat_np = _numba_inplace_flexible_kernel(
            post_np, full_tgt_ids, self.blank, self.prune_threshold, PH, U_phi, self.need_full_matrix
        )
        
        device = posteriors.device
        
        # 归一化得分计算
        t_len = target_lens.item() if isinstance(target_lens, torch.Tensor) else target_lens
        normed_logscores = [(log_a_t[t] / t_len) for t in range(T)]
        
        # 处理返回张量
        if self.need_full_matrix:
            res_matrix = torch.from_numpy(full_mat_np).unsqueeze(0).to(device)
        else:
            res_matrix = torch.zeros((1, T, U_phi), device=device)
            
        return (
            res_matrix,
            normed_logscores,
            torch.from_numpy(log_a_t).to(device),
            torch.from_numpy(start_a_t.astype(np.float32)).to(device),
            torch.from_numpy(total_a_t.astype(np.float32)).to(device)
        )