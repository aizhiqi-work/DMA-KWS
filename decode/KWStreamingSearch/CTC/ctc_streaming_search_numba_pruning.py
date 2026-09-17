import torch
import numpy as np
from numba import njit
from KWStreamingSearch.base_search import KWSBaseSearch

# 定义常量
PH = -1e35 

# ================== 1. 灵活性能内核 (Numba JIT) ==================
@njit(cache=True)
def _numba_flexible_streaming_kernel(post, full_tgt_ids, blank, prune_threshold, ph_val, U_phi, need_full_matrix):
    """
    灵活内核：
    - need_full_matrix=False: 极致速度，仅保留当前帧状态。
    - need_full_matrix=True: 记录全过程，用于轨迹可视化。
    """
    T, V = post.shape
    
    # 状态变量（始终保持最新一帧）
    prev_log_alpha = np.full(U_phi, -1e35, dtype=np.float32)
    prev_start_alpha = np.full(U_phi, -1, dtype=np.int64)
    
    # 逐帧结果记录
    log_alpha_each_t = np.full(T, ph_val, dtype=np.float32)
    start_alpha_each_t = np.full(T, -1, dtype=np.int64)
    
    # 条件分配大矩阵
    if need_full_matrix:
        full_log_alpha = np.full((T, U_phi), -1e35, dtype=np.float32)
    else:
        # 即使不需要，也要返回一个微小数组以保持 Numba 返回类型一致
        full_log_alpha = np.zeros((1, 1), dtype=np.float32)

    # 初始化搜索起点
    prev_log_alpha[0:2] = 0.0
    prev_start_alpha[0:2] = 0
    if need_full_matrix:
        full_log_alpha[0, 0:2] = 0.0

    for t in range(T):
        # PSD (Posterior Skip Decoding) 策略
        if np.exp(post[t, blank]) > prune_threshold:
            if t > 0:
                log_alpha_each_t[t] = log_alpha_each_t[t-1]
                start_alpha_each_t[t] = start_alpha_each_t[t-1]
                if need_full_matrix:
                    full_log_alpha[t] = full_log_alpha[t-1]
            continue

        # 当前帧计算（零申请，重用局部变量）
        curr_log_alpha = np.empty(U_phi, dtype=np.float32)
        curr_start_alpha = np.empty(U_phi, dtype=np.int64)

        for u in range(U_phi):
            # 来源 0: 自环 (u)
            v0, s0 = prev_log_alpha[u], prev_start_alpha[u]
            # 来源 1: 跳转 (u-1)
            v1 = prev_log_alpha[u-1] if u >= 1 else -1e35
            s1 = prev_start_alpha[u-1] if u >= 1 else -1
            
            # 基础 2 路比较
            if v0 >= v1:
                res_v, res_s = v0, s0
            else:
                res_v, res_s = v1, s1
            
            # 来源 2: 跨越跳转 (u-2) - 仅限 Vocab 节点 (奇数索引)
            if u % 2 != 0 and u >= 2:
                v2 = prev_log_alpha[u-2]
                if v2 > res_v:
                    res_v, res_s = v2, prev_start_alpha[u-2]

            # 加上发射概率
            curr_log_alpha[u] = res_v + post[t, full_tgt_ids[u]]
            curr_start_alpha[u] = res_s

        # 边界重置：允许每一帧作为潜在起点
        curr_log_alpha[0:2] = 0.0
        curr_start_alpha[0:2] = t

        # 原地拷贝状态到 prev
        prev_log_alpha[:] = curr_log_alpha
        prev_start_alpha[:] = curr_start_alpha

        # 如果需要可视化，记录当前行
        if need_full_matrix:
            full_log_alpha[t] = curr_log_alpha

        # 记录当前时刻最佳得分 (u-1 vs u-2)
        idx = U_phi - 1 if curr_log_alpha[U_phi-1] >= curr_log_alpha[U_phi-2] else U_phi - 2
        log_alpha_each_t[t] = curr_log_alpha[idx]
        start_alpha_each_t[t] = curr_start_alpha[idx]

    return log_alpha_each_t, start_alpha_each_t, full_log_alpha

# ================== 2. 包装类定义 ==================
class CTCFsdStreamingSearch(KWSBaseSearch):
    def __init__(self, blank: int = 0, max_keep_blank_threshold=1.0, need_full_matrix=False):
        """
        :param need_full_matrix: 是否保留 log_alpha 全矩阵。
                                 False: 生产/压测模式，速度最快。
                                 True: 调试模式，支持热图可视化。
        """
        super().__init__(blank)
        self.prune_threshold = max_keep_blank_threshold
        self.blank = blank
        self.need_full_matrix = need_full_matrix

    def forward(self, log_posteriors, targets, logits_lens, target_lens):
        """
        符合 KWS 接口的 forward
        """
        # 执行内核搜索
        log_alpha_matrix, logalpha_tlist, start_tlist, total_tlist, log_alpha_final = \
            self.streaming_search(log_posteriors, targets, logits_lens, target_lens)
        
        T = log_posteriors.shape[1]
        target_len = target_lens.item() if isinstance(target_lens, torch.Tensor) else target_lens
        
        # 归一化得分 (Log-sum / length)
        # 注意：这里为了兼容你的脚本，logalpha_tlist 现在是 Tensor
        normed_logscores = [(logalpha_tlist[t].item() / target_len) for t in range(T)]

        return log_alpha_matrix, normed_logscores, logalpha_tlist, start_tlist, total_tlist

    def streaming_search(self, posteriors, targets, logits_lens, target_lens):
        B, T, V = posteriors.shape
        U = int(max(target_lens).item())
        U_phi = 2 * U + 1
        
        # 1. 准备目标 ID (加上 Blank 间隔)
        tgt_np = targets.detach().view(-1).cpu().numpy().astype(np.int32)
        full_tgt_ids = np.zeros(U_phi, dtype=np.int32)
        full_tgt_ids[0::2] = self.blank
        full_tgt_ids[1::2] = tgt_np
        
        # 2. 准备 Log Posteriors
        post_np = posteriors.detach().squeeze(0).cpu().numpy().astype(np.float32)
        
        # 3. 调用 JIT 内核
        log_a_t, start_a_t, full_matrix = _numba_flexible_streaming_kernel(
            post_np, full_tgt_ids, self.blank, self.prune_threshold, PH, U_phi, 
            self.need_full_matrix
        )
        
        # 4. 封装结果回 Torch
        device = posteriors.device
        
        # 处理全矩阵输出
        if self.need_full_matrix:
            res_matrix = torch.from_numpy(full_matrix).unsqueeze(0).to(device)
        else:
            res_matrix = torch.zeros((1, T, U_phi), device=device) # 占位
            
        return (
            res_matrix,                                     # 可视化用的 [1, T, U_phi]
            torch.from_numpy(log_a_t).to(device),           # 每帧分 [T]
            torch.from_numpy(start_a_t.astype(np.float32)).to(device), # 每帧起点 [T]
            torch.zeros(T, device=device),                  # total_tlist 占位
            log_a_t[-1].item()                              # 最终分
        )

# ================== 3. 使用示例 ==================
# 如果你要可视化，这样初始化：
# kws_search = CTCFsdStreamingSearch(blank=0, need_full_matrix=True)

# 如果你要性能压测，这样初始化：
# kws_search = CTCFsdStreamingSearch(blank=0, need_full_matrix=False)