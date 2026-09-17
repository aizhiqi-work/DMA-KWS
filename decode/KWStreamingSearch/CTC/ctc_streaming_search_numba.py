import torch
import numpy as np
from numba import njit
from KWStreamingSearch.base_search import KWSBaseSearch

# 定义常量，保持与你的环境一致
PH = -1e35 

# ================== 1. Numba 加速内核 (核心引擎) ==================
@njit(cache=True)
def _numba_streaming_search_kernel(posteriors, targets, blank, prune_threshold, ph_val, U_phi):
    """
    使用 Numba JIT 编译的计算核心。
    去除了所有 Torch 依赖，只使用 NumPy 基础算术。
    """
    T, V = posteriors.shape
    # 预分配矩阵
    log_alpha = np.full((T, U_phi), -1000.0, dtype=np.float32)
    start_alpha = np.zeros((T, U_phi), dtype=np.float32)
    total_alpha = np.zeros((T, U_phi), dtype=np.float32)

    log_alpha_each_t = np.full(T, ph_val, dtype=np.float32)
    start_alpha_each_t = np.full(T, ph_val, dtype=np.float32)
    total_alpha_each_t = np.full(T, ph_val, dtype=np.float32)

    psd_skips = 0
    t = 0
    while t < T:
        # PSD 策略：跳过高 Blank 概率帧
        if np.exp(posteriors[t, blank]) > prune_threshold:
            psd_skips += 1
            t += 1
            continue
        
        # 对应你原始代码的 U 维度循环
        for u in range(U_phi):
            if u == 0 or u == 1:
                if t - psd_skips == 0:
                    log_alpha[t, u] = 0.0
                    start_alpha[t, u] = 0
                    total_alpha[t, u] = 1
                else:
                    log_alpha[t, u] = 0.0
                    start_alpha[t, u] = t
                    total_alpha[t, u] = 1
            else:
                if t - psd_skips == 0:
                    log_alpha[t, u] = -1e35
                    start_alpha[t, u] = -1
                    total_alpha[t, u] = 0
                else:
                    prev_t = t - 1 - psd_skips
                    if u % 2 == 0:
                        # Blank 节点转移逻辑
                        f_blank = log_alpha[prev_t, u] + posteriors[prev_t, blank]
                        f_vocab = log_alpha[prev_t, u-1] + posteriors[prev_t, blank]
                        if f_blank >= f_vocab:
                            log_alpha[t, u] = f_blank
                            start_alpha[t, u] = start_alpha[prev_t, u]
                            total_alpha[t, u] = total_alpha[prev_t, u] + 1
                        else:
                            log_alpha[t, u] = f_vocab
                            start_alpha[t, u] = start_alpha[prev_t, u-1]
                            total_alpha[t, u] = total_alpha[prev_t, u-1] + 1
                    else:
                        # Vocab 节点转移逻辑 (3路最大值)
                        target_id = targets[u // 2]
                        f_last_v = log_alpha[prev_t, u-2] + posteriors[prev_t, target_id]
                        f_blank  = log_alpha[prev_t, u-1] + posteriors[prev_t, target_id]
                        f_now_v  = log_alpha[prev_t, u]   + posteriors[prev_t, target_id]
                        
                        if f_last_v >= f_blank and f_last_v >= f_now_v:
                            log_alpha[t, u] = f_last_v
                            start_alpha[t, u] = start_alpha[prev_t, u-2]
                            total_alpha[t, u] = total_alpha[prev_t, u-2] + 1
                        elif f_blank >= f_last_v and f_blank >= f_now_v:
                            log_alpha[t, u] = f_blank
                            start_alpha[t, u] = start_alpha[prev_t, u-1]
                            total_alpha[t, u] = total_alpha[prev_t, u-1] + 1
                        else:
                            log_alpha[t, u] = f_now_v
                            start_alpha[t, u] = start_alpha[prev_t, u]
                            total_alpha[t, u] = total_alpha[prev_t, u] + 1

        # 确定当前时刻的最佳输出行 (u-1 vs u-2)
        if log_alpha[t, U_phi-1] >= log_alpha[t, U_phi-2]:
            out_row = U_phi - 1
        else:
            out_row = U_phi - 2
        
        log_alpha_each_t[t] = log_alpha[t, out_row]
        start_alpha_each_t[t] = start_alpha[t, out_row]
        total_alpha_each_t[t] = total_alpha[t, out_row]
        
        psd_skips = 0
        t += 1

    return log_alpha_each_t, start_alpha_each_t, total_alpha_each_t, log_alpha

# ================== 2. 加速版类定义 ==================
class CTCFsdStreamingSearch(KWSBaseSearch):
    def __init__(self, blank: int = 0, max_keep_blank_threshold=1.0):
        super().__init__(blank)
        self.prune_threshold = max_keep_blank_threshold
        self.blank = blank

    def forward(self, log_posteriors: torch.Tensor, targets: torch.Tensor, 
                logits_lens: torch.Tensor, target_lens: torch.Tensor):
        
        # 调用核心搜索
        forward_logprob, logalpha_tlist, start_tlist, total_tlist, log_alpha = \
            self.streaming_search(log_posteriors, targets, logits_lens, target_lens)
        
        T = log_posteriors.shape[1]
        target_len = target_lens.item() if isinstance(target_lens, torch.Tensor) else target_lens

        # ---- 归一化分数 (根据你的需求保留) ----
        # 这里把 log 分数除以 target_len，并转为列表
        normed_logscores = [logalpha_tlist[t] / target_len for t in range(T)]

        return log_alpha, normed_logscores, logalpha_tlist, start_tlist, total_tlist

    def streaming_search(self, posteriors: torch.Tensor, targets: torch.Tensor, 
                         logits_lens: torch.tensor, target_lens: torch.Tensor):
        
        B, T, V = posteriors.shape
        U = int(max(target_lens).item())
        U_phi = 2 * U + 1
        
        # 数据转换：Torch -> NumPy (Numba 必须使用 NumPy)
        post_np = posteriors.detach().squeeze(0).cpu().numpy().astype(np.float32)
        targets_np = targets.detach().squeeze(0).cpu().numpy().astype(np.int32)
        
        # 执行加速内核
        log_a_t, start_a_t, total_a_t, log_alpha_full = _numba_streaming_search_kernel(
            post_np, targets_np, self.blank, self.prune_threshold, PH, U_phi
        )
        
        # 结果封装回 Torch (保持接口兼容)
        device = posteriors.device
        # 将结果从 numpy 转回 torch tensor 或标量
        return (
            log_a_t[-1].item(),
            torch.from_numpy(log_a_t).to(device),
            torch.from_numpy(start_a_t).to(device),
            torch.from_numpy(total_a_t).to(device),
            torch.from_numpy(log_alpha_full).unsqueeze(0).to(device)
        )