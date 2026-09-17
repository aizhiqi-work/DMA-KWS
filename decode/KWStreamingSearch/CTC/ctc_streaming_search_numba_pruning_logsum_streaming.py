import torch
import numpy as np
from numba import njit

PH = -1e35 

# ================== 1. Numba 逐帧步进内核 (极致性能) ==================
@njit(cache=True)
def _numba_kws_step_core(frame_post, full_tgt_ids, prev_log_alpha, prev_start_alpha, prev_total_alpha, blank, U_phi):
    """
    内部核心函数：计算单帧状态转移
    """
    curr_log_alpha = np.empty(U_phi, dtype=np.float32)
    curr_start_alpha = np.empty(U_phi, dtype=np.int64)
    curr_total_alpha = np.empty(U_phi, dtype=np.int64)

    for u in range(U_phi):
        # 路径 0: 自环 (Stay)
        v0, s0, t0 = prev_log_alpha[u], prev_start_alpha[u], prev_total_alpha[u]
        # 路径 1: 标准跳转 (Move)
        v1 = prev_log_alpha[u-1] if u >= 1 else -1e35
        s1 = prev_start_alpha[u-1] if u >= 1 else -1
        t1 = prev_total_alpha[u-1] if u >= 1 else 0
        
        # 基础 2 路比较
        if v0 >= v1:
            res_v, res_s, res_t = v0, s0, t0
        else:
            res_v, res_s, res_t = v1, s1, t1
        
        # 路径 2: 跨越跳转 (Skip Blank) - 仅限 Vocab 节点 (奇数索引)
        if u % 2 != 0 and u >= 2:
            v2 = prev_log_alpha[u-2]
            if v2 > res_v:
                res_v, res_s, res_t = v2, prev_start_alpha[u-2], prev_total_alpha[u-2]
        
        # 累加当前帧发射概率
        curr_log_alpha[u] = res_v + frame_post[full_tgt_ids[u]]
        curr_start_alpha[u] = res_s
        curr_total_alpha[u] = res_t + 1

    return curr_log_alpha, curr_start_alpha, curr_total_alpha

# ================== 2. 流式搜索包装类 ==================
class CTCFsdStreamingSearch:
    def __init__(self, blank=0, need_full_matrix=False):
        self.blank = blank
        self.need_full_matrix = need_full_matrix
        
        # 状态变量
        self.prev_log_alpha = None
        self.prev_start_alpha = None
        self.prev_total_alpha = None
        self.history_log_alpha = [] # 仅在 need_full_matrix=True 时使用

    def reset(self):
        """重置搜索状态，通常在处理新音频流时调用"""
        self.prev_log_alpha = None
        self.prev_start_alpha = None
        self.prev_total_alpha = None
        self.history_log_alpha = []

    def _init_state(self, U_phi):
        """延迟初始化状态向量"""
        self.prev_log_alpha = np.full(U_phi, -1e35, dtype=np.float32)
        self.prev_start_alpha = np.full(U_phi, -1, dtype=np.int64)
        self.prev_total_alpha = np.full(U_phi, 0, dtype=np.int64)
        # 起点初始化：允许从前两个节点（Blank 或 第一个音素）开始
        self.prev_log_alpha[0:2] = 0.0
        self.prev_start_alpha[0:2] = 0
        self.prev_total_alpha[0:2] = 1

    def step(self, frame_log_post, full_tgt_ids, t, prune_threshold=0.95):
        """
        真正的流式步进：处理单帧数据
        :param frame_log_post: [Vocab_Size] 的 numpy 数组
        :param t: 当前全局帧索引
        """
        U_phi = len(full_tgt_ids)
        if self.prev_log_alpha is None:
            self._init_state(U_phi)

        # 1. PSD 策略：如果 Blank 概率极高，则跳过复杂计算，维持上一帧状态
        if np.exp(frame_log_post[self.blank]) > prune_threshold:
            # 状态保持不变 (Implicit Stay)
            pass
        else:
            # 2. 调用 Numba 内核计算状态转移
            curr_v, curr_s, curr_t = _numba_kws_step_core(
                frame_log_post, full_tgt_ids, 
                self.prev_log_alpha, self.prev_start_alpha, self.prev_total_alpha, 
                self.blank, U_phi
            )
            
            # 3. 边界重置：每一帧都可以是关键词的潜在起点
            curr_v[0:2] = 0.0
            curr_s[0:2] = t
            curr_t[0:2] = 1
            
            # 4. 更新持久化状态
            self.prev_log_alpha = curr_v
            self.prev_start_alpha = curr_s
            self.prev_total_alpha = curr_t

        # 5. 如果开启了 flag，记录历史记录用于可视化
        if self.need_full_matrix:
            self.history_log_alpha.append(self.prev_log_alpha.copy())

        # 6. 返回当前时刻的最佳候选分值 (在最后两个节点中选最大的)
        idx = U_phi - 1 if self.prev_log_alpha[U_phi-1] >= self.prev_log_alpha[U_phi-2] else U_phi - 2
        return self.prev_log_alpha[idx], self.prev_start_alpha[idx], self.prev_total_alpha[idx]