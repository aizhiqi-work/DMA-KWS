import numpy as np
from numba import njit

@njit(cache=True)
def _numba_step_fast_kernel(
    frame_post, 
    all_tgt_ids, 
    all_tgt_lens, 
    prev_v, 
    prev_s, 
    t_curr
):
    """
    极致优化的单核 JIT 内核。
    对于 2000-5000 词量级，单核串行避开了线程同步开销，速度最快。
    """
    N, U_max = all_tgt_ids.shape
    new_v = np.empty_like(prev_v)
    new_s = np.empty_like(prev_s)
    
    scores = np.empty(N, dtype=np.float32)
    starts = np.empty(N, dtype=np.int64)

    for n in range(N):
        U_phi = all_tgt_lens[n]
        for u in range(U_phi):
            # 路径 0: Stay
            res_v = prev_v[n, u]
            res_s = prev_s[n, u]
            
            # 路径 1: Move
            if u >= 1:
                v1 = prev_v[n, u-1]
                if v1 > res_v:
                    res_v, res_s = v1, prev_s[n, u-1]
            
            # 路径 2: Skip Blank
            if u % 2 != 0 and u >= 2:
                v2 = prev_v[n, u-2]
                if v2 > res_v:
                    res_v, res_s = v2, prev_s[n, u-2]
            
            new_v[n, u] = res_v + frame_post[all_tgt_ids[n, u]]
            new_s[n, u] = res_s

        # 边界起点重置：每一帧都可能产生新起点
        v0 = frame_post[all_tgt_ids[n, 0]]
        v1 = frame_post[all_tgt_ids[n, 1]]
        if v0 > new_v[n, 0]:
            new_v[n, 0], new_s[n, 0] = v0, t_curr
        if v1 > new_v[n, 1]:
            new_v[n, 1], new_s[n, 1] = v1, t_curr

        # 结果提取：取最后两个状态的最大值
        idx1, idx2 = U_phi - 1, U_phi - 2
        if new_v[n, idx1] >= new_v[n, idx2]:
            scores[n], starts[n] = new_v[n, idx1], new_s[n, idx1]
        else:
            scores[n], starts[n] = new_v[n, idx2], new_s[n, idx2]

    return new_v, new_s, scores, starts

class BatchCTCFsdStreamingSearch:
    def __init__(self, keywords_ids_list, blank=0):
        """
        :param keywords_ids_list: List[np.array] 每一个元素的 full_tgt_ids
        """
        self.blank = blank
        self.num_keywords = len(keywords_ids_list)
        self.target_lens = np.array([len(x) for x in keywords_ids_list], dtype=np.int32)
        self.max_u = np.max(self.target_lens)
        
        # 填充 ID 矩阵 (用 blank 填充)
        self.all_tgt_ids = np.full((self.num_keywords, self.max_u), blank, dtype=np.int32)
        for i, ids in enumerate(keywords_ids_list):
            self.all_tgt_ids[i, :len(ids)] = ids
            
        self.prev_log_alpha = np.full((self.num_keywords, self.max_u), -1e35, dtype=np.float32)
        self.prev_start_alpha = np.full((self.num_keywords, self.max_u), -1, dtype=np.int64)

    def reset(self):
        self.prev_log_alpha.fill(-1e35)
        self.prev_start_alpha.fill(-1)

    def step(self, frame_log_post, t):
        """
        批量处理所有关键词
        :return: scores (N,), starts (N,)
        """
        self.prev_log_alpha, self.prev_start_alpha, scores, starts = _numba_step_fast_kernel(
            frame_log_post, 
            self.all_tgt_ids, 
            self.target_lens,
            self.prev_log_alpha, 
            self.prev_start_alpha,
            t
        )
        return scores, starts