import torch

from KWStreamingSearch.base_search import KWSBaseSearch
from KWStreamingSearch.fusion_strategy import PH

class CTCFsdStreamingSearch(KWSBaseSearch):
    def __init__(self, blank: int = 0, max_keep_blank_threshold=1.0):
        super().__init__(blank)
        self.prune_threshold = max_keep_blank_threshold
        self.blank = blank
        

    def forward(
        self, log_posteriors: torch.Tensor, targets: torch.Tensor, logits_lens: torch.Tensor, target_lens: torch.Tensor
    ):
        forward_logprob, logalpha_tlist, start_tlist, total_tlist, log_alpha \
            = self.streaming_search(log_posteriors, targets, logits_lens, target_lens)
            
        # print(log_posteriors.shape)

        T = log_posteriors.shape[1]

        # ---- 长度归一化 ----
        target_len = target_lens.item() if isinstance(target_lens, torch.Tensor) else target_lens
        length_normed_logprob = forward_logprob / target_len

        # ---- 每帧归一化分数 ----
        normed_logscores = [
            logalpha_tlist[t]/ target_len for t in range(T)
        ]

        return log_alpha, normed_logscores, logalpha_tlist, start_tlist, total_tlist



        
    def streaming_search(
        self, posteriors: torch.Tensor, targets: torch.Tensor,  logits_lens: torch.tensor, target_lens: torch.Tensor
    ):
        B, T, _ = posteriors.shape
        U = max(target_lens).item()
        assert B == 1, f"decoding batch size must be 1, get B = {B}"
        assert isinstance(target_lens, torch.Tensor)
        # target = abc, vocab: u % 2 != 1, blank: u % 2 == 0, 
        # processed_target = [blank, a, blank, b, blank, c, blank]
        U_phi = 2 * U + 1       
        # 统一 device，避免 CPU/GPU 混用
        log_alpha = torch.zeros(B, T, U_phi, device=posteriors.device) - 1000
        start_alpha = torch.zeros(B, T, U_phi, device=posteriors.device)
        total_alpha = torch.zeros(B, T, U_phi, device=posteriors.device)
        
        log_alpha = log_alpha.to(posteriors.device)

        log_prob = PH
        log_alpha_each_t = [PH for _ in range(T)] 
        start_alpha_each_t = [PH for _ in range(T)]
        total_alpha_each_t = [PH for _ in range(T)]

        b, t, psd_skips = 0, 0, 0 # B = 1
        while t < T:
            # for psd: if p_{t, `blank`} > self.prune_threshold, then skip current frame
            # 注意：.item() 转标量，避免张量参与 if 判断报错；默认阈值为 1.0 表示几乎不会触发
            if torch.exp(posteriors[b, t, self.blank]).item() > self.prune_threshold:
                psd_skips += 1
                t += 1
                continue
            
            for u in range(U_phi):
                if u == 0 or u == 1:
                    # the initial entries for each time step (the first and second row.)
                    if t-psd_skips == 0:
                        # the first entry
                        log_alpha[b, t, u] = 0.0
                        start_alpha[b, t, u] = 0
                        total_alpha[b, t, u] = 1
                    else:
                        # other entries
                        log_alpha[b, t, u] = 0.0
                        start_alpha[b, t, u] = t 
                        total_alpha[b, t, u] = 1
                else:
                    # not initial entries
                    if t-psd_skips == 0:
                        # the first column
                        # not valid.
                        log_alpha[b, t, u] = -1e35
                        start_alpha[b, t, u] = -1
                        total_alpha[b, t, u] = 0
                    else:
                        # each time step
                        if u % 2 == 0:
                            # blank
                            from_blank = log_alpha[b, t-1-psd_skips, u] + posteriors[b, t-1-psd_skips, self.blank]
                            from_vocab = log_alpha[b, t-1-psd_skips, u-1] + posteriors[b, t-1-psd_skips, self.blank] # u or u-1?
                            
                            if (from_blank >= from_vocab).item():
                                log_alpha[b, t, u] = from_blank
                                start_alpha[b, t, u] = start_alpha[b, t-1-psd_skips, u]
                                total_alpha[b, t, u] = total_alpha[b, t-1-psd_skips, u] + 1
                            else:
                                log_alpha[b, t, u] = from_vocab
                                start_alpha[b, t, u] = start_alpha[b, t-1-psd_skips, u-1]
                                total_alpha[b, t, u] = total_alpha[b, t-1-psd_skips, u-1] + 1
                        else:
                            # vocab
                            from_last_vocab = log_alpha[b, t-1-psd_skips, u-2] + posteriors[b, t-1-psd_skips, targets[b, u//2]] # processed_target = [blank, a, blank, b, blank, c, blank]
                            from_blank = log_alpha[b, t-1-psd_skips, u-1] + posteriors[b, t-1-psd_skips, targets[b, u//2]]      # idx: u // 2 (\floor{u / 2})
                            from_now_vocab = log_alpha[b, t-1-psd_skips, u] + posteriors[b, t-1-psd_skips, targets[b, u//2]]
                            
                            _, max_index = torch.max(
                                torch.stack([from_last_vocab, from_blank, from_now_vocab], dim=0), dim=0)
                            max_index = int(max_index.item())
                            
                            if max_index == 0:
                                # from last token
                                log_alpha[b, t, u] = from_last_vocab
                                start_alpha[b, t, u] = start_alpha[b, t-1-psd_skips, u-2]
                                total_alpha[b, t, u] = total_alpha[b, t-1-psd_skips, u-2] + 1
                            elif max_index == 1:
                                # from blank
                                log_alpha[b, t, u] = from_blank 
                                start_alpha[b, t, u] = start_alpha[b, t-1-psd_skips, u-1]
                                total_alpha[b, t, u] = total_alpha[b, t-1-psd_skips, u-1] + 1
                            else:
                                # from current token
                                log_alpha[b, t, u] = from_now_vocab 
                                start_alpha[b, t, u] = start_alpha[b, t-1-psd_skips, u]
                                # 修复这里的 batch 维度索引错误（0 -> b）
                                total_alpha[b, t, u] = total_alpha[b, t-1-psd_skips, u] + 1

            # output for each time step
            _, max_out_index = torch.max(
                torch.stack([log_alpha[b, t, U_phi-1], log_alpha[b, t, U_phi-2]], dim=0),dim=0)
            max_out_index = int(max_out_index.item())
            
            if max_out_index == 0:
                #  max path ends in a vocab token.
                out_row = U_phi - 1 
            else:
                # max path ends in a blk token.
                out_row = U_phi - 2
            
            out_log_alpha = log_alpha[b, t, out_row]
            out_start_alpha = start_alpha[b, t, out_row]
            out_total_alpha = total_alpha[b, t, out_row]

            log_prob = out_log_alpha.item()
            log_alpha_each_t[t] = out_log_alpha
            start_alpha_each_t[t] = out_start_alpha
            total_alpha_each_t[t] = out_total_alpha
            
            psd_skips = 0
            t += 1
        
        return log_prob, log_alpha_each_t, start_alpha_each_t, total_alpha_each_t, log_alpha