from models.encoder import ConformerEncoder
from models.ctc import CTC
from models.processor import compute_fbank
import torch
import torchaudio
import torch.nn as nn
from models.text.char_tokenizer import CharTokenizer
from models.search import ctc_greedy_search
from KWStreamingSearch.CTC.ctc_streaming_search import CTCFsdStreamingSearch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json

class Stage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConformerEncoder(
            input_size=80,
            output_size=144,
            attention_heads=4,
            linear_units=576,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.0,
            use_cnn_module=True,
            input_layer="conv2d",
            pos_enc_layer_type="rel_pos",
            selfattention_layer_type="rel_selfattn",
            cnn_module_kernel=3,
        )
        self.ctc = CTC(
            odim=73,
            encoder_output_size=144,
            blank_id=0
        )
        
    @torch.jit.unused
    def ctc_logprobs(self, encoder_out: torch.Tensor, blank_penalty: float = 0.0, blank_id: int = 0):
        if blank_penalty > 0.0:
            logits = self.ctc.ctc_lo(encoder_out)
            logits[:, :, blank_id] -= blank_penalty
            logits = logits.log_softmax(dim=2)
        else:
            logits = self.ctc.log_softmax(encoder_out)
        return logits
    
    @torch.jit.unused
    def decode(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc_logprobs(encoder_out, blank_penalty=0.0, blank_id=0)
        # greedy_search_results = ctc_greedy_search(ctc_probs, encoder_lens, blank_id=0)
        return ctc_probs, None


def load_model(ckpt_path: str, device: torch.device) -> Stage1:
    """加载模型"""
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # 提取编码器权重
    encoder_ckpt = {k.replace('encoder.', ''): v 
                   for k, v in ckpt.items() if k.startswith('encoder')}
    
    # 提取CTC权重
    ctc_ckpt = {k.replace('ctc.', ''): v 
               for k, v in ckpt.items() if k.startswith('ctc')}
    
    model = Stage1()
    model.encoder.load_state_dict(encoder_ckpt)
    model.ctc.load_state_dict(ctc_ckpt)
    model.eval()
    model.to(device)
    
    return model


if __name__ == "__main__":
    # 配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = "/nvme01/openkws/wenet/examples/librispeech-g2p/s0/exp/ls-gs-1460-ckpts/avg_10.pt"
    tokenizer_path = '/nvme01/openkws/wenet/examples/librispeech-g2p/s0/data/dict/lang_char.txt'
    
    model = load_model(ckpt_path, device)
    tokenizer = CharTokenizer(tokenizer_path, None, split_with_space=' ')
    

    import json
    with open('/nvme01/openkws/wwd-data/heysnips/hey_snips_research_6k_en_train_eval_clean_ter/test.json', 'r') as f:
        samples = json.load(f)
    
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import numpy as np
    import os


    batch_size = 32
    max_workers = 32
    target_word_list = [
        "S N IH1 P S" ,
    ]
    token_dict = {}
    for word in target_word_list:
        _, token_ids = tokenizer.tokenize(word)
        token_dict[word] = token_ids

    # --- Step1: 批量特征提取 + 编码 ---
    features_list = []  # 每个样本的 ctc_probs
    sample_keys = []    # 样本索引或 key
    labels_dict = {word: [] for word in target_word_list}  # 每个词对应label列表

    samples = samples[:5000]

    for i in tqdm(range(0, len(samples), batch_size)):
        batch_samples = samples[i:i+batch_size]
        feats_batch = []
        feat_lens_batch = []
        
        for sample in batch_samples:
            audio_path = os.path.join("/nvme01/openkws/wwd-data/heysnips/hey_snips_research_6k_en_train_eval_clean_ter/", sample['audio_file_path'])
            waveform, sample_rate = torchaudio.load(audio_path)

            

            sample['key'] = sample['worker_id']
            sample['wav'] = waveform
            sample['sample_rate'] = sample_rate
            sample = compute_fbank(sample, num_mel_bins=80, frame_shift=10, frame_length=25, dither=0.1)
            
            feats_batch.append(sample["feat"])
            feat_lens_batch.append(sample["feat"].shape[0])
            sample_keys.append(sample['key'])
            
            # 为每个词生成 label
            for word in target_word_list:
                labels_dict[word].append(1 if sample['is_hotword'] else 0)
        
        # pad feats
        max_len = max(feat_lens_batch)
        padded_feats = [torch.cat([f, torch.zeros(max_len - f.shape[0], f.shape[1])], dim=0)
                        for f in feats_batch]
        feats_tensor = torch.stack(padded_feats).to(device)
        feat_lens_tensor = torch.tensor(feat_lens_batch, device=device)
        
        # 模型编码
        with torch.no_grad():
            ctc_probs_batch, _ = model.decode(feats_tensor, feat_lens_tensor)  # (B, T, V)
        
        # 转 CPU 存储
        features_list.extend([ctc_probs_batch[b].cpu() for b in range(len(batch_samples))])


    # --- Step2: 多进程做每个词的流式搜索 ---
    def kws_streaming_search(ctc_probs, token_ids):
        kws_tokens = torch.tensor(token_ids).unsqueeze(0)
        kws_tokens_lens = torch.tensor([len(token_ids)])
        ctc_probs_lens = torch.tensor([ctc_probs.shape[0]])
        kws_search = CTCFsdStreamingSearch(blank=0)
        
        _, logalpha_tlist, _, _, _ = kws_search(ctc_probs.unsqueeze(0), kws_tokens, ctc_probs_lens, kws_tokens_lens)
        max_score = max([s.exp().item() for s in logalpha_tlist])
        return max_score

    scores_dict = {word: [] for word in target_word_list}

    for word in target_word_list:
        token_ids = token_dict[word]
        max_scores = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(kws_streaming_search, features_list[i], token_ids)
                    for i in range(len(features_list))]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {word}"):
                max_scores.append(fut.result())
        scores_dict[word] = max_scores


    # --- Step3: 每个词计算无误报 ACC ---
    acc_dict = {}
    threshold_dict = {}

    for word in target_word_list:
        max_scores = np.array(scores_dict[word])
        print(max_scores)
        labels = np.array(labels_dict[word])
        
        neg_scores = max_scores[labels == 0]
        threshold = sorted(neg_scores)[25]
        preds = (max_scores > threshold).astype(int)
        acc = (preds == labels).mean()
        
        threshold_dict[word] = threshold
        acc_dict[word] = acc

    # 输出结果
    for word in target_word_list:
        print(f"{word}: Threshold={threshold_dict[word]:.5f}, ACC={acc_dict[word]:.4f}")

    # mean ACC
    mean_acc = np.mean(list(acc_dict.values()))
    print(f"Mean ACC: {mean_acc:.4f}")