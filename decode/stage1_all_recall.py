import torch, json, torchaudio
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import numpy as np
from models.encoder import ConformerEncoder
from models.ctc import CTC
from models.processor import compute_fbank
from models.text.char_tokenizer import CharTokenizer
from models.search import ctc_greedy_search

from KWStreamingSearch.CTC.ctc_streaming_search import CTCFsdStreamingSearch

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
        return ctc_probs

    @torch.jit.unused
    def decode_with_greedy_search(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc_logprobs(encoder_out, blank_penalty=0.0, blank_id=0)
        greedy_search_results = ctc_greedy_search(ctc_probs, encoder_lens, blank_id=0)
        return ctc_probs, greedy_search_results


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



def get_wavs(data_dir: str):
    wavs = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                wavs.append(os.path.join(root, file))
    return wavs

import json
from torch.utils.data import Dataset, DataLoader
class HeySips(Dataset):
    def __init__(self):
        self.data = get_wavs("/nvme01/openkws_github/eval/data_wwd/neg")
        # self.data = self.data[:10000]
        tokenizer_path = '/nvme01/openkws/wenet/examples/librispeech-g2p/s0/data/dict/lang_char.txt'
        self.tokenizer = CharTokenizer(tokenizer_path, None, split_with_space=' ')
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {}
        audio_path = self.data[idx]
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            sample['wav'] = waveform
            sample['sample_rate'] = sample_rate
            sample['key'] = audio_path
            # sample = compute_fbank(sample, num_mel_bins=80, frame_shift=10, frame_length=25, dither=0.1)
            # sample['feat'] = sample['feat']
            # print('1', sample['feat'].shape)
            feat_path = audio_path.replace("neg", "neg-fbank").replace('.wav', '.npy')
            sample['feat'] = torch.from_numpy(np.load(feat_path).astype(np.float32))
            sample['label'] = 0
            target_word = "HH EY1 S N IH1 P S"
        except:
            return None
        _, target_word_token_ids = self.tokenizer.tokenize(target_word)
        return {
            'anchor_seq': torch.tensor(target_word_token_ids, dtype=torch.long),
            'feat': sample['feat'],
            'waveform': sample['wav'].squeeze(0),
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }
    

from torch.nn.utils.rnn import pad_sequence
def test_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    feats = [item['feat'] for item in batch]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)  # feat 填充0
    feat_lengths = torch.tensor([f.size(0) for f in feats])  # 记录原始长度
    anchor = [item['anchor_seq'] for item in batch]
    anchor = pad_sequence(anchor, batch_first=True, padding_value=0)  # anchor_seq 填充0 类似<blank>
    labels = torch.tensor([item['label'] for item in batch])  # 直接转为Tensor
    waveform = [item['waveform'] for item in batch]
    padded_waveform = pad_sequence(waveform, batch_first=True, padding_value=0)  # waveform 填充0
    return {
        "anchor": anchor,
        "feat": padded_feats,
        "feat_lengths": feat_lengths,  # 原始feat长度
        "waveform": padded_waveform,
        "label": labels,
    }


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = "/nvme01/openkws/wenet/examples/librispeech-g2p/s0/exp/ls-gs-1460-ckpts/avg_10.pt"
    model = load_model(ckpt_path, device)
    model.eval()
    data = HeySips()
    dataloader = DataLoader(data, batch_size=256, num_workers=16, shuffle=False, collate_fn=test_collate_fn)
    from tqdm import tqdm    
    kws_streaming_search = CTCFsdStreamingSearch(blank=0)
    from multiprocessing import Pool
    from tqdm import tqdm

    def kws_search_single(args):
        ctc_prob, kws_tokens, device = args
        kws_streaming_search_local = CTCFsdStreamingSearch(blank=0)  # 每进程实例化
        ctc_probs_lens = torch.tensor([ctc_prob.shape[0]], device=device)  # (1,)
        kws_tokens_lens = torch.tensor([kws_tokens.shape[1]], device=device)  # (1,)
        _, normed_logscores, _, start_tlist, _ = kws_streaming_search_local(
            ctc_prob.unsqueeze(0), kws_tokens, ctc_probs_lens, kws_tokens_lens
        )
        max_idx = np.array(normed_logscores).argmax().item()
        max_score = normed_logscores[max_idx].exp().item()
        max_start_frame = start_tlist[max_idx]
        return max([s.exp().item() for s in normed_logscores])

    labels = []
    preds = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        anchor, feat, feat_lengths, label, waveform = batch['anchor'], batch['feat'], batch['feat_lengths'], batch['label'], batch['waveform']
        anchor = anchor.to(device)
        feat = feat.to(device)
        feat_lengths = feat_lengths.to(device)
        labels.extend(label.tolist())
        
        with torch.no_grad():
            ctc_probs = model.decode(feat, feat_lengths)  # (B, T, V)

        # --- 构造 multiprocessing 参数列表 ---
        args_list = [(ctc_probs[i].cpu(), anchor[i].unsqueeze(0).cpu(), "cpu") 
                    for i in range(ctc_probs.shape[0])]


        with Pool(processes=64) as pool:  # 根据机器调节进程数
            batch_preds = list(tqdm(pool.imap(kws_search_single, args_list), total=len(args_list), leave=False))
        
        preds.extend(batch_preds)

        del anchor, feat, feat_lengths, label, ctc_probs

    
    print(sorted(preds)[:10])
    import matplotlib.pyplot as plt
    plt.hist(preds, bins=50)
    plt.show()
    plt.savefig('preds_hist.png')