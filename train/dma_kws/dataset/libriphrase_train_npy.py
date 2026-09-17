import random
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append("/nvme01/openkws/qbyt")
from models.text.char_tokenizer import CharTokenizer
from tqdm import tqdm
tqdm.pandas()  # 注册 pandas tqdm 扩展



class LibriPhrasetTRAIN(Dataset):
    def __init__(
        self,
        parquet_file="/nvme01/openkws/libriphrase/counts/ls-460/78k/aggregated_segments_with_g2p_distance.parquet",
        wav_dir="/nvme01/openkws/libriphrase/segments",
    ):
        # 初始化 tokenizer
        self.tokenizer = CharTokenizer(
            "/nvme01/openkws/wenet/examples/librispeech-g2p/s0/data/dict/lang_char.txt",
            None,
            split_with_space=" ",
        )

        # 只读必要列
        self.df = pd.read_parquet(
            parquet_file, columns=["ngram", "ngram_g2p", "clips", "distances"]
        )

        # 用 ngram 作为索引，加速查找，避免 df_dict 占内存
        self.df.set_index("ngram", inplace=True)

        # anchor 列表和映射
        self.anchor_lists = self.df.index.tolist()
        self.anchor_lens = len(self.anchor_lists)
        self.anchor2idx = {anchor: idx for idx, anchor in enumerate(self.anchor_lists)}

        self.wav_dir = wav_dir
        self.cycle_sample = 1000 * self.anchor_lens

    def __len__(self):
        return self.cycle_sample

    def _get_row(self, ngram):
        """惰性解析 row"""
        row = self.df.loc[ngram]
        clips_list = (
            json.loads(row["clips"]) if isinstance(row["clips"], str) else row["clips"]
        )
        distances = list(row["distances"])
        return {
            "ngram_g2p": row["ngram_g2p"],
            "clips_list": clips_list,
            "distances": distances,
        }

    def get_negative(self, index):
        """获取随机负样本"""
        rand = random.randint(0, self.anchor_lens - 2)
        random_index = rand if rand < index else rand + 1

        neg_ngram = self.anchor_lists[random_index]
        neg_inform = self._get_row(neg_ngram)

        neg_clip = random.choice(neg_inform["clips_list"])
        neg_wav = neg_clip["audio_path"]
        neg_g2p = neg_inform["ngram_g2p"]

        return neg_wav, neg_g2p, neg_ngram

    def get_hard_negative(self, hard_neg_lists):
        """获取难负样本"""
        hard_negtive = random.choice(hard_neg_lists)
        hard_ngram = hard_negtive["ngram"]

        hard_inform = self._get_row(hard_ngram)
        hard_clip = random.choice(hard_inform["clips_list"])
        hard_wav = hard_clip["audio_path"]
        hard_g2p = hard_inform["ngram_g2p"]

        return hard_wav, hard_g2p, hard_ngram

    def get_samples(self, anchor_wav, query_wav, anchor_seq, query_g2p, label):
        feats = torch.from_numpy(
            np.load(
                os.path.join(self.wav_dir, query_wav)
                .replace("LP-100", "LP-100-fbank")
                .replace("LP-460", "LP-460-fbank")
                .replace("GP-1000", "GP-1000-fbank")
                .replace(".wav", ".npy")
            )
        )
        
        anchor_feats = torch.from_numpy(
            np.load(
                os.path.join(self.wav_dir, anchor_wav)
                .replace("LP-100", "LP-100-fbank")
                .replace("LP-460", "LP-460-fbank")
                .replace("GP-1000", "GP-1000-fbank")
                .replace(".wav", ".npy")
            )
        )
        
        _, query_seq = self.tokenizer.tokenize(query_g2p)
        seq_label = [1 if x in query_seq else 0 for x in anchor_seq]

        sample = {
            "anchor_seq": torch.tensor(anchor_seq, dtype=torch.long),  # text
            "feat": feats,  # audio
            "anchor_feat": anchor_feats,
            "label": torch.tensor(label, dtype=torch.long),  # label
            "seq_label": torch.tensor(seq_label, dtype=torch.long),  # seq_label
        }
        return sample

    def __getitem__(self, index):
        
        index = index % self.anchor_lens

        mini_batchs = []
        
        # 获取 anchor 信息
        anchor = self.anchor_lists[index]
        anchor_inform = self._get_row(anchor)
        anchor_g2p = anchor_inform["ngram_g2p"]
        anchor_clips = anchor_inform["clips_list"]  # list，直接可用
        
        _, anchor_seq = self.tokenizer.tokenize(anchor_g2p)
        
        # anchor_wav
        anchor_wav = random.choice(anchor_clips)['audio_path']
        
        # pos_1
        query = anchor
        query_wav = random.choice(anchor_clips)['audio_path']
        query_g2p = anchor_g2p
        label = 1
        
        mini_batchs.append(self.get_samples(anchor_wav, query_wav, anchor_seq, query_g2p, label))

        # pos_2
        query = anchor
        query_wav = random.choice(anchor_clips)['audio_path']
        query_g2p = anchor_g2p
        label = 1

        mini_batchs.append(self.get_samples(anchor_wav, query_wav, anchor_seq, query_g2p, label))
        
        # random neg_1
        negative_wav, negative_g2p, negative = self.get_negative(index)
        label = 0
        
        mini_batchs.append(self.get_samples(anchor_wav, negative_wav, anchor_seq, negative_g2p, label))
        
        hard_neg_lists = anchor_inform['distances']
        negative_wav, negative_g2p, negative = self.get_hard_negative(hard_neg_lists)
        label = 0
        
        mini_batchs.append(self.get_samples(anchor_wav, negative_wav, anchor_seq, negative_g2p, label))
        return mini_batchs


def train_collate_fn(mini_batchs):
    
    batch = [item for mini_batch in mini_batchs for item in mini_batch]
    # shuffle
    random.shuffle(batch)

    feats = [item['feat'] for item in batch]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)  # feat 填充0
    feat_lengths = torch.tensor([f.size(0) for f in feats])  # 记录原始长度
    
    anchor_feats = [item['anchor_feat'] for item in batch]
    padded_anchor_feats = pad_sequence(anchor_feats, batch_first=True, padding_value=0)  # anchor_feat 填充0
    anchor_feat_lengths = torch.tensor([f.size(0) for f in anchor_feats])  # 记录原始长度
    
    anchor = [item['anchor_seq'] for item in batch]
    anchor = pad_sequence(anchor, batch_first=True, padding_value=0)  # anchor_seq 填充0 类似<blank>
   
    labels = torch.tensor([item['label'] for item in batch])  # 直接转为Tensor

    seq_labels = [item['seq_label'] for item in batch]
    padded_seq_labels = pad_sequence(seq_labels, batch_first=True, padding_value=-1)  # seq_label 填充-1
    seq_label_mask = (padded_seq_labels != -1).float()  # 生成 mask，-1 填充的部分为 0，其他为 1    

    return {
        "anchor": anchor,
        "anchor_feat": padded_anchor_feats,
        "anchor_feat_lengths": anchor_feat_lengths,  # 原始anchor_feat长度
        "feat": padded_feats,
        "feat_lengths": feat_lengths,  # 原始feat长度
        "label": labels,
        "seq_label": padded_seq_labels,
        "seq_label_mask": seq_label_mask
    }


if __name__ == "__main__":
    dataset = LibriPhrasetTRAIN()
    # 测试下dataloader
    dataloader = DataLoader(dataset, batch_size=128, num_workers=8, shuffle=True, collate_fn=train_collate_fn) 
    from tqdm import tqdm
    for i, data in enumerate(tqdm(dataloader)):
        pass