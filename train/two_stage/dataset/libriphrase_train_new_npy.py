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
        parquet_file="/nvme01/openkws/libriphrase/counts/ls-gs-1460/processed_data.parquet",
        negative_ratio=1, 
        hard_negative_ratio=1,
        augment=True,
        wav_dir="/nvme01/openkws/libriphrase/segments",
        sample_lens=5000  # 允许用户指定数据集的长度
    ):
        self.tokenizer = CharTokenizer('/nvme01/openkws/wenet/examples/librispeech-g2p/s0/data/dict/lang_char.txt', None, split_with_space=' ')
        self.df = pd.read_parquet(
            parquet_file, 
            columns=["ngram", "ngram_g2p", "clips_file", "distances_file"]
        )
        self.anchor_lists = self.df["ngram"].tolist()
        print(len(self.anchor_lists))
        self.anchor2idx = {anchor: idx for idx, anchor in enumerate(self.anchor_lists)}
        self.negative_ratio = negative_ratio
        self.hard_negative_ratio = hard_negative_ratio
        self.sample_lens = sample_lens
        self.wav_dir = wav_dir


    def __len__(self):
        return self.sample_lens

    
    def get_random_clips(self, clips_file):
        number = int(os.path.basename(clips_file).split('-')[-2])
        random_number = random.randint(0, number - 1)
        clips_data = np.load(clips_file, allow_pickle=True)
        return clips_data[random_number]

    def get_random_distances(self, distances_file):
        number = int(os.path.basename(distances_file).split('-')[-2])
        if number == 0:
            return None
        random_number = random.randint(0, number - 1)
        distances_data = np.load(distances_file, allow_pickle=True)
        return distances_data[random_number]



    def get_negative(self, index):
        """ 获取随机负样本 """
        remaining_indices = [i for i in range(len(self.anchor_lists)) if i != index]
        random_index = random.choice(remaining_indices)
        
        negtive = self.anchor_lists[random_index]
        negtive_inform = self.df.iloc[random_index]
        negtive_clips = negtive_inform['clips_file']
        negtive_wav = self.get_random_clips(negtive_clips)
        negtive_g2p = negtive_inform['ngram_g2p']
        return negtive_wav, negtive_g2p, negtive


    def get_hard_negative(self, hard_negtive):
        """ 获取难负样本 """
        hard_ngram = hard_negtive['ngram']

        idx = self.anchor2idx[hard_ngram]
        hard_ngram_inform = self.df.iloc[idx]

        hard_ngram_wav = self.get_random_clips(hard_ngram_inform['clips_file'])
        hard_ngram_g2p = hard_ngram_inform['ngram_g2p']
        return hard_ngram_wav, hard_ngram_g2p, hard_ngram


    def __getitem__(self, index):
        # % index 循环
        index = index % len(self.anchor_lists)
        # anchor
        anchor = self.anchor_lists[index]
        anchor_inform = self.df.iloc[index]
        anchor_g2p = anchor_inform['ngram_g2p']
        label = 1
        # 随机正负
        if random.random() < 0.5:
            label = 1
            anchor_clips = anchor_inform['clips_file']

            query = anchor
            query_wav = self.get_random_clips(anchor_clips)['audio_path']
            query_g2p = anchor_g2p
        else:
            label = 0
            hard_neg = self.get_random_distances(anchor_inform['distances_file'])
            # 随机选择负样本类型 (1: 随机负样本, 2: 难负样本)
            negative_type = random.choices([1, 2], weights=[self.negative_ratio, self.hard_negative_ratio], k=1)[0]
            if negative_type == 1 or hard_neg is None: # 随机负样本
                negative_wav, negative_g2p, negative = self.get_negative(index)
            else: # 难负样本
                negative_wav, negative_g2p, negative = self.get_hard_negative(hard_neg)
                
            query = negative
            query_wav = negative_wav['audio_path']
            query_g2p = negative_g2p

        feats = torch.from_numpy(np.load(os.path.join(self.wav_dir, query_wav).replace('LP-460', 'LP-460-fbank').replace('GP-1000', 'GP-1000-fbank').replace('.wav', '.npy')))

        
        _, anchor_seq = self.tokenizer.tokenize(anchor_g2p)
        _, query_seq = self.tokenizer.tokenize(query_g2p)

        # 做一个序列存在与否的标签，即如果序列存在，标签为1，否则为0
        seq_label = []
        for i in range(len(anchor_seq)):
            if anchor_seq[i] in query_seq:
                seq_label.append(1)
            else:
                seq_label.append(0)

        # print(query_wav)
        sample = {
            "anchor_seq": torch.tensor(anchor_seq, dtype=torch.long), # text
            "feat": feats, # audio
            "label": torch.tensor(label, dtype=torch.long),   # label
            "seq_label": torch.tensor(seq_label, dtype=torch.long) # seq_label
        }
        return sample


def train_collate_fn(batch):
   
    # Padding 特征
    feats = [item['feat'] for item in batch]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)  # feat 填充0
    feat_lengths = torch.tensor([f.size(0) for f in feats])  # 记录原始长度

    anchor = [item['anchor_seq'] for item in batch]
    anchor = pad_sequence(anchor, batch_first=True, padding_value=0)  # anchor_seq 填充0 类似<blank>
   
    labels = torch.tensor([item['label'] for item in batch])  # 直接转为Tensor

    seq_labels = [item['seq_label'] for item in batch]
    padded_seq_labels = pad_sequence(seq_labels, batch_first=True, padding_value=-1)  # seq_label 填充-1
    seq_label_mask = (padded_seq_labels != -1).float()  # 生成 mask，-1 填充的部分为 0，其他为 1    

    return {
        "anchor": anchor,
        "feat": padded_feats,
        "feat_lengths": feat_lengths,  # 原始feat长度
        "label": labels,
        "seq_label": padded_seq_labels,
        "seq_label_mask": seq_label_mask
    }


if __name__ == "__main__":
    dataset = LibriPhrasetTRAIN(sample_lens=50000)
    print(dataset[0])
    
    # # # # 测试下dataloader
    # dataloader = DataLoader(dataset, batch_size=256, num_workers=16, shuffle=True, collate_fn=train_collate_fn) 
    # # # for batch in dataloader:
    # # #     pass

    # from tqdm import tqdm
    # for i, data in enumerate(tqdm(dataloader)):
    #     pass
