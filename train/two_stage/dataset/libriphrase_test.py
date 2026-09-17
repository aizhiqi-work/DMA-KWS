import torch
from torch.utils.data import Dataset, DataLoader

import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import torchaudio
import pickle
import numpy as np
from torchaudio.compliance.kaldi import fbank
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import re

import sys
sys.path.append("/nvme01/openkws/qbyt")
from dataset.features import FeatureExtractor
from models.text.char_tokenizer import CharTokenizer
from g2p_en import G2p


class LibriPhrasetTEST(Dataset):
    def __init__(
        self,
        test_dir="/nvme01/openkws/libriphrase/eval/",
        csv=[
            "evaluation_set/libriphrase_diffspk_all_1word.csv",
            "evaluation_set/libriphrase_diffspk_all_2word.csv",
            "evaluation_set/libriphrase_diffspk_all_3word.csv",
            "evaluation_set/libriphrase_diffspk_all_4word.csv"
        ],
        types='easy',
        save_path='/nvme01/openkws/libriphrase/eval/evaluation_set/test_all_phrase.csv',
        augment=False,
    ):
        if not os.path.exists(save_path):
            self.data = pd.DataFrame(columns=['anchor_text', 'anchor', 'anchor_dur', 'comparison_text', 'comparison', 'comparison_dur', 'target', 'type'])
            for path in csv:
                n_word = os.path.join(test_dir, path)
                df = pd.read_csv(n_word)
                anc = df[['anchor_text', 'anchor', 'anchor_dur', 'comparison_text', 'comparison', 'comparison_dur', 'target', 'type']]
                com = df[['comparison_text', 'comparison', 'comparison_dur', 'anchor_text', 'anchor', 'anchor_dur', 'target', 'type']]
                self.data = self.data._append(anc.rename(columns={y: x for x, y in zip(self.data.columns, anc.columns)}), ignore_index=True)
                self.data = self.data._append(com.rename(columns={y: x for x, y in zip(self.data.columns, com.columns)}), ignore_index=True)
            self.data.to_csv(save_path, index=False)
        else:
            self.data = pd.read_csv(save_path)
        if types == 'easy':
            self.data = self.data.loc[self.data['type'].isin(['diffspk_easyneg', 'diffspk_positive'])]
        elif types == 'hard':
            self.data = self.data.loc[self.data['type'].isin(['diffspk_hardneg', 'diffspk_positive'])]
        
        self.data = self.data.values.tolist()
        self.test_dir = test_dir

        self.tokenizer = CharTokenizer('/nvme01/openkws/wenet/examples/librispeech-g2p/s0/data/dict/lang_char.txt', None, split_with_space=' ')
        self.feature_extractor = FeatureExtractor(augment=augment, wav_dir=test_dir)
        self.g2p = G2p()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        _anchor_text, _anchor, _, _comparison_text, _comparison, _, _target, _ = self.data[index]
        # carriage train-other-500/train-other-500/1006/135212/1006-135212-0068_0.wav recognized train-other-500/train-other-500/2541/185493/2541-185493-0021_0.wav 0
        anchor = _anchor_text
        anchor_phones = self.g2p(re.sub(r'[^\w\s]', '', _anchor_text.lower()))
        anchor_g2p = ' '.join([phone for phone in anchor_phones if phone != ' '])

        query = _comparison_text
        query_wav = _comparison
        feats = self.feature_extractor.process(query_wav)

        _, anchor_seq = self.tokenizer.tokenize(anchor_g2p)

        sample = {
            "anchor_seq": torch.tensor(anchor_seq, dtype=torch.long), # text            
            "feat": feats['feat'], # audio
            "label": torch.tensor(_target, dtype=torch.long),   # label
        }

        return sample


def test_collate_fn(batch):
   
    # Padding 特征
    feats = [item['feat'] for item in batch]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)  # feat 填充0
    feat_lengths = torch.tensor([f.size(0) for f in feats])  # 记录原始长度

    anchor = [item['anchor_seq'] for item in batch]
    anchor = pad_sequence(anchor, batch_first=True, padding_value=0)  # anchor_seq 填充0 类似<blank>
   
    labels = torch.tensor([item['label'] for item in batch])  # 直接转为Tensor

    
    return {
        "anchor": anchor,
        "feat": padded_feats,
        "feat_lengths": feat_lengths,  # 原始feat长度
        "label": labels,
    }




if __name__ == '__main__':
    test_dataset = LibriPhrasetTEST()
    # 测试下dataloader
    dataloader = DataLoader(test_dataset, batch_size=256, num_workers=8, shuffle=True, collate_fn=test_collate_fn) 
    # for batch in dataloader:
    #     pass

    from tqdm import tqdm
    for i, data in enumerate(tqdm(dataloader)):
        pass
