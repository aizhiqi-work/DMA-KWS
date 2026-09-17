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
sys.path.append("/nvme01/openkws_github/eval/dskws")
from dataset.features import FeatureExtractor
from models.text.char_tokenizer import CharTokenizer
from g2p_en import G2p


class QualComm_TEST(Dataset):
    def __init__(
        self,
        eval_path='/nvme01/openkws_github/eval/Q.csv',
        test_dir='/nvme01/openkws_github/eval/data/Qcomm/Qualcomm-fbank',
        augment=False
    ):
        self.data = pd.read_csv(eval_path)
        self.data = self.data.values.tolist()
        self.test_dir = test_dir
        self.tokenizer = CharTokenizer('/nvme01/openkws/wenet/examples/librispeech-g2p/s0/data/dict/lang_char.txt', None, split_with_space=' ')
        self.feature_extractor = FeatureExtractor(augment=augment, wav_dir=test_dir)
        self.g2p = G2p()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor, _, anchor_text, comparison, _,  comparison_text, label = self.data[index]
        
        anchor_phones = self.g2p(re.sub(r'[^\w\s]', '', anchor_text.lower()))
        anchor_g2p = ' '.join([phone for phone in anchor_phones if phone != ' '])
        _, anchor_seq = self.tokenizer.tokenize(anchor_g2p)
        
        query_phones = self.g2p(re.sub(r'[^\w\s]', '', comparison_text.lower()))
        query_g2p = ' '.join([phone for phone in query_phones if phone != ' '])
        _, query_seq = self.tokenizer.tokenize(query_g2p)
        
        
        # feats = self.feature_extractor.process(comparison)
        feats = torch.from_numpy(np.load(os.path.join(self.test_dir, comparison).replace('.wav', '.npy')))
        
        # anchor_feats = self.feature_extractor.process(anchor)
        anchor_feats = torch.from_numpy(np.load(os.path.join(self.test_dir, anchor).replace('.wav', '.npy')))

        
        sample = {    
            "anchor_seq": torch.tensor(anchor_seq, dtype=torch.long), # text            
            "feat": feats, # audio
            "anchor_feats": anchor_feats,
            "label": torch.tensor(label, dtype=torch.long),   # label
        }

        return sample


def test_collate_fn(batch):
   
    # Padding 特征
    feats = [item['feat'] for item in batch]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)  # feat 填充0
    feat_lengths = torch.tensor([f.size(0) for f in feats])  # 记录原始长度

    anchor = [item['anchor_seq'] for item in batch]
    anchor = pad_sequence(anchor, batch_first=True, padding_value=0)  # anchor_seq 填充0 类似<blank>
   
    anchor_feats = [item['anchor_feats'] for item in batch]
    padded_anchor_feats = pad_sequence(anchor_feats, batch_first=True, padding_value=0)  # feat 填充0
    anchor_feat_lengths = torch.tensor([f.size(0) for f in anchor_feats])  # 记录原始长度
   
   
    labels = torch.tensor([item['label'] for item in batch])  # 直接转为Tensor

    
    return {
        "anchor": anchor,
        "feat": padded_feats,
        "feat_lengths": feat_lengths,  # 原始feat长度
        "anchor_feat": padded_anchor_feats,
        "anchor_feat_lengths": anchor_feat_lengths,
        "label": labels,
    }




if __name__ == '__main__':
    test_dataset = QualComm_TEST()
    print(test_dataset[0])
    # 测试下dataloader
    dataloader = DataLoader(test_dataset, batch_size=256, num_workers=8, shuffle=True, collate_fn=test_collate_fn) 
    # for batch in dataloader:
    #     pass

    from tqdm import tqdm
    for i, data in enumerate(tqdm(dataloader)):
        pass
