import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import torchaudio
import numpy as np
import re
import sys
sys.path.append("/nvme01/openkws/qbyt")

import os
from dataset.features import FeatureExtractor
from models.text.char_tokenizer import CharTokenizer
from g2p_en import G2p

class LibriPhraset(Dataset):
    def __init__(self):
        self.feature_extractor = FeatureExtractor(augment=False, wav_dir='')
        self.data = self.read_file_list('/nvme01/openkws_github/eval/data_wwd/hey_snips_pos_test')


    def read_file_list(self, folder_path, ext='.wav'):
        file_list = []
        for root, dirs, files in os.walk(folder_path):
            for _file in files:
                if _file.endswith(ext):
                    file_list.append(os.path.join(root, _file))
        return file_list


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, index):
        query_wav = self.data[index]
        save_path = query_wav.replace("hey_snips_pos_test", "hey_snips_pos_test-fbank").replace('.wav', '.npy')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            return torch.tensor(0)
        try:
            feat = self.feature_extractor.process(query_wav)['feat'].numpy()
            np.save(save_path, feat.astype(np.float32))
            return torch.tensor(0)
        except:
            return torch.tensor(0)


# ---------------- Main ----------------
if __name__ == '__main__':
    test_dataset = LibriPhraset()
    print(len(test_dataset))
    
    dataloader = DataLoader(
        test_dataset,
        batch_size=256,
        num_workers=12,
        shuffle=False,
    )

    from tqdm import tqdm
    for _ in tqdm(dataloader, desc="Saving fbank features"):
        pass

    print("全部 feat 已保存完毕！")