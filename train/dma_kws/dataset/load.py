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



df = pd.read_parquet(
        "/nvme01/openkws/libriphrase/counts/ls-100/aggregated_segments_with_g2p_distance.parquet", 
        columns=["ngram", "ngram_g2p", "clips", "distances"]
    )

print(df)