
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models.text.char_tokenizer import CharTokenizer
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import get_cosine_schedule_with_warmup
from stage2.models.encoder import ConformerEncoder
from stage2.model import QbyT
import torchmetrics
import numpy as np
import torch
from collections import OrderedDict
from stage2.models.processor import compute_fbank
import torchaudio


class Wrapper(LightningModule):
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
        self.qbyt = QbyT(
            encoder_output_size=144,
            num_embeds=73, # 其中1,2不会使用<unk> <sos/eos>, 0用来做padding
            embed_dim=128,
            post_num_layers=2,
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.auc_metric = torchmetrics.AUROC(task="binary")  # AUC计算
        self.eer_metric = torchmetrics.classification.EER(task="binary") # EER计算
    

    def forward(self, feat, feat_lengths, text):
        encoder_out, _ = self.encoder(feat, feat_lengths)
        logits, text_logits = self.qbyt(encoder_out, text)
        return logits, text_logits


    def training_step(self, batch, batch_idx):
        anchor = batch['anchor']
        feat = batch['feat']
        feat_lengths = batch['feat_lengths']
        label = batch['label']
        seq_label = batch['seq_label']
        seq_label_mask = batch['seq_label_mask']

        # 获取模型输出
        logits, seq_logits = self(feat, feat_lengths, anchor)  # logits：[B], seq_logits: [1B, T]
        
        # # 句子级别的损失
        utt_loss = self.criterion(logits, label.float())

        seq_loss = F.binary_cross_entropy_with_logits(
                seq_logits, seq_label.float(), weight=seq_label_mask, reduction='sum'
        ) / (seq_label_mask.sum() + 1e-6)


        # # 总损失
        total_loss = utt_loss + seq_loss

        # 记录损失
        self.log('train/loss', total_loss, on_step=True, prog_bar=True)
        self.log('train/utt_loss', utt_loss, on_step=True, prog_bar=True)
        self.log('train/seq_loss', seq_loss, on_step=True, prog_bar=True)
        # 学习率
        lr = self.optimizers().param_groups[0]['lr']
        self.log('train/lr', lr, on_step=True, prog_bar=True)
        return total_loss


    def validation_step(self, batch, batch_idx):
        # 从batch中提取信息
        anchor = batch['anchor']
        feat = batch['feat']
        feat_lengths = batch['feat_lengths']
        label = batch['label']

        logits, _ = self(feat, feat_lengths, anchor)

        preds = torch.sigmoid(logits)
        # 更新 AUC 和 EER
        self.auc_metric.update(preds, label)
        self.eer_metric.update(preds, label)

    def on_validation_epoch_end(self):
        # 计算 AUC 和 EER
        auc = self.auc_metric.compute()
        eer = self.eer_metric.compute()

        # 记录结果
        self.log('val/auc', auc, prog_bar=True)
        self.log('val/eer', eer, prog_bar=True)

        # 重置指标
        self.auc_metric.reset()
        self.eer_metric.reset()
        
        
    def test_step(self, batch, batch_idx):
        # 从batch中提取信息
        anchor = batch['anchor']
        feat = batch['feat']
        feat_lengths = batch['feat_lengths']
        label = batch['label']

        logits, _ = self(feat, feat_lengths, anchor)

        preds = torch.sigmoid(logits)
        # 更新 AUC 和 EER
        self.auc_metric.update(preds, label)
        self.eer_metric.update(preds, label)
        
    
    def on_test_epoch_end(self):
        # 计算 AUC 和 EER
        auc = self.auc_metric.compute()
        eer = self.eer_metric.compute()

        # 记录结果
        self.log('test/auc', auc, prog_bar=True)
        self.log('test/eer', eer, prog_bar=True)

        # 重置指标
        self.auc_metric.reset()
        self.eer_metric.reset()


    def configure_optimizers(self):
        # qbyt + encoder
        optimizer = torch.optim.Adam(
            list(self.qbyt.parameters()) + list(self.encoder.parameters()), 
            lr=1e-3
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=2500,  # warmup步骤
            num_training_steps=50000  # 总训练步骤
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 按步更新
                "frequency": 1
            },
        }

        
if __name__ == "__main__":
    pl.seed_everything(2025)
    wrapper = Wrapper.load_from_checkpoint('/nvme01/openkws/qbyt_460v2/ckpts/155k-v2-ft/avg_10.ckpt')
    wrapper.eval()
    print(wrapper)

    anchor_g2p = "HH EY1 S N IH1 P S"
    tokenizer_path = '/nvme01/openkws/wenet/examples/librispeech-g2p/s0/data/dict/lang_char.txt'
    tokenizer = CharTokenizer(tokenizer_path, None, split_with_space=' ')
    _, anchor_seq = tokenizer.tokenize(anchor_g2p)
    

    # /nvme01/openkws/wwd-data/heysnips/hey_snips_research_6k_en_train_eval_clean_ter
    # 读取全部的音频
    import glob
    audio_dir = '/nvme01/openkws/wwd-data/heysnips/hey_snips_research_6k_en_train_eval_clean_ter/audio_files_no_silence'
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    
    preds = []
    from tqdm import tqdm
    for audio_file in tqdm(audio_files):
        sample = {
            'key': 'test',
            'txt': "xxx"
        }
        audio_path = audio_file
        waveform, sample_rate = torchaudio.load(audio_path)
        sample['wav'] = waveform
        sample['sample_rate'] = sample_rate
        sample = compute_fbank(sample, num_mel_bins=80, frame_shift=10, frame_length=25, dither=0.1)
        

        anchor = torch.tensor(anchor_seq).unsqueeze(0)
        feat = sample['feat'].unsqueeze(0)
        feat_lengths = torch.tensor([feat.size(1)])

        logits, _ = wrapper(feat, feat_lengths, anchor)

        pred = torch.sigmoid(logits)

        print(pred)
        preds.append(pred.item())
    
    print(preds)
    print(sum(preds) / len(preds))
    print(max(preds))
    print(min(preds))
    # 画图
    import matplotlib.pyplot as plt
    plt.hist(preds, bins=100)
    plt.show()
    plt.savefig('preds.png')