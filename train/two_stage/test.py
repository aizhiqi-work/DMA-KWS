
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import get_cosine_schedule_with_warmup

from models.encoder import ConformerEncoder
from model import QbyT
import torchmetrics
import numpy as np


import torch
from collections import OrderedDict

def average_checkpoints_lightning(model_paths, output_path):
    # 先加载第一个作为模板
    template = torch.load(model_paths[0], map_location="cpu")
    
    # 拿出 state_dict 部分
    state_dicts = []
    for path in model_paths:
        ckpt = torch.load(path, map_location="cpu")
        state_dicts.append(ckpt["state_dict"])
    
    # 计算平均
    avg_state_dict = OrderedDict()
    for k in state_dicts[0].keys():
        avg_state_dict[k] = sum(d[k].float() for d in state_dicts) / len(state_dicts)
    
    # 用 template 填回
    template["state_dict"] = avg_state_dict
    
    # 保存
    torch.save(template, output_path)
    print(f"✅ 融合后的 Lightning checkpoint 已保存到 {output_path}")



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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    from dataset.libriphrase_test_npy import LibriPhrasetTEST, test_collate_fn
    # 测试集
    test_dataset = LibriPhrasetTEST(types='hard')
    test_dataloader = DataLoader(test_dataset, batch_size=1024, collate_fn=test_collate_fn, shuffle=False, num_workers=16)
    trainer = Trainer(devices=1, accelerator='gpu')
    # for i in range(11000, 28001, 1000):
    #     print("*"*100)
    #     model_path = f'/nvme01/openkws/qbyt/ckpts/init-ls-100/step_step={i:06d}.ckpt'
    #     print(model_path)
    #     wrapper = Wrapper.load_from_checkpoint(model_path)
    #     trainer.test(wrapper, test_dataloader)
    #     print("*"*100)
        
        # 11000: 0.9219094514846802
        # 13000: 0.9186399579048157
        # 14000: 0.9175159335136414
        # 15000: 0.9164063930511475
        # 16000: 0.9170455932617188
        # 17000: 0.9171217679977417
        # 22000: 0.9191782474517822
        # 23000: 0.9172936677932739
        # 25000: 0.9178267121315002
        # 28000: 0.9171013832092285
    
    model_paths = [
        '/nvme01/openkws/qbyt_460/ckpts/init-ls-460/step_step=041000.ckpt', # 0.9219
        '/nvme01/openkws/qbyt_460/ckpts/init-ls-460/step_step=042000.ckpt', # 0.9186
        '/nvme01/openkws/qbyt_460/ckpts/init-ls-460/step_step=043000.ckpt', # 0.9175
        '/nvme01/openkws/qbyt_460/ckpts/init-ls-460/step_step=044000.ckpt', # 0.9164
        '/nvme01/openkws/qbyt_460/ckpts/init-ls-460/step_step=045000.ckpt', # 0.9170
        '/nvme01/openkws/qbyt_460/ckpts/init-ls-460/step_step=046000.ckpt', # 0.9171
        '/nvme01/openkws/qbyt_460/ckpts/init-ls-460/step_step=047000.ckpt', # 0.9192
        '/nvme01/openkws/qbyt_460/ckpts/init-ls-460/step_step=048000.ckpt', # 0.9173
        '/nvme01/openkws/qbyt_460/ckpts/init-ls-460/step_step=049000.ckpt', # 0.9178
        '/nvme01/openkws/qbyt_460/ckpts/init-ls-460/step_step=050000.ckpt', # 0.9171
        
    ]

    avg_state_dict = average_checkpoints_lightning(model_paths, 
                                        output_path="/nvme01/openkws/qbyt_460/ckpts/init-ls-460/avg_10.ckpt")


    wrapper = Wrapper.load_from_checkpoint('/nvme01/openkws/qbyt_460/ckpts/init-ls-460/avg_10.ckpt')
    trainer.test(wrapper, test_dataloader)

    # ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    #        Test metric             DataLoader 0
    # ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    #         test/auc             0.930998682975769
    #         test/eer            0.13705535233020782
    # ──────────────────────────────────────────────────
    
    # 测试集
    test_dataset = LibriPhrasetTEST(types='easy')
    test_dataloader = DataLoader(test_dataset, batch_size=1024, collate_fn=test_collate_fn, shuffle=False, num_workers=16)
    trainer.test(wrapper, test_dataloader)




