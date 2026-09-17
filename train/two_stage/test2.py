
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

        preds = torch.sigmoid(logits) # [B]

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

    # import os
    # import re
    # import glob

    # ckpt_dir = "/nvme01/openkws/qbyt_1460/ckpts/ft-ls-gs-1460->ls-gs-1460"

    # # 匹配 val_auc 数值，允许结尾有一个 .
    # pattern = re.compile(r"val_auc=([0-9.]+)")

    # ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    # # /nvme01/openkws/qbyt_1460/ckpts/ft-ls-gs-1460->ls-gs-1460/step_step=108000_auc_val_auc=0.974367.ckpt
    # # 按照steps进行排序
    # # print(ckpt_files)
    
    # # # 修复排序逻辑：先提取文件名，再解析步数
    # # def extract_step(filepath):
    # #     filename = os.path.basename(filepath)
    # #     if 'step_step=' in filename:
    # #         step_part = filename.split('step_step=')[1].split('_')[0]
    # #         return int(step_part)
    # #     return 0
    
    # # ckpt_files = sorted(ckpt_files, key=extract_step)
    # # ckpt_files = ckpt_files[::3]=
    
    # ckpt_with_auc = []
    # for ckpt in ckpt_files:
    #     match = pattern.search(os.path.basename(ckpt))
    #     if match:
    #         auc_str = match.group(1).rstrip(".")  # 去掉右侧多余的小数点
    #         auc = float(auc_str)
    #         ckpt_with_auc.append((auc, ckpt))
            

    # # 按 auc 值降序排列
    # ckpt_with_auc.sort(key=lambda x: x[0], reverse=True)

    # # # 取前10个
    # top10 = ckpt_with_auc[:10]
    # model_paths = []

    # print("=== Top 10 ckpts by val_auc ===")
    # for auc, ckpt in top10:
    #     print(f"{ckpt}  --> val_auc={auc:.6f}")
    #     model_paths.append(ckpt)


    
    pl.seed_everything(2025)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
    from dataset.libriphrase_test import LibriPhrasetTEST, test_collate_fn
    # 测试集
    test_dataset = LibriPhrasetTEST(types='hard')
    test_dataloader = DataLoader(test_dataset, batch_size=1024, collate_fn=test_collate_fn, shuffle=False, num_workers=16)
    trainer = Trainer(devices=1, accelerator='gpu')
    
    # avg_state_dict = average_checkpoints_lightning(model_paths, 
    #                                     output_path=f"{ckpt_dir}/avg_10.ckpt")


    wrapper = Wrapper.load_from_checkpoint(f'/nvme01/openkws/qbyt_460v2/ckpts/155k-v2/avg_10.ckpt')
    trainer.test(wrapper, test_dataloader)

    # # # # ──────────────────────────────────────────────────
    # # # #        Test metric             DataLoader 0
    # # # # ──────────────────────────────────────────────────
    # # # #         test/auc             0.930998682975769
    # # # #         test/eer            0.13705535233020782
    # # # # ──────────────────────────────────────────────────
    # # # 测试集
    # test_dataset = LibriPhrasetTEST(types='easy')
    # test_dataloader = DataLoader(test_dataset, batch_size=1024, collate_fn=test_collate_fn, shuffle=False, num_workers=16)
    # trainer.test(wrapper, test_dataloader)




