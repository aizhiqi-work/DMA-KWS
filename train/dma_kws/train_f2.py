
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
from model_f2 import QbyT
import torchmetrics
import numpy as np



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
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.qbyt = QbyT(
            encoder_output_size=144,
            num_embeds=73, # 其中1,2不会使用<unk> <sos/eos>, 0用来做padding
            embed_dim=128,
            post_num_layers=2,
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.auc_metric = torchmetrics.AUROC(task="binary")  # AUC计算
        self.eer_metric = torchmetrics.classification.EER(task="binary") # EER计算
    

    def forward(self, feat, feat_lengths, text, anchor_feat, anchor_feat_lengths):
        encoder_out, _ = self.encoder(feat, feat_lengths)
        anchor_encoder_out, _ = self.encoder(anchor_feat, anchor_feat_lengths)
        logits, text_logits = self.qbyt(encoder_out, text, anchor_encoder_out)
        return logits, text_logits


    def training_step(self, batch, batch_idx):
        anchor = batch['anchor']
        feat = batch['feat']
        feat_lengths = batch['feat_lengths']
        anchor_feat = batch['anchor_feat']
        anchor_feat_lengths = batch['anchor_feat_lengths']
        label = batch['label']
        seq_label = batch['seq_label']
        seq_label_mask = batch['seq_label_mask']

        # 获取模型输出
        logits, seq_logits = self(feat, feat_lengths, anchor, anchor_feat, anchor_feat_lengths)  # logits：[B], seq_logits: [1B, T]
        
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
        cross_lr = self.optimizers().param_groups[1]['lr']
        self.log('train/lr', lr, on_step=True, prog_bar=True)
        self.log('train/cross_lr', cross_lr, on_step=True, prog_bar=True)
        return total_loss


    def validation_step(self, batch, batch_idx):
        # 从batch中提取信息
        anchor = batch['anchor']
        feat = batch['feat']
        feat_lengths = batch['feat_lengths']
        anchor_feat = batch['anchor_feat']
        anchor_feat_lengths = batch['anchor_feat_lengths']
        label = batch['label']

        logits, _ = self(feat, feat_lengths, anchor, anchor_feat, anchor_feat_lengths)

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
        self.log('val_auc', auc, prog_bar=True)
        # 重置指标
        self.auc_metric.reset()
        self.eer_metric.reset()


    def configure_optimizers(self):

        # cross-attn 的参数
        cross_attn_params = list(self.qbyt.cross_attn.parameters())

        # 其他参数（不包括 cross_attn）
        other_params = []
        for name, param in self.qbyt.named_parameters():
            # 跳过 cross_attn
            if not name.startswith("cross_attn"):
                other_params.append(param)

        optimizer = torch.optim.Adam(
            [
                {"params": other_params, "lr": 4e-4},
                {"params": cross_attn_params, "lr": 1e-3},   # cross_attn 学习率更大
            ]
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=2500,
            num_training_steps=50000
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


        
if __name__ == "__main__":
    pl.seed_everything(2025)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

    from dataset.libriphrase_train_npy import LibriPhrasetTRAIN, train_collate_fn
    from dataset.libriphrase_test_npy import SDLibriPhrasetTEST, test_collate_fn

    # 使用 sample_lens 来控制数据集的长度
    train_dataset = LibriPhrasetTRAIN(
        parquet_file="/nvme01/openkws/libriphrase/counts/ls-gs-1460/aggregated_segments_with_g2p_distance.parquet",
    )

    # 测试集
    test_dataset = SDLibriPhrasetTEST(
        types='hard'
    )

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=512, collate_fn=train_collate_fn, shuffle=True, num_workers=16, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, collate_fn=test_collate_fn, shuffle=False, num_workers=8, drop_last=True)

    # 模型检查点
    model_checkpoint = ModelCheckpoint(
        dirpath="/nvme01/openkws_github/train/ckpts/155k-mm-f2",
        filename="step_{step:06d}_auc_{val_auc:.6f}",
        save_top_k=-1,
        save_on_train_epoch_end=False,  # 按训练步数保存
        every_n_train_steps=1000       # 每1000步保存一次
    )

    # TensorBoard日志
    logger = pl.loggers.TensorBoardLogger('/nvme01/openkws_github/train/logs', name='155k-mm-f2')

    # 包装器
    wrapper = Wrapper.load_from_checkpoint(
        "/nvme01/openkws/qbyt_460v2/ckpts/155k-v2-ft/avg_10.ckpt",
        strict=False,
    )

    # Trainer 设置
    trainer = Trainer(
        devices=4, 
        accelerator='gpu',
        strategy='ddp',  # 使用分布式数据并行训练
        logger=logger,
        max_steps=50000,  # 训练500000步
        callbacks=[model_checkpoint],
        accumulate_grad_batches=1,  # 每两步累积一次梯度，2048 batchsize 512 * 4 * 1
        gradient_clip_val=1.0,  # 设置梯度裁剪阈值为1.0
        check_val_every_n_epoch=None,  # 不按 epoch 验证
        val_check_interval=1000,  # 每1000步验证一次
    )

    # 开始训练
    trainer.fit(wrapper, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)





