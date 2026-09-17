# OpenKWS — User-Defined Keyword Spotting

本仓库是以下两篇工作的开源实现（代码与 checkpoint，不含数据）：

- **DMA-KWS (TASLP, under review)** — *Effective User-defined Keyword Spotting with Dual-stage Matching, Multi-modal Enrollment, and Continual Adaptation*
- **Dual Data Scaling (ICASSP)** — *Dual Data Scaling for Robust Two-Stage User-Defined Keyword Spotting*

现有 KWS 方法往往在易混淆词、说话人相关发音差异、以及数据需求高这三个问题上表现不佳。本工作用一个 **两阶段匹配 + 适配** 的框架来解决：

---

## 方法简介

**两阶段匹配（Dual-stage matching）**

- **Stage I**：流式 CTC 音素搜索定位候选片段。支持多种解码策略：
  - **CDC-KWS**：*Streaming Keyword Spotting Boosted by Cross-layer Discrimination Consistency*（跨层判别一致性融合）
  - **MFA-KWS**：*MFA-KWS: Effective Keyword Spotting with Multi-head Frame-asynchronous Decoding*（CTC + TDT 融合）
  - **MFS-KWS**：CTC + RNN-T 融合
  - 底层 CTC 帧同步搜索做了 **numba 加速 + 竞争解码（多关键词批量）**
- **Stage II**：QbyT（query-by-text）音素匹配，对候选做细粒度校验，有效区分易混淆关键词。

**多模态注册（Multi-modal enrollment）**

- 融合用户语音（enrollment audio）与文本嵌入，提升说话人相关的关键词识别效果。

**参数高效适配（Parameter-efficient adaptation）**

- 轻量持续适配，仅约 **187k 可训练参数**，支持合成/真实数据，适合端侧部署。

---

## 性能

- **LibriPhrase (hard)**：**97.85% AUC**，**6.13% EER**（SOTA）。
- Dual Data Scaling 在各数据规模下的 P-WER / AUC / EER 对比见 `openkws` 论文与 `train/two_stage` 实验目录。

---

## 目录结构

```
openkws_github/
├── README.md
├── decode/                      # Stage-1 解码 + 评估
│   ├── KWStreamingSearch/        #   搜索库
│   │   ├── CTC/                  #     CTC 帧同步搜索（torch + numba + 竞争批量）
│   │   │   └── cdc_streaming_search.py   # CDC-KWS
│   │   ├── MFA/                  #     MFA-KWS / MFS-KWS 包装
│   │   └── Transducer/           #     RNN-T / TDT 搜索
│   ├── models/                   #   Conformer CTC ASR（加载 stage-1 ckpt）
│   ├── visual-v*.py              #   流式 / 竞争解码可视化与评估
│   ├── stage1_*.py               #   stage-1 召回 / 精确率评估
│   ├── save/inference*.py        #   端到端推理脚本
│   └── stage2/                   #   stage-2（QbyT）测试
├── train/
│   ├── stage1_wenet/             # Stage-1 WeNet 训练 recipe + 库
│   ├── two_stage/                # Dual Data Scaling 两阶段训练
│   └── dma_kws/                  # DMA-KWS 多模态注册训练（enr_audio）
└── ckpts/
    ├── stage1/ls-gs-1460.pt      # 1460h Conformer CTC 声学模型
    └── stage2/
        ├── 155k-v2-ft.ckpt        # Dual Data Scaling stage2 最优（155k anchors）
        ├── 155k-mm-f1.ckpt       # DMA-KWS 多模态融合 f1
        └── 155k-mm-f2.ckpt       # DMA-KWS 多模态融合 f2
```

---

## Stage-1 解码与测速

底层 CTC 帧同步搜索 `CTC/ctc_streaming_search.py` 为纯 PyTorch 实现，`ctc_streaming_search_numba*.py` 为对应的 numba 加速版本（含 pruning、logsum、流式、多关键词批量）。

CPU 基准（V=73，关键词长度 U=5，T=1000 帧，均值）：

| 方法 | 耗时 | 吞吐 | 加速比 |
|---|---|---|---|
| torch full-seq（基线） | 591.3 ms | 1.69 kfps | 1.0x |
| numba full-seq | 0.07 ms | 13.8 Mfps | ~8167x |
| numba streaming | 5.00 ms | 200 kfps | ~118x |

竞争解码（多关键词并行，numba batch）：

| 关键词数 K | 耗时 | 吞吐 |
|---|---|---|
| 10（torch 逐词基线 6560.8 ms） | 2.83 ms | 3.54 M 词-帧/s（约 23205x） |
| 100 | 5.02 ms | 19.9 M 词-帧/s |
| 1000 | 23.88 ms | 41.9 M 词-帧/s |
| 5000 | 109.73 ms | 45.6 M 词-帧/s |

> 说明：MFA/MFS 包装层当前引用 `CTCPsdStreamingSearch`（prefix-synchronous CTC 解码），该分支未随本仓库提供；实际提供并做了 numba 加速与竞争解码的是帧同步搜索 `CTCFsdStreamingSearch`。

---

## 数据集（需自行获取）

- **GigaPhrase-1000**（155k anchors）：https://github.com/aizhiqi-work/GigaPhrase-1000
- **LibriPhrase-100**（12k anchors）：HuggingFace `ZhiqiAi/LibriPhrase-100`
- **LibriPhrase-460**（78k anchors）：HuggingFace `ZhiqiAi/LibriPhrase-460`

---

## Checkpoint 说明

- `ckpts/stage1/ls-gs-1460.pt`：Stage-1 声学模型（LibriSpeech clean-100/360 + GigaSpeech-1000 共 1460 小时，Conformer CTC，avg-10）。
- `ckpts/stage2/155k-v2-ft.ckpt`：Dual Data Scaling 的 Stage-2 最优模型（155k anchors，微调）。
- `ckpts/stage2/155k-mm-f1.ckpt` / `155k-mm-f2.ckpt`：DMA-KWS 多模态注册模型（两个融合变体）。


---

## Stage-1 训练（WeNet）

Stage-1 是 Conformer CTC 声学模型，训练复用 WeNet（`wenet-e2e/wenet`，基于 commit `d2156645`），并新增自定义的 `librispeech-g2p` recipe（G2P 音素化）。

- 代码位置：`train/stage1_wenet/`
  - `wenet/`：WeNet Python 库（含本项目对 `bin/train.py`、`dataset/dataset.py`、`dataset/processor.py`、`text/char_tokenizer.py`、`utils/init_dataset.py`、`utils/init_model.py` 的改动）
  - `examples/librispeech-g2p/s0/`：训练 recipe
    - `conf/train_conformer_1460.yaml`：1460h（LibriSpeech clean-100/360 + GigaSpeech-1000）配置
    - `conf/train_conformer_460.yaml` / `train_conformer_710.yaml` / `train_conformer.yaml` 等其它规模配置
    - `run_1460.sh` / `run_460.sh` / `run.sh` 等训练脚本
    - `data/dict/lang_char.txt`：音素字典（tokenizer 符号表）
    - `local/`：数据准备脚本
- 未包含：数据清单 `data/lists/*.list`（约 956M）与训练产物 `exp/`、`tensorboard/`，需按 `local/` 脚本从 LibriSpeech / GigaSpeech 重新生成。
- 对应 checkpoint：`ckpts/stage1/ls-gs-1460.pt`（由 `exp/ls-gs-1460-ckpts/avg_10.pt` 重命名）。
