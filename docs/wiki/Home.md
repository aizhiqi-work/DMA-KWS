# DMA-KWS Wiki

Welcome to the DMA-KWS wiki! Operational guides for the [DMA-KWS](https://github.com/ty4045681/DMA-KWS) repository.

## Pages

- **[Environment Setup](Environment-Setup)** — recommended `uv`-based install with the
  version pins this codebase needs (Python 3.10, torch/torchaudio 2.7.1+cu126,
  torchmetrics >= 1.9), CUDA/V100 notes, verification and troubleshooting.
- **[Stage II Training Pipeline](Stage-II-Training-Pipeline)** — verified end-to-end
  route: data prep (GigaPhrase-1000 + LibriPhrase-460, eval set, MUSAN, LibriSpeech
  train-other-500), icefall + k2 + cuDNN + PYTHONPATH setup, encoder checkpoints and
  configs, single-step smoke results, and the measured time budget.
- **[False-Alarm Evaluation](False-Alarm-Evaluation)** — how to measure 误唤醒率
  (false accepts per 24 h) with `eval_musan_fa.py` on MUSAN and LibriSpeech
  train-other-500, plus the recorded results for the zh-en v3 / v4 / v4.1 trio
  (all 12 remaining checkpoints measured 2026-10-03: MUSAN 0 for every model, LibriSpeech
  0.29 次/24 h only for GS-finetune v2 and GS-base full-context v4.1).
- **[Experiment Log](Experiment-Log)** — archive of **all 15 completed 50k-step runs**:
  the readout ladder (v2 / v3 / v4 / v4.1) × three frozen encoders (zh-en-3M avg-2,
  GigaSpeech KWS finetune, GigaSpeech KWS base at two operating points), with full
  validation curves, best-checkpoint metrics, the cross-encoder ranking (v2 wins
everywhere; zh-en > GS-base > GS-finetune), throughput / co-location measurements,
  data-prep lessons and a record template for new runs.

- **[Paper Stage II Recipe](Paper-Stage-II-Recipe)** — what the authors' code actually does:
  the per-sample 50/50 positive/negative switch, the hard : normal negative weights
  (1:1 by default, 1:100 in the two-stage finetune), and the full initialisation lineage
  (which scripts start from an already trained QbyT via `load_from_checkpoint`) — plus how
  this repository mirrors it and how to reproduce that stage locally.

## Repository docs

- [README](https://github.com/ty4045681/DMA-KWS#readme) — pipeline overview, configs,
  training and evaluation commands.
- `docs/` in the repository — paper reproduction, Stage-II multi-source background,
  QbyT sink-attention diagnostics, offline MUSAN mixing for keyword adaptation
  (`docs/adapt-offline-musan-mix.md`).

---

*These pages are versioned in the repository under `docs/wiki/` and published with
`bash scripts/publish_wiki.sh`.*
