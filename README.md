# DMA-KWS

**DMA-KWS (Under review at TASLP)** Effective User-defined Keyword Spotting with Dual-stage Matching, Multi-modal Enrollment, and Continual Adaptation**.

Existing KWS methods often suffer from poor discrimination among confusable words, speaker-dependent pronunciation variations, and high data requirements. DMA-KWS addresses these issues with a **dual-stage matching and adaptation framework**.

---

## 🔑 Key Features

* **Dual-stage matching**

  * **Stage I**: CTC decoding with streaming phoneme search to locate candidate segments
  * **Stage II**: QbyT-based phoneme matching for fine-grained verification
  * Effectively distinguishes highly confusable keywords

* **Multi-modal enrollment**

  * Fuses user speech and text embeddings
  * Improves speaker-dependent keyword spotting performance

* **Parameter-efficient adaptation**

  * Lightweight continual adaptation with only **187k trainable parameters**
  * Supports both synthetic and real data
  * Suitable for on-device deployment

---

## 📊 Performance

* **LibriPhrase (Hard subset)**

  * **97.85% AUC**, **6.13% EER** (SOTA)

* Consistently outperforms text-only enrollment in speaker-dependent settings

---

## 📂 Open-source Contents

This repository provides:

* ✅ Two-stage training code for DMA-KWS
* ✅ Data processing scripts
* 🚧 Pre-trained checkpoints (coming soon)

### Datasets

* **GigaPhrase-1000**: [https://github.com/aizhiqi-work/GigaPhrase-1000](https://github.com/aizhiqi-work/GigaPhrase-1000)
* **LibriPhrase**: used for evaluation

---