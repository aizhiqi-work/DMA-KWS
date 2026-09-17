import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
from g2p_en import G2p

# 模型相关导入
from models.encoder import ConformerEncoder
from models.ctc import CTC
from models.processor import compute_fbank
from models.text.char_tokenizer import CharTokenizer
from KWStreamingSearch.CTC.ctc_streaming_search import CTCFsdStreamingSearch

# ================== 1. 配置与模型加载 ==================
device = torch.device("cpu")
ckpt_path = "/nvme01/soundlab/kws/openkws/wenet/examples/librispeech-g2p/s0/exp/ls-gs-1460-ckpts/avg_10.pt"
dict_path = "/nvme01/soundlab/kws/openkws/wenet/examples/librispeech-g2p/s0/data/dict/lang_char.txt"
audio_path = "/nvme01/soundlab/kws/openkws/librispeech/test/test-clean/4970/29095/4970-29095-0038.wav"

# 输入配置
input_text = "EMMA WOODHOUSE HANDSOME CLEVER AND RICH WITH A COMFORTABLE HOME AND HAPPY DISPOSITION SEEMED TO UNITE SOME OF THE BEST BLESSINGS OF EXISTENCE"
target_word = "HANDSOME"

class Stage1(nn.Module):
    def __init__(self):
        super().__init__()
        # 保持你原始代码中的所有参数，特别是 cnn_module_kernel
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
        self.ctc = CTC(
            odim=73,
            encoder_output_size=144,
            blank_id=0
        )
    
    def ctc_logprobs(self, encoder_out: torch.Tensor):
        return self.ctc.log_softmax(encoder_out)

def load_system(ckpt_path, dict_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = Stage1()
    
    # 保持你原始代码的权重提取逻辑
    encoder_ckpt = {k.replace('encoder.', ''): v for k, v in ckpt.items() if k.startswith('encoder')}
    ctc_ckpt = {k.replace('ctc.', ''): v for k, v in ckpt.items() if k.startswith('ctc')}
    
    model.encoder.load_state_dict(encoder_ckpt)
    model.ctc.load_state_dict(ctc_ckpt)
    model.eval().to(device)
    
    tokenizer = CharTokenizer(dict_path, None, split_with_space=' ')
    return model, tokenizer

model, tokenizer = load_system(ckpt_path, dict_path, device)
g2p = G2p()

# ================== 2. 音频预处理与特征提取 ==================
waveform, sample_rate = torchaudio.load(audio_path)
if sample_rate != 16000:
    waveform = T.Resample(sample_rate, 16000)(waveform)

silence_duration = 0.5 
num_silence_samples = int(silence_duration * sample_rate)
# 保持与原音频相同的通道数 (Channels, Time)
silence = torch.zeros((waveform.shape[0], num_silence_samples))

# 4. 拼接静音 (加在音频末尾)
waveform = torch.cat([waveform, silence], dim=1)

sample = {'wav': waveform, 'sample_rate': 16000, 'key': 'a3023974-4fb9-4468-a702-5799f96d20c8'}
feat_dict = compute_fbank(sample, num_mel_bins=80, frame_shift=10, frame_length=25, dither=0.0)
feat = feat_dict['feat'].unsqueeze(0).to(device)
feat_lengths = torch.tensor([feat.size(1)], device=device)

# ================== 3. 前向计算 ==================
with torch.no_grad():
    encoder_out, encoder_mask = model.encoder(feat, feat_lengths)
    ctc_probs = model.ctc_logprobs(encoder_out)

T_mel, T_enc = feat.size(1), encoder_out.size(1)
subsample_rate = int(round(T_mel / T_enc))

# ================== 4. 强制对齐与边界计算 ==================
# 自动合并 G2P 音素
full_phonemes = " ".join(g2p(input_text))
_, full_ids = tokenizer.tokenize(full_phonemes)

def get_word_boundaries(emission, tokens, words_text, subsample_rate):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
    token_spans = torchaudio.functional.merge_tokens(alignments[0], scores[0].exp())
    
    boundaries = []
    token_ptr = 0
    for w in words_text.split():
        p_list = g2p(w)
        num_tokens = len(p_list)
        start = token_spans[token_ptr].start * subsample_rate
        end = token_spans[token_ptr + num_tokens - 1].end * subsample_rate
        boundaries.append({"word": w, "start": start, "end": end})
        token_ptr += num_tokens
    
    # 柔化边界
    for i in range(len(boundaries)-1):
        mid = (boundaries[i]["end"] + boundaries[i+1]["start"]) / 2
        boundaries[i]["end"] = boundaries[i+1]["start"] = mid
    return boundaries

word_boundaries = get_word_boundaries(ctc_probs, full_ids, input_text, subsample_rate)

# ================== 5. KWS 流式搜索与评分 ==================
kws_streaming_search = CTCFsdStreamingSearch(blank=0)
target_phonemes = g2p(target_word)
target_phonemes_str = " ".join(target_phonemes)
_, target_ids = tokenizer.tokenize(target_phonemes_str)
target_ids_tensor = torch.tensor(target_ids, dtype=torch.long).unsqueeze(0)

log_posteriors, normed_logscores, _, _, _ = kws_streaming_search(
    ctc_probs, target_ids_tensor, 
    torch.tensor([ctc_probs.shape[1]]), torch.tensor([target_ids_tensor.shape[1]])
)

# 打印分数
max_idx = np.array(normed_logscores).argmax().item()
max_score = np.exp(normed_logscores[max_idx])
print(f"\n[Target]: {target_word} | [Max Confidence]: {max_score:.4f}")

# ================== 6. 可视化 ==================
# 自动生成带 Φ 的 ylabel
ylabel = ['Φ']
for p in target_phonemes:
    ylabel.extend([p, 'Φ'])

log_alpha_matrix = np.exp(log_posteriors.detach().cpu().numpy()[0])

plt.figure(figsize=(18, 5))
ax = sns.heatmap(log_alpha_matrix.T, cmap="inferno", xticklabels=10, yticklabels=ylabel)
ax.figure.axes[-1].set_position([0.76, 0.15, 0.05, 0.7])
ax.invert_yaxis()

for w in word_boundaries:
    plt.axvline(w["start"] / subsample_rate, color='white', linestyle='--', linewidth=1.5)
    # plt.text((w["start"] + w["end"]) / (2 * subsample_rate), log_alpha_matrix.shape[1] + 0.5, 
    #          w["word"], color='black', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('kws_analysis_v2.png', dpi=300)
plt.show()

import time

import time

# ================== 7. [进阶版] 100次迭代性能基准分析 ==================
print("\n" + "🔥" * 15 + " 正在进行高精度性能压测 (50次迭代) " + "🔥" * 15)

# 1. 准备测试数据
test_logits = ctc_probs  # (1, T, D)
test_targets = target_ids_tensor
test_logits_lens = torch.tensor([ctc_probs.shape[1]])
test_target_lens = torch.tensor([target_ids_tensor.shape[1]])

# --- Step A: 预热 (Warm-up) ---
# Numba 会在第一次调用时进行 LLVM 编译，必须排除在计时之外
_ = kws_streaming_search(test_logits, test_targets, test_logits_lens, test_target_lens)
torch.cuda.synchronize() if device.type == 'cuda' else None

# --- Step B: 循环测试 ---
num_iterations = 50
latencies = []

for i in range(num_iterations):
    start_time = time.perf_counter()
    
    # 核心搜索函数
    results = kws_streaming_search(test_logits, test_targets, test_logits_lens, test_target_lens)
    
    # 如果你在 GPU 上运行，取消下面这一行的注释以确保计时准确
    # torch.cuda.synchronize() 
    
    end_time = time.perf_counter()
    latencies.append((end_time - start_time) * 1000) # 转换为毫秒

# --- Step C: 数据统计 ---
avg_latency = np.mean(latencies)
min_latency = np.min(latencies)
max_latency = np.max(latencies)
std_latency = np.std(latencies)
p99_latency = np.percentile(latencies, 99) # 99分位数，衡量长尾延迟

audio_len_sec = waveform.shape[1] / 16000
rtf = (avg_latency / 1000) / audio_len_sec

# --- Step D: 结果解析与区间锁定 (取最后一次结果即可) ---
log_post, normed_scores, alpha_t, start_tlist, total_t = results
scores_np = np.array([s.item() if torch.is_tensor(s) else s for s in normed_scores])
max_idx = scores_np.argmax()
final_max_score = np.exp(scores_np[max_idx])
best_start_frame = int(start_tlist[max_idx].item())

# 时间转换
time_per_frame = (10 * subsample_rate) / 1000.0 # 假设 10ms frame_shift
actual_start_sec = best_start_frame * time_per_frame
actual_end_sec = max_idx * time_per_frame

# --- Step E: 打印性能报告 ---
print(f"📊 [样本总量]: {num_iterations} 次迭代")
print(f"⏱️  [平均延迟]: {avg_latency:.3f} ms")
print(f"📉 [最低延迟]: {min_latency:.3f} ms")
print(f"📈 [最高延迟]: {max_latency:.3f} ms")
print(f"📏 [标准偏差]: {std_latency:.3f} ms")
print(f"⚡ [P99 延迟]: {p99_latency:.3f} ms")
print(f"🚀 [RTF 指标]: {rtf:.6f} (越小越好)")
print("-" * 60)
print(f"🎯 [搜索结果]: 关键词 '{target_word}' | 置信度 = {final_max_score:.4f}")
print(f"📍 [锁定区间]: {actual_start_sec:.2f}s -> {actual_end_sec:.2f}s ({best_start_frame} -> {max_idx} 帧)")
print("🔥" * 40)