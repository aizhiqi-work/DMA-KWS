import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from g2p_en import G2p

# 1. 模型组件导入 (请确保路径正确)
from models.encoder import ConformerEncoder
from models.ctc import CTC
from models.processor import compute_fbank
from models.text.char_tokenizer import CharTokenizer

# 核心：导入你的流式搜索类
# 路径：/nvme01/soundlab/kws/openkws_ctc/KWStreamingSearch/CTC/ctc_streaming_search_numba_pruning_logsum_streaming.py
from KWStreamingSearch.CTC.ctc_streaming_search_numba_pruning_logsum_streaming import CTCFsdStreamingSearch

# ================== 1. 系统配置与加载 ==================
device = torch.device("cpu")
ckpt_path = "/nvme01/soundlab/kws/openkws/wenet/examples/librispeech-g2p/s0/exp/ls-gs-1460-ckpts/avg_10.pt"
dict_path = "/nvme01/soundlab/kws/openkws/wenet/examples/librispeech-g2p/s0/data/dict/lang_char.txt"
audio_path = "/nvme01/soundlab/kws/openkws/librispeech/test/test-clean/4970/29095/4970-29095-0038.wav"

class Stage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConformerEncoder(
            input_size=80, output_size=144, attention_heads=4, linear_units=576, num_blocks=6,
            dropout_rate=0.1, positional_dropout_rate=0.1, attention_dropout_rate=0.0,
            use_cnn_module=True, input_layer="conv2d", pos_enc_layer_type="rel_pos",
            selfattention_layer_type="rel_selfattn", cnn_module_kernel=3
        )
        self.ctc = CTC(odim=73, encoder_output_size=144, blank_id=0)
    
    def ctc_logprobs(self, encoder_out: torch.Tensor):
        return self.ctc.log_softmax(encoder_out)

def load_system(ckpt_path, dict_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = Stage1()
    model.encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in ckpt.items() if k.startswith('encoder')})
    model.ctc.load_state_dict({k.replace('ctc.', ''): v for k, v in ckpt.items() if k.startswith('ctc')})
    model.eval().to(device)
    return model, CharTokenizer(dict_path, None, split_with_space=' ')

model, tokenizer = load_system(ckpt_path, dict_path, device)
g2p = G2p()

# ================== 2. 音频预处理与特征流模拟 ==================
waveform, sr = torchaudio.load(audio_path)
if sr != 16000: waveform = T.Resample(sr, 16000)(waveform)

silence_samples = 16000 * 2
silence = torch.zeros((1, silence_samples))

# Append it to your existing waveform
waveform = torch.cat((waveform, silence), dim=1)


# 模拟输入文本和关键词
input_text = "RUTH WAS GLAD TO HEAR THAT PHILIP HAD MADE A PUSH INTO THE WORLD AND SHE WAS SURE THAT HIS TALENT AND COURAGE WOULD MAKE A WAY FOR HIM"
target_word = "RUTH WAS"

# 准备关键词序列 (Phoneme IDs)
target_phonemes = g2p(target_word)
target_phonemes = [p for p in target_phonemes if p.strip()]
_, target_ids = tokenizer.tokenize(" ".join(target_phonemes))
U_phi = 2 * len(target_ids) + 1
full_tgt_ids = np.zeros(U_phi, dtype=np.int32)
full_tgt_ids[0::2] = 0 # Blank
full_tgt_ids[1::2] = target_ids

# 模拟特征提取和全量推理（实际流式中可改为 chunk 前向）
sample = {'wav': waveform, 'sample_rate': 16000, 'key': 'test'}
feat = compute_fbank(sample, num_mel_bins=80)['feat'].unsqueeze(0).to(device)
with torch.no_grad():
    ctc_probs = model.ctc_logprobs(model.encoder(feat, torch.tensor([feat.size(1)]))[0])
post_np = ctc_probs.squeeze(0).cpu().numpy()
T_frames = post_np.shape[0]

# ================== 3. 多点流式解码触发逻辑 ==================
kws_engine = CTCFsdStreamingSearch(blank=0, need_full_matrix=True)
kws_engine.reset()

# 策略参数
THRESHOLD = 0.5          # 唤醒阈值
MIN_INTERVAL = 20        # 触发冷却时间（防止同一单词重复触发）
detected_events = []     # 存储所有成功的触发点 [(start, end, score)]
all_conf_history = []    # 记录完整置信度曲线用于绘图
last_trigger_t = -MIN_INTERVAL

print(f"\n🚀 正在启动流式检测: 目标词 [{target_word}]")

for t in range(T_frames):
    frame_log_post = post_np[t]
    
    # 执行流式单步搜索
    # 假设你的 step 返回: score, start_frame, total_frame
    score, s_frame, _ = kws_engine.step(frame_log_post, full_tgt_ids, t, prune_threshold=0.98)
    
    # 转换为置信度 (0 ~ 1)
    confidence = np.exp(score / len(target_ids))
    all_conf_history.append(confidence)
    
    # 多触发点逻辑：超过阈值且不在冷却期内
    if confidence > THRESHOLD and (t - last_trigger_t) > MIN_INTERVAL:
        event = {"start": s_frame, "end": t, "conf": confidence}
        detected_events.append(event)
        last_trigger_t = t
        print(f"✨ [触发点 {len(detected_events)}] 置信度: {confidence:.4f} | 时间: {t*10}ms | 帧区间: {s_frame} -> {t}")

# ================== 4. 可视化所有触发点 ==================
if kws_engine.need_full_matrix:
    log_alpha_matrix = np.stack(kws_engine.history_log_alpha)
    
    plt.figure(figsize=(18, 8))
    
    # 上图：热力图
    ax1 = plt.subplot(2, 1, 1)
    ylabel = ['Φ']
    for p in target_phonemes: ylabel.extend([p, 'Φ'])
    sns.heatmap(np.exp(log_alpha_matrix).T, cmap="magma", yticklabels=ylabel, ax=ax1, cbar_kws={'label': 'Prob'})
    ax1.invert_yaxis()
    
    # 画出所有触发边界
    colors = plt.cm.rainbow(np.linspace(0, 1, len(detected_events)))
    for i, (event, color) in enumerate(zip(detected_events, colors)):
        ax1.axvline(event["start"], color=color, linestyle='--', alpha=0.7)
        ax1.axvline(event["end"], color=color, linestyle='-', linewidth=2)
        ax1.text(event["end"], len(ylabel)+0.5, f"Hit {i+1}", color=color, fontweight='bold')

    # 下图：实时置信度曲线
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(all_conf_history, color='tab:blue', linewidth=2, label='Streaming Confidence')
    ax2.axhline(THRESHOLD, color='red', linestyle=':', label='Wake-up Threshold')
    ax2.fill_between(range(T_frames), all_conf_history, THRESHOLD, 
                     where=(np.array(all_conf_history) > THRESHOLD), color='red', alpha=0.2)
    
    ax2.set_xlabel("Time Frames (10ms)")
    ax2.set_ylabel("Confidence Score")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle(f"Multi-Trigger KWS Analysis: '{target_word}'", fontsize=16)
    plt.tight_layout()
    plt.savefig('multi_trigger_analysis.png', dpi=300)
    print(f"\n📊 可视化结果已保存至: multi_trigger_analysis.png")
    plt.show()

print(f"\n✅ 检测任务结束。共捕捉到 {len(detected_events)} 个实例。")