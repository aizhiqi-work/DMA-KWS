import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from g2p_en import G2p

# ================== 1. 系统组件与模型加载 ==================
from models.encoder import ConformerEncoder
from models.ctc import CTC
from models.processor import compute_fbank
from models.text.char_tokenizer import CharTokenizer
from KWStreamingSearch.CTC.ctc_streaming_search_numba_pruning_logsum_streaming import CTCFsdStreamingSearch

device = torch.device("cpu")
ckpt_path = "/nvme01/soundlab/kws/openkws/wenet/examples/librispeech-g2p/s0/exp/ls-gs-1460-ckpts/avg_10.pt"
dict_path = "/nvme01/soundlab/kws/openkws/wenet/examples/librispeech-g2p/s0/data/dict/lang_char.txt"
audio_path = "/nvme01/soundlab/kws/openkws/librispeech/test/test-clean/672/122797/672-122797-0020.wav"

class KWSStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConformerEncoder(
            input_size=80, output_size=144, attention_heads=4, linear_units=576, num_blocks=6,
            dropout_rate=0.1, positional_dropout_rate=0.1, attention_dropout_rate=0.0,
            use_cnn_module=True, input_layer="conv2d", pos_enc_layer_type="rel_pos",
            selfattention_layer_type="rel_selfattn", cnn_module_kernel=3
        )
        self.ctc = CTC(odim=73, encoder_output_size=144, blank_id=0)
    def forward(self, x, x_len):
        enc_out, _ = self.encoder(x, x_len)
        return self.ctc.log_softmax(enc_out)

def load_all():
    ckpt = torch.load(ckpt_path, map_location=device)
    model = KWSStage()
    model.encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in ckpt.items() if k.startswith('encoder')})
    model.ctc.load_state_dict({k.replace('ctc.', ''): v for k, v in ckpt.items() if k.startswith('ctc')})
    model.eval()
    return model, CharTokenizer(dict_path, None, split_with_space=' '), G2p()

model, tokenizer, g2p = load_all()

# ================== 2. 音频特征预处理 ==================
waveform, sr = torchaudio.load(audio_path)
if sr != 16000: waveform = T.Resample(sr, 16000)(waveform)
# 拼接静音以稳定流式初始状态
silence = torch.zeros((1, 16000 * 2))
waveform = torch.cat((silence, waveform, silence), dim=1)

sample = {'wav': waveform, 'sample_rate': 16000, 'key': 'test'}
feat = compute_fbank(sample, num_mel_bins=80)['feat'].unsqueeze(0).to(device)

with torch.no_grad():
    post_np = model(feat, torch.tensor([feat.size(1)])).squeeze(0).cpu().numpy()
T_frames = post_np.shape[0]

# ================== 3. 多负例冲突预分析 ==================
target_word = "GREEN"
# negative_words = ["REJOICE", "RAINBOW"] 
negative_words = ["GREEN BOTH"]

# 核心冲突分析逻辑
is_prefix = any(word.startswith(target_word) and word != target_word for word in negative_words)
is_suffix = any(word.endswith(target_word) and word != target_word for word in negative_words)
is_inside = any(target_word in word and word != target_word for word in negative_words)

# 根据冲突类型分配资源
# 如果是后缀冲突(JOICE)，回溯必须深；如果是前缀冲突(RAIN)，等待必须久
WAIT_FRAMES = 12 if (is_prefix or is_inside) else 2
LOOK_BACK = 25 if (is_suffix or is_inside) else 10
MARGIN = 0.15
THRESHOLD = 0.5
RECOVERY = 15

print(f"--- 配置分析: [{target_word}] ---")
print(f"冲突类型: 前缀={is_prefix}, 后缀={is_suffix}, 包含={is_inside}")
print(f"探测策略: 等待={WAIT_FRAMES}f, 回溯={LOOK_BACK}f, 阈值={THRESHOLD}")

def get_word_ids(word):
    phonemes = g2p(word)
    phonemes = [p for p in phonemes if p.strip()]
    _, ids = tokenizer.tokenize(" ".join(phonemes))
    u_phi = 2 * len(ids) + 1
    full_ids = np.zeros(u_phi, dtype=np.int32)
    full_ids[1::2] = ids
    return full_ids, len(ids)

pos_ids, pos_len = get_word_ids(target_word)
neg_engines = {w: {"engine": CTCFsdStreamingSearch(blank=0), "data": get_word_ids(w)} for w in negative_words}
pos_engine = CTCFsdStreamingSearch(blank=0)

# ================== 4. 非对称流式竞争判定 ==================
pos_history, neg_history = [], []
final_triggers = []
pending = None
last_trigger_t = -RECOVERY

for t in range(T_frames):
    frame_post = post_np[t]
    
    # 正例得分
    p_score, p_start, _ = pos_engine.step(frame_post, pos_ids, t)
    p_conf = np.exp(p_score / pos_len)
    
    # 负例表最大得分
    cur_n_max = 0.0
    for w, obj in neg_engines.items():
        n_s, _, _ = obj["engine"].step(frame_post, obj["data"][0], t)
        cur_n_max = max(cur_n_max, np.exp(n_s / obj["data"][1]))
    
    pos_history.append(p_conf)
    neg_history.append(cur_n_max)

    # 1. 触发判定
    if p_conf > THRESHOLD and pending is None and (t - last_trigger_t) > RECOVERY:
        if not (is_prefix or is_suffix or is_inside):
            # 完全清白的情况，0 延迟输出
            print(f"✨ [Direct] t={t} | {p_conf:.3f}")
            final_triggers.append(t)
            last_trigger_t = t
        else:
            pending = {'t_trig': t, 'p_max': p_conf}
            print(f"⏳ [Pending] t={t} | 进入观察窗...")

    # 2. 处于观察窗逻辑
    if pending is not None:
        pending['p_max'] = max(pending['p_max'], p_conf)
        
        # 3. 观察期结束（非对称窗口核心）
        if t - pending['t_trig'] >= WAIT_FRAMES:
            # 双向扫描：从过去 LOOK_BACK 到现在 WAIT_FRAMES
            search_start = max(0, pending['t_trig'] - LOOK_BACK)
            search_end = t + 1
            win_n_max = max(neg_history[search_start : search_end])
            
            if pending['p_max'] > (win_n_max + MARGIN):
                print(f"✨ [Success] t={pending['t_trig']} | 压制对手({win_n_max:.2f})成功！")
                final_triggers.append(pending['t_trig'])
                last_trigger_t = t
            else:
                print(f"🚫 [Inhibited] t={pending['t_trig']} | 被拦截，最强负例竞争者得分: {win_n_max:.2f}")
            pending = None

print(f"\n✅ 检测结束。有效触发: {len(final_triggers)} 次。")