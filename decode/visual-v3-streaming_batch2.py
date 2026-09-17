import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import time
import random
import os
from g2p_en import G2p

# 导入刚才定义的 Batch 类
from KWStreamingSearch.CTC.ctc_streaming_search_numba_pruning_logsum_streaming_batch import BatchCTCFsdStreamingSearch

# # 限制 CPU 核心数
# torch.set_num_threads(4) 
# os.environ["OMP_NUM_THREADS"] = "4"

# ================== 1. 模型加载 ==================
device = torch.device("cpu")
ckpt_path = "/nvme01/soundlab/kws/openkws/wenet/examples/librispeech-g2p/s0/exp/ls-gs-1460-ckpts/avg_10.pt"
dict_path = "/nvme01/soundlab/kws/openkws/wenet/examples/librispeech-g2p/s0/data/dict/lang_char.txt"
audio_path = "/nvme01/soundlab/kws/openkws/librispeech/test/test-clean/4970/29095/4970-29095-0038.wav"

class Stage1(nn.Module):
    def __init__(self):
        super().__init__()
        from models.encoder import ConformerEncoder
        from models.ctc import CTC
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

def load_system():
    from models.text.char_tokenizer import CharTokenizer
    ckpt = torch.load(ckpt_path, map_location=device)
    model = Stage1()
    model.encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in ckpt.items() if k.startswith('encoder')})
    model.ctc.load_state_dict({k.replace('ctc.', ''): v for k, v in ckpt.items() if k.startswith('ctc')})
    model.eval()
    return model, CharTokenizer(dict_path, None, split_with_space=' '), G2p()

model, tokenizer, g2p = load_system()

# ================== 2. 准备 2000 个关键词 ==================
input_text = "RUTH WAS GLAD TO HEAR THAT PHILIP HAD MADE A PUSH INTO THE WORLD AND SHE WAS SURE"
words_pool = input_text.split()
num_test_keywords = 10000
keywords_ids_list = []

print(f"--- 正在生成 {num_test_keywords} 个测试关键词 ---")
for i in range(num_test_keywords):
    n = random.randint(1, 2)
    k_word = "".join(random.sample(words_pool, min(n, len(words_pool))))
    phonemes = [p for p in g2p(k_word) if p.strip()]
    _, ids = tokenizer.tokenize(" ".join(phonemes))
    full_ids = np.zeros(2 * len(ids) + 1, dtype=np.int32)
    full_ids[1::2] = ids
    keywords_ids_list.append(full_ids)

# 初始化 Batch 引擎
batch_engine = BatchCTCFsdStreamingSearch(keywords_ids_list, blank=0)

# ================== 3. 预热与推理 ==================
from models.processor import compute_fbank
waveform, sr = torchaudio.load(audio_path)
feat = compute_fbank({'wav': waveform, 'sample_rate': 16000, 'key': ">>>"}, num_mel_bins=80)['feat'].unsqueeze(0)

print("\n🔥 系统深度预热...")
with torch.no_grad():
    post_np = model(feat, torch.tensor([feat.size(1)])).squeeze(0).cpu().numpy()
    # 内核预热
    for _ in range(10): _ = batch_engine.step(post_np[0], 0)
    batch_engine.reset()

# ================== 4. 正式压力测试 ==================
print(f"🚀 启动 Batch 压测 | 词数: {num_test_keywords} | 帧数: {post_np.shape[0]}")

frame_times = []
t_total_start = time.perf_counter()

for t in range(post_np.shape[0]):
    f_post = post_np[t]
    
    t_f_start = time.perf_counter()
    # 批量步进，单次调用处理 2000 词
    scores, starts = batch_engine.step(f_post, t)
    t_f_end = time.perf_counter()
    
    frame_times.append(t_f_end - t_f_start)

t_total_duration = time.perf_counter() - t_total_start

# ================== 5. 结果分析 ==================
avg_ms = np.mean(frame_times) * 1000
p99_ms = np.percentile(frame_times, 99) * 1000
rtf = t_total_duration / (post_np.shape[0] * 0.01)

print("\n" + "📊 " + "="*45)
print(f"Batch 模式压测报告 (串行 JIT 版)")
print(f"平均每帧耗时: {avg_ms:.3f} ms")
print(f"P99 峰值耗时:  {p99_ms:.3f} ms")
print(f"实时率 (RTF):  {rtf:.4f}")
print("-" * 47)
if rtf < 1.0:
    print(f"✅ 成功：支持 {num_test_keywords} 词实时解码。")
    print(f"💡 理论上限：约 {int(num_test_keywords / rtf)} 词。")
else:
    print(f"❌ 警告：RTF > 1，无法实时。")
print("="*47)