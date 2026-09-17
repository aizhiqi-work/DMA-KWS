import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import time
import random
from g2p_en import G2p

# 导入必要的组件
from models.encoder import ConformerEncoder
from models.ctc import CTC
from models.processor import compute_fbank
from models.text.char_tokenizer import CharTokenizer
from KWStreamingSearch.CTC.ctc_streaming_search_numba_pruning_logsum_streaming import CTCFsdStreamingSearch

# 限制 CPU 核心数，模拟受限环境
torch.set_num_threads(4) 
import os
os.environ["OMP_NUM_THREADS"] = "4"

# ================== 1. 系统加载 ==================
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
    def forward(self, x, x_len):
        enc_out, _ = self.encoder(x, x_len)
        return self.ctc.log_softmax(enc_out)

def load_system():
    ckpt = torch.load(ckpt_path, map_location=device)
    model = Stage1()
    model.encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in ckpt.items() if k.startswith('encoder')})
    model.ctc.load_state_dict({k.replace('ctc.', ''): v for k, v in ckpt.items() if k.startswith('ctc')})
    model.eval()
    return model, CharTokenizer(dict_path, None, split_with_space=' '), G2p()

model, tokenizer, g2p = load_system()

# ================== 2. 构造 100 个随机关键词 ==================
input_text = "RUTH WAS GLAD TO HEAR THAT PHILIP HAD MADE A PUSH INTO THE WORLD AND SHE WAS SURE THAT HIS TALENT AND COURAGE WOULD MAKE A WAY FOR HIM"
words_pool = input_text.split()
num_test_keywords = 2000
test_keywords = []

print(f"--- 正在生成 {num_test_keywords} 个测试关键词 ---")
for i in range(num_test_keywords):
    # 随机组合 1-2 个词，模拟不同长度的关键词
    n = random.randint(1, 2)
    k_word = "".join(random.sample(words_pool, min(n, len(words_pool))))
    
    phonemes = [p for p in g2p(k_word) if p.strip()]
    _, ids = tokenizer.tokenize(" ".join(phonemes))
    U_phi = 2 * len(ids) + 1
    full_ids = np.zeros(U_phi, dtype=np.int32)
    full_ids[1::2] = ids
    
    test_keywords.append({
        "word": k_word,
        "ids": full_ids,
        "engine": CTCFsdStreamingSearch(blank=0),
        "target_len": len(ids)
    })

# ================== 3. 系统深度预热 (Warm-up) ==================
waveform, sr = torchaudio.load(audio_path)
if sr != 16000: waveform = T.Resample(sr, 16000)(waveform)
sample = {'wav': waveform, 'sample_rate': 16000, 'key': 'test'}
feat = compute_fbank(sample, num_mel_bins=80)['feat'].unsqueeze(0).to(device)

print("\n🔥 正在进行深度预热（触发 JIT 编译与算子初始化）...")
with torch.no_grad():
    # 神经网络预热
    for _ in range(5):
        _ = model(feat, torch.tensor([feat.size(1)]))

    # 引擎预热：拿一个词跑 10 帧，触发 Numba 内部编译逻辑
    dummy_frame = np.zeros(73, dtype=np.float32)
    for _ in range(10):
        test_keywords[0]["engine"].step(dummy_frame, test_keywords[0]["ids"], 0)
    # 重置所有引擎状态
    for kw in test_keywords: kw["engine"].reset()

print("✅ 预热完毕。")

# ================== 4. 正式神经网络推理 ==================
print("\n--- 执行单次神经网络推理 ---")
start_nn = time.time()
with torch.no_grad():
    post_np = model(feat, torch.tensor([feat.size(1)])).squeeze(0).cpu().numpy()
nn_time = time.time() - start_nn
T_frames = post_np.shape[0]

# ================== 5. 压力测试：流式搜索 ==================
print(f"--- 启动多词并行压力测试 | 帧数: {T_frames} | 词数: {num_test_keywords} ---")

total_search_time = 0
per_frame_times = []

for t in range(T_frames):
    frame_post = post_np[t]
    
    # 记录该帧处理 100 个词的总时间
    t_start = time.perf_counter()
    for kw in test_keywords:
        # 这里是核心：100 个词共享同一个 frame_post 内存引用
        kw["engine"].step(frame_post, kw["ids"], t)
    t_end = time.perf_counter()
    
    duration = t_end - t_start
    per_frame_times.append(duration)
    total_search_time += duration

# ================== 6. 结果深度分析 ==================
avg_frame_time_ms = np.mean(per_frame_times) * 1000
p99_frame_time_ms = np.percentile(per_frame_times, 99) * 1000
std_frame_time_ms = np.std(per_frame_times) * 1000
real_time_factor = total_search_time / (T_frames * 0.01) # 10ms 每帧

print("\n" + "📊 " + "="*45)
print(f"压力测试报告 (CPU核心: 4 | 线程限制: 4)")
print(f"NN 推理总耗时:   {nn_time:.4f} s")
print(f"搜索总耗时:     {total_search_time:.4f} s")
print("-" * 47)
print(f"平均每帧 (100词): {avg_frame_time_ms:.3f} ms")
print(f"P99 峰值帧:      {p99_frame_time_ms:.3f} ms")
print(f"帧耗时标准差:     {std_frame_time_ms:.3f} ms")
print(f"实时率 (RTF):    {real_time_factor:.4f}")
print("="*47)

if real_time_factor < 0.5:
    print(f"🌟 性能卓越：仅占用约 {real_time_factor*100:.1f}% 的单核等效算力。")
elif real_time_factor < 1.0:
    print(f"✅ 性能达标：可以满足实时流式需求。")
else:
    print(f"❌ 性能不足：RTF > 1，系统会出现音频堆积延迟。")

# 计算理论承载极限
theoretical_max = int(num_test_keywords / real_time_factor)
print(f"💡 理论上限：该配置下，最大可支持约 {theoretical_max} 个词同时解码。")