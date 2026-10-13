# 增训正样本离线 MUSAN 混噪

关键词增训 adapt 在 prepare 阶段把 MUSAN 的 noise、music、speech 按配置电平混入 train 正样本，只把镜像 fbank 写到磁盘。训练代码不用 wav，因此 adapt 管线、数据集、joint 校验和训练入口都没有改动。

实现见 `dma_kws/stage2/prepare_adapt_noise.py`，测试见 `tests/test_prepare_adapt_noise.py`。设计过程与取舍见 [2026-10-13-adapt-offline-musan-mix.md](superpowers/plans/2026-10-13-adapt-offline-musan-mix.md)。

## 为什么需要

现场 SNR=10 基线，二阶段与端到端唤醒率：

| 场景 | 二阶段 | 端到端 |
| --- | --- | --- |
| 噪声 40 / 50 / 60 dB | 0.979 / 0.974 / 0.964 | 0.954 / 0.944 / 0.939 |
| 音乐噪声 40 / 50 / 60 dB | 0.949 / 0.949 / 0.949 | 0.940 / 0.920 / 0.920 |
| 人声混杂噪声 40 / 50 / 60 dB | 0.636 / 0.667 / 0.698 | 0.633 / 0.657 / 0.666 |

人声与噪声、音乐的差距是 31 到 34 个百分点，置信区间不含 0，是唯一确凿的差距。噪声和音乐已经在 0.95 附近。人声条件下二阶段与端到端只差 0.2 到 3 个百分点，损失在 encoder/QbyT 声学侧。

训练侧此前没有见过人声干扰。adapt 的 `_validate_lora_runtime_config` 明确拒绝在线混噪，而 v4.1 配方文档里的 MUSAN 切分是 `--train-categories music,noise`，speech 全部留在 eval。本模块补上这个缺口。

## 能力边界

做这些事：

- 只对 train 且 label 为 1 的样本生成变体，每个启用的类别各生成 `variants_per_sample` 个。
- 在 prepare 阶段完成解码、RMS 功率比混音和 fbank 计算，只写 `.npy`。
- 把配方写进 manifest 和账本，包括抽中的 MUSAN 录音、抽样电平、随机种子。
- 对配置、切分重叠、切分摘要做 fail-closed 校验。

不做这些事：

- 不改 `adapt_dataset.py`、`adapt.py`、`joint_manifest.py`、`run_keyword_adaptation.py`。
- 不打开 `stage2.noise_augmentation`，那个开关对 adapt 仍然报错。
- 不落盘加噪 wav。
- 不做 burst、stationary、volume 变换，也不做多分量叠加。评估侧的 `prep.musan_mix` 才有这些。

## 前置条件

MUSAN 根目录按 `<root>/{noise,music,speech}/...` 布局。切分目录需要两份白名单：

- `train_background.list`，一行一个绝对或相对路径，相对路径按列表文件所在目录解析。
- `eval_musan.list`，与 train 列表必须不相交。
- `split.json` 可选。存在时，每个类别池的路径摘要必须等于 `splits.train.by_category[<类别>].catalog_sha256`。

speech 必须出现在 train 白名单里。当前 v4.1 的切分把 speech 全部放进 eval，所以想训 speech 要先重建：

~~~bash
python scripts/split_musan.py \
  --musan-root /data/MUSAN \
  --output-dir data/dma-kws/processed/musan_split \
  --train-categories music,noise,speech
~~~

这个重建会把 eval speech 缩到约 40%，SNR=10 基线需要在新的 eval 列表上重测。只训 noise 和 music 时，现有切分可以直接用。

fbank 的 dither 建议为 0。v41/v42 链通过 `override /fbank: icefall_kws` 已经是 0；普通 `adapt_hey_eva` 继承 `fbank: default` 的 0.1，用 dither 大于 0 的配置重新渲染变体会得到不同的特征。已有特征不会被重新渲染，除非删除文件或设置 `prep.no_skip_existing=true`。

## 配置

完整默认值在 `configs/adapt/default.yaml` 的 `adapt.noise_augmentation`：

~~~yaml
adapt:
  noise_augmentation:
    enabled: false
    generation: g1
    seed: 2025
    musan_split_dir: ""
    train_list: ""
    eval_list: ""
    variants_per_sample: 1
    keep_original: true
    dry_run: false
    phases: []
    categories:
      noise:  {enabled: false, snr_db_min: 0.0,        snr_db_max: 15.0}
      music:  {enabled: false, snr_db_min: 5.0,        snr_db_max: 20.0}
      speech: {enabled: false, relative_db_min: -15.0, relative_db_max: -5.0}
~~~

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `enabled` | false | 总开关。关闭时解析结果为 None，prepare 行为与改动前一致。 |
| `generation` | g1 | 配方和特征的命名空间。配置指纹变化而 generation 不变会报错。 |
| `seed` | 2025 | 配方种子。每条变体的种子由 seed、generation、keyword、phase、源路径、序号、类别派生。 |
| `musan_root` | 空 | 留空则取 `prep.musan_root`。 |
| `musan_split_dir` | 空 | 启用时必填，放两份白名单和可选 `split.json`。 |
| `train_list` | 空 | 留空取 `<musan_split_dir>/train_background.list`。 |
| `eval_list` | 空 | 留空取 `<musan_split_dir>/eval_musan.list`。 |
| `variants_per_sample` | 1 | 每个启用类别、每个 train 正样本的副本数。总变体数 = 正样本数 × 启用类别数 × 该值。 |
| `keep_original` | true | 保留未加噪的正样本行。false 只保留变体，旧特征不删。 |
| `dry_run` | false | 只写计划预览，不渲染特征，不改 manifest。 |
| `phases` | [] | 空表示本次 prepare 的所有 phase。joint 只训 real，建议填 `[real]`。 |
| `categories.noise/music` | 全关 | 用 `snr_db_min`、`snr_db_max`，单位是干净与干扰的 RMS 功率比。 |
| `categories.speech` | 关 | 用 `relative_db_min`、`relative_db_max`，0 表示等响，-10 表示干扰比关键词低 10 dB，也就是 SNR=10。 |

电平在计划时从区间内均匀抽取并写进账本。混音公式与在线 `TrainingNoiseAugmenter` 共用同一份实现，即 `dma_kws/stage2/features.py` 里的 `mix_with_source`，noise 和 music 按 SNR 定标，speech 按相对 RMS 定标。

## 运行

先出计划，检查变体数、类别分布和存储量：

~~~bash
python scripts/run_keyword_adaptation.py +experiment=adapt_hey_eva_v42 \
  adapt.stage=prepare \
  prep.musan_root=/data/MUSAN \
  adapt.noise_augmentation.enabled=true \
  adapt.noise_augmentation.musan_split_dir=data/dma-kws/processed/musan_split \
  adapt.noise_augmentation.dry_run=true
~~~

确认后去掉 dry_run，正式 prepare：

~~~bash
python scripts/run_keyword_adaptation.py +experiment=adapt_hey_eva_v42 \
  adapt.stage=prepare \
  prep.musan_root=/data/MUSAN \
  adapt.noise_augmentation.enabled=true \
  adapt.noise_augmentation.musan_split_dir=data/dma-kws/processed/musan_split \
  adapt.noise_augmentation.categories.noise.enabled=true \
  adapt.noise_augmentation.categories.music.enabled=true \
  adapt.noise_augmentation.categories.speech.enabled=true
~~~

speech 范围覆盖 SNR=10 时用 `relative_db_min=-15`、`relative_db_max=-5`。只关心人声时，只开 speech，这样每个正样本的变体全是人声干扰。

控制台会打印一行 `musan_mix` 计划、一个 "Mix MUSAN variants" 进度条，汇总表里会出现 `aug_status`、`aug_generation`、`aug_planned`、`aug_written`、`aug_skipped`、`aug_frames`、`aug_storage` 和每类别计数。

## 产出

~~~text
<data_root>/
  fbank/<phase>/<positive|negative>/....npy          干净特征，不变
  fbank/<phase>/_aug/<generation>/....npy            变体特征
  manifests/<phase>_train.csv                        干净行在前，变体行追加在后
  manifests/<phase>_eval.csv                         不含任何 aug_* 列
  augmentation/<generation>/
    recipes.json      计划，含配置指纹
    features.jsonl    账本，含 sha256、帧数、实际 SNR
    <category>.list   该类别使用的 train 池
    plan_preview.json 仅 dry-run
~~~

变体行的 audio_path 形如 `raw/tts/_aug/g1/pos_0001__a1b2c3d4__aug000_noise.wav`，磁盘上没有这个文件。fbank 落在 `fbank/tts/_aug/g1/pos_0001__a1b2c3d4__aug000_noise.npy`，数据集用同一个镜像函数解析，因此不需要改 `adapt_dataset.py`。文件名里的 8 位十六进制来自源路径摘要，避免不同目录下的同名文件撞车。

变体行在 manifest 里新增这些列，干净行留空：

| 列 | 内容 |
| --- | --- |
| `aug_recipe_id` | 配方 id，关联 recipes.json 和 features.jsonl |
| `aug_generation` | generation |
| `aug_category` | noise、music 或 speech |
| `aug_metric` | snr_db 或 relative_db |
| `aug_level_db` | 计划时抽定的电平 |
| `aug_musan_path` | 抽中的 MUSAN 录音 |
| `original_audio_path` | 父行的干净路径。joint 的泄漏分组会读这一列。 |

`recipes.json` 每条配方记录 recipe_id、phase、源路径、变体路径、fbank 相对路径、类别、电平、应用后的 SNR、MUSAN 路径和种子。`features.jsonl` 每条记录 recipe_id、fbank 相对路径、sha256、字节数、帧数、维度、实际 SNR 和 MUSAN 路径，按 recipe_id 排序。

`prepare_keyword_adaptation` 的返回 stats 里有 `augmentation` 块：status、generation、config_fingerprint、planned、written、skipped、frames、bytes、by_category、recipes、features。每个 phase 的 stats 增加 `train_augmented`。

## 机制与保证

只扩 train 正样本。规划器只遍历每个 phase 的 train 行并筛选 label 为 1，eval 行按原对象返回。负样本不加噪。

变体身份取决于配方。种子由源路径、序号和类别摘要派生，配方文件名里也带序号和类别。改 seed、改电平区间、改 MUSAN 列表都会产生新变体，不会复用旧特征。

配置指纹防止静默换配方。指纹覆盖 generation、seed、副本数、keep_original、两个白名单的 sha256、每个类别池的摘要和 fbank 参数，写在 recipes.json 里。指纹不一致而 generation 未变时 prepare 直接报错。

泄漏守卫没有旁路。train 白名单必须与 eval 白名单不相交；每个类别池的摘要必须等于 split.json 记录的 train 分区摘要；启用一个没有 train 录音的类别会报错。

写入是幂等的。变体 `.npy` 先写临时文件再 `os.replace`。recipes.json 和 features.jsonl 每次重写但内容逐字节一致。features.jsonl 合并已有条目和本次新渲染的条目。manifest 在特征写完之后才写，所以 manifest 里不会出现指向缺失特征的变体行。

训练侧接口不变。`TrainingNoiseAugmenter.mix` 保持二元组返回值，内部委托给 `mix_with_source`；`tests/test_stage2_training_noise.py` 保持全绿。

## 测试与验证

~~~bash
.venv/bin/python -m pytest tests/test_prepare_adapt_noise.py -q
.venv/bin/python -m pytest tests/test_stage2_training_noise.py tests/test_adapt_dataset.py \
  tests/test_adapt_console.py tests/test_joint_adapt_manifest.py tests/test_run_keyword_adaptation.py -q
~~~

新测试覆盖这些行为：关闭或缺省返回 None、未知键、缺目录、区间反向、类别配错度量、副本数为 0、train 列表指向 eval、NaN 电平、eval 重叠、split.json 摘要不符、没有 speech 池、计划确定性、只扩 train 正样本、keep_original 为 false、抽中来源和电平的 SNR 实测、渲染幂等、指纹变化报错、dry-run 不写特征、端到端 manifest 与 `KeywordAdaptationDataset` 加载。

## 故障排查

| 报错 | 原因 | 处理 |
| --- | --- | --- |
| `musan_split_dir is required` | 启用了但没给切分目录 | 填 `musan_split_dir`，或先跑 `scripts/split_musan.py` |
| `no speech recordings under ...` | train 白名单里没有 speech | 用 `--train-categories music,noise,speech` 重建切分，或关掉 speech 类别 |
| `overlaps the eval allowlist` | 两份白名单有交集 | 用 `scripts/split_musan.py` 重新生成，不要手工拼接 |
| `does not match the recorded split partition` | 白名单与 split.json 记录不一致 | 用同一批产出的列表文件，或删掉过期的 split.json |
| `different configuration fingerprint` | 改了配置但沿用同一个 generation | 改 `generation`，或把配置改回去，或删掉该 generation 目录 |
| `uses snr_db_min; it takes relative_db_min/relative_db_max` | speech 配了 SNR 字段 | 按类别换用对应字段 |
| `Synthetic augmented audio path already exists on disk` | 该路径被真实文件占用 | 检查 raw 树下是否有同名文件，通常会换 generation |
| `phases does not match any prepared phase` | 指定的 phase 本次没有准备 | 去掉 `phases` 或改成实际准备的 phase |

## 已知限制与后续

没有多分量叠加。评估里 "人声混杂噪声" 是人声和噪声同时存在，当前每个变体只混一条录音。要覆盖这个组合条件，需要让变体支持 1 到 N 个分量，每个分量独立对干净信号定标后相加。这是下一步优先级最高的改动。

负样本侧没有加 speech。只给正样本加人声干扰会引入 "有人声就是正样本" 的偏置，可能抬高 speech 背景上的误唤醒。配套做法是把 speech 纳入 `stage2.background_negative` 的缓存池。

跳过时不重算内容哈希。`skip_existing` 命中的文件按字节大小与账本比对，大小相同就保留旧条目，不会重新读取文件内容。要强制全部重渲染，设 `prep.no_skip_existing=true`。独立的 verify 命令和 generation 清理命令还没做。

dither 不是硬校验。dither 大于 0 时重新渲染的特征不可复现，代码只在文档里提示，没有报错，原因是 `adapt_hey_eva` 这类实验仍继承 `fbank: default`。

存储按变体数线性增长。一个 2 秒片段约 200 帧、80 维、float32，约 64 KB。一万个正样本、三个类别、每类一份约 1.9 GB，dry-run 会先给出准确数字。

正类先验会偏移。变体行会抬高 keyword 数据集里正样本的比例，`MixedAdaptationDataset` 按行均匀抽取。先接受偏移并观察目标域指标，需要时再考虑给负样本复制同样份数，或把 `keep_original` 与副本数调小。

## 代码地图

| 位置 | 内容 |
| --- | --- |
| `dma_kws/stage2/prepare_adapt_noise.py` | `AugmentationRequest.from_config`、`resolve_adapt_noise_augmentation`、`plan_offline_musan_variants`、`preview_offline_musan_variants`、`materialize_offline_musan_variants`、`describe_offline_musan_request` |
| `dma_kws/stage2/features.py` | `TrainingNoiseAugmenter.mix_with_source`，`mix` 委托它 |
| `dma_kws/stage2/prepare_adapt.py` | `prepare_keyword_adaptation` 的 `noise_augmentation` 参数，规划与物化的两个挂点 |
| `dma_kws/stage2/adapt_console.py` | 计划行、进度阶段、汇总计数 |
| `configs/adapt/default.yaml` | `adapt.noise_augmentation` 默认块 |
| `tests/test_prepare_adapt_noise.py` | 11 项测试 |

## 相关文档

- [改动方案与基线分析](superpowers/plans/2026-10-13-adapt-offline-musan-mix.md)
- [Stage II 多来源背景负样本](stage2-multisource-background.md)
- [Hey Eva 工作点对照表](hey-eva-operating-points.md)
