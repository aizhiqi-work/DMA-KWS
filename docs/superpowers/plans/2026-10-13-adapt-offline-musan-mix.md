---
title: "关键词增训离线 MUSAN 混噪（Path A）改动方案"
description: "prepare 阶段把 MUSAN noise/music/speech 按 SNR 离线混入增训 train 正样本，只物化 fbank，训练运行时零改动；含配置面、模块类型、溯源/幂等/泄漏守卫、测试计划。"
tags: [adapt, musan, prep, augmentation, plan]
status: "proposed / not implemented"
baseline_commit: "d792f148e26165aa865f61ba9558418f8d98ad45"
created: "2026-10-13"
---

# 关键词增训离线 MUSAN 混噪改动方案

Path A 的实施合同：在 prepare 阶段把 MUSAN 的 noise / music / speech 按配置 SNR 离线混入**增训 train 正样本**，只产出镜像 fbank（.npy），训练侧（adapt.py / adapt_dataset.py / joint_manifest.py）零改动。

方案由两个独立设计候选综合而来（最小改动面 vs 可审计数据契约），取舍见 §11。

## 1. 目标与边界

**做**

- prepare_keyword_adaptation 在扫描/切分之后，为每个 train 正样本生成 N 个加噪变体行；
- 混音与 fbank 计算在 prepare 阶段完成，变体行带完整溯源写进 manifests/<phase>_train.csv；
- eval 行、负样本行保持干净；MUSAN 只用泄漏安全的 train 白名单。

**不做**

- 不改 adapt_dataset.py / adapt.py / joint_manifest.py / run_keyword_adaptation.py；
- 不打开 stage2.noise_augmentation（adapt 明确禁止，见 §2）；
- 不落盘加噪 wav，只落 fbank；
- 第一版只做单条干扰录音的稳态叠加，不做多类别叠加、不做 burst/stationary/volume（那是评估侧 prep.musan_mix 的能力）。

## 2. 已核对的代码事实

| 事实 | 依据 | 设计影响 |
| --- | --- | --- |
| adapt 训练时无法在线混噪 | adapt.py _validate_lora_runtime_config :152-161，调用点 :1326 | 只能 prep 离线混；新配置不得触发 stage2.noise_augmentation.enabled |
| adapt manifest 走 fbank 镜像，数据集从不打开 wav | adapt_dataset.py _resolve_adapt_fbank_path :26、np.load :138；adapt_paths.py wav_to_fbank_mirror :122 | 变体 audio_path 可以是"只存在 .npy"的合成路径 |
| fbank_root 与 train manifest 路径固定 | adapt.py :111、:117-118 | 独立 fbank 根/独立 manifest 方案不可达，只能在现有 manifest 扩行 |
| 每次 prepare 都重写 <phase>_<split>.csv | prepare_adapt.py :475-498 | 变体必须在同一次 prepare 内确定性派生，否则 adapt.stage=all 会丢 |
| skip_existing 按文件跳过 | prepare_adapt.py :292-306 | 变体身份必须编码配方，否则换配置会静默复用旧特征 |
| joint 校验只要求 audio_path 非空；original_audio_path 参与泄漏分组 | joint_manifest.py :27、:42-56、:89-105、:115-133 | 变体沿用父行 split/phase/speaker 元数据，父路径写进 original_audio_path 即保持同组 |
| 混音数学已存在且被测试锁定 | features.py TrainingNoiseAugmenter :28 / mix :132-202 / _load_audio :309；tests/test_stage2_training_noise.py :326 | 复用，不重写 RMS-SNR/重采样/静音重抽 |
| 评估侧 schema 与泄漏工具 | musan_mix.py；musan_fa.py musan_catalog_sha256 :25 / detect_subset :71；musan_split.py _split_summary :613-648 | 配置命名对齐 noise.snr_db / speech.relative_db；用 split.json 的 by_category catalog_sha256 做 fail-closed 泄漏证明 |
| fbank 配置链决定可复现性 | configs/config.yaml fbank: default（dither 0.1）；fbank/icefall_kws.yaml dither 0.0；experiment/icefall_zipformer_stage2.yaml :18 override /fbank: icefall_kws | v41/v42 链已是 dither 0；普通 adapt_hey_eva 不是，需要守卫（§8.9） |
| orchestration 无需改动 | run_keyword_adaptation.py :305-306 只转发整个 adapt/prep dict | 只改 prepare_with_console 内部 |

### 两次实测（方案关键假设，均已运行验证）

1. 镜像路径映射（调用真实函数）：

   ~~~text
   raw/tts/positive/foo.wav              => <data_root>/fbank/tts/positive/foo.npy
   raw/tts/positive/foo__mix01.wav       => <data_root>/fbank/tts/positive/foo__mix01.npy
   /data/tts/hey_eva/foo__mix01.wav      => <data_root>/fbank/external/data/tts/hey_eva/foo__mix01.npy
   ~~~

2. KeywordAdaptationDataset 加载 wav 不存在的变体行成功（label=1, feat=(4,80)），且 mix_* 元数据列经 manifest 往返保留。证明"合成 audio_path + 只物化 fbank"对训练代码透明。

补充：变体路径只要包含字面量 raw 分量就走稳定分支。**因此变体一律用 data_root 相对的 raw/<phase>/_aug/<gen>/... 形式，即使源来自外部绝对路径**——避免 fbank/external 分支把机器绝对路径烘进特征树。

## 3. 总体数据流

~~~text
scan / split                          （现有）
  -> phase_splits: {phase: (train_rows, eval_rows)}
plan_offline_musan_variants           （新，纯函数，无音频 I/O）
  仅 phase ∈ config.phases 且 row.label == 1：
    for i in range(variants_per_sample):
      cat  = enabled_categories[i % n]
      seed = sha256(base_seed, generation, keyword, phase, audio_path, i, cat)
      vpath = raw/<phase>/_aug/<gen>/<stem>__aug<i:03d>_<cat>.wav   （磁盘上不存在）
      fbank = _resolve_adapt_fbank_path(...)     （复用数据集自己的解析函数）
      变体行 = dataclasses.replace(row, audio_path=vpath, metadata={...溯源列})
  -> train_rows = [原行 if keep_original] + 变体行
  -> 写 augmentation/<gen>/recipes.csv（确定性计划，无时间戳）
report("fbank_total", 干净样本数 + 变体数)
  -> 现有干净 fbank 循环（不改）
materialize_offline_musan_variants    （新，音频 I/O）
  逐配方：ledger 已有且 .npy 存在且 hash 一致 -> 跳过
          torchaudio.load(干净 wav) -> augmenter.mix_with_source(rng=Random(recipe_seed))
            -> waveform_to_fbank(..., **fbank_params) -> 原子写 np.save
  写 augmentation/<gen>/features.csv（观测账本：hash、功率、增益）
  -> 现有 manifest 写入器（不改）写出扩行后的 train CSV
~~~

单调发布规则：**指向某个变体特征的 manifest 行，只在该特征已存在且已入账本之后才写。** manifest 是最后的内容写入。

## 4. 配置面

默认值追加到 configs/adapt/default.yaml（全部默认关闭，enabled=false 时现有运行逐字节不变）：

~~~yaml
adapt:
  # prepare 阶段离线把 MUSAN 混入 TRAIN 正样本；这是 adapt 里 stage2.noise_augmentation
  # 的离线对应物——干扰烘进镜像 fbank，训练仍然不需要 wav。
  noise_augmentation:
    enabled: false
    generation: g1              # 配方与特征命名空间；改配置必须换 generation（§7）
    seed: 2025
    musan_split_dir: ""         # 含 train_background.list / eval_musan.list / split.json
    variants_per_sample: 1      # 每个 train 正样本的变体数；类别按 noise->music->speech 轮转
    keep_original: true         # 保留干净正样本行
    dry_run: false              # 只出计划与成本，不物化、不发布
    phases: []                  # [] = 本次 prepare 的所有 phase；joint 建议 [real]
    categories:
      noise:  {enabled: false, snr_db_min: 0.0,  snr_db_max: 15.0}
      music:  {enabled: false, snr_db_min: 5.0,  snr_db_max: 20.0}
      speech: {enabled: false, relative_db_min: -10.0, relative_db_max: -5.0}
~~~

hey eva v42 启用示例（不新增 CLI 参数，纯 override）：

~~~bash
# 1) 先出计划（不落特征）
python scripts/prepare_keyword_adaptation.py +experiment=adapt_hey_eva_v42 \
  prep.musan_root=/data/corpora/MUSAN \
  adapt.noise_augmentation.enabled=true \
  adapt.noise_augmentation.musan_split_dir=data/dma-kws/processed/musan_split \
  adapt.noise_augmentation.categories.noise.enabled=true \
  adapt.noise_augmentation.categories.music.enabled=true \
  adapt.noise_augmentation.dry_run=true

# 2) 真正 prepare（同一组 override，dry_run=false）；
#    经 run_keyword_adaptation.py adapt.stage=prepare 走也可以，脚本无需改动。
~~~

字段语义：

| 字段 | 说明 |
| --- | --- |
| enabled | 总开关；关闭时 AugmentationRequest.from_config 返回 None，prepare 行为与今天一致 |
| generation | 配方/特征命名空间。配置指纹变化而 generation 未变时直接拒绝（§7） |
| musan_split_dir | 必须同时含 train 白名单与 eval 白名单，用于泄漏证明；相对路径按仓库解析 |
| variants_per_sample | 变体总数；类别按启用顺序轮转，保证每类都覆盖而非靠概率 |
| keep_original | true 保留原行 + 变体；false 只保留变体（旧 .npy 不删） |
| categories.*.snr_db_min/max | 干净/干扰 RMS 功率比，dB；混音时 rng.uniform(min, max) |
| categories.speech.relative_db_min/max | 与评估侧同义（0 = 等响）；内部取负作为 SNR 交给混音器 |

## 5. 模块与类型

新文件 dma_kws/stage2/prepare_adapt_noise.py（约 300 行；名字与 stage2 子系统的其它 prep 模块一致）：

~~~python
MIX_CATEGORIES = ("noise", "music", "speech")

@dataclass(frozen=True)
class AdaptMixCategory:
    kind: str                      # noise | music | speech
    level_name: str                # "snr_db" | "relative_db"（配置单位）
    level_min: float
    level_max: float
    pool: tuple[Path, ...]         # 排序后的规范 MUSAN train 录音
    pool_sha256: str               # musan_catalog_sha256(pool, musan_root)

    @property
    def snr_db_min(self) -> float: ...   # speech 取 -level_max
    @property
    def snr_db_max(self) -> float: ...   # speech 取 -level_min

@dataclass(frozen=True)
class AugmentationRequest:         # 已校验、可信
    generation: str
    seed: int
    musan_root: Path
    musan_split_dir: Path
    variants_per_sample: int
    keep_original: bool
    phases: tuple[str, ...]        # () = 全部
    categories: tuple[AdaptMixCategory, ...]   # 仅 enabled 且池非空
    config_snapshot: Mapping[str, Any]

    @classmethod
    def from_config(cls, block, *, prep) -> "AugmentationRequest | None":
        """唯一解析边界：未知键、非有限电平、min>max、份数 <=0、
        启用类别集合为空、speech 配 snr_db* / noise|music 配 relative_db*、
        缺 musan_root / musan_split_dir —— 全部在碰文件系统之前报错。"""

@dataclass(frozen=True)
class MixVariant:
    phase: str
    row: Any                       # dataclasses.replace 出来的 AdaptSample
    source_wav: Path               # 要加载的干净 wav（外部源时为绝对路径）
    fbank_path: Path               # 数据集将来解析到的同一路径
    variant_audio_path: str
    category: str
    index: int
    recipe_seed: int
    level_name: str
    level_db: float                # 计划时抽定的电平
    musan_path: Path               # 计划时抽定的干扰录音
    musan_relative_path: str
    musan_sha256: str              # 可选（hash_musan_content）

@dataclass(frozen=True)
class MixPlan:
    request: AugmentationRequest
    config_fingerprint: str
    phase_splits: dict[str, tuple[list[Any], list[Any]]]  # train 已含变体行
    variants: tuple[MixVariant, ...]
    recipes_path: Path             # data_root/augmentation/<gen>/recipes.csv
    features_path: Path            # data_root/augmentation/<gen>/features.csv

def resolve_adapt_noise_augmentation(adapt, prep) -> AugmentationRequest | None: ...
def describe_adapt_noise_augmentation(request) -> str: ...        # 计划表一行
def load_musan_train_pool(request) -> MusanTrainPool: ...         # 泄漏证明
def plan_offline_musan_variants(phase_splits, *, keyword, data_root, request,
                                fbank_params) -> MixPlan: ...      # 纯函数
def materialize_offline_musan_variants(plan, *, skip_existing, fbank_params,
                                       on_progress=None) -> MaterializationResult: ...
~~~

关键实现细节：

- 抽样确定性：recipe_seed = sha256(base_seed, generation, keyword, phase, audio_path, index, category) 前 8 字节；与行顺序、worker 数、其它类别无关。
- 每条变体只调一次 features.TrainingNoiseAugmenter.mix_with_source（新增方法，mix 委托它，保持二元组契约，tests/test_stage2_training_noise.py:326 不动）：

~~~python
def mix(self, waveform, sample_rate, *, rng) -> tuple[torch.Tensor, float]:
    mixed, snr_db, _source = self.mix_with_source(waveform, sample_rate, rng=rng)
    return mixed, snr_db

def mix_with_source(self, waveform, sample_rate, *, rng) -> tuple[torch.Tensor, float, Path]:
    """mix() 加上真正抽中的干扰录音路径（离线溯源用）。"""
~~~

- MixVariant.fbank_path 用数据集的 _resolve_adapt_fbank_path 计算（延迟 import、只读），不复制镜像逻辑。
- 变体路径固定为 raw/<phase>/_aug/<generation>/<stem>__aug<NNN>_<cat>.wav；即使源是外部绝对路径也用它，fbank 落在 fbank/<phase>/_aug/<gen>/ 下，机器路径不进入特征树。

接线（4 处小改）：

| 文件 | 改动 |
| --- | --- |
| dma_kws/stage2/prepare_adapt.py prepare_keyword_adaptation :360 | 新增 keyword-only 参数 augmentation=None；phase_splits 完成后 plan（fail fast），干净 fbank 循环之后 materialize + 追加行；fbank_total 计入变体 |
| dma_kws/stage2/adapt_console.py prepare_with_console :134 | resolve 传入；计划表加一行；3 个新进度阶段（augment_plan_total/plan、augment_total/augment、augment_skipped）；stats 增加 augmentation 块与 per-category 表 |
| dma_kws/stage2/features.py | 增加 mix_with_source，mix 委托 |
| configs/adapt/default.yaml | 追加 §4 默认块 |

## 6. 产出物与溯源

~~~text
<data_root>/
  fbank/<phase>/<positive|negative>/....npy         干净特征（不变）
  fbank/<phase>/_aug/<gen>/<stem>__augNNN_cat.npy   变体特征（原子 tmp+fsync+replace）
  manifests/<phase>_train.csv                       干净行在前，变体块追加在后
  manifests/<phase>_eval.csv                        永远干净，不含任何 aug_* 列
  augmentation/<gen>/
    recipes.csv    确定性计划（无时间戳，逐字节可复现）
    features.csv   观测账本（变体 hash/字节数/帧数/功率/增益/源特征 hash）
    state.json     运行日志（有时间戳、计数、状态、配置快照）
~~~

变体行新增的 manifest 元数据列（干净行为空，load_manifest_csv 会丢弃空值，往返稳定）：
aug_recipe_id、aug_recipe_sha256、aug_generation、aug_category、aug_metric、aug_level_db、aug_musan_path、aug_musan_sha256、original_audio_path。

original_audio_path 是刻意选的列名：joint_manifest.py :27-56 已把它当身份键，变体因此留在干净父行的泄漏分组里；语义上也正是"这一行来自哪条干净录音"。

回答"某一行是拿哪条 MUSAN、什么 SNR 混的"：

~~~text
tts_train.csv 行: audio_path=raw/tts/_aug/g1/pos_0001__aug000_noise.wav
  aug_recipe_id=9f2c… aug_category=noise aug_level_db=7.3125
  aug_musan_path=/data/MUSAN/noise/free-sound/noise-0001.wav aug_musan_sha256=1b7e…
  original_audio_path=raw/tts/positive/pos_0001.wav
  -> recipes.csv: recipe_seed=… config_fingerprint=4c81… crop/replay 参数
  -> features.csv: variant_fbank_sha256=c0d3… applied_gain=0.1832
  -> verify --deep: 用记录的 seed/路径重放 -> 同 gain、同 hash
~~~

## 7. 幂等与 generation 契约

- 每个变体由 sha256 派生种子：与行顺序、worker 数、其它类别无关；同一输入重跑，manifest 与 recipes.csv 逐字节一致，fbank 全跳过。
- 删掉某个变体 .npy：只重算那一个（种子相同 -> 同配方、同字节）；孤儿文件不会被引用。
- **generation 是配置血缘**：augmentation/<gen>/recipes.csv 里记录 config_fingerprint（配置快照 + fbank 参数 + MUSAN list 摘要 + 混音策略版本）。配置指纹变化而 generation 未变 -> 拒绝并列出变化字段，提示三条出路（换 generation / 改回配置 / 删该 generation 目录）。这样 `skip_existing` 永远不会把旧配方的特征当成新配方。
- 崩溃矩阵：planning 崩溃无痕；materialize 崩溃留下孤儿 .npy + 部分 recipes/features，重跑按配方补齐；manifest 在特征与账本齐备后才写。

## 8. 不变量与守卫

| # | 不变量 | 机制 |
| --- | --- | --- |
| 1 | 只加 train 正样本 | planner 只遍历 phase_splits[phase][0] 且 label==1；auditor 发现 eval 行或 label==0 行带 aug_* 即 fail |
| 2 | 训练运行时零改动 | 只有 features.py 加一个方法；数据集/adapt/ joint/入口脚本不动 |
| 3 | 变体通过镜像契约可加载 | MixVariant.fbank_path 来自数据集自己的解析函数；round-trip 测试用真实 KeywordAdaptationDataset 加载 |
| 4 | joint 元数据完整 | dataclasses.replace 保留 split/phase/metadata（含 speaker_id、keyword_phonemes）；original_audio_path 保持分组 |
| 5 | 确定性与幂等 | §7；skip_existing 按文件 + ledger 校验 |
| 6 | MUSAN 泄漏 fail-closed | 无开关可绕过：musan_root 必填；train 列表条目必须落在 root 下且 detect_subset 映射到启用类别；必须存在非空 eval 白名单且与各类别池不相交（目录同名 eval_musan.list 直接拒绝当 train_list）；split.json 存在时每个池的 pool_sha256 必须等于 splits.train.by_category[kind].catalog_sha256（两侧都是排序后的 root 相对路径摘要）；启用类别池为空报错 |
| 7 | 类别语义 | 每类别 enabled / level 范围；variants_per_sample 按类别轮转；一个变体只抽一条录音 |
| 8 | 特征参数一致 | 变体与干净特征用同一份 fbank_kwargs(get_fbank_config(config)) |
| 9 | dither 守卫 | fbank.dither != 0 时启用混音直接报错，提示 fbank.dither=0 或 override /fbank: icefall_kws。理由：dither>0 时同一配方两次物化特征不同，"确定性"承诺失效；v41/v42 链已是 0，普通 adapt_hey_eva 需显式设置 |
| 10 | 发布顺序 | manifest 行只在特征已存在且已入账本后写；features.csv 在 manifest 之前提交 |

规划期关系型检查：启用后变体数 0、同行重复 audio_path、两个变体落到同一 fbank_path、合成 wav 路径已真实存在、文件名超 255 字节、源 manifest 已带 aug_* 元数据 —— 全部报错。

## 9. 类别平衡（需要拍板）

MixedAdaptationDataset 在 keyword 数据集内按行均匀抽取（adapt_dataset.py :236-239），损失是逐句 BCE（losses.py :260）。加变体后正样本行数变成 (1 if keep_original else 0) 倍以上，正负先验偏移；同时"带噪 => 正样本"可能成为可学习的伪相关，而 checkpoint 选的 MUSAN FA/hour 正是会被它抬高的指标。

选项：

1. 第一步推荐：variants_per_sample=1、keep_original=true、只开 noise（+music），speech 最后再开；先测 target AUC 与 eval_musan_fa FA/hour 相对无增广基线；
2. 若 FA 变差：先提高干净占比（keep_original=true 固定，减小 variants），再考虑给负样本也加噪（需要显式放宽不变量 1，并给负样本走波形路径）；
3. 可选 balance 开关：给负样本行复制同样份数（纯 manifest 操作），把先验恢复 1:1；默认不做。

## 10. 测试计划（tests/test_prepare_adapt_noise.py 等）

1. 配置校验：关闭 -> None；未知键；缺 musan_split_dir/musan_root；min > max；speech 配 snr_db*、noise 配 relative_db*；启用类别池为空；variants_per_sample <= 0；train_list 指向 eval_musan.list；dither>0 报错。
2. 泄漏守卫：临时 MUSAN 树 + train/eval list + split.json；断言不相交失败点名；断言同目录 split.json 对不上摘要时拒绝。
3. SNR 数学：常量干净 + 常量噪声 + monkeypatch _load_audio，min==max 时实测 10*log10(P_clean/P_delta) 等于配置值；speech relative_db=r 得到 snr_db=-r；与 TrainingNoiseAugmenter.mix 数值 parity。
4. 确定性：重复 resolve/plan 一致；种子与行顺序、类别轮转顺序无关；换 generation/seed 换种子与路径。
5. 幂等/skip_existing：连续两次 prepare，train CSV 与 recipes.csv 逐字节一致、第二次 0 写入；删一个变体 .npy 只补一个、CSV 不变；配置指纹变化未换 generation -> 拒绝。
6. manifest round-trip：真实 KeywordAdaptationDataset 加载每条 train 行（含变体、无 wav）；eval manifest 无 aug_* 列。
7. joint：真实正样本变体带 speaker_id；prepare 后 validate_joint_manifests(data_root) 通过。

## 11. 综合决策与被否决方案

综合：**以最小改动面候选为骨架**（合成 audio_path + 只物化 fbank；复用 TrainingNoiseAugmenter；只在 prepare_adapt/adapt_console/features/config 四处动刀），**移植可审计候选的三项**：

1. 变体统一放到 raw/<phase>/_aug/<gen>/ 命名空间（修复外部绝对路径经 fbank/external 分支烘入机器路径的问题；generation 让陈旧文件清理与血缘判断变简单）；
2. config_fingerprint + 拒绝机制（补上"换配置不换文件名"的静默复用漏洞）；
3. recipes.csv / features.csv 双层账本与 original_audio_path 列（joint 分组 + 可重放溯源）。

被否决：

- **(a) 物化真实加噪 wav**：下一次 scan_raw_tree / _wav_paths 会把加噪文件当新数据再混一遍（数据级幂等破坏）；放 raw 外则镜像退化，不同 phase 同名互相覆盖；磁盘/IO 翻倍。
- **(b) 独立 manifest + 独立 fbank 根**：adapt.py 固定了路径，不可达（除非改训练代码）。
- **(c) fbank 域直接相加**：log-mel 能量不可加，训练条件与评估（波形混合）不可比。
- **(d) 新写混音器**：会重复 RMS-SNR/重采样/静音重抽约 70 行，必然与在线路径漂移。**但记录裁剪偏移、让重放完全无需 RNG 的更强审计版本**（新 noise_mix.py 纯函数 + features.py 委托）作为后续升级项，不在 v1：v1 记录 seed + 抽中录音 + 电平，同环境可重放；跨版本重放依赖 RNG 序列不变，deep verify 会暴露差异。
- **(e) 评估侧 MusanWaveformMixer 当引擎**：整段配方引擎（burst/stationary/volume、固定采样率），超出训练需要；只借鉴其配置 schema 与 recipe_metadata 思路。

## 12. 实施顺序与验证

1. prepare_adapt_noise.py + 测试 1-4（纯函数，无音频 I/O）：配置、泄漏、SNR 数学、确定性；
2. features.py 加 mix_with_source，跑 tests/test_stage2_training_noise.py 确认行为不变；
3. prepare_adapt.py 接线 + 测试 5-7；adapt_console.py + configs/adapt/default.yaml + 统计/进度；
4. 端到端：小 MUSAN 子集跑 prepare，检查 tts_train.csv 的 aug_* 列、fbank/**/_aug/** 文件、eval manifest 无 aug_* 列；再跑 adapt.stage=train 证明运行时无感。

~~~bash
.venv/bin/python -m pytest tests/test_prepare_adapt_noise.py tests/test_stage2_training_noise.py \
  tests/test_adapt_dataset.py tests/test_joint_adapt_manifest.py -q
bash scripts/run_smoke.sh     # 注意 AGENTS.md 列出的 6 个缺依赖既有失败
~~~

## 13. 开放问题与风险

- **有没有用**：未知；按 §9 的推荐矩阵先做小实验。
- **speech 混入正样本**：relative_db 接近 0 时关键词可能被同音色竞争语音盖住，等于给正样本注入标噪；默认 speech 关闭、范围 [-10, -5]。
- **与背景负例重复使用 MUSAN**：v42 已把 MUSAN 当背景负样本，同一录音可能同时出现在正样本变体与背景负样本；两者都在 train，不算泄漏，但信号相关，后续可把背景缓存池从 train 列表里扣掉。
- **深重放跨环境**：不同 torchaudio/lhotse 版本可能末位不一致；verify 需区分"结构/溯源错误"与"版本差异提示"。
- **静音/坏文件**：静音正样本在 prepare 期报错（mix 抛异常）；坏 MUSAN 在 _load_audio 报错；可能让原本绿色的 prepare 变红。
- **存储**：每变体约 T/10ms x 80 x 4B，2 秒片段约 64KB；1 万正样本 x 2 变体约 1.3GB；dry_run 先报数。
- **孤儿特征**：换 generation 后旧目录不自动清理，可用 augmentation/<gen> 目录粒度删除。
- **joint 浪费**：joint 只训 real，默认 phases: [] 也会给 tts 生成变体，建议显式 [real]。
- **manifest 写入原子性**：现有 write_train_manifest 原地写；可选的小硬化（tmp+os.replace）值得单独一条小改动。

## 14. 下一步

先写 dma_kws/stage2/prepare_adapt_noise.py 的 from_config / load_musan_train_pool / plan_offline_musan_variants（类型、泄漏守卫、配方种子、命名、recipes.csv），配套 tests/test_prepare_adapt_noise.py 用例 1-4，再动 features.py / prepare_adapt.py / adapt_console.py。

## 15. 基线复核（2026-10-13 现场实测）与方案修订

SNR=10 下的现场基线（二阶段 / 端到端唤醒率）：

| 场景 | 二阶段 | 端到端 |
| --- | --- | --- |
| 噪声 40 dB | 0.979 (189/193) | 0.9538 |
| 音乐噪声 40 dB | 0.949 (186/196) | 0.9395 |
| 人声混杂噪声 40 dB | 0.636 (126/198) | 0.6332 |
| 噪声 50 dB | 0.974 (189/194) | 0.9440 |
| 音乐噪声 50 dB | 0.949 (186/196) | 0.9196 |
| 人声混杂噪声 50 dB | 0.667 (130/195) | 0.6567 |
| 噪声 60 dB | 0.964 (187/194) | 0.9389 |
| 音乐噪声 60 dB | 0.949 (186/196) | 0.9196 |
| 人声混杂噪声 60 dB | 0.698 (139/199) | 0.6664 |

统计读法（Wilson / Newcombe 95% CI，n≈195）：

- 人声 vs 噪声/音乐：+31~34pp，CI 不含 0，是唯一确凿的差距；
- 噪声 40 vs 60（+1.5pp）、噪声 vs 音乐（+3.0pp）、人声 40 vs 60（+6.2pp）均不显著；
- 人声条件下 n≈195 的 CI 宽约 ±13pp；检出 0.65→0.75 需要每组 ≥330 条，配对设计（同一批片段）需求低一个量级。

三个必须先处理的观察：

1. **音乐三行计数完全相同（186/196 ×3）**：几乎可以确定电平旋钮没有作用到 music 分量（或复用了同一批片段）。这三行是同一个条件，不是三个独立条件；先修混音脚本再谈结论。
2. **训练侧从未见过人声干扰**：adapt 禁止在线混噪；且按 v4.1 配方文档，泄漏安全切分是 --train-categories music,noise（speech 100% 在 eval）。人声只出现在评估侧，"人声混杂"的失败没有对应训练监督——这是根因候选，请核对现场 split.json 的实际 train_categories。
3. **人声条件下二阶段与端到端只差 0.2~3pp**，损失在 encoder/QbyT，不在后端。

由此对方案的修订（按优先级）：

1. **先做零训练成本判定**：读人声条件下的 AUC 与分数分布。若 AUC 仍高、只是固定阈值召回低，则是标定问题（现行阈值由 music+noise 背景拟合，speech 全在 eval），在含 speech 的背景池上重标定可能直接回收大部分差距。
2. **speech 提为第一优先类别**，范围覆盖 SNR=10（relative_db = -10），建议 [-15, -3]，份数 ≥2；noise/music 已 0.95+，先不投入混噪预算。
3. **新增组合类别 speech_noise**：失败条件是"人声+噪声"同时存在，当前"每变体一条录音"的模型无法表达该条件。MixVariant 改为 1..N 个分量，每个分量独立对干净信号定增益后相加（与评估侧多分量混音同构）。
4. **人声来源与切分**：现有 train 白名单只有 music,noise，需重建（--train-categories music,noise,speech，代价是 eval speech 缩到约 40%，基线必须重测），或引入独立的 train-only 语音语料。注意 LibriSpeech other-500 已在定阈值背景池里，不能直接拿来当训练干扰。
5. **负样本侧同步加 speech**：把 speech 纳入背景负样本（重建背景缓存），否则"有 speech ⇒ 正样本"的偏置会抬高 speech 背景上的 FA；保留 keep_original 干净占比。
6. **容量**：本仓库实测放开 encoder 在干净条件下有 +5~20pp（§2），冻结 encoder 只靠正样本混噪预计只能回收部分差距；本轮建议至少与 encoder 解冻（或 C1 全参）同批比较。

验收口径（新增，替代"跑一次看看"）：

- 每个条件 ≥400 条**配对**片段；同时报 AUC、固定 FA 预算下的召回（阈值在含 speech 的背景池上定）与相对基线的 McNemar 检验；
- noise/music 不得回退（当前 0.949~0.979，回退即判负）；
- 明确目标值，例如 1 FA/h 预算下人声混杂 SNR=10 二阶段唤醒率 ≥0.90、端到端 ≥0.88。

## 16. 实施记录（2026-10-13）

完整实现文档见 [../../adapt-offline-musan-mix.md](../../adapt-offline-musan-mix.md)。本节只记录与计划的差异。

已按 §4/§5 落地。实际接口：

| 项 | 实施 |
| --- | --- |
| 新模块 | dma_kws/stage2/prepare_adapt_noise.py |
| 配置 | adapt.noise_augmentation（默认 enabled=false；musan_root 缺省取 prep.musan_root；train_list/eval_list 可覆盖，默认取 musan_split_dir 下的 train_background.list / eval_musan.list） |
| 类型 | AugmentationRequest / AdaptMixCategory / MixVariant / MixPlan / MixStats |
| 函数 | resolve_adapt_noise_augmentation、describe_offline_musan_request、plan_offline_musan_variants、materialize_offline_musan_variants、preview_offline_musan_variants |
| features.py | TrainingNoiseAugmenter.mix_with_source(noise_path=, snr_db=)；mix() 委托它，二元组契约不变（tests/test_stage2_training_noise.py 全绿） |
| 变体路径 | raw/<phase>/_aug/<gen>/<stem>__<key8>__augNNN_<cat>.wav（key8 = 源 audio_path 的 sha256 前 8 位，避免同名撞车） |
| 产出 | augmentation/<gen>/{recipes.json, features.jsonl, <cat>.list, plan_preview.json（仅 dry-run）} |
| 溯源列 | aug_recipe_id / aug_generation / aug_category / aug_metric / aug_level_db / aug_musan_path / original_audio_path |
| 幂等 | sha256 派生配方种子 + 配置指纹（存在 recipes.json 内）；指纹变化而 generation 未变则报错；skip_existing 逐文件 |
| 测试 | tests/test_prepare_adapt_noise.py 11 项：配置校验、泄漏守卫、确定性、幂等/指纹、dry-run、端到端 manifest + KeywordAdaptationDataset 加载 |

与计划的偏离（简化，不改变语义）：

1. 账本用 recipes.json + features.jsonl（不是 recipes.csv / features.csv）；features.jsonl 按 recipe_id 排序并合并既有条目，重跑逐字节一致。
2. 没有单独的 verify CLI 与 generation 清理命令（v1 不做）。
3. 每个变体仍只混一条录音；§15 修订 3 的 speech+noise 组合类别未实现，留作下一步。
4. dither 守卫没有做成硬错误：adapt_hey_eva 等实验继承 fbank: default（dither 0.1），硬错误会挡住现有运行；v41/v42 链已是 dither 0，文档提示为先。
5. phases 只做交集：请求的 phase 未准备就跳过，全部不匹配才报错。
6. 类别覆盖改成每个启用类别各生成 variants_per_sample 个变体。计划里的轮转写法在三个类别都开、variants_per_sample=1 时只会生成 noise 变体，speech 拿不到，与目标相反。

使用（hey eva v42）：

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

前置条件：speech 必须出现在 train 白名单里。按 v4.1 配方文档，现有切分是 --train-categories music,noise（speech 100% 在 eval）；启用一个没有 train 录音的类别会直接报错并给出重建命令。要训 speech，先重建切分（--train-categories music,noise,speech，eval speech 缩到约 40%，基线需重测）。

验证：.venv/bin/python -m pytest tests/test_prepare_adapt_noise.py tests/test_stage2_training_noise.py tests/test_adapt_dataset.py tests/test_adapt_console.py -q；全量 2194 passed，8 个失败均为缺 whisper/torchcodec/lhotse 等可选依赖的既有失败。


