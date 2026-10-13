"""Tests for offline MUSAN mixing into keyword-adaptation train positives."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from dma_kws.inference.musan_fa import detect_subset, musan_catalog_sha256
from dma_kws.stage2 import features as features_module
from dma_kws.stage2.adapt_dataset import KeywordAdaptationDataset
from dma_kws.stage2.features import TrainingNoiseAugmenter
from dma_kws.stage2.prepare_adapt import (
    AdaptSample,
    load_manifest_csv,
    prepare_keyword_adaptation,
)
from dma_kws.stage2.prepare_adapt_noise import (
    materialize_offline_musan_variants,
    plan_offline_musan_variants,
    preview_offline_musan_variants,
    resolve_adapt_noise_augmentation,
)

SR = 16000
SMALL_FBANK = {
    "num_mel_bins": 8,
    "frame_length": 25,
    "frame_shift": 10,
    "dither": 0.0,
    "window_type": "povey",
    "backend": "torchaudio_kaldi",
    "target_sample_rate": SR,
    "snip_edges": False,
    "low_freq": 20.0,
    "high_freq": -400.0,
}
DICT_PATH = Path(__file__).resolve().parents[1] / "data" / "dict" / "lang_char.txt"


def _write_wav(path: Path, *, value: float = 0.1, seconds: float = 0.3, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(sr * seconds), dtype=np.float32) / sr
    sf.write(str(path), (value * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32), sr)
    return path


def _write_list(path: Path, entries) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{Path(entry)}\n" for entry in entries), encoding="utf-8")
    return path


def _musan_root(tmp_path: Path, *, per_category: int = 2) -> Path:
    root = tmp_path / "MUSAN"
    for category in ("noise", "music", "speech"):
        for index in range(per_category):
            _write_wav(root / category / f"{category}-{index:02d}.wav", value=0.05 * (index + 1))
    return root


def _all_audio(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.rglob("*.wav"))


def _split_dir(
    tmp_path: Path,
    root: Path,
    *,
    train_categories=("noise", "music", "speech"),
    write_split_json: bool = True,
) -> Path:
    split_dir = tmp_path / "musan_split"
    everything = _all_audio(root)
    train: list[Path] = []
    evaluation: list[Path] = []
    for path in everything:
        category = detect_subset(path, root)
        take_train = category in train_categories and not any(
            detect_subset(candidate, root) == category for candidate in train
        )
        (train if take_train else evaluation).append(path)
    _write_list(split_dir / "train_background.list", train)
    _write_list(split_dir / "eval_musan.list", evaluation)
    if write_split_json:
        by_category = {}
        for category in ("noise", "music", "speech"):
            pool = [path for path in train if detect_subset(path, root) == category]
            by_category[category] = {
                "catalog_sha256": musan_catalog_sha256(pool, root),
                "recordings": len(pool),
            }
        payload = {
            "schema_version": 2,
            "musan_root": str(root),
            "policy": {"train_categories": list(train_categories)},
            "splits": {"train": {"by_category": by_category}},
        }
        (split_dir / "split.json").write_text(json.dumps(payload), encoding="utf-8")
    return split_dir


def _block(split_dir: Path, **overrides) -> dict:
    block = {
        "enabled": True,
        "generation": "g1",
        "seed": 2025,
        "musan_split_dir": str(split_dir),
        "variants_per_sample": 1,
        "keep_original": True,
        "categories": {
            "noise": {"enabled": True, "snr_db_min": 0.0, "snr_db_max": 15.0},
            "music": {"enabled": True, "snr_db_min": 5.0, "snr_db_max": 20.0},
            "speech": {"enabled": True, "relative_db_min": -15.0, "relative_db_max": -5.0},
        },
    }
    block.update(overrides)
    return block


def _request(tmp_path: Path, **overrides):
    root = _musan_root(tmp_path)
    split_dir = _split_dir(tmp_path, root)
    request = resolve_adapt_noise_augmentation(
        {"noise_augmentation": _block(split_dir, **overrides)}, {"musan_root": str(root)}
    )
    assert request is not None
    return request, root


def _rows() -> tuple[list[AdaptSample], list[AdaptSample]]:
    train = [
        AdaptSample(audio_path="raw/tts/positive/a.wav", text="hey eva", label=1, phase="tts", split="train"),
        AdaptSample(audio_path="raw/tts/positive/b.wav", text="hey eva", label=1, phase="tts", split="train"),
        AdaptSample(audio_path="raw/tts/negative/hey_ava.wav", text="hey ava", label=0, phase="tts", split="train"),
    ]
    evaluation = [
        AdaptSample(audio_path="raw/tts/eval/positive/c.wav", text="hey eva", label=1, phase="tts", split="eval")
    ]
    return train, evaluation


def test_disabled_or_missing_block_returns_none():
    assert resolve_adapt_noise_augmentation({}, {}) is None
    assert resolve_adapt_noise_augmentation({"noise_augmentation": {"enabled": False}}, {}) is None
    assert resolve_adapt_noise_augmentation({"noise_augmentation": {}}, {}) is None


def test_config_validation_rejects_bad_inputs(tmp_path: Path):
    request, root = _request(tmp_path)
    assert [category.kind for category in request.categories] == ["noise", "music", "speech"]

    split_dir = Path(request.musan_split_dir)
    with pytest.raises(ValueError, match="Unknown adapt.noise_augmentation"):
        resolve_adapt_noise_augmentation(
            {"noise_augmentation": _block(split_dir, bogus=1)}, {"musan_root": str(root)}
        )
    with pytest.raises(ValueError, match="musan_split_dir"):
        resolve_adapt_noise_augmentation(
            {"noise_augmentation": _block(split_dir, musan_split_dir="")}, {"musan_root": str(root)}
        )
    with pytest.raises(ValueError, match="must be <= snr_db_max"):
        resolve_adapt_noise_augmentation(
            {
                "noise_augmentation": _block(
                    split_dir,
                    categories={"noise": {"enabled": True, "snr_db_min": 20.0, "snr_db_max": 10.0}},
                )
            },
            {"musan_root": str(root)},
        )
    with pytest.raises(ValueError, match="it takes relative_db_min/relative_db_max"):
        resolve_adapt_noise_augmentation(
            {
                "noise_augmentation": _block(
                    split_dir,
                    categories={"speech": {"enabled": True, "snr_db_min": -10.0, "snr_db_max": -5.0}},
                )
            },
            {"musan_root": str(root)},
        )
    with pytest.raises(ValueError, match="requires snr_db_min, snr_db_max"):
        resolve_adapt_noise_augmentation(
            {
                "noise_augmentation": _block(
                    split_dir,
                    categories={"noise": {"enabled": True}},
                )
            },
            {"musan_root": str(root)},
        )
    with pytest.raises(ValueError, match="variants_per_sample"):
        resolve_adapt_noise_augmentation(
            {"noise_augmentation": _block(split_dir, variants_per_sample=0)}, {"musan_root": str(root)}
        )
    with pytest.raises(ValueError, match="must be the train allowlist"):
        resolve_adapt_noise_augmentation(
            {"noise_augmentation": _block(split_dir, train_list=str(split_dir / "eval_musan.list"))},
            {"musan_root": str(root)},
        )
    with pytest.raises(ValueError, match="must be finite"):
        resolve_adapt_noise_augmentation(
            {
                "noise_augmentation": _block(
                    split_dir,
                    categories={"noise": {"enabled": True, "snr_db_min": float("nan"), "snr_db_max": 10.0}},
                )
            },
            {"musan_root": str(root)},
        )


def test_leakage_guard(tmp_path: Path):
    request, root = _request(tmp_path)
    split_dir = Path(request.musan_split_dir)
    train_paths = [line for line in (split_dir / "train_background.list").read_text().splitlines() if line]
    _write_list(split_dir / "train_background.list", train_paths + [str(_all_audio(root)[0])])
    _write_list(split_dir / "eval_musan.list", [str(_all_audio(root)[0])])
    with pytest.raises(ValueError, match="overlaps the eval allowlist"):
        resolve_adapt_noise_augmentation(
            {"noise_augmentation": _block(split_dir)}, {"musan_root": str(root)}
        )

    split_dir = _split_dir(tmp_path / "mismatch", root)
    payload = json.loads((split_dir / "split.json").read_text(encoding="utf-8"))
    payload["splits"]["train"]["by_category"]["noise"]["catalog_sha256"] = "0" * 64
    (split_dir / "split.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the recorded split partition"):
        resolve_adapt_noise_augmentation(
            {"noise_augmentation": _block(split_dir)}, {"musan_root": str(root)}
        )


def test_enabled_category_without_train_pool_is_actionable(tmp_path: Path):
    root = _musan_root(tmp_path)
    split_dir = _split_dir(tmp_path, root, train_categories=("noise", "music"))
    with pytest.raises(ValueError, match="no speech recordings"):
        resolve_adapt_noise_augmentation(
            {"noise_augmentation": _block(split_dir)}, {"musan_root": str(root)}
        )


def test_plan_is_deterministic_and_only_expands_train_positives(tmp_path: Path):
    request, _root = _request(tmp_path)
    train_rows, eval_rows = _rows()
    splits = {"tts": (train_rows, eval_rows)}
    first = plan_offline_musan_variants(
        splits, keyword="hey eva", data_root=tmp_path, request=request, fbank_params=SMALL_FBANK
    )
    second = plan_offline_musan_variants(
        splits, keyword="hey eva", data_root=tmp_path, request=request, fbank_params=SMALL_FBANK
    )
    assert [
        (v.variant_audio_path, v.recipe_seed, v.level_db, str(v.musan_path)) for v in first.variants
    ] == [
        (v.variant_audio_path, v.recipe_seed, v.level_db, str(v.musan_path)) for v in second.variants
    ]
    # Two train positives, three enabled categories, one copy each.
    assert len(first.variants) == 6
    assert first.phase_splits["tts"][1] == eval_rows
    planned_rows = first.phase_splits["tts"][0]
    assert len(planned_rows) == 9  # keep_original: 2 clean positives + 1 negative + 6 variants
    variant_rows = [row for row in planned_rows if row.metadata.get("aug_recipe_id")]
    assert len(variant_rows) == 6
    assert {row.metadata["aug_category"] for row in variant_rows} == {"noise", "music", "speech"}
    assert all(row.audio_path.startswith("raw/tts/_aug/g1/") for row in variant_rows)
    assert all(row.metadata["original_audio_path"].startswith("raw/tts/positive/") for row in variant_rows)
    assert all(v.fbank_path.parent == tmp_path / "fbank" / "tts" / "_aug" / "g1" for v in first.variants)


def test_plan_can_drop_clean_positives(tmp_path: Path):
    request, _root = _request(tmp_path, keep_original=False)
    train_rows, eval_rows = _rows()
    plan = plan_offline_musan_variants(
        {"tts": (train_rows, eval_rows)},
        keyword="hey eva",
        data_root=tmp_path,
        request=request,
        fbank_params=SMALL_FBANK,
    )
    kept = plan.phase_splits["tts"][0]
    assert len(kept) == 7  # 6 variants + the untouched negative
    assert not [row for row in kept if row.label == 1 and "aug_recipe_id" not in row.metadata]


def test_mix_with_source_honours_recorded_source_and_level(monkeypatch, tmp_path: Path):
    noise = tmp_path / "noise.wav"
    noise.touch()
    noise_list = _write_list(tmp_path / "noise.list", [noise])
    waveform_dir = tmp_path / "wav"
    waveform_dir.mkdir()
    augmenter = TrainingNoiseAugmenter(
        waveform_dir=waveform_dir,
        noise_list_path=noise_list,
        snr_db_min=0.0,
        snr_db_max=30.0,
    )
    noise_waveform = torch.tensor([[1.0, -1.0] * 8])
    monkeypatch.setattr(
        features_module, "_load_audio", lambda path, **kwargs: (noise_waveform.clone(), SR)
    )
    clean = torch.full((1, 16), 0.5)
    mixed, sampled_snr, drawn = augmenter.mix_with_source(
        clean, SR, rng=random.Random(0), noise_path=noise, snr_db=10.0
    )
    measured = 10.0 * math.log10(
        float(clean.square().mean() / (mixed - clean).square().mean())
    )
    assert sampled_snr == pytest.approx(10.0)
    assert measured == pytest.approx(10.0, abs=1e-4)
    assert Path(drawn) == noise


def test_materialize_is_idempotent_and_guards_fingerprint(tmp_path: Path):
    request, _root = _request(tmp_path, variants_per_sample=2)
    train_rows, eval_rows = _rows()
    plan = plan_offline_musan_variants(
        {"tts": (train_rows, eval_rows)},
        keyword="hey eva",
        data_root=tmp_path,
        request=request,
        fbank_params=SMALL_FBANK,
    )
    for variant in plan.variants:
        _write_wav(variant.source_wav)

    first = materialize_offline_musan_variants(
        plan, fbank_params=SMALL_FBANK, skip_existing=True
    )
    assert first.written == 12 and first.skipped == 0
    assert all(variant.fbank_path.is_file() for variant in plan.variants)
    assert all(np.load(variant.fbank_path).shape[1] == 8 for variant in plan.variants)
    recipes = plan.recipes_path.read_bytes()
    features = plan.features_path.read_bytes()

    second = materialize_offline_musan_variants(
        plan, fbank_params=SMALL_FBANK, skip_existing=True
    )
    assert second.written == 0 and second.skipped == 12
    assert plan.recipes_path.read_bytes() == recipes
    assert plan.features_path.read_bytes() == features

    changed, _root = _request(tmp_path / "other", variants_per_sample=2)
    changed_plan = plan_offline_musan_variants(
        {"tts": (train_rows, eval_rows)},
        keyword="hey eva",
        data_root=tmp_path,
        request=changed,
        fbank_params=SMALL_FBANK,
    )
    with pytest.raises(ValueError, match="different configuration fingerprint"):
        materialize_offline_musan_variants(
            changed_plan, fbank_params=SMALL_FBANK, skip_existing=True
        )


def test_preview_writes_no_features(tmp_path: Path):
    request, _root = _request(tmp_path, dry_run=True)
    train_rows, eval_rows = _rows()
    plan = plan_offline_musan_variants(
        {"tts": (train_rows, eval_rows)},
        keyword="hey eva",
        data_root=tmp_path,
        request=request,
        fbank_params=SMALL_FBANK,
    )
    stats = preview_offline_musan_variants(plan)
    assert stats.status == "dry_run" and stats.planned == 6 and stats.written == 0
    assert (plan.generation_dir / "plan_preview.json").is_file()
    assert not plan.recipes_path.exists()
    assert not any(variant.fbank_path.exists() for variant in plan.variants)


def test_prepare_keyword_adaptation_dry_run_keeps_manifests_clean(tmp_path: Path, monkeypatch):
    from dma_kws.stage2 import prepare_adapt as prepare_adapt_module

    monkeypatch.setattr(prepare_adapt_module, "make_g2p", lambda: object())
    monkeypatch.setattr(prepare_adapt_module, "validate_g2p", lambda *_args, **_kwargs: None)

    def fake_clean_fbank(wav_path, fbank_path, *, fbank_params, skip_existing=True):
        del wav_path, fbank_params, skip_existing
        Path(fbank_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(fbank_path, np.zeros((4, 8), dtype=np.float32))
        return True

    monkeypatch.setattr(prepare_adapt_module, "compute_and_save_fbank", fake_clean_fbank)

    root = _musan_root(tmp_path)
    split_dir = _split_dir(tmp_path, root)
    request = resolve_adapt_noise_augmentation(
        {"noise_augmentation": _block(split_dir, dry_run=True)}, {"musan_root": str(root)}
    )
    data_root = tmp_path / "adapt" / "hey_eva"
    _write_wav(data_root / "raw" / "tts" / "positive" / "pos_a.wav")
    _write_wav(data_root / "raw" / "tts" / "positive" / "pos_b.wav")
    _write_wav(data_root / "raw" / "tts" / "eval" / "positive" / "held.wav")

    stats = prepare_keyword_adaptation(
        keyword="hey eva",
        data_root=data_root,
        fbank_params=SMALL_FBANK,
        noise_augmentation=request,
    )
    assert stats["augmentation"]["status"] == "dry_run"
    assert stats["phases"]["tts"]["train"] == 2
    assert stats["phases"]["tts"]["train_augmented"] == 0
    train_rows = load_manifest_csv(data_root / "manifests" / "tts_train.csv")
    assert len(train_rows) == 2
    assert not any(row.metadata.get("aug_recipe_id") for row in train_rows)
    assert (data_root / "augmentation" / "g1" / "plan_preview.json").is_file()
    assert not list((data_root / "fbank").rglob("*_aug*"))


def test_prepare_keyword_adaptation_end_to_end(tmp_path: Path, monkeypatch):
    from dma_kws.stage2 import prepare_adapt as prepare_adapt_module

    monkeypatch.setattr(prepare_adapt_module, "make_g2p", lambda: object())
    monkeypatch.setattr(prepare_adapt_module, "validate_g2p", lambda *_args, **_kwargs: None)

    def fake_clean_fbank(wav_path, fbank_path, *, fbank_params, skip_existing=True):
        del wav_path
        if skip_existing and Path(fbank_path).is_file():
            return False
        Path(fbank_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(
            fbank_path,
            np.zeros((4, int(fbank_params.get("num_mel_bins", 8))), dtype=np.float32),
        )
        return True

    # The local venv has no TorchCodec, which torchaudio.load now requires; the
    # clean-feature pass is not what this test exercises.
    monkeypatch.setattr(prepare_adapt_module, "compute_and_save_fbank", fake_clean_fbank)

    root = _musan_root(tmp_path)
    split_dir = _split_dir(tmp_path, root)
    request = resolve_adapt_noise_augmentation(
        {"noise_augmentation": _block(split_dir)}, {"musan_root": str(root)}
    )
    data_root = tmp_path / "adapt" / "hey_eva"
    _write_wav(data_root / "raw" / "tts" / "positive" / "pos_a.wav")
    _write_wav(data_root / "raw" / "tts" / "positive" / "pos_b.wav")
    _write_wav(data_root / "raw" / "tts" / "eval" / "positive" / "held.wav")

    stats = prepare_keyword_adaptation(
        keyword="hey eva",
        data_root=data_root,
        fbank_params=SMALL_FBANK,
        noise_augmentation=request,
    )
    augmentation = stats["augmentation"]
    assert augmentation["status"] == "complete"
    assert augmentation["written"] == 6 and augmentation["planned"] == 6
    assert stats["phases"]["tts"]["train_augmented"] == 6

    train_rows = load_manifest_csv(data_root / "manifests" / "tts_train.csv")
    eval_rows = load_manifest_csv(data_root / "manifests" / "tts_eval.csv")
    assert len(train_rows) == 8
    assert len(eval_rows) == 1
    assert all("aug_recipe_id" not in row.metadata for row in eval_rows)
    variant_rows = [row for row in train_rows if row.metadata.get("aug_recipe_id")]
    assert len(variant_rows) == 6
    assert all(row.metadata["original_audio_path"] in {"raw/tts/positive/pos_a.wav", "raw/tts/positive/pos_b.wav"} for row in variant_rows)

    if not DICT_PATH.is_file():
        pytest.skip("dict not available")
    from dma_kws.stage2 import adapt_dataset as adapt_dataset_module
    from dma_kws.tokenizer import load_char_tokenizer

    monkeypatch.setattr(adapt_dataset_module, "text_to_phonemes", lambda _g2p, text: text.split())
    monkeypatch.setattr(adapt_dataset_module, "make_g2p", lambda: (lambda text: text.split()))
    dataset = KeywordAdaptationDataset(
        manifest_path=data_root / "manifests" / "tts_train.csv",
        keyword="hey eva",
        fbank_root=data_root / "fbank",
        tokenizer=load_char_tokenizer(DICT_PATH),
        manifest_root=data_root,
    )
    loaded = dataset[3]  # appended variant row; no wav exists for it
    assert loaded["label"].item() == 1
    assert loaded["feat"].shape[1] == 8
