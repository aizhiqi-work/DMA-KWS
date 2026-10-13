"""Prepare keyword adaptation data: fbank features and train/eval manifests."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np

from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.stage2.adapt_paths import directory_name_to_text, neg_slug_to_text, slugify, wav_to_fbank_mirror
from dma_kws.stage2.features import waveform_to_fbank
from dma_kws.tokenizer import unsupported_phones

if TYPE_CHECKING:
    from dma_kws.stage2.prepare_adapt_noise import AugmentationRequest


@dataclass
class AdaptSample:
    audio_path: str
    text: str
    label: int
    phase: str
    split: str | None = None
    # Preserve source-manifest attributes such as speaker_id, device and session.
    # They are not model inputs, but they are required to audit domain balance and
    # prove that an explicitly supplied train/eval split is speaker-disjoint.
    metadata: dict[str, str] = field(default_factory=dict)


def _import_torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "Missing torchaudio. Install CUDA PyTorch/torchaudio on the training machine first."
        ) from exc
    return torchaudio


def validate_g2p(text: str, g2p: Any) -> None:
    """Fail early when a keyword/negative text cannot be tokenized cleanly.

    Phonemes outside the vocabulary would otherwise be silently mapped to
    ``<unk>`` at training and inference time.
    """
    phonemes = text_to_phonemes(g2p, text)
    if not phonemes:
        raise ValueError(f"G2P produced empty phoneme sequence for text: {text!r}")
    unsupported = unsupported_phones(phonemes)
    if unsupported:
        raise ValueError(
            f"G2P produced phonemes outside the vocabulary for text {text!r}: "
            f"{', '.join(unsupported)} (full sequence: {' '.join(phonemes)})"
        )


def scan_raw_tree(data_root: Path, keyword: str) -> list[AdaptSample]:
    """Scan explicit train and eval trees under ``raw/{tts,real}``."""
    keyword = keyword.strip()
    raw_root = data_root / "raw"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw adaptation directory not found: {raw_root}")

    samples: list[AdaptSample] = []
    for phase_dir in sorted(raw_root.iterdir()):
        if not phase_dir.is_dir():
            continue
        phase = phase_dir.name
        for split, split_dir in (("train", phase_dir), ("eval", phase_dir / "eval")):
            positive_dir = split_dir / "positive"
            if positive_dir.is_dir():
                for wav_path in sorted(positive_dir.rglob("*.wav")):
                    rel = wav_path.relative_to(data_root)
                    samples.append(
                        AdaptSample(
                            audio_path=str(rel),
                            text=keyword,
                            label=1,
                            phase=phase,
                            split=split,
                        )
                    )

            negative_root = split_dir / "negative"
            if negative_root.is_dir():
                for neg_dir in sorted(path for path in negative_root.iterdir() if path.is_dir()):
                    neg_text = neg_slug_to_text(neg_dir.name)
                    for wav_path in sorted(neg_dir.rglob("*.wav")):
                        rel = wav_path.relative_to(data_root)
                        samples.append(
                            AdaptSample(
                                audio_path=str(rel),
                                text=neg_text,
                                label=0,
                                phase=phase,
                                split=split,
                            )
                        )
    if not samples:
        raise ValueError(f"No wav files found under {raw_root}")
    return samples


def _wav_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() == ".wav")


def scan_external_source(
    *,
    phase: str,
    keyword: str,
    positive_dir: Path,
    negative_root: Path,
) -> list[AdaptSample]:
    if not positive_dir.is_dir():
        raise FileNotFoundError(f"Positive audio directory not found for phase {phase!r}: {positive_dir}")
    if not negative_root.is_dir():
        raise FileNotFoundError(f"Negative audio directory not found for phase {phase!r}: {negative_root}")

    positive_paths = _wav_paths(positive_dir)
    if not positive_paths:
        raise ValueError(f"No wav files found in positive audio directory for phase {phase!r}: {positive_dir}")

    samples = [
        AdaptSample(audio_path=str(path.resolve()), text=keyword, label=1, phase=phase)
        for path in positive_paths
    ]
    negative_dirs = sorted(path for path in negative_root.iterdir() if path.is_dir())
    if not negative_dirs:
        raise ValueError(f"No negative phrase subdirectories found for phase {phase!r}: {negative_root}")

    negative_count = 0
    for negative_dir in negative_dirs:
        text = directory_name_to_text(negative_dir.name)
        for wav_path in _wav_paths(negative_dir):
            samples.append(AdaptSample(audio_path=str(wav_path.resolve()), text=text, label=0, phase=phase))
            negative_count += 1
    if not negative_count:
        raise ValueError(f"No wav files found below negative audio directory for phase {phase!r}: {negative_root}")
    return samples


def scan_external_sources(keyword: str, sources: dict[str, Any]) -> list[AdaptSample]:
    expected_phases = {"tts", "real"}
    configured_phases = {phase for phase, source in sources.items() if isinstance(source, dict) and source}
    if configured_phases != expected_phases:
        raise ValueError("External adaptation sources must configure both 'tts' and 'real' phases")

    samples: list[AdaptSample] = []
    for phase in sorted(expected_phases):
        source = sources[phase]
        positive_dir = str(source.get("positive_dir", "")).strip()
        negative_root = str(source.get("negative_root", "")).strip()
        if not positive_dir or not negative_root:
            raise ValueError(
                f"External adaptation source for phase {phase!r} requires positive_dir and negative_root"
            )
        samples.extend(
            scan_external_source(
                phase=phase,
                keyword=keyword,
                positive_dir=Path(positive_dir),
                negative_root=Path(negative_root),
            )
        )
    return samples


def load_manifest_csv(path: Path) -> list[AdaptSample]:
    rows: list[AdaptSample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"audio_path", "text", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest {path} must contain columns: {sorted(required)}")
        phase = "custom"
        reserved = {"audio_path", "text", "keyword", "label", "phase", "split"}
        for row in reader:
            row_phase = str(row.get("phase", "")).strip() or phase
            row_split = str(row.get("split", "")).strip().casefold() or None
            rows.append(
                AdaptSample(
                    audio_path=str(row["audio_path"]),
                    text=str(row["text"]),
                    label=int(row["label"]),
                    phase=row_phase,
                    split=row_split,
                    metadata={
                        str(key): str(value)
                        for key, value in row.items()
                        if key is not None
                        and key not in reserved
                        and value is not None
                        and str(value).strip()
                    },
                )
            )
    return rows


def split_explicit_manifest(
    samples: list[AdaptSample],
    *,
    phase: str,
) -> tuple[list[AdaptSample], list[AdaptSample]] | None:
    """Return a declared train/eval split and reject speaker leakage.

    A manifest remains backward compatible when no row has ``split``: the caller
    can use the historical seeded random split.  Once one row declares a split,
    every row in that phase must do so; silently randomizing the remainder would
    defeat the purpose of a speaker-controlled manifest.
    """

    declared = [sample for sample in samples if sample.split is not None]
    if not declared:
        return None
    if len(declared) != len(samples):
        raise ValueError(
            f"Manifest phase {phase!r} mixes explicit and missing split values; "
            "set every row to 'train' or 'eval', or omit split from every row."
        )

    invalid = sorted({str(sample.split) for sample in samples} - {"train", "eval"})
    if invalid:
        raise ValueError(
            f"Manifest phase {phase!r} has invalid split values: {invalid}; "
            "expected 'train' or 'eval'."
        )

    train = [sample for sample in samples if sample.split == "train"]
    eval_set = [sample for sample in samples if sample.split == "eval"]
    if not train or not eval_set:
        raise ValueError(
            f"Manifest phase {phase!r} requires at least one train and one eval row "
            "when explicit splits are used."
        )

    train_speakers = {
        sample.metadata["speaker_id"].strip().casefold()
        for sample in train
        if sample.metadata.get("speaker_id", "").strip()
    }
    eval_speakers = {
        sample.metadata["speaker_id"].strip().casefold()
        for sample in eval_set
        if sample.metadata.get("speaker_id", "").strip()
    }
    overlap = sorted(train_speakers & eval_speakers)
    if overlap:
        raise ValueError(
            f"Manifest phase {phase!r} leaks speaker_id across train/eval: {overlap}"
        )
    return train, eval_set


def validate_manifest_speaker_splits(samples: Iterable[AdaptSample]) -> None:
    """Reject a speaker assigned to train anywhere and eval anywhere."""

    train_speakers = {
        sample.metadata["speaker_id"].strip().casefold()
        for sample in samples
        if sample.split == "train" and sample.metadata.get("speaker_id", "").strip()
    }
    eval_speakers = {
        sample.metadata["speaker_id"].strip().casefold()
        for sample in samples
        if sample.split == "eval" and sample.metadata.get("speaker_id", "").strip()
    }
    overlap = sorted(train_speakers & eval_speakers)
    if overlap:
        raise ValueError(f"Manifest leaks speaker_id across train/eval: {overlap}")


def split_train_eval(
    samples: list[AdaptSample],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[list[AdaptSample], list[AdaptSample]]:
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    eval_count = max(1, int(round(len(samples) * eval_fraction))) if samples else 0
    eval_indices = set(indices[:eval_count])
    train = [samples[i] for i in range(len(samples)) if i not in eval_indices]
    eval_set = [samples[i] for i in range(len(samples)) if i in eval_indices]
    return train, eval_set


def compute_and_save_fbank(
    wav_path: Path,
    fbank_path: Path,
    *,
    fbank_params: dict[str, Any],
    skip_existing: bool = True,
) -> bool:
    if skip_existing and fbank_path.is_file():
        return False
    fbank_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio = _import_torchaudio()
    waveform, sample_rate = torchaudio.load(str(wav_path))
    feat = waveform_to_fbank(waveform, sample_rate=sample_rate, **fbank_params)
    np.save(fbank_path, feat.cpu().numpy())
    return True


def _materialize_rows(rows: Iterable[AdaptSample]) -> list[AdaptSample]:
    return list(rows)


def _metadata_fields(rows: Iterable[AdaptSample]) -> list[str]:
    return sorted({key for row in rows for key in row.metadata})


def write_train_manifest(path: Path, rows: Iterable[AdaptSample]) -> None:
    rows = _materialize_rows(rows)
    metadata_fields = _metadata_fields(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["audio_path", "text", "label", *metadata_fields],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "audio_path": row.audio_path,
                    "text": row.text,
                    "label": row.label,
                    **{key: row.metadata.get(key, "") for key in metadata_fields},
                }
            )


def write_eval_manifest(path: Path, rows: Iterable[AdaptSample], keyword: str) -> None:
    rows = _materialize_rows(rows)
    metadata_fields = _metadata_fields(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["audio_path", "text", "keyword", "label", *metadata_fields],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "audio_path": row.audio_path,
                    "text": row.text,
                    "keyword": keyword,
                    "label": row.label,
                    **{key: row.metadata.get(key, "") for key in metadata_fields},
                }
            )


def prepare_keyword_adaptation(
    *,
    keyword: str,
    data_root: Path,
    fbank_params: dict[str, Any],
    eval_fraction: float = 0.2,
    eval_seed: int = 2025,
    manifest_csv: Path | None = None,
    sources: dict[str, Any] | None = None,
    skip_existing: bool = True,
    on_progress: Callable[[str, int], None] | None = None,
    joint: bool = False,
    noise_augmentation: "AugmentationRequest | None" = None,
) -> dict[str, Any]:
    """Prepare fbank features and manifests for one keyword slug tree.

    ``on_progress(stage, value)`` reports work as it happens; stages are
    ``scan``, ``g2p_total``/``g2p``, ``fbank_total``/``fbank`` and, when offline
    mixing is enabled, ``augment_total``/``augment``/``augment_skipped``
    (``*_total`` carries a count, the others an increment).

    ``joint`` requires both source phases, keeps real speakers and original
    recording groups within one split, and preserves every explicit assignment.

    ``noise_augmentation`` is an already validated
    ``prepare_adapt_noise.AugmentationRequest``; ``None`` keeps today's behavior.
    Variants are derived deterministically and only train positives are expanded.
    """

    def report(stage: str, value: int) -> None:
        if on_progress is not None:
            on_progress(stage, value)

    if joint and manifest_csv is not None and not manifest_csv.is_file():
        raise FileNotFoundError(f"Joint adaptation source manifest not found: {manifest_csv}")
    uses_manifest = manifest_csv is not None and manifest_csv.is_file()
    source_config = sources or {}
    uses_external_sources = any(
        isinstance(source, dict)
        and any(str(source.get(key, "")).strip() for key in ("positive_dir", "negative_root"))
        for source in source_config.values()
    )
    if uses_manifest:
        all_samples = load_manifest_csv(manifest_csv)
        if not joint:
            validate_manifest_speaker_splits(all_samples)
    elif uses_external_sources:
        all_samples = scan_external_sources(keyword, source_config)
    else:
        all_samples = scan_raw_tree(data_root, keyword)
    report("scan", len(all_samples))

    by_phase: dict[str, list[AdaptSample]] = {}
    for sample in all_samples:
        by_phase.setdefault(sample.phase, []).append(sample)

    phase_splits: dict[str, tuple[list[AdaptSample], list[AdaptSample]]] = {}
    if joint:
        from dma_kws.stage2.joint_manifest import split_joint_samples

        phase_splits = split_joint_samples(
            all_samples, data_root=data_root, eval_fraction=eval_fraction, seed=eval_seed,
        )
    for phase, phase_samples in (() if joint else sorted(by_phase.items())):
        explicit_split = (
            split_explicit_manifest(phase_samples, phase=phase)
            if uses_manifest
            else None
        )
        if explicit_split is not None:
            train_rows, eval_rows = explicit_split
        elif uses_manifest or uses_external_sources:
            train_rows, eval_rows = split_train_eval(
                phase_samples,
                eval_fraction=eval_fraction,
                seed=eval_seed,
            )
        else:
            unexpected = sorted({sample.split for sample in phase_samples} - {"train", "eval"})
            if unexpected:
                raise ValueError(f"Unexpected adaptation splits for phase {phase!r}: {unexpected}")
            train_rows = [sample for sample in phase_samples if sample.split == "train"]
            eval_rows = [sample for sample in phase_samples if sample.split == "eval"]
            if not train_rows:
                raise ValueError(
                    f"No training wav files found for phase {phase!r}; expected positive/ or negative/ "
                    f"under {data_root / 'raw' / phase}"
                )
            if not eval_rows:
                raise ValueError(
                    f"No evaluation wav files found for phase {phase!r}; expected positive/ or negative/ "
                    f"under {data_root / 'raw' / phase / 'eval'}"
                )
        phase_splits[phase] = (train_rows, eval_rows)

    base_phase_splits = phase_splits
    mix_plan = None
    if noise_augmentation is not None:
        from dma_kws.stage2.prepare_adapt_noise import plan_offline_musan_variants

        mix_plan = plan_offline_musan_variants(
            base_phase_splits,
            keyword=keyword,
            data_root=data_root,
            request=noise_augmentation,
            fbank_params=fbank_params,
        )

    unique_texts = sorted({sample.text for sample in all_samples})
    report("g2p_total", len(unique_texts))
    g2p = make_g2p()
    for text in unique_texts:
        validate_g2p(text, g2p)
        report("g2p", 1)

    materialize_variants = mix_plan is not None and not noise_augmentation.dry_run
    report(
        "fbank_total",
        len(all_samples) + (len(mix_plan.variants) if materialize_variants else 0),
    )
    written = 0
    skipped = 0
    for sample in all_samples:
        wav_path = data_root / sample.audio_path
        if not wav_path.is_file():
            raise FileNotFoundError(f"Missing wav file: {wav_path}")
        fbank_path = wav_to_fbank_mirror(data_root / "fbank", wav_path)
        if compute_and_save_fbank(
            wav_path,
            fbank_path,
            fbank_params=fbank_params,
            skip_existing=skip_existing,
        ):
            written += 1
        else:
            skipped += 1
        report("fbank", 1)

    mix_stats = None
    if mix_plan is not None:
        from dma_kws.stage2.prepare_adapt_noise import (
            materialize_offline_musan_variants,
            preview_offline_musan_variants,
        )

        if noise_augmentation.dry_run:
            mix_stats = preview_offline_musan_variants(mix_plan)
        else:
            mix_stats = materialize_offline_musan_variants(
                mix_plan,
                fbank_params=fbank_params,
                skip_existing=skip_existing,
                on_progress=on_progress,
            )
            phase_splits = mix_plan.phase_splits

    manifest_dir = data_root / "manifests"
    stats: dict[str, Any] = {
        "keyword": keyword,
        "slug": slugify(keyword),
        "phases": {},
        "unique_texts": len(unique_texts),
    }

    for phase, (train_rows, eval_rows) in phase_splits.items():
        train_path = manifest_dir / f"{phase}_train.csv"
        eval_path = manifest_dir / f"{phase}_eval.csv"
        write_train_manifest(train_path, train_rows)
        write_eval_manifest(eval_path, eval_rows, keyword)
        stats["phases"][phase] = {
            "total": len(train_rows) + len(eval_rows),
            "train": len(train_rows),
            "eval": len(eval_rows),
            "train_positive": sum(1 for row in train_rows if row.label == 1),
            "train_negative": sum(1 for row in train_rows if row.label == 0),
            "eval_positive": sum(1 for row in eval_rows if row.label == 1),
            "eval_negative": sum(1 for row in eval_rows if row.label == 0),
            "train_augmented": sum(
                1 for row in train_rows if row.metadata.get("aug_recipe_id")
            ),
            "train_manifest": str(train_path),
            "eval_manifest": str(eval_path),
        }

    if mix_stats is not None:
        stats["augmentation"] = mix_stats.to_stats_dict()
    stats["fbank_written"] = written
    stats["fbank_skipped"] = skipped
    return stats
