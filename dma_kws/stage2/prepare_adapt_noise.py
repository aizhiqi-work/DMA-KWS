"""Offline MUSAN mixing for keyword adaptation training positives.

The adaptation trainer reads precomputed fbank mirrors and never opens a wav, so
noise robustness has to be baked in at prepare time. This module plans
deterministic variants for train-split positives, materializes only their fbank
features through the dataset's own mirror resolver, and records which MUSAN
recording and level produced every row.

Leakage is fail-closed: the MUSAN train allowlist must be disjoint from the eval
allowlist, and when split.json is present every category pool must hash to the
recorded train partition.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from dma_kws.inference.musan_fa import detect_subset, musan_catalog_sha256
from dma_kws.stage2 import features as _features
from dma_kws.stage2.fbank import FbankExtractor

MIX_CATEGORIES = ("noise", "music", "speech")
_AUG_PATH_COMPONENT = "_aug"
_AUGMENTATION_DIR = "augmentation"
_TRAIN_LIST_NAME = "train_background.list"
_EVAL_LIST_NAME = "eval_musan.list"
_SPLIT_JSON_NAME = "split.json"
_RECIPE_SCHEMA_VERSION = 1
_GENERATION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TOP_LEVEL_KEYS = frozenset(
    {
        "enabled",
        "generation",
        "seed",
        "musan_root",
        "musan_split_dir",
        "train_list",
        "eval_list",
        "variants_per_sample",
        "keep_original",
        "dry_run",
        "phases",
        "categories",
    }
)
_CATEGORY_KEYS = frozenset(
    {
        "enabled",
        "variants_per_sample",
        "snr_db_min",
        "snr_db_max",
        "relative_db_min",
        "relative_db_max",
    }
)
_LEVEL_FIELDS = {
    "noise": ("snr_db_min", "snr_db_max"),
    "music": ("snr_db_min", "snr_db_max"),
    "speech": ("relative_db_min", "relative_db_max"),
}


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _reject_unknown_keys(mapping: Mapping[str, Any], allowed: frozenset[str], *, field: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown {field} fields: {', '.join(unknown)}")


def _finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _load_path_list(path: Path) -> tuple[Path, ...]:
    """Read one audio path per line; relative entries resolve from the list dir."""
    if not path.is_file():
        raise FileNotFoundError(f"MUSAN allowlist not found: {path}")
    base = path.parent
    paths: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entry = Path(line).expanduser()
        if not entry.is_absolute():
            entry = base / entry
        paths.append(entry.resolve())
    if not paths:
        raise ValueError(f"MUSAN allowlist is empty: {path}")
    return tuple(paths)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = _tempfile_name(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if Path(temporary_name).exists():
            Path(temporary_name).unlink()


def _tempfile_name(path: Path) -> tuple[int, str]:
    import tempfile

    return tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = _tempfile_name(path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if Path(temporary_name).exists():
            Path(temporary_name).unlink()


def _to_mono(waveform: Any, *, source: Path) -> Any:
    """Average channels to (1, samples), mirroring TrainingNoiseAugmenter."""
    if waveform.dim() == 1:
        return waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError(f"Expected a 1-D or 2-D waveform for {source}, got {tuple(waveform.shape)}")
    if waveform.size(0) == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def _recipe_seed(base_seed: int, generation: str, keyword: str, phase: str, audio_path: str, index: int, category: str) -> int:
    digest = hashlib.sha256(
        "\0".join((str(base_seed), generation, keyword, phase, audio_path, str(index), category)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def _recipe_id(phase: str, audio_path: str, index: int, category: str, generation: str) -> str:
    digest = hashlib.sha256(
        "\0".join((generation, phase, audio_path, str(index), category)).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def _safe_stem(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    return cleaned or "clip"


@dataclass(frozen=True)
class AdaptMixCategory:
    """One enabled MUSAN category with its validated pool and level policy."""

    kind: str
    level_name: str
    level_min: float
    level_max: float
    pool: tuple[Path, ...]
    pool_sha256: str

    @property
    def snr_db_min(self) -> float:
        return -self.level_max if self.level_name == "relative_db" else self.level_min

    @property
    def snr_db_max(self) -> float:
        return -self.level_min if self.level_name == "relative_db" else self.level_max

    def snr_db_for(self, level_db: float) -> float:
        return -float(level_db) if self.level_name == "relative_db" else float(level_db)

    def describe(self) -> str:
        unit = "rel" if self.level_name == "relative_db" else "snr"
        return f"{self.kind} {unit} {self.level_min:g}..{self.level_max:g}"


@dataclass(frozen=True)
class AugmentationRequest:
    """Validated, trusted offline-mixing configuration."""

    generation: str
    seed: int
    musan_root: Path
    musan_split_dir: Path
    train_list: Path
    eval_list: Path
    variants_per_sample: int
    keep_original: bool
    dry_run: bool
    phases: tuple[str, ...]
    categories: tuple[AdaptMixCategory, ...]

    @classmethod
    def from_config(
        cls,
        block: Mapping[str, Any] | None,
        *,
        prep: Mapping[str, Any] | None,
    ) -> "AugmentationRequest | None":
        """Parse adapt.noise_augmentation; None when absent or disabled."""
        config = _require_mapping(block, field="adapt.noise_augmentation")
        if not config:
            return None
        _reject_unknown_keys(config, _TOP_LEVEL_KEYS, field="adapt.noise_augmentation")
        if not bool(config.get("enabled", False)):
            return None

        generation = str(config.get("generation", "g1") or "g1").strip()
        if not _GENERATION_RE.match(generation):
            raise ValueError(
                "adapt.noise_augmentation.generation must start with a letter or digit and "
                "contain only letters, digits, '_', '.', '-'"
            )
        try:
            seed = int(config.get("seed", 2025))
            variants_per_sample = int(config.get("variants_per_sample", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("adapt.noise_augmentation.seed/variants_per_sample must be integers") from exc
        if variants_per_sample < 1:
            raise ValueError("adapt.noise_augmentation.variants_per_sample must be >= 1")

        prep_map = _require_mapping(prep, field="prep")
        raw_root = str(config.get("musan_root") or prep_map.get("musan_root") or "").strip()
        if not raw_root:
            raise ValueError(
                "adapt.noise_augmentation requires adapt.noise_augmentation.musan_root or prep.musan_root"
            )
        musan_root = Path(raw_root).expanduser().resolve()
        if not musan_root.is_dir():
            raise ValueError(f"MUSAN root is not a directory: {musan_root}")

        raw_split_dir = str(config.get("musan_split_dir") or "").strip()
        if not raw_split_dir:
            raise ValueError(
                "adapt.noise_augmentation.musan_split_dir is required; it must hold "
                f"{_TRAIN_LIST_NAME}, {_EVAL_LIST_NAME} and (optionally) {_SPLIT_JSON_NAME}"
            )
        split_dir = Path(raw_split_dir).expanduser().resolve()
        if not split_dir.is_dir():
            raise ValueError(f"MUSAN split directory is not a directory: {split_dir}")

        train_list = Path(str(config.get("train_list") or split_dir / _TRAIN_LIST_NAME)).expanduser().resolve()
        eval_list = Path(str(config.get("eval_list") or split_dir / _EVAL_LIST_NAME)).expanduser().resolve()
        if train_list == eval_list or train_list.name == _EVAL_LIST_NAME:
            raise ValueError(
                "adapt.noise_augmentation.train_list must be the train allowlist, not the eval allowlist"
            )

        train_paths = _load_path_list(train_list)
        eval_paths = _load_path_list(eval_list)
        eval_set = set(eval_paths)
        overlap = sorted(str(path) for path in train_paths if path in eval_set)
        if overlap:
            raise ValueError(
                "MUSAN train allowlist overlaps the eval allowlist: " + ", ".join(overlap[:3])
            )
        other = sorted({path for path in train_paths if detect_subset(path, musan_root) not in MIX_CATEGORIES})
        if other:
            raise ValueError(
                "MUSAN train allowlist entries must live under musan_root/<category>: "
                + ", ".join(str(path) for path in other[:3])
            )

        categories_cfg = _require_mapping(config.get("categories"), field="adapt.noise_augmentation.categories")
        unknown_categories = sorted(set(categories_cfg) - set(MIX_CATEGORIES))
        if unknown_categories:
            raise ValueError(
                "Unknown adapt.noise_augmentation.categories entries: " + ", ".join(unknown_categories)
            )

        split_summary = _read_split_summary(split_dir)
        categories: list[AdaptMixCategory] = []
        for kind in MIX_CATEGORIES:
            category_cfg = _require_mapping(
                categories_cfg.get(kind), field=f"adapt.noise_augmentation.categories.{kind}"
            )
            if not category_cfg:
                continue
            _reject_unknown_keys(
                category_cfg, _CATEGORY_KEYS, field=f"adapt.noise_augmentation.categories.{kind}"
            )
            if not bool(category_cfg.get("enabled", False)):
                continue
            level_fields = _LEVEL_FIELDS[kind]
            wrong_metric = sorted(
                name
                for name in ("snr_db_min", "snr_db_max", "relative_db_min", "relative_db_max")
                if name in category_cfg and name not in level_fields
            )
            if wrong_metric:
                raise ValueError(
                    f"adapt.noise_augmentation.categories.{kind} uses {wrong_metric[0]}; "
                    f"it takes {level_fields[0]}/{level_fields[1]}"
                )
            missing = [name for name in level_fields if name not in category_cfg]
            if missing:
                raise ValueError(
                    f"adapt.noise_augmentation.categories.{kind} requires {', '.join(missing)}"
                )
            level_min = _finite_float(
                category_cfg[level_fields[0]], field=f"adapt.noise_augmentation.categories.{kind}.{level_fields[0]}"
            )
            level_max = _finite_float(
                category_cfg[level_fields[1]], field=f"adapt.noise_augmentation.categories.{kind}.{level_fields[1]}"
            )
            if level_min > level_max:
                raise ValueError(
                    f"adapt.noise_augmentation.categories.{kind}.{level_fields[0]} must be <= {level_fields[1]}"
                )
            pool = tuple(path for path in train_paths if detect_subset(path, musan_root) == kind)
            if not pool:
                raise ValueError(
                    f"MUSAN train allowlist has no {kind} recordings under {musan_root}; rebuild the "
                    f"leakage-safe split with scripts/split_musan.py --train-categories including {kind}"
                )
            pool_sha256 = musan_catalog_sha256(pool, musan_root)
            _assert_train_partition(split_summary, kind=kind, pool_sha256=pool_sha256, split_dir=split_dir)
            categories.append(
                AdaptMixCategory(
                    kind=kind,
                    level_name="relative_db" if kind == "speech" else "snr_db",
                    level_min=level_min,
                    level_max=level_max,
                    pool=pool,
                    pool_sha256=pool_sha256,
                )
            )
        if not categories:
            raise ValueError("adapt.noise_augmentation.enabled is true but no category is enabled")

        raw_phases = config.get("phases") or ()
        if isinstance(raw_phases, str):
            phases = tuple(part.strip().casefold() for part in raw_phases.split(",") if part.strip())
        elif isinstance(raw_phases, (list, tuple)):
            phases = tuple(str(part).strip().casefold() for part in raw_phases if str(part).strip())
        else:
            raise ValueError("adapt.noise_augmentation.phases must be a list or comma-separated string")

        return cls(
            generation=generation,
            seed=seed,
            musan_root=musan_root,
            musan_split_dir=split_dir,
            train_list=train_list,
            eval_list=eval_list,
            variants_per_sample=variants_per_sample,
            keep_original=bool(config.get("keep_original", True)),
            dry_run=bool(config.get("dry_run", False)),
            phases=phases,
            categories=tuple(categories),
        )


def _read_split_summary(split_dir: Path) -> Mapping[str, Any] | None:
    path = split_dir / _SPLIT_JSON_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid MUSAN split summary: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid MUSAN split summary: {path}")
    return payload


def _assert_train_partition(
    summary: Mapping[str, Any] | None,
    *,
    kind: str,
    pool_sha256: str,
    split_dir: Path,
) -> None:
    if summary is None:
        return
    recorded = (
        summary.get("splits", {}).get("train", {}).get("by_category", {}).get(kind, {})
        if isinstance(summary.get("splits"), Mapping)
        else {}
    )
    expected = recorded.get("catalog_sha256") if isinstance(recorded, Mapping) else None
    if not expected:
        raise ValueError(
            f"{split_dir / _SPLIT_JSON_NAME} does not record a train catalog digest for {kind}; "
            "rebuild the split with scripts/split_musan.py"
        )
    if str(expected) != pool_sha256:
        raise ValueError(
            f"MUSAN {kind} train pool does not match the recorded split partition "
            f"({split_dir / _SPLIT_JSON_NAME}); use the lists produced by scripts/split_musan.py"
        )


def describe_offline_musan_request(request: AugmentationRequest | None) -> str:
    if request is None:
        return ""
    parts = [category.describe() for category in request.categories]
    files = sum(len(category.pool) for category in request.categories)
    return (
        f"{', '.join(parts)} x{request.variants_per_sample} per category "
        f"({files} files, gen {request.generation})"
    )


@dataclass(frozen=True)
class MixVariant:
    """One planned augmented positive row."""

    phase: str
    row: Any
    source_wav: Path
    fbank_path: Path
    variant_audio_path: str
    category: str
    index: int
    recipe_id: str
    recipe_seed: int
    level_name: str
    level_db: float
    applied_snr_db: float
    musan_path: Path


@dataclass(frozen=True)
class MixPlan:
    request: AugmentationRequest
    keyword: str
    data_root: Path
    config_fingerprint: str
    phase_splits: dict[str, tuple[list[Any], list[Any]]]
    variants: tuple[MixVariant, ...]

    @property
    def generation_dir(self) -> Path:
        return Path(self.data_root) / _AUGMENTATION_DIR / self.request.generation

    @property
    def recipes_path(self) -> Path:
        return self.generation_dir / "recipes.json"

    @property
    def features_path(self) -> Path:
        return self.generation_dir / "features.jsonl"


@dataclass(frozen=True)
class MixStats:
    status: str
    generation: str
    config_fingerprint: str
    planned: int
    written: int
    skipped: int
    frames: int
    total_bytes: int
    by_category: Mapping[str, int]
    recipes_path: Path
    features_path: Path

    def to_stats_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "generation": self.generation,
            "config_fingerprint": self.config_fingerprint,
            "planned": self.planned,
            "written": self.written,
            "skipped": self.skipped,
            "frames": self.frames,
            "bytes": self.total_bytes,
            "by_category": dict(self.by_category),
            "recipes": str(self.recipes_path),
            "features": str(self.features_path),
        }


def resolve_adapt_noise_augmentation(
    adapt: Mapping[str, Any], prep: Mapping[str, Any]
) -> AugmentationRequest | None:
    """Build the validated request from the composed adapt/prep config."""
    return AugmentationRequest.from_config(adapt.get("noise_augmentation"), prep=prep)


def _config_fingerprint(request: AugmentationRequest, fbank_params: Mapping[str, Any]) -> str:
    payload = {
        "schema": _RECIPE_SCHEMA_VERSION,
        "generation": request.generation,
        "seed": request.seed,
        "variants_per_sample": request.variants_per_sample,
        "keep_original": request.keep_original,
        "musan_root": str(request.musan_root),
        "train_list_sha256": _sha256_file(request.train_list),
        "eval_list_sha256": _sha256_file(request.eval_list),
        "categories": [
            {
                "kind": category.kind,
                "level_name": category.level_name,
                "level_min": category.level_min,
                "level_max": category.level_max,
                "pool_sha256": category.pool_sha256,
            }
            for category in request.categories
        ],
        "fbank_params": _canonical_json(dict(fbank_params)),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def plan_offline_musan_variants(
    phase_splits: Mapping[str, tuple[list[Any], list[Any]]],
    *,
    keyword: str,
    data_root: str | Path,
    request: AugmentationRequest,
    fbank_params: Mapping[str, Any],
) -> MixPlan:
    """Derive augmented train-positive rows; pure, deterministic, no audio I/O."""
    from dma_kws.stage2.adapt_dataset import _resolve_adapt_fbank_path

    root = Path(data_root)
    fbank_root = root / "fbank"
    requested = request.phases or tuple(phase_splits)
    selected = [phase for phase in requested if phase in phase_splits]
    if not selected:
        raise ValueError(
            "adapt.noise_augmentation.phases does not match any prepared phase: "
            f"{sorted(requested)} vs {sorted(phase_splits)}"
        )

    categories = request.categories
    variant_paths: set[str] = set()
    fbank_paths: set[Path] = set()
    variants: list[MixVariant] = []
    new_splits: dict[str, tuple[list[Any], list[Any]]] = {}

    for phase, (train_rows, eval_rows) in phase_splits.items():
        if phase not in selected:
            new_splits[phase] = (train_rows, eval_rows)
            continue
        kept: list[Any] = []
        added: list[Any] = []
        for row in train_rows:
            if int(row.label) != 1:
                kept.append(row)
                continue
            if request.keep_original:
                kept.append(row)
            source_wav = Path(str(row.audio_path))
            if not source_wav.is_absolute():
                source_wav = root / source_wav
            key8 = hashlib.sha256(str(row.audio_path).encode("utf-8")).hexdigest()[:8]
            stem = _safe_stem(Path(str(row.audio_path)).stem)
            # Every enabled category gets its own copies. Cycling categories would
            # silently drop the later ones whenever variants_per_sample is smaller
            # than the number of enabled categories.
            ordinal = 0
            for category in categories:
                for _copy in range(request.variants_per_sample):
                    index = ordinal
                    ordinal += 1
                    recipe_seed = _recipe_seed(
                        request.seed, request.generation, keyword, phase, str(row.audio_path), index, category.kind
                    )
                    rng = random.Random(recipe_seed)
                    musan_path = Path(rng.choice(category.pool))
                    level_db = float(rng.uniform(category.level_min, category.level_max))
                    recipe_id = _recipe_id(phase, str(row.audio_path), index, category.kind, request.generation)
                    name = f"{stem}__{key8}__aug{index:03d}_{category.kind}.wav"
                    if len(name.encode("utf-8")) > 255:
                        raise ValueError(f"Augmented variant filename is too long: {name}")
                    variant_audio_path = (
                        f"raw/{phase}/{_AUG_PATH_COMPONENT}/{request.generation}/{name}"
                    )
                    fbank_path = _resolve_adapt_fbank_path(
                        fbank_root, variant_audio_path, manifest_root=root
                    )
                    if variant_audio_path in variant_paths:
                        raise ValueError(f"Duplicate augmented audio path: {variant_audio_path}")
                    if fbank_path in fbank_paths:
                        raise ValueError(f"Two variants resolve to the same fbank: {fbank_path}")
                    if (root / variant_audio_path).exists():
                        raise ValueError(
                            f"Synthetic augmented audio path already exists on disk: {root / variant_audio_path}"
                        )
                    variant_paths.add(variant_audio_path)
                    fbank_paths.add(fbank_path)
                    metadata = dict(row.metadata)
                    metadata.update(
                        {
                            "aug_recipe_id": recipe_id,
                            "aug_generation": request.generation,
                            "aug_category": category.kind,
                            "aug_metric": category.level_name,
                            "aug_level_db": f"{level_db:.6f}",
                            "aug_musan_path": str(musan_path),
                            "original_audio_path": str(row.audio_path),
                        }
                    )
                    variants.append(
                        MixVariant(
                            phase=phase,
                            row=replace(row, audio_path=variant_audio_path, metadata=metadata),
                            source_wav=source_wav,
                            fbank_path=fbank_path,
                            variant_audio_path=variant_audio_path,
                            category=category.kind,
                            index=index,
                            recipe_id=recipe_id,
                            recipe_seed=recipe_seed,
                            level_name=category.level_name,
                            level_db=level_db,
                            applied_snr_db=category.snr_db_for(level_db),
                            musan_path=musan_path,
                        )
                    )
                    added.append(variants[-1].row)
        new_splits[phase] = (kept + added, eval_rows)

    return MixPlan(
        request=request,
        keyword=keyword,
        data_root=root,
        config_fingerprint=_config_fingerprint(request, fbank_params),
        phase_splits=new_splits,
        variants=tuple(variants),
    )


def _recipes_payload(plan: MixPlan) -> dict[str, Any]:
    recipes = []
    for variant in plan.variants:
        try:
            fbank_relpath = variant.fbank_path.relative_to(plan.data_root).as_posix()
        except ValueError:
            fbank_relpath = str(variant.fbank_path)
        recipes.append(
            {
                "recipe_id": variant.recipe_id,
                "phase": variant.phase,
                "source_audio_path": str(variant.row.metadata.get("original_audio_path", "")),
                "variant_audio_path": variant.variant_audio_path,
                "variant_fbank_relpath": fbank_relpath,
                "category": variant.category,
                "metric": variant.level_name,
                "level_db": variant.level_db,
                "applied_snr_db": variant.applied_snr_db,
                "musan_path": str(variant.musan_path),
                "recipe_seed": variant.recipe_seed,
            }
        )
    recipes.sort(key=lambda item: item["recipe_id"])
    return {
        "schema": _RECIPE_SCHEMA_VERSION,
        "generation": plan.request.generation,
        "fingerprint": plan.config_fingerprint,
        "keyword": plan.keyword,
        "recipes": recipes,
    }


def _read_existing_recipes(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Existing augmentation recipes file is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Existing augmentation recipes file has an unexpected shape: {path}")
    return payload


def _read_features(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Existing augmentation features ledger is not valid JSONL: {path}") from exc
        if isinstance(entry, Mapping) and entry.get("recipe_id"):
            entries[str(entry["recipe_id"])] = dict(entry)
    return entries


def _build_augmenters(plan: MixPlan, fbank_params: Mapping[str, Any]) -> dict[str, Any]:
    generation_dir = plan.generation_dir
    generation_dir.mkdir(parents=True, exist_ok=True)
    augmenters: dict[str, Any] = {}
    for category in plan.request.categories:
        list_path = generation_dir / f"{category.kind}.list"
        _atomic_write_text(list_path, "".join(f"{path}\n" for path in category.pool))
        augmenters[category.kind] = _features.TrainingNoiseAugmenter(
            waveform_dir=plan.data_root,
            noise_list_path=list_path,
            snr_db_min=category.snr_db_min,
            snr_db_max=category.snr_db_max,
            fbank_kwargs=fbank_params,
        )
    return augmenters


def preview_offline_musan_variants(plan: MixPlan) -> MixStats:
    """Dry-run: publish the deterministic plan without touching features."""
    generation_dir = plan.generation_dir
    generation_dir.mkdir(parents=True, exist_ok=True)
    by_category: dict[str, int] = {}
    for variant in plan.variants:
        by_category[variant.category] = by_category.get(variant.category, 0) + 1
    payload = _recipes_payload(plan)
    payload["status"] = "dry_run"
    payload["by_category"] = by_category
    _atomic_write_text(
        generation_dir / "plan_preview.json", _canonical_json(payload) + "\n"
    )
    return MixStats(
        status="dry_run",
        generation=plan.request.generation,
        config_fingerprint=plan.config_fingerprint,
        planned=len(plan.variants),
        written=0,
        skipped=0,
        frames=0,
        total_bytes=0,
        by_category=by_category,
        recipes_path=plan.recipes_path,
        features_path=plan.features_path,
    )


def materialize_offline_musan_variants(
    plan: MixPlan,
    *,
    fbank_params: Mapping[str, Any],
    skip_existing: bool = True,
    on_progress: Callable[[str, int], None] | None = None,
) -> MixStats:
    """Render every planned variant to fbank and publish the provenance ledger."""
    existing = _read_existing_recipes(plan.recipes_path)
    if existing is not None and existing.get("fingerprint") not in (None, plan.config_fingerprint):
        raise ValueError(
            f"adapt.noise_augmentation generation {plan.request.generation!r} was produced with a "
            "different configuration fingerprint; bump adapt.noise_augmentation.generation "
            "(or revert the config) before preparing again"
        )
    _atomic_write_text(plan.recipes_path, _canonical_json(_recipes_payload(plan)) + "\n")

    augmenters = _build_augmenters(plan, fbank_params)
    extractor = FbankExtractor(**dict(fbank_params))
    ledger = _read_features(plan.features_path)
    previous = dict(ledger)
    ledger = {}

    def report(stage: str, value: int) -> None:
        if on_progress is not None:
            on_progress(stage, value)

    report("augment_total", len(plan.variants))
    written = 0
    skipped = 0
    frames = 0
    total_bytes = 0
    by_category: dict[str, int] = {}
    for variant in plan.variants:
        category_count = by_category.get(variant.category, 0)
        by_category[variant.category] = category_count + 1
        entry = previous.get(variant.recipe_id)
        if skip_existing and variant.fbank_path.is_file():
            if entry is None or int(entry.get("bytes", -1)) != variant.fbank_path.stat().st_size:
                entry = _ledger_entry(variant, plan)
            ledger[variant.recipe_id] = entry
            skipped += 1
            report("augment_skipped", 1)
            report("augment", 1)
            continue

        raw, sample_rate = _features._load_audio(variant.source_wav)
        waveform = _to_mono(raw, source=variant.source_wav)
        waveform, sample_rate = extractor.prepare_waveform(waveform, int(sample_rate))
        mixed, applied_snr_db, _drawn = augmenters[variant.category].mix_with_source(
            waveform,
            int(sample_rate),
            rng=random.Random(variant.recipe_seed),
            noise_path=variant.musan_path,
            snr_db=variant.applied_snr_db,
        )
        feature = extractor.extract(mixed, int(sample_rate)).cpu().numpy()
        _atomic_save_npy(variant.fbank_path, feature)
        sha256 = _sha256_file(variant.fbank_path)
        ledger[variant.recipe_id] = {
            "recipe_id": variant.recipe_id,
            "variant_fbank_relpath": _fbank_relpath(variant, plan),
            "sha256": sha256,
            "bytes": int(feature.nbytes),
            "frames": int(feature.shape[0]),
            "dim": int(feature.shape[1]),
            "applied_snr_db": float(applied_snr_db),
            "musan_path": str(variant.musan_path),
        }
        written += 1
        frames += int(feature.shape[0])
        total_bytes += int(feature.nbytes)
        report("augment", 1)

    lines = [
        _canonical_json(ledger[recipe_id])
        for recipe_id in sorted(ledger)
    ]
    _atomic_write_text(plan.features_path, "".join(f"{line}\n" for line in lines))

    return MixStats(
        status="complete",
        generation=plan.request.generation,
        config_fingerprint=plan.config_fingerprint,
        planned=len(plan.variants),
        written=written,
        skipped=skipped,
        frames=frames,
        total_bytes=total_bytes,
        by_category=by_category,
        recipes_path=plan.recipes_path,
        features_path=plan.features_path,
    )


def _fbank_relpath(variant: MixVariant, plan: MixPlan) -> str:
    try:
        return variant.fbank_path.relative_to(plan.data_root).as_posix()
    except ValueError:
        return str(variant.fbank_path)


def _ledger_entry(variant: MixVariant, plan: MixPlan) -> dict[str, Any]:
    payload = np.load(variant.fbank_path, allow_pickle=False)
    try:
        return {
            "recipe_id": variant.recipe_id,
            "variant_fbank_relpath": _fbank_relpath(variant, plan),
            "sha256": _sha256_file(variant.fbank_path),
            "bytes": int(payload.nbytes),
            "frames": int(payload.shape[0]),
            "dim": int(payload.shape[1]),
            "applied_snr_db": float(variant.applied_snr_db),
            "musan_path": str(variant.musan_path),
        }
    finally:
        del payload
