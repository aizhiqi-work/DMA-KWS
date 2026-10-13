"""Console helpers for Stage II keyword adaptation.

Row builders stay free of rich so they can be unit tested; rendering goes through
:class:`~dma_kws.stage2.prep_console.Stage2PrepReporter`.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from dma_kws.stage2.prep_console import Stage2PrepReporter
from dma_kws.training.adapt_params import resolve_adapt_lr

METRIC_KEYS = ("auc", "eer", "accuracy", "f1", "recall", "fpr")


def adapt_reporter(config: dict[str, Any], prep: dict[str, Any] | None = None) -> Stage2PrepReporter:
    """Build a reporter honouring ``prep.use_rich`` then ``stage2.console.rich``."""
    prep_cfg = prep if prep is not None else (config.get("prep") or {})
    if "use_rich" in prep_cfg:
        use_rich = bool(prep_cfg["use_rich"])
    else:
        console_cfg = (config.get("stage2") or {}).get("console") or {}
        use_rich = bool(console_cfg.get("rich", True))
    return Stage2PrepReporter(use_rich=use_rich)


def _count(value: Any) -> str:
    return f"{int(value):,}"


def _metric(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _file_size(path: Path) -> str:
    if not Path(path).is_file():
        return "missing"
    size = Path(path).stat().st_size
    return f"{size / (1024 * 1024):.1f} MiB" if size >= 1024 * 1024 else f"{size / 1024:.1f} KiB"


def prep_plan_rows(
    *,
    keyword: str,
    slug: str,
    data_root: Path,
    source: str,
    eval_fraction: float,
    eval_seed: int,
    skip_existing: bool,
    augmentation: str = "",
) -> list[tuple[str, str]]:
    """Settings table for keyword adaptation data preparation."""
    rows = [
        ("keyword", keyword),
        ("slug", slug),
        ("data_root", str(data_root)),
        ("source", source),
        ("eval_fraction", str(eval_fraction)),
        ("eval_seed", str(eval_seed)),
        ("skip_existing_fbank", str(skip_existing)),
    ]
    if augmentation:
        rows.append(("musan_mix", augmentation))
    return rows


def prep_phase_table(stats: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    """Per-phase train/eval composition table from ``prepare_keyword_adaptation`` stats."""
    columns = ["phase", "train (pos/neg)", "eval (pos/neg)", "total", "train manifest"]
    rows: list[list[str]] = []
    for phase, phase_stats in sorted((stats.get("phases") or {}).items()):
        rows.append(
            [
                phase,
                f"{_count(phase_stats['train'])} "
                f"({phase_stats.get('train_positive', 0)}/{phase_stats.get('train_negative', 0)})",
                f"{_count(phase_stats['eval'])} "
                f"({phase_stats.get('eval_positive', 0)}/{phase_stats.get('eval_negative', 0)})",
                _count(phase_stats["total"]),
                str(phase_stats.get("train_manifest", "")),
            ]
        )
    return columns, rows


def prep_stats_rows(stats: dict[str, Any]) -> list[tuple[str, str]]:
    """Aggregate counters from ``prepare_keyword_adaptation`` stats."""
    phases = stats.get("phases") or {}
    total = sum(int(phase["total"]) for phase in phases.values())
    rows = [
        ("samples", _count(total)),
        ("unique_texts", _count(stats.get("unique_texts", 0))),
        ("fbank_written", _count(stats.get("fbank_written", 0))),
        ("fbank_skipped", _count(stats.get("fbank_skipped", 0))),
    ]
    augmentation = stats.get("augmentation") or {}
    if augmentation:
        rows.extend(
            [
                ("aug_status", str(augmentation.get("status", ""))),
                ("aug_generation", str(augmentation.get("generation", ""))),
                ("aug_planned", _count(augmentation.get("planned", 0))),
                ("aug_written", _count(augmentation.get("written", 0))),
                ("aug_skipped", _count(augmentation.get("skipped", 0))),
                ("aug_frames", _count(augmentation.get("frames", 0))),
                (
                    "aug_storage",
                    f"{int(augmentation.get('bytes', 0)) / (1024 * 1024):.1f} MiB",
                ),
            ]
        )
        by_category = augmentation.get("by_category") or {}
        for kind in ("noise", "music", "speech"):
            if kind in by_category:
                rows.append((f"aug_{kind}", _count(by_category[kind])))
    return rows


@contextmanager
def prep_progress(reporter: Stage2PrepReporter) -> Iterator[Callable[[str, int], None]]:
    """Yield an ``on_progress`` callback driving G2P and fbank progress bars."""
    with reporter.tasks() as tasks:
        g2p_task = tasks.add("Validate G2P (unique texts)", total=None)
        fbank_task = tasks.add("Compute fbank", total=None)
        augment_task = tasks.add("Mix MUSAN variants", total=None)

        def on_progress(stage: str, value: int) -> None:
            if stage == "scan":
                reporter.info(f"Scanned {_count(value)} wav samples")
            elif stage == "g2p_total":
                tasks.set_total(g2p_task, value)
            elif stage == "g2p":
                tasks.advance(g2p_task, value)
            elif stage == "fbank_total":
                tasks.set_total(fbank_task, value)
            elif stage == "fbank":
                tasks.advance(fbank_task, value)
            elif stage == "augment_total":
                tasks.set_total(augment_task, value)
            elif stage == "augment":
                tasks.advance(augment_task, value)

        yield on_progress


def describe_prep_source(manifest_csv: Path | None, sources: dict[str, Any] | None, data_root: Path) -> str:
    """Human-readable description of where adaptation wavs are read from."""
    if manifest_csv is not None and Path(manifest_csv).is_file():
        return f"manifest csv: {manifest_csv}"
    if isinstance(sources, dict) and any(
        isinstance(source, dict)
        and any(str(source.get(key, "")).strip() for key in ("positive_dir", "negative_root"))
        for source in sources.values()
    ):
        phases = ", ".join(sorted(phase for phase, source in sources.items() if source))
        return f"external directories ({phases})"
    return f"raw tree: {data_root / 'raw'}"


def prepare_with_console(
    config: dict[str, Any],
    adapt: dict[str, Any],
    prep: dict[str, Any],
) -> dict[str, Any]:
    """Run keyword adaptation data prep with plan, progress bars and summary tables."""
    from dma_kws.config import fbank_kwargs, get_fbank_config
    from dma_kws.stage2.adapt_paths import adapt_data_root, slugify
    from dma_kws.stage2.prepare_adapt import prepare_keyword_adaptation
    from dma_kws.stage2.prepare_adapt_noise import (
        describe_offline_musan_request,
        resolve_adapt_noise_augmentation,
    )

    keyword = str(adapt.get("keyword", "")).strip()
    if not keyword:
        raise SystemExit("adapt.keyword is required")

    data_root = Path(adapt["data_root"]) if adapt.get("data_root") else adapt_data_root(config, keyword)
    manifest_csv = prep.get("manifest_csv") or adapt.get("manifest_csv")
    manifest_path = Path(manifest_csv) if manifest_csv else None
    sources = adapt.get("sources") if isinstance(adapt.get("sources"), dict) else None
    eval_fraction = float(adapt.get("eval_fraction", 0.2))
    eval_seed = int(adapt.get("eval_seed", (config.get("training") or {}).get("seed", 2025)))
    skip_existing = not bool(prep.get("no_skip_existing", False))

    fbank_params = fbank_kwargs(get_fbank_config(config))
    noise_augmentation = resolve_adapt_noise_augmentation(adapt, prep)

    reporter = adapt_reporter(config, prep)
    reporter.section(f"Prepare keyword adaptation data · {keyword}")
    reporter.print_plan(
        prep_plan_rows(
            keyword=keyword,
            slug=slugify(keyword),
            data_root=data_root,
            source=describe_prep_source(manifest_path, sources, data_root),
            eval_fraction=eval_fraction,
            eval_seed=eval_seed,
            skip_existing=skip_existing,
            augmentation=describe_offline_musan_request(noise_augmentation),
        ),
        title="Preparation Plan",
    )

    with prep_progress(reporter) as on_progress:
        stats = prepare_keyword_adaptation(
            keyword=keyword,
            data_root=data_root,
            fbank_params=fbank_params,
            eval_fraction=eval_fraction,
            eval_seed=eval_seed,
            manifest_csv=manifest_path,
            sources=sources,
            skip_existing=skip_existing,
            on_progress=on_progress,
            joint=(
                str(adapt.get("phase", "")).strip().casefold() == "joint"
                or adapt.get("train_phases") == ["joint"]
                or adapt.get("train_phases") == ("joint",)
            ),
            noise_augmentation=noise_augmentation,
        )

    reporter.print_table(*prep_phase_table(stats), title="Phase Splits")
    reporter.print_stats(prep_stats_rows(stats), title="Preparation Summary")
    reporter.done(f"Adaptation data ready under {data_root}")
    return stats


def adapt_plan_rows(
    *,
    adapt_paths: dict[str, Any],
    accelerator: str,
    devices: int,
    init_checkpoint: str,
    adapter_resume: str | None,
    resume_path: Path | None,
    params_file: str = "",
    method: str = "lora",
) -> list[tuple[str, str]]:
    """Identity and path table printed before adaptation training starts."""
    rows = [
        ("keyword", adapt_paths["keyword_str"]),
        ("slug", adapt_paths["slug_str"]),
        ("phase", adapt_paths["phase_str"]),
        ("accelerator", f"{accelerator} x{devices}"),
        ("data_root", str(adapt_paths["data_root"])),
        ("exp_root", str(adapt_paths["exp_root"])),
        ("train_manifest", str(adapt_paths["train_manifest"])),
        ("eval_manifest", str(adapt_paths["eval_manifest"])),
        (
            "init_checkpoint",
            init_checkpoint if init_checkpoint else "restored from full checkpoint",
        ),
    ]
    if method == "lora":
        rows.append(("adapter_resume", str(adapter_resume) if adapter_resume else "none (fresh LoRA)"))
    else:
        rows.append(("adapt_method", method))
    if resume_path is not None:
        rows.append(("resume_ckpt", str(resume_path)))
    if params_file:
        rows.append(("params_file", params_file))
    return rows


def dataset_table(
    entries: list[tuple[str, int, str]],
) -> tuple[list[str], list[list[str]]]:
    """Dataset composition table: ``(name, samples, detail)`` entries."""
    columns = ["dataset", "samples", "detail"]
    return columns, [[name, _count(samples), detail] for name, samples, detail in entries]


def label_breakdown(dataset: Any) -> str:
    """``pos=/neg=`` summary for a manifest-backed dataset, when labels are known."""
    frame = getattr(dataset, "df", None)
    if frame is None or "label" not in getattr(frame, "columns", []):
        return ""
    labels = frame["label"].astype(int)
    return f"pos={int((labels == 1).sum())} neg={int((labels == 0).sum())}"


def lora_rows(
    *,
    rank: int,
    alpha: float,
    targets: tuple[str, ...] | list[str],
    injected: list[str] | None,
    param_counts: dict[str, int],
) -> list[tuple[str, str]]:
    """LoRA configuration and parameter-budget table."""
    total = int(param_counts.get("total", 0))
    trainable = int(param_counts.get("trainable", 0))
    share = f" ({100.0 * trainable / total:.3f}% of model)" if total else ""
    return [
        ("rank", str(rank)),
        ("alpha", str(alpha)),
        ("scaling (alpha/rank)", f"{float(alpha) / rank:.3f}" if rank else "n/a"),
        ("targets", ", ".join(str(target) for target in targets)),
        ("injected_matrices", str(len(injected)) if injected is not None else "unknown"),
        ("trainable_params", f"{trainable:,}{share}"),
        ("lora_params", f"{int(param_counts.get('lora_trainable', 0)):,}"),
        ("total_params", f"{total:,}"),
    ]


def artifact_rows(paths: dict[str, Path]) -> tuple[list[str], list[list[str]]]:
    """Saved-artifact table with on-disk sizes."""
    columns = ["artifact", "size", "path"]
    return columns, [[name, _file_size(path), str(path)] for name, path in paths.items()]


def sweep_baseline_rows(
    *,
    keyword: str,
    slug: str,
    base_checkpoint: str,
    lph_auc_base: float,
    n_trials: int,
    lambda_forget: float,
    lph_subset: int,
    search_mix: bool,
    single_phase: bool,
    storage: str,
    study_name: str,
) -> list[tuple[str, str]]:
    """Sweep configuration and baseline LibriPhrase AUC table."""
    return [
        ("keyword", keyword),
        ("slug", slug),
        ("base_checkpoint", base_checkpoint),
        ("lph_auc_base", _metric(lph_auc_base)),
        ("completed_trial_target", str(n_trials)),
        ("failure_policy", "stop on first error; rerun resumes completed trials"),
        ("lambda_forget", str(lambda_forget)),
        ("lph_subset", str(lph_subset) if lph_subset else "all"),
        ("search_mix", str(search_mix)),
        ("single_phase", str(single_phase)),
        ("study_name", study_name),
        ("storage", storage),
    ]


def trial_param_rows(params: dict[str, Any], *, adapt: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    """Effective hyperparameters for one sweep trial."""
    rows = [(key, str(value)) for key, value in sorted(params.items()) if not key.startswith("_")]
    if adapt is not None:
        lr, source = resolve_adapt_lr(adapt)
        rows.append(("effective_lr", f"{lr} (from {source})"))
    return rows


def sweep_results_table(trials: list[dict[str, Any]]) -> tuple[list[str], list[list[str]]]:
    """Score table across completed sweep trials, best first."""
    columns = ["trial", "score", "target_auc", "lph_auc", "forget", "params"]
    ordered = sorted(trials, key=lambda trial: trial.get("score", 0.0), reverse=True)
    rows = []
    for trial in ordered:
        params = trial.get("params") or {}
        rows.append(
            [
                str(trial.get("number", "?")),
                _metric(trial.get("score")),
                _metric(trial.get("target_auc")),
                _metric(trial.get("lph_auc")),
                _metric(trial.get("forget_penalty")),
                " ".join(f"{key}={value}" for key, value in sorted(params.items())),
            ]
        )
    return columns, rows


def eval_comparison_table(report: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    """Base vs adapted metric comparison with deltas, for the eval report."""
    columns = ["metric", "base", "adapted", "delta"]
    rows: list[list[str]] = []
    groups = (
        ("target", report.get("target_base"), report.get("target_adapted"), METRIC_KEYS),
        ("tts", report.get("tts_base"), report.get("tts_adapted"), METRIC_KEYS),
        ("lph", report.get("lph_base"), report.get("lph_adapted"), METRIC_KEYS),
        ("musan", (report.get("musan_base") or {}).get("metrics"),
         (report.get("musan_adapted") or {}).get("metrics"), ("fa_per_hour",)),
    )
    for prefix, base, adapted, metric_keys in groups:
        if not isinstance(base, dict) or not isinstance(adapted, dict):
            continue
        for key in metric_keys:
            if key not in base and key not in adapted:
                continue
            base_value = base.get(key)
            adapted_value = adapted.get(key)
            if base_value is None or adapted_value is None:
                delta = "-"
            else:
                delta = f"{float(adapted_value) - float(base_value):+.4f}"
            rows.append([f"{prefix}/{key}", _metric(base_value), _metric(adapted_value), delta])
    return columns, rows
