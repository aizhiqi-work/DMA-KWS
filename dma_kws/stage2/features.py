"""Stage II online fbank extraction with optional speed/noise augmentation."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from dma_kws.stage2.background_sampling import (
    BackgroundSourceInfo,
    draw_crop_spec,
    materialize_crop,
    probe_source_info,
)
from dma_kws.stage2.fbank import FbankExtractor

DEFAULT_NUM_MEL_BINS = 80
DEFAULT_DITHER = 0.1
DEFAULT_FRAME_LENGTH = 25
DEFAULT_FRAME_SHIFT = 10


class TrainingNoiseAugmenter:
    """Mix a random noise recording into a Stage-II training waveform.

    The augmenter deliberately owns no RNG state.  The caller supplies the
    dataset's ``random.Random`` instance so DataLoader worker/DDP reseeding also
    controls the augmentation gate, source choice, crop and SNR draw.
    """

    def __init__(
        self,
        *,
        waveform_dir: str | Path,
        noise_list_path: str | Path,
        snr_db_min: float,
        snr_db_max: float,
        fbank_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.snr_db_min = float(snr_db_min)
        self.snr_db_max = float(snr_db_max)
        if not math.isfinite(self.snr_db_min) or not math.isfinite(self.snr_db_max):
            raise ValueError("Stage II noise SNR bounds must be finite")
        if self.snr_db_min > self.snr_db_max:
            raise ValueError("Stage II noise snr_db_min must be <= snr_db_max")

        self.waveform_dir = Path(waveform_dir)
        if not self.waveform_dir.is_dir():
            raise FileNotFoundError(
                f"Stage II waveform directory not found: {self.waveform_dir}"
            )

        self.noise_list_path = Path(noise_list_path)
        self.noise_paths = self._load_noise_files(self.noise_list_path)
        self._fbank_kwargs = dict(fbank_kwargs or {})
        self._fbank_extractor: FbankExtractor | None = None

    @staticmethod
    def _load_noise_files(noise_list_path: Path) -> tuple[Path, ...]:
        if not noise_list_path.is_file():
            raise FileNotFoundError(
                f"Stage II noise list file not found: {noise_list_path}"
            )

        paths: list[Path] = []
        base = noise_list_path.parent
        with noise_list_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                path = Path(line).expanduser()
                if not path.is_absolute():
                    path = base / path
                paths.append(path)

        if not paths:
            raise ValueError(f"Stage II noise list is empty: {noise_list_path}")
        missing = next((path for path in paths if not path.is_file()), None)
        if missing is not None:
            raise FileNotFoundError(f"Stage II noise audio not found: {missing}")
        return tuple(paths)

    @staticmethod
    def _as_mono(waveform: torch.Tensor, *, source: Path) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2 or waveform.size(0) == 0:
            raise ValueError(
                f"Expected audio shaped (channels, samples) for {source}, "
                f"got {tuple(waveform.shape)}"
            )
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.size(1) == 0:
            raise ValueError(f"Audio contains no samples: {source}")
        return waveform.to(dtype=torch.float32).contiguous()

    @staticmethod
    def _match_noise_length(
        noise: torch.Tensor,
        target_samples: int,
        *,
        rng: random.Random,
    ) -> torch.Tensor:
        noise_samples = int(noise.size(1))
        if noise_samples >= target_samples:
            max_offset = noise_samples - target_samples
            offset = rng.randint(0, max_offset) if max_offset else 0
            return noise[:, offset : offset + target_samples]

        # Repeat one extra period so a random phase still yields a complete
        # target-length slice instead of always starting at noise sample zero.
        offset = rng.randrange(noise_samples) if noise_samples > 1 else 0
        repeats = math.ceil((target_samples + offset) / noise_samples)
        tiled = noise.repeat(1, repeats)
        return tiled[:, offset : offset + target_samples]

    def _fbank(self) -> FbankExtractor:
        # Construct lazily inside the DataLoader worker.  Some backends keep
        # native extractor objects that should not be created in the parent and
        # then pickled/forked into every worker.
        if self._fbank_extractor is None:
            self._fbank_extractor = FbankExtractor(**self._fbank_kwargs)
        return self._fbank_extractor

    def mix(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        rng: random.Random,
    ) -> tuple[torch.Tensor, float]:
        """Return ``(mixed_waveform, sampled_snr_db)`` at the clean sample rate."""
        mixed, snr_db, _source = self.mix_with_source(waveform, sample_rate, rng=rng)
        return mixed, snr_db

    def mix_with_source(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        rng: random.Random,
        noise_path: str | Path | None = None,
        snr_db: float | None = None,
    ) -> tuple[torch.Tensor, float, Path]:
        """``mix`` plus the interference recording actually drawn.

        Offline callers pass an already drawn ``noise_path`` and ``snr_db`` so the
        recipe written to disk is exactly what was rendered. When they are omitted
        the source and level are drawn from ``rng`` in the online order (source and
        crop draws inside the retry loop, then the level draw).
        """
        if int(sample_rate) <= 0:
            raise ValueError("Stage II clean sample rate must be positive")
        waveform = self._as_mono(waveform, source=Path("<clean waveform>"))
        signal_power = waveform.square().mean()
        tiny = torch.finfo(torch.float32).tiny
        if not torch.isfinite(signal_power) or float(signal_power) <= tiny:
            raise ValueError(
                "Cannot apply SNR noise augmentation to a silent clean waveform"
            )

        noise = None
        drawn_path = None
        noise_power = None
        # A long recording can be valid overall while one randomly selected crop
        # is silent. Retry a bounded number of source/crop draws so an occasional
        # quiet region does not kill a persistent DataLoader worker.
        for _attempt in range(8):
            candidate_path = (
                Path(noise_path)
                if noise_path is not None
                else rng.choice(self.noise_paths)
            )
            candidate, noise_sample_rate = _load_audio(
                candidate_path,
                rng=rng,
                target_samples=int(waveform.size(1)),
                target_sample_rate=int(sample_rate),
            )
            candidate = self._as_mono(candidate, source=candidate_path)
            if int(noise_sample_rate) != int(sample_rate):
                torchaudio = _import_torchaudio()
                candidate = torchaudio.functional.resample(
                    candidate,
                    int(noise_sample_rate),
                    int(sample_rate),
                )
                candidate = self._as_mono(candidate, source=candidate_path)
            candidate = self._match_noise_length(
                candidate, int(waveform.size(1)), rng=rng
            )
            candidate_power = candidate.square().mean()
            if torch.isfinite(candidate_power) and float(candidate_power) > tiny:
                noise = candidate
                drawn_path = candidate_path
                noise_power = candidate_power
                break
        if noise is None or drawn_path is None or noise_power is None:
            raise ValueError(
                "Could not draw a non-silent Stage II noise crop after 8 attempts"
            )

        sampled_snr_db = (
            rng.uniform(self.snr_db_min, self.snr_db_max)
            if snr_db is None
            else float(snr_db)
        )
        try:
            snr_linear = 10.0 ** (sampled_snr_db / 10.0)
        except OverflowError as exc:
            raise ValueError(
                f"Stage II noise SNR is outside numeric range: {sampled_snr_db}"
            ) from exc
        if not math.isfinite(snr_linear) or snr_linear <= 0.0:
            raise ValueError(
                f"Stage II noise SNR is outside numeric range: {sampled_snr_db}"
            )
        scale = torch.sqrt(signal_power / (noise_power * snr_linear))
        mixed = waveform + noise * scale
        if not torch.isfinite(mixed).all():
            raise ValueError(
                f"Stage II noise augmentation produced non-finite samples: {drawn_path}"
            )
        return mixed.contiguous(), float(sampled_snr_db), drawn_path

    def extract(self, wav_path: str, *, rng: random.Random) -> torch.Tensor:
        """Load one clean clip, mix noise, and compute configured Stage-II fbank."""
        relative_path = Path(wav_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Stage II waveform path must be relative to waveform_dir: {wav_path}"
            )
        source_path = self.waveform_dir / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Stage II cached waveform not found: {source_path}. "
                "Run prepare_stage2_paper.py with prep.waveform_dir set, or point "
                "stage2.noise_augmentation.waveform_dir at an existing WAV tree."
            )
        waveform, sample_rate = _load_audio(source_path)
        waveform = self._as_mono(waveform, source=source_path)
        extractor = self._fbank()
        waveform, sample_rate = extractor.prepare_waveform(
            waveform, int(sample_rate)
        )
        mixed, _snr_db = self.mix(waveform, int(sample_rate), rng=rng)
        return extractor.extract(mixed, int(sample_rate))


class TrainingBackgroundSampler:
    """Draw a fixed-duration pure-background crop and compute Stage-II fbank.

    Unlike :class:`TrainingNoiseAugmenter`, this source does not mix background
    into speech. It supplies a genuine negative clip, so the dataset can teach
    the readout that music/noise alone must be explained as background. The RNG
    remains owned by the dataset for worker/DDP reproducibility, and both the
    fbank extractor and audio decode are lazy/partial inside each worker.
    """

    def __init__(
        self,
        *,
        audio_list_path: str | Path,
        duration_seconds_min: float,
        duration_seconds_max: float,
        fbank_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.duration_seconds_min = float(duration_seconds_min)
        self.duration_seconds_max = float(duration_seconds_max)
        if not math.isfinite(self.duration_seconds_min) or not math.isfinite(
            self.duration_seconds_max
        ):
            raise ValueError("Stage II background duration bounds must be finite")
        if self.duration_seconds_min <= 0.0:
            raise ValueError(
                "Stage II background duration_seconds_min must be positive"
            )
        if self.duration_seconds_min > self.duration_seconds_max:
            raise ValueError(
                "Stage II background duration_seconds_min must be <= "
                "duration_seconds_max"
            )

        self.audio_list_path = Path(audio_list_path)
        self.audio_paths = TrainingNoiseAugmenter._load_noise_files(
            self.audio_list_path
        )
        self._fbank_kwargs = dict(fbank_kwargs or {})
        self._fbank_extractor: FbankExtractor | None = None
        # Worker-local header cache. Probe sample_rate/frames/channels without
        # decoding audio or consuming dataset RNG.
        self._source_info_cache: dict[str, BackgroundSourceInfo] = {}

    def _fbank(self) -> FbankExtractor:
        if self._fbank_extractor is None:
            self._fbank_extractor = FbankExtractor(**self._fbank_kwargs)
        return self._fbank_extractor

    def _source_info(self, path: Path) -> BackgroundSourceInfo:
        key = str(path)
        cached = self._source_info_cache.get(key)
        if cached is not None:
            return cached
        info = probe_source_info(path)
        self._source_info_cache[key] = info
        return info

    def extract(self, *, rng: random.Random) -> torch.Tensor:
        """Sample one background crop and return configured fbank features."""
        duration_seconds = rng.uniform(
            self.duration_seconds_min,
            self.duration_seconds_max,
        )
        source_path = rng.choice(self.audio_paths)
        source = self._source_info(source_path)
        spec = draw_crop_spec(source, duration_seconds, rng=rng)
        waveform, sample_rate = materialize_crop(source, spec)
        return self._fbank().extract(waveform, sample_rate)


def _import_torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "Missing torchaudio. Install CUDA PyTorch/torchaudio on the training machine first."
        ) from exc
    return torchaudio


def _load_audio(
    path: str | Path,
    *,
    rng: random.Random | None = None,
    target_samples: int | None = None,
    target_sample_rate: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Load audio without torchaudio's optional TorchCodec dependency.

    Noise callers pass a target duration and RNG. In that mode, long recordings
    are decoded from a random span instead of reading the entire file for every
    augmented sample. Clean waveform-cache callers omit these arguments and read
    the complete clip.
    """
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "Missing soundfile. Install the declared dma-kws runtime dependencies."
        ) from exc

    partial_args = (rng, target_samples, target_sample_rate)
    if any(value is not None for value in partial_args) and not all(
        value is not None for value in partial_args
    ):
        raise ValueError(
            "rng, target_samples, and target_sample_rate must be provided together"
        )

    with sf.SoundFile(str(path)) as handle:
        sample_rate = int(handle.samplerate)
        frames = -1
        if rng is not None:
            assert target_samples is not None
            assert target_sample_rate is not None
            if target_samples <= 0 or target_sample_rate <= 0:
                raise ValueError("Target noise duration and sample rate must be positive")
            # Decode just enough source-rate samples to cover the target-rate
            # clean clip. A final crop/repeat after resampling handles rounding.
            frames = max(
                1,
                math.ceil(target_samples * sample_rate / target_sample_rate),
            )
            if len(handle) > frames:
                handle.seek(rng.randint(0, len(handle) - frames))
            else:
                frames = -1
        array = handle.read(
            frames=frames,
            dtype="float32",
            always_2d=True,
        )
    # soundfile is [samples, channels]; all Stage-II waveform code uses
    # [channels, samples]. copy() avoids a non-writable/strided NumPy view.
    waveform = torch.from_numpy(np.asarray(array, dtype=np.float32).T.copy())
    return waveform, int(sample_rate)


def compute_fbank(
    sample: dict[str, Any],
    *,
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    frame_shift: int = DEFAULT_FRAME_SHIFT,
    dither: float = DEFAULT_DITHER,
    window_type: str = "povey",
    backend: str = "torchaudio_kaldi",
    target_sample_rate: int | None = None,
    snip_edges: bool = True,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    extractor: FbankExtractor | None = None,
) -> dict[str, Any]:
    """Compute fbank features with the configured Stage II backend."""
    sample_rate = sample["sample_rate"]
    waveform = sample["wav"]
    if extractor is None:
        extractor = FbankExtractor(
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            window_type=window_type,
            backend=backend,
            target_sample_rate=target_sample_rate,
            snip_edges=snip_edges,
            low_freq=low_freq,
            high_freq=high_freq,
        )
    sample["feat"] = extractor.extract(waveform, sample_rate)
    return sample


def waveform_to_fbank(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    frame_shift: int = DEFAULT_FRAME_SHIFT,
    dither: float = DEFAULT_DITHER,
    window_type: str = "povey",
    backend: str = "torchaudio_kaldi",
    target_sample_rate: int | None = None,
    snip_edges: bool = True,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    extractor: FbankExtractor | None = None,
) -> torch.Tensor:
    """Compute configured fbank features from a mono waveform tensor."""
    sample = compute_fbank(
        {"wav": waveform, "sample_rate": sample_rate, "key": ""},
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=dither,
        window_type=window_type,
        backend=backend,
        target_sample_rate=target_sample_rate,
        snip_edges=snip_edges,
        low_freq=low_freq,
        high_freq=high_freq,
        extractor=extractor,
    )
    return sample["feat"]


class FeatureExtractor:
    """Load raw wav, optionally augment, and compute 80-dim fbank features."""

    def __init__(
        self,
        *,
        augment: bool = True,
        wav_dir: str | Path,
        noise_list_path: str | Path | None = None,
    ) -> None:
        self.data_aug = {
            "speed_perturb": augment,
            "add_noise": augment,
        }
        self.wav_dir = str(wav_dir)
        if augment:
            if noise_list_path is None:
                raise ValueError("noise_list_path is required when augment=True")
            self.noise_lists = self._load_noise_files(noise_list_path)
        else:
            self.noise_lists = []

    def _load_noise_files(self, noise_file_path: str | Path) -> list[str]:
        path = Path(noise_file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Noise list file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return handle.readlines()

    def _apply_speed_perturbation(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        speeds: list[float] | tuple[float, ...] = (0.9, 1.0, 1.1),
    ) -> torch.Tensor:
        speed = random.choice(speeds)
        if speed != 1.0:
            torchaudio = _import_torchaudio()
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform,
                sample_rate,
                [["speed", str(speed)], ["rate", str(sample_rate)]],
            )
        return waveform

    def _add_noise_with_snr(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        target_snr_db: float,
    ) -> torch.Tensor:
        torchaudio = _import_torchaudio()
        noise_path = random.choice(self.noise_lists).strip()
        noise_wav, noise_sr = torchaudio.load(noise_path)

        if noise_sr != sample_rate:
            noise_wav = torchaudio.transforms.Resample(noise_sr, sample_rate)(noise_wav)

        if noise_wav.shape[1] < waveform.shape[1]:
            repeat_times = (waveform.shape[1] // noise_wav.shape[1]) + 1
            noise_wav = noise_wav.repeat(1, repeat_times)[:, : waveform.shape[1]]
        else:
            noise_wav = noise_wav[:, : waveform.shape[1]]

        signal_power = torch.mean(waveform**2)
        noise_power = torch.mean(noise_wav**2)
        snr_linear = 10 ** (target_snr_db / 10)
        scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))
        noise_wav *= scaling_factor
        return waveform + noise_wav

    def load_npy_fbank(self, query_wav: str) -> torch.Tensor:
        """Load precomputed fbank from the main LibriPhrase npy layout."""
        from dma_kws.stage2.dataset import _resolve_fbank_path

        fbank_path = _resolve_fbank_path(self.wav_dir, query_wav)
        return torch.from_numpy(np.load(fbank_path))

    def process(self, wav_path: str) -> dict[str, Any]:
        """Load raw wav, optionally augment, and return a sample dict with ``feat``."""
        torchaudio = _import_torchaudio()
        full_path = os.path.join(self.wav_dir, wav_path)
        waveform, sr = torchaudio.load(full_path)

        if self.data_aug["speed_perturb"]:
            waveform = self._apply_speed_perturbation(waveform, sr)

        if self.data_aug["add_noise"] and random.random() < 0.5:
            snr = random.choice(range(5, 20))
            waveform = self._add_noise_with_snr(waveform, sr, snr)

        return compute_fbank(
            {
                "key": full_path,
                "wav": waveform,
                "sample_rate": sr,
            }
        )
