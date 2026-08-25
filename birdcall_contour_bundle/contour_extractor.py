"""Python port of the manually supplied-interval MATLAB contour extractor.

The implementation deliberately keeps the signal-processing stages and output
names close to ``extract_birdcall_contours.m``. Automatic interval detection is
not part of the version-1 Python API: callers must supply suspected intervals.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal
from scipy.io import wavfile


@dataclass
class OutputConfig:
    dir: str = ""
    prefix: str = ""
    write_csv: bool = True
    save_npz: bool = True
    save_metadata_json: bool = True


@dataclass
class AudioConfig:
    channel_mode: str = "mean"
    min_sample_rate_hz: float = 24_000.0


@dataclass
class FilterConfig:
    band_hz: tuple[float, float] = (3_500.0, 10_500.0)
    order: int = 6


@dataclass
class StftConfig:
    win_samples: int = 256
    overlap_frac: float = 0.85
    nfft: int | None = None
    nfft_factor: int = 8
    min_nfft: int = 4_096
    max_nfft: int = 16_384
    detect_band_hz: tuple[float, float] = (3_500.0, 10_500.0)


@dataclass
class EnhanceConfig:
    time_bg_sec: float = 0.10
    freq_bg_hz: float = 500.0
    min_mad_db: float = 0.75
    # The one-second synthetic whistle is continuous. Per-column contrast must
    # dominate slower background subtraction so the tracker does not prefer
    # enhancement side-lobes through the middle of the chirp.
    w_time: float = 0.20
    w_freq: float = 0.10
    w_col: float = 0.70
    gauss_size: int = 0
    gauss_sigma: float = 0.75


@dataclass
class IntervalConfig:
    read_pad_sec: float = 0.05
    track_pad_sec: float = 0.015
    warn_duration_sec: float = 1.25


@dataclass
class ViterbiConfig:
    max_slope_hz_per_sec: float = 150_000.0
    jump_penalty_linear: float = 0.010
    jump_penalty_quadratic: float = 0.0005
    min_confidence: float = 1.2
    smooth_median_sec: float = 0.004
    smooth_mean_sec: float = 0.005


@dataclass
class FigureConfig:
    save_png: bool = True
    save_rejected: bool = False
    max_figures: int = 100
    context_sec: float = 0.10
    plot_band_hz: tuple[float, float] = (3_000.0, 11_000.0)
    dynamic_range_db: float = 55.0
    resolution_dpi: int = 250


@dataclass
class DebugConfig:
    verbose: bool = True


@dataclass
class ContourConfig:
    output: OutputConfig = field(default_factory=OutputConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    stft: StftConfig = field(default_factory=StftConfig)
    enhance: EnhanceConfig = field(default_factory=EnhanceConfig)
    interval: IntervalConfig = field(default_factory=IntervalConfig)
    viterbi: ViterbiConfig = field(default_factory=ViterbiConfig)
    figures: FigureConfig = field(default_factory=FigureConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


@dataclass
class Contour:
    file_name: str
    file_path: str
    interval_index: int
    contour_index: int
    requested_start_sec: float
    requested_end_sec: float
    time_sec_abs: np.ndarray
    freq_hz: np.ndarray
    raw_freq_hz: np.ndarray
    path_scores: np.ndarray
    confidence: float
    score_max: float
    total_dp_score: float
    method: str = "bandpass_scipy_stft_robust_constrained_viterbi"

    @property
    def t_start_sec(self) -> float:
        return float(self.time_sec_abs[0])

    @property
    def t_end_sec(self) -> float:
        return float(self.time_sec_abs[-1])

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.t_end_sec - self.t_start_sec)

    @property
    def f_start_hz(self) -> float:
        return float(self.freq_hz[0])

    @property
    def f_end_hz(self) -> float:
        return float(self.freq_hz[-1])

    @property
    def f_min_hz(self) -> float:
        return float(np.min(self.freq_hz))

    @property
    def f_max_hz(self) -> float:
        return float(np.max(self.freq_hz))

    @property
    def f_median_hz(self) -> float:
        return float(np.median(self.freq_hz))

    @property
    def bandwidth_hz(self) -> float:
        return self.f_max_hz - self.f_min_hz


@dataclass
class IntervalStatus:
    interval_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    status: str
    contour_index: int | None = None
    error_message: str = ""


@dataclass
class ContourResults:
    wav_file: str
    file_name: str
    fs: int
    duration_sec: float
    cfg: ContourConfig
    intervals: list[tuple[float, float]]
    interval_statuses: list[IntervalStatus]
    contours: list[Contour]
    output_dir: str
    summary_csv_path: str
    points_csv_path: str
    interval_csv_path: str
    npz_path: str
    metadata_json_path: str


@dataclass
class _Ridge:
    valid: bool = False
    time_sec_abs: np.ndarray = field(default_factory=lambda: np.empty(0))
    freq_hz: np.ndarray = field(default_factory=lambda: np.empty(0))
    raw_freq_hz: np.ndarray = field(default_factory=lambda: np.empty(0))
    path_scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    confidence: float = math.nan
    score_max: float = math.nan
    total_dp_score: float = math.nan


@dataclass
class _Diagnostic:
    y: np.ndarray
    fs: int
    t0_abs_sec: float
    frequencies_hz: np.ndarray
    times_sec: np.ndarray
    power_db: np.ndarray
    enhanced: np.ndarray


def birdcall_contour_default_config() -> ContourConfig:
    """Return independent defaults tuned for the 48 kHz synthetic whistle."""
    return ContourConfig()


def extract_birdcall_contours(
    wav_file: str | Path,
    intervals: Sequence[Sequence[float]] | np.ndarray,
    cfg: ContourConfig | None = None,
) -> ContourResults:
    """Extract one ridge from each manually supplied ``(start, end)`` interval.

    Processing failures in an individual interval are recorded in the interval
    status table so other requested intervals can still be processed. Invalid
    API inputs and unreadable audio fail fast.
    """
    cfg = cfg if cfg is not None else birdcall_contour_default_config()
    if not isinstance(cfg, ContourConfig):
        raise TypeError("cfg must be a ContourConfig or None.")
    wav_path = Path(wav_file).expanduser().resolve()
    if not wav_path.is_file():
        raise FileNotFoundError(f"Recording not found: {wav_path}")

    fs, audio = wavfile.read(wav_path)
    fs = int(fs)
    if fs <= cfg.audio.min_sample_rate_hz:
        raise ValueError(
            f"Sample rate {fs:.1f} Hz is too low for the configured "
            f"{cfg.filter.band_hz[0]:.1f}-{cfg.filter.band_hz[1]:.1f} Hz band."
        )
    audio = _audio_to_float(audio)
    duration_sec = audio.shape[0] / fs
    normalized_intervals = _normalize_intervals(intervals, duration_sec)

    output_dir = Path(cfg.output.dir).expanduser() if cfg.output.dir else (
        wav_path.parent / f"{wav_path.stem}_birdcall_contours"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    if cfg.figures.save_png:
        figures_dir.mkdir(parents=True, exist_ok=True)
    prefix = _sanitize_name(cfg.output.prefix or wav_path.stem)

    if cfg.debug.verbose:
        print(f"birdcontour: file={wav_path}")
        print(f"birdcontour: mode=provided, intervals={len(normalized_intervals)}, fs={fs} Hz")
        print(f"birdcontour: output={output_dir}")

    contours: list[Contour] = []
    statuses: list[IntervalStatus] = []
    figure_count = 0
    for interval_index, (start_sec, end_sec) in enumerate(normalized_intervals, start=1):
        if end_sec - start_sec > cfg.interval.warn_duration_sec:
            warnings.warn(
                f"Interval {interval_index} is {end_sec - start_sec:.3f} s long; "
                "one Viterbi ridge will be returned.",
                RuntimeWarning,
                stacklevel=2,
            )
        status = IntervalStatus(
            interval_index, start_sec, end_sec, end_sec - start_sec, "error"
        )
        try:
            ridge, diagnostic = _extract_one_interval(
                audio, fs, start_sec, end_sec, cfg
            )
            if ridge.valid:
                contour = Contour(
                    file_name=wav_path.name,
                    file_path=str(wav_path),
                    interval_index=interval_index,
                    contour_index=len(contours) + 1,
                    requested_start_sec=start_sec,
                    requested_end_sec=end_sec,
                    time_sec_abs=ridge.time_sec_abs,
                    freq_hz=ridge.freq_hz,
                    raw_freq_hz=ridge.raw_freq_hz,
                    path_scores=ridge.path_scores,
                    confidence=ridge.confidence,
                    score_max=ridge.score_max,
                    total_dp_score=ridge.total_dp_score,
                )
                contours.append(contour)
                status.status = "accepted"
                status.contour_index = contour.contour_index
            else:
                status.status = "rejected_low_confidence"

            should_save = ridge.valid or cfg.figures.save_rejected
            if cfg.figures.save_png and should_save and figure_count < cfg.figures.max_figures:
                figure_path = figures_dir / f"{prefix}_interval{interval_index:04d}.png"
                _save_interval_figure(
                    diagnostic, ridge, start_sec, end_sec, interval_index, figure_path, cfg
                )
                figure_count += 1
        except Exception as exc:  # keep MATLAB's per-interval fault isolation
            status.status = "error"
            status.error_message = str(exc)
            warnings.warn(
                f"Interval {interval_index} [{start_sec:.6f}, {end_sec:.6f}] failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        statuses.append(status)

    summary_path = output_dir / f"{prefix}_contour_summary.csv"
    points_path = output_dir / f"{prefix}_contour_points.csv"
    interval_path = output_dir / f"{prefix}_interval_status.csv"
    npz_path = output_dir / f"{prefix}_contours.npz"
    metadata_path = output_dir / f"{prefix}_run_metadata.json"

    if cfg.output.write_csv:
        _write_csv_artifacts(contours, statuses, summary_path, points_path, interval_path)
    if cfg.output.save_npz:
        _write_npz(npz_path, fs, duration_sec, normalized_intervals, statuses, contours)
    if cfg.output.save_metadata_json:
        metadata = {
            "schema_version": 1,
            "wav_file": str(wav_path),
            "sample_rate_hz": fs,
            "duration_sec": duration_sec,
            "interval_mode": "provided",
            "accepted_contours": len(contours),
            "configuration": asdict(cfg),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    results = ContourResults(
        wav_file=str(wav_path),
        file_name=wav_path.name,
        fs=fs,
        duration_sec=duration_sec,
        cfg=cfg,
        intervals=normalized_intervals,
        interval_statuses=statuses,
        contours=contours,
        output_dir=str(output_dir),
        summary_csv_path=str(summary_path),
        points_csv_path=str(points_path),
        interval_csv_path=str(interval_path),
        npz_path=str(npz_path),
        metadata_json_path=str(metadata_path),
    )
    if cfg.debug.verbose:
        point_count = sum(c.time_sec_abs.size for c in contours)
        print(
            f"birdcontour: accepted={len(contours)}/{len(normalized_intervals)}, "
            f"points={point_count}"
        )
    return results


def load_contours_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a version-1 pickle-free contour archive."""
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.array(archive[name], copy=True) for name in archive.files}
    required = {
        "schema_version", "fs", "duration_sec", "interval_start_sec",
        "interval_end_sec", "interval_status", "interval_contour_index",
        "contour_offsets", "contour_interval_index", "contour_confidence",
        "time_sec_abs", "freq_hz", "raw_freq_hz", "path_scores",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Contour NPZ is missing fields: {sorted(missing)}")
    if int(data["schema_version"].reshape(-1)[0]) != 1:
        raise ValueError("Unsupported contour NPZ schema version.")
    offsets = data["contour_offsets"].astype(int, copy=False)
    point_count = data["time_sec_abs"].size
    if offsets.ndim != 1 or offsets.size == 0 or offsets[0] != 0 or offsets[-1] != point_count:
        raise ValueError("Invalid contour_offsets in contour NPZ.")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("contour_offsets must be nondecreasing.")
    for name in ("freq_hz", "raw_freq_hz", "path_scores"):
        if data[name].size != point_count:
            raise ValueError(f"{name} does not match the flattened point count.")
    return data


def _audio_to_float(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.floating):
        result = audio.astype(np.float64, copy=False)
    elif np.issubdtype(audio.dtype, np.signedinteger):
        info = np.iinfo(audio.dtype)
        result = audio.astype(np.float64) / max(abs(info.min), info.max)
    elif np.issubdtype(audio.dtype, np.unsignedinteger):
        info = np.iinfo(audio.dtype)
        midpoint = (info.max + 1) / 2
        result = (audio.astype(np.float64) - midpoint) / midpoint
    else:
        raise TypeError(f"Unsupported WAV sample dtype: {audio.dtype}")
    result[~np.isfinite(result)] = 0.0
    return result


def _normalize_intervals(
    intervals: Sequence[Sequence[float]] | np.ndarray, duration_sec: float
) -> list[tuple[float, float]]:
    if intervals is None:
        raise ValueError(
            "intervals is required in the version-1 Python extractor; "
            "automatic interval detection is not implemented."
        )
    values = np.asarray(intervals, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("intervals must be a nonempty N-by-2 sequence of (start, end) seconds.")
    if not np.all(np.isfinite(values)):
        raise ValueError("interval bounds must be finite.")
    values[:, 0] = np.maximum(values[:, 0], 0.0)
    values[:, 1] = np.minimum(values[:, 1], duration_sec)
    if np.any(values[:, 1] <= values[:, 0]):
        raise ValueError("every interval must overlap the recording and have positive duration.")
    values = values[np.argsort(values[:, 0], kind="stable")]
    return [(float(start), float(end)) for start, end in values]


def _extract_one_interval(
    audio: np.ndarray,
    fs: int,
    start_sec: float,
    end_sec: float,
    cfg: ContourConfig,
) -> tuple[_Ridge, _Diagnostic]:
    read_start = max(0, math.floor((start_sec - cfg.interval.read_pad_sec) * fs))
    read_end = min(audio.shape[0], math.ceil((end_sec + cfg.interval.read_pad_sec) * fs))
    segment = np.asarray(audio[read_start:read_end])
    if segment.ndim == 2:
        if cfg.audio.channel_mode.lower() == "first":
            segment = segment[:, 0]
        elif cfg.audio.channel_mode.lower() == "mean":
            segment = np.mean(segment, axis=1)
        else:
            raise ValueError("audio.channel_mode must be 'mean' or 'first'.")
    elif segment.ndim != 1:
        raise ValueError(f"WAV audio must be mono or samples-by-channels; got {segment.shape}.")
    segment = segment.astype(np.float64, copy=False).reshape(-1)
    if segment.size == 0:
        raise ValueError("interval produced an empty audio segment.")
    segment = segment - np.mean(segment)
    filtered = _bandpass_audio(segment, fs, cfg)
    t0_abs_sec = read_start / fs
    frequencies, times, power_db = _compute_spectrogram(filtered, fs, t0_abs_sec, cfg)
    enhanced = _enhance_spectrogram(power_db, frequencies, times, cfg)
    ridge = _track_ridge(enhanced, frequencies, times, start_sec, end_sec, cfg)
    return ridge, _Diagnostic(
        filtered, fs, t0_abs_sec, frequencies, times, power_db, enhanced
    )


def _bandpass_audio(audio: np.ndarray, fs: int, cfg: ContourConfig) -> np.ndarray:
    low, high = sorted(map(float, cfg.filter.band_hz))
    nyquist = fs / 2
    low = max(1.0, low)
    high = min(0.99 * nyquist, high)
    if low >= high:
        raise ValueError(f"Invalid band {low:.1f}-{high:.1f} Hz for fs {fs:.1f} Hz.")
    sos = signal.butter(cfg.filter.order, (low, high), btype="bandpass", fs=fs, output="sos")
    try:
        filtered = signal.sosfiltfilt(sos, audio)
    except ValueError as exc:
        raise ValueError(f"Zero-phase bandpass filtering failed: {exc}") from exc
    filtered[~np.isfinite(filtered)] = 0.0
    return filtered


def _compute_spectrogram(
    audio: np.ndarray, fs: int, t0_abs_sec: float, cfg: ContourConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nwin = max(128, int(round(cfg.stft.win_samples)))
    if audio.size < nwin:
        audio = np.pad(audio, (0, nwin - audio.size))
    noverlap = int(round(cfg.stft.overlap_frac * nwin))
    noverlap = min(max(0, noverlap), nwin - 1)
    if cfg.stft.nfft is None or cfg.stft.nfft <= 0:
        desired = cfg.stft.nfft_factor * nwin
        nfft = 1 << int(math.ceil(math.log2(max(1, desired))))
    else:
        nfft = int(round(cfg.stft.nfft))
    nfft = max(nwin, cfg.stft.min_nfft, nfft)
    nfft = min(cfg.stft.max_nfft, nfft)
    window = signal.windows.hann(nwin, sym=False)
    frequencies, relative_times, spectrum = signal.stft(
        audio,
        fs=fs,
        window=window,
        nperseg=nwin,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
        scaling="spectrum",
    )
    low, high = cfg.stft.detect_band_hz
    keep = (frequencies >= low) & (frequencies <= high)
    if not np.any(keep):
        raise ValueError(f"No STFT bins remain in {low:.1f}-{high:.1f} Hz.")
    power = np.abs(spectrum[keep]) ** 2
    power_db = 10 * np.log10(power + np.finfo(float).eps)
    return frequencies[keep], t0_abs_sec + relative_times, power_db


def _enhance_spectrogram(
    power_db: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    cfg: ContourConfig,
) -> np.ndarray:
    power = np.asarray(power_db, dtype=float).copy()
    finite = power[np.isfinite(power)]
    power[~np.isfinite(power)] = np.median(finite) if finite.size else 0.0
    power = ndimage.median_filter(power, size=(3, 3), mode="reflect")
    dt = _safe_spacing(times, 1.0)
    df = _safe_spacing(frequencies, 1.0)
    time_frames = _odd_number(cfg.enhance.time_bg_sec / max(dt, np.finfo(float).eps))
    freq_bins = _odd_number(cfg.enhance.freq_bg_hz / max(df, np.finfo(float).eps))
    time_floor = ndimage.median_filter(power, size=(1, time_frames), mode="nearest")
    freq_floor = ndimage.median_filter(power, size=(freq_bins, 1), mode="nearest")
    enhanced_time = power - time_floor
    enhanced_freq = power - freq_floor

    col_median = np.median(power, axis=0, keepdims=True)
    col_mad = 1.4826 * np.median(np.abs(power - col_median), axis=0, keepdims=True)
    z_col = (power - col_median) / np.maximum(col_mad, cfg.enhance.min_mad_db)
    row_median = np.median(enhanced_time, axis=1, keepdims=True)
    row_mad = 1.4826 * np.median(
        np.abs(enhanced_time - row_median), axis=1, keepdims=True
    )
    z_time = (enhanced_time - row_median) / np.maximum(row_mad, cfg.enhance.min_mad_db)
    freq_median = np.median(enhanced_freq)
    freq_mad = 1.4826 * np.median(np.abs(enhanced_freq - freq_median))
    z_freq = (enhanced_freq - freq_median) / max(freq_mad, cfg.enhance.min_mad_db)
    enhanced = (
        cfg.enhance.w_time * z_time
        + cfg.enhance.w_freq * z_freq
        + cfg.enhance.w_col * z_col
    )
    if cfg.enhance.gauss_size > 1:
        truncate = ((cfg.enhance.gauss_size - 1) / 2) / max(cfg.enhance.gauss_sigma, 1e-12)
        enhanced = ndimage.gaussian_filter(
            enhanced, cfg.enhance.gauss_sigma, mode="nearest", truncate=truncate
        )
    enhanced[~np.isfinite(enhanced)] = 0.0
    return enhanced


def _track_ridge(
    enhanced: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    start_sec: float,
    end_sec: float,
    cfg: ContourConfig,
) -> _Ridge:
    ridge = _Ridge()
    t1 = max(times[0], start_sec - cfg.interval.track_pad_sec)
    t2 = min(times[-1], end_sec + cfg.interval.track_pad_sec)
    keep_time = (times >= t1) & (times <= t2)
    if np.count_nonzero(keep_time) < 2:
        return ridge
    selected = enhanced[:, keep_time]
    selected_times = times[keep_time]
    rows, total_score = _viterbi_dp(selected, frequencies, selected_times, cfg)
    if rows.size == 0:
        return ridge
    columns = np.arange(rows.size)
    path_scores = selected[rows, columns]
    raw_frequency = frequencies[rows]
    frequency = _refine_subbin(selected, frequencies, rows)
    dt = _safe_spacing(selected_times, 1.0)
    frequency = _moving_median(
        frequency, _odd_number(cfg.viterbi.smooth_median_sec / max(dt, 1e-12))
    )
    frequency = _moving_mean(
        frequency, _odd_number(cfg.viterbi.smooth_mean_sec / max(dt, 1e-12))
    )
    inside = (selected_times >= start_sec) & (selected_times <= end_sec)
    if np.count_nonzero(inside) < 2:
        return ridge
    ridge.time_sec_abs = selected_times[inside]
    ridge.freq_hz = frequency[inside]
    ridge.raw_freq_hz = raw_frequency[inside]
    ridge.path_scores = path_scores[inside]
    ridge.confidence = float(np.mean(ridge.path_scores))
    ridge.score_max = float(np.max(ridge.path_scores))
    ridge.total_dp_score = float(total_score)
    ridge.valid = bool(
        np.isfinite(ridge.confidence) and ridge.confidence >= cfg.viterbi.min_confidence
    )
    return ridge


def _viterbi_dp(
    emission: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    cfg: ContourConfig,
) -> tuple[np.ndarray, float]:
    emission = np.asarray(emission, dtype=float).copy()
    n_freq, n_time = emission.shape
    if n_freq == 0 or n_time == 0:
        return np.empty(0, dtype=int), math.nan
    finite = emission[np.isfinite(emission)]
    if finite.size == 0:
        return np.empty(0, dtype=int), math.nan
    emission[~np.isfinite(emission)] = np.min(finite) - 10
    if n_time == 1:
        row = int(np.argmax(emission[:, 0]))
        return np.array([row]), float(emission[row, 0])
    df = _safe_spacing(frequencies, 1.0)
    dt = _safe_spacing(times, 1.0)
    max_jump = int(math.ceil(cfg.viterbi.max_slope_hz_per_sec * dt / max(df, 1e-12)))
    max_jump = max(1, min(n_freq - 1, max_jump))
    previous_score = emission[:, 0].copy()
    back_pointer = np.zeros((n_freq, n_time), dtype=np.int32)
    for column in range(1, n_time):
        best_transition = np.full(n_freq, -np.inf)
        best_previous = np.zeros(n_freq, dtype=np.int32)
        for jump in range(-max_jump, max_jump + 1):
            penalty = (
                cfg.viterbi.jump_penalty_linear * abs(jump)
                + cfg.viterbi.jump_penalty_quadratic * jump * jump
            )
            if jump < 0:
                current = np.arange(0, n_freq + jump)
            elif jump > 0:
                current = np.arange(jump, n_freq)
            else:
                current = np.arange(n_freq)
            prior = current - jump
            transition = previous_score[prior] - penalty
            better = transition > best_transition[current]
            selected_current = current[better]
            best_transition[selected_current] = transition[better]
            best_previous[selected_current] = prior[better]
        previous_score = emission[:, column] + best_transition
        back_pointer[:, column] = best_previous
    row = int(np.argmax(previous_score))
    final_score = float(previous_score[row])
    rows = np.zeros(n_time, dtype=int)
    rows[-1] = row
    for column in range(n_time - 1, 0, -1):
        row = int(back_pointer[row, column])
        rows[column - 1] = row
    return rows, final_score


def _refine_subbin(
    enhanced: np.ndarray, frequencies: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    refined = frequencies[rows].astype(float, copy=True)
    df = _safe_spacing(frequencies, 0.0)
    if df <= 0:
        return refined
    for column, row in enumerate(rows):
        if row <= 0 or row >= frequencies.size - 1:
            continue
        below, center, above = enhanced[row - 1 : row + 2, column]
        denominator = below - 2 * center + above
        if np.isfinite(denominator) and abs(denominator) > np.finfo(float).eps:
            delta = 0.5 * (below - above) / denominator
            refined[column] = frequencies[row] + np.clip(delta, -0.75, 0.75) * df
    return refined


def _moving_median(values: np.ndarray, size: int) -> np.ndarray:
    if size <= 1 or values.size <= 1:
        return values.copy()
    return ndimage.median_filter(values, size=size, mode="nearest")


def _moving_mean(values: np.ndarray, size: int) -> np.ndarray:
    if size <= 1 or values.size <= 1:
        return values.copy()
    return ndimage.uniform_filter1d(values, size=size, mode="nearest")


def _write_csv_artifacts(
    contours: list[Contour],
    statuses: list[IntervalStatus],
    summary_path: Path,
    points_path: Path,
    interval_path: Path,
) -> None:
    summary_fields = [
        "fileName", "intervalIndex", "contourIndex", "requestedStartSec",
        "requestedEndSec", "tStartSec", "tEndSec", "durationSec", "fStartHz",
        "fEndHz", "fMinHz", "fMaxHz", "fMedianHz", "bandwidthHz",
        "confidence", "scoreMax", "method",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for contour in contours:
            writer.writerow({
                "fileName": contour.file_name,
                "intervalIndex": contour.interval_index,
                "contourIndex": contour.contour_index,
                "requestedStartSec": contour.requested_start_sec,
                "requestedEndSec": contour.requested_end_sec,
                "tStartSec": contour.t_start_sec,
                "tEndSec": contour.t_end_sec,
                "durationSec": contour.duration_sec,
                "fStartHz": contour.f_start_hz,
                "fEndHz": contour.f_end_hz,
                "fMinHz": contour.f_min_hz,
                "fMaxHz": contour.f_max_hz,
                "fMedianHz": contour.f_median_hz,
                "bandwidthHz": contour.bandwidth_hz,
                "confidence": contour.confidence,
                "scoreMax": contour.score_max,
                "method": contour.method,
            })
    point_fields = [
        "fileName", "intervalIndex", "contourIndex", "pointIndex", "timeSec",
        "freqHz", "rawFreqHz", "pathScore", "confidence",
    ]
    with points_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=point_fields)
        writer.writeheader()
        for contour in contours:
            for point_index, values in enumerate(
                zip(
                    contour.time_sec_abs,
                    contour.freq_hz,
                    contour.raw_freq_hz,
                    contour.path_scores,
                    strict=True,
                ),
                start=1,
            ):
                time_sec, freq_hz, raw_freq_hz, path_score = values
                writer.writerow({
                    "fileName": contour.file_name,
                    "intervalIndex": contour.interval_index,
                    "contourIndex": contour.contour_index,
                    "pointIndex": point_index,
                    "timeSec": time_sec,
                    "freqHz": freq_hz,
                    "rawFreqHz": raw_freq_hz,
                    "pathScore": path_score,
                    "confidence": contour.confidence,
                })
    interval_fields = [
        "intervalIndex", "tStartSec", "tEndSec", "durationSec", "status",
        "contourIndex", "errorMessage",
    ]
    with interval_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=interval_fields)
        writer.writeheader()
        for status in statuses:
            writer.writerow({
                "intervalIndex": status.interval_index,
                "tStartSec": status.start_sec,
                "tEndSec": status.end_sec,
                "durationSec": status.duration_sec,
                "status": status.status,
                "contourIndex": "" if status.contour_index is None else status.contour_index,
                "errorMessage": status.error_message,
            })


def _write_npz(
    path: Path,
    fs: int,
    duration_sec: float,
    intervals: list[tuple[float, float]],
    statuses: list[IntervalStatus],
    contours: list[Contour],
) -> None:
    counts = np.array([contour.time_sec_abs.size for contour in contours], dtype=np.int64)
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts)))

    def concatenate(field_name: str) -> np.ndarray:
        if not contours:
            return np.empty(0, dtype=np.float64)
        return np.concatenate([np.asarray(getattr(c, field_name), dtype=float) for c in contours])

    np.savez_compressed(
        path,
        schema_version=np.array(1, dtype=np.int64),
        fs=np.array(fs, dtype=np.int64),
        duration_sec=np.array(duration_sec, dtype=np.float64),
        interval_start_sec=np.array([value[0] for value in intervals], dtype=np.float64),
        interval_end_sec=np.array([value[1] for value in intervals], dtype=np.float64),
        interval_status=np.array([status.status for status in statuses], dtype="U32"),
        interval_contour_index=np.array(
            [-1 if status.contour_index is None else status.contour_index for status in statuses],
            dtype=np.int64,
        ),
        contour_offsets=offsets,
        contour_interval_index=np.array([c.interval_index for c in contours], dtype=np.int64),
        contour_confidence=np.array([c.confidence for c in contours], dtype=np.float64),
        time_sec_abs=concatenate("time_sec_abs"),
        freq_hz=concatenate("freq_hz"),
        raw_freq_hz=concatenate("raw_freq_hz"),
        path_scores=concatenate("path_scores"),
    )


def _save_interval_figure(
    diagnostic: _Diagnostic,
    ridge: _Ridge,
    start_sec: float,
    end_sec: float,
    interval_index: int,
    path: Path,
    cfg: ContourConfig,
) -> None:
    waveform_time = diagnostic.t0_abs_sec + np.arange(diagnostic.y.size) / diagnostic.fs
    x1 = max(float(waveform_time[0]), start_sec - cfg.figures.context_sec)
    x2 = min(float(waveform_time[-1]), end_sec + cfg.figures.context_sec)
    figure, axes = plt.subplots(3, 1, figsize=(14, 9.5), constrained_layout=True)
    axes[0].plot(waveform_time, diagnostic.y, color="black", linewidth=0.6)
    axes[0].axvline(start_sec, color="tab:blue", linestyle="--")
    axes[0].axvline(end_sec, color="tab:blue", linestyle="--")
    axes[0].set(xlim=(x1, x2), ylabel="Amplitude", title=f"Interval {interval_index}: {start_sec:.6f}-{end_sec:.6f} s")
    axes[0].grid(True)
    maximum_db = float(np.max(diagnostic.power_db))
    axes[1].pcolormesh(
        diagnostic.times_sec,
        diagnostic.frequencies_hz / 1_000,
        diagnostic.power_db,
        shading="auto",
        cmap="viridis",
        vmin=maximum_db - cfg.figures.dynamic_range_db,
        vmax=maximum_db,
    )
    axes[1].set(xlim=(x1, x2), ylim=np.array(cfg.figures.plot_band_hz) / 1_000, ylabel="Frequency (kHz)", title="SciPy STFT power (dB)")
    axes[2].pcolormesh(
        diagnostic.times_sec,
        diagnostic.frequencies_hz / 1_000,
        diagnostic.enhanced,
        shading="auto",
        cmap="viridis",
    )
    if ridge.time_sec_abs.size:
        axes[2].plot(ridge.time_sec_abs, ridge.freq_hz / 1_000, color="white", linewidth=2.0)
        axes[2].plot(ridge.time_sec_abs, ridge.freq_hz / 1_000, color="black", linewidth=0.7)
    axes[2].set(
        xlim=(x1, x2),
        ylim=np.array(cfg.figures.plot_band_hz) / 1_000,
        xlabel="Time from recording start (s)",
        ylabel="Frequency (kHz)",
        title=f"Enhanced TF map and contour; confidence={ridge.confidence:.3f}",
    )
    figure.savefig(path, dpi=cfg.figures.resolution_dpi)
    plt.close(figure)


def _safe_spacing(values: np.ndarray, fallback: float) -> float:
    if values.size > 1:
        spacing = float(np.median(np.diff(values)))
        if np.isfinite(spacing) and spacing > 0:
            return spacing
    return fallback


def _odd_number(value: float) -> int:
    result = max(1, int(round(value)))
    return result if result % 2 else result + 1


def _sanitize_name(name: str) -> str:
    sanitized = "".join(character if character.isalnum() or character in "_.-" else "_" for character in str(name))
    return sanitized or "birdcall"
