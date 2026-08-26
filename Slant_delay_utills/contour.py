"""Time-frequency ridge extraction for manually selected WAV intervals."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import ndimage, signal
from scipy.io import wavfile


@dataclass(frozen=True)
class ContourConfig:
    """Physical extraction settings shared by every supported sample rate."""

    channel_mode: str = "mean"
    band_hz: tuple[float, float] = (3_500.0, 10_500.0)
    filter_order: int = 6
    window_seconds: float = 256 / 48_000
    overlap_fraction: float = 0.85
    fft_padding_factor: int = 16
    time_background_seconds: float = 0.10
    frequency_background_hz: float = 500.0
    min_mad_db: float = 0.75
    enhancement_weights: tuple[float, float, float] = (0.20, 0.10, 0.70)
    read_padding_seconds: float = 0.05
    track_padding_seconds: float = 0.015
    slope_margin: float = 5.0
    jump_penalty: tuple[float, float] = (0.010, 0.0005)
    min_confidence: float = 1.2
    smoothing_seconds: tuple[float, float] = (0.004, 0.005)
    figure_dynamic_range_db: float = 55.0
    figure_dpi: int = 150

    def resolve_stft(self, sample_rate: int) -> tuple[int, int]:
        """Return window and FFT sizes for a concrete sample rate."""
        if sample_rate <= 0 or self.window_seconds <= 0 or self.fft_padding_factor < 1:
            raise ValueError("Sample rate, window duration, and FFT padding must be positive.")
        window_samples = max(64, round(sample_rate * self.window_seconds))
        target_nfft = max(window_samples, self.fft_padding_factor * window_samples)
        nfft = 1 << math.ceil(math.log2(target_nfft))
        return window_samples, nfft


@dataclass(frozen=True)
class Contour:
    interval_index: int
    interval: tuple[float, float]
    accepted: bool
    confidence: float
    time_seconds: np.ndarray
    frequency_hz: np.ndarray
    raw_frequency_hz: np.ndarray
    path_scores: np.ndarray


@dataclass(frozen=True)
class Extraction:
    input_path: Path
    sample_rate: int
    contours: tuple[Contour, ...]
    output_dir: Path


@dataclass(frozen=True)
class _Diagnostic:
    samples: np.ndarray
    start_seconds: float
    frequencies_hz: np.ndarray
    times_seconds: np.ndarray
    power_db: np.ndarray
    enhanced: np.ndarray


def extract_contours(
    wav_file: str | Path,
    intervals: Sequence[Sequence[float]],
    output_dir: str | Path,
    config: ContourConfig | None = None,
) -> Extraction:
    """Extract one ridge per interval and write a compact NPZ plus diagnostics."""
    config = config or ContourConfig()
    path = Path(wav_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Recording not found: {path}")

    sample_rate, samples = wavfile.read(path)
    sample_rate = int(sample_rate)
    low_hz, high_hz = config.band_hz
    if low_hz <= 0 or high_hz <= low_hz or high_hz >= sample_rate / 2:
        raise ValueError(
            f"Frequency band {config.band_hz} Hz is invalid for a {sample_rate} Hz WAV."
        )
    samples = _as_float(samples)
    duration = samples.shape[0] / sample_rate
    normalized_intervals = _validate_intervals(intervals, duration)

    destination = Path(output_dir).expanduser().resolve()
    figures_dir = destination / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for stale in figures_dir.glob("interval_*.png"):
        stale.unlink()

    contours: list[Contour] = []
    for index, interval in enumerate(normalized_intervals, start=1):
        contour, diagnostic = _extract_interval(
            samples, sample_rate, interval, index, config
        )
        contours.append(contour)
        if contour.accepted:
            _plot_diagnostic(
                diagnostic,
                contour,
                sample_rate,
                figures_dir / f"interval_{index:04d}.png",
                config,
            )

    result = Extraction(path, sample_rate, tuple(contours), destination)
    _save_npz(result, config)
    return result


def _as_float(samples: np.ndarray) -> np.ndarray:
    if np.issubdtype(samples.dtype, np.floating):
        result = samples.astype(np.float64, copy=False)
    elif np.issubdtype(samples.dtype, np.signedinteger):
        limit = max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
        result = samples.astype(np.float64) / limit
    elif np.issubdtype(samples.dtype, np.unsignedinteger):
        midpoint = (np.iinfo(samples.dtype).max + 1) / 2
        result = (samples.astype(np.float64) - midpoint) / midpoint
    else:
        raise TypeError(f"Unsupported WAV dtype: {samples.dtype}")
    result[~np.isfinite(result)] = 0.0
    return result


def _validate_intervals(
    intervals: Sequence[Sequence[float]], duration: float
) -> list[tuple[float, float]]:
    if intervals is None:
        raise ValueError("Manual intervals are required.")
    values = np.asarray(intervals, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 2:
        raise ValueError("intervals must be a nonempty N-by-2 sequence.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Interval bounds must be finite.")
    values[:, 0] = np.maximum(values[:, 0], 0.0)
    values[:, 1] = np.minimum(values[:, 1], duration)
    if np.any(values[:, 1] <= values[:, 0]):
        raise ValueError("Every interval must overlap the WAV and have positive duration.")
    values = values[np.argsort(values[:, 0], kind="stable")]
    return [(float(start), float(end)) for start, end in values]


def _extract_interval(
    audio: np.ndarray,
    sample_rate: int,
    interval: tuple[float, float],
    index: int,
    config: ContourConfig,
) -> tuple[Contour, _Diagnostic]:
    start, end = interval
    first = max(0, math.floor((start - config.read_padding_seconds) * sample_rate))
    last = min(audio.shape[0], math.ceil((end + config.read_padding_seconds) * sample_rate))
    samples = np.asarray(audio[first:last])
    if samples.ndim == 2:
        if config.channel_mode == "mean":
            samples = samples.mean(axis=1)
        elif config.channel_mode == "first":
            samples = samples[:, 0]
        else:
            raise ValueError("channel_mode must be 'mean' or 'first'.")
    elif samples.ndim != 1:
        raise ValueError(f"Expected mono or samples-by-channels WAV, got {samples.shape}.")
    samples = samples.astype(float, copy=False).reshape(-1)
    if not samples.size:
        raise ValueError("Interval produced no audio samples.")
    samples -= samples.mean()

    filtered = _bandpass(samples, sample_rate, config)
    frequencies, times, power_db = _spectrogram(
        filtered, sample_rate, first / sample_rate, config
    )
    enhanced = _enhance(power_db, frequencies, times, config)
    contour = _track(enhanced, frequencies, times, interval, index, config)
    diagnostic = _Diagnostic(filtered, first / sample_rate, frequencies, times, power_db, enhanced)
    return contour, diagnostic


def _bandpass(samples: np.ndarray, sample_rate: int, config: ContourConfig) -> np.ndarray:
    low, high = sorted(config.band_hz)
    high = min(high, 0.99 * sample_rate / 2)
    if low <= 0 or low >= high:
        raise ValueError(f"Invalid frequency band {config.band_hz} for {sample_rate} Hz.")
    sos = signal.butter(
        config.filter_order, (low, high), btype="bandpass", fs=sample_rate, output="sos"
    )
    try:
        return signal.sosfiltfilt(sos, samples)
    except ValueError as error:
        raise ValueError(f"Interval is too short for zero-phase filtering: {error}") from error


def _spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    config: ContourConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_size, nfft = config.resolve_stft(sample_rate)
    if samples.size < window_size:
        samples = np.pad(samples, (0, window_size - samples.size))
    overlap = min(window_size - 1, max(0, round(config.overlap_fraction * window_size)))
    frequencies, times, spectrum = signal.stft(
        samples,
        fs=sample_rate,
        window=signal.windows.hann(window_size, sym=False),
        nperseg=window_size,
        noverlap=overlap,
        nfft=nfft,
        detrend=False,
        boundary=None,
        padded=False,
        scaling="spectrum",
    )
    low, high = config.band_hz
    band = (frequencies >= low) & (frequencies <= high)
    if not np.any(band):
        raise ValueError(f"No STFT bins fall inside {config.band_hz} Hz.")
    power_db = 10 * np.log10(np.abs(spectrum[band]) ** 2 + np.finfo(float).eps)
    return frequencies[band], start_seconds + times, power_db


def _enhance(
    power_db: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    config: ContourConfig,
) -> np.ndarray:
    power = np.nan_to_num(power_db, nan=float(np.nanmedian(power_db)))
    power = ndimage.median_filter(power, size=(3, 3), mode="reflect")
    dt, df = _spacing(times), _spacing(frequencies)
    time_size = _odd(config.time_background_seconds / dt)
    frequency_size = _odd(config.frequency_background_hz / df)
    by_time = power - ndimage.median_filter(power, size=(1, time_size), mode="nearest")
    by_frequency = power - ndimage.median_filter(
        power, size=(frequency_size, 1), mode="nearest"
    )

    column_median = np.median(power, axis=0, keepdims=True)
    column_mad = 1.4826 * np.median(abs(power - column_median), axis=0, keepdims=True)
    column_z = (power - column_median) / np.maximum(column_mad, config.min_mad_db)
    row_median = np.median(by_time, axis=1, keepdims=True)
    row_mad = 1.4826 * np.median(abs(by_time - row_median), axis=1, keepdims=True)
    time_z = (by_time - row_median) / np.maximum(row_mad, config.min_mad_db)
    median = np.median(by_frequency)
    mad = 1.4826 * np.median(abs(by_frequency - median))
    frequency_z = (by_frequency - median) / max(mad, config.min_mad_db)
    weights = config.enhancement_weights
    return np.nan_to_num(weights[0] * time_z + weights[1] * frequency_z + weights[2] * column_z)


def _track(
    enhanced: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    interval: tuple[float, float],
    index: int,
    config: ContourConfig,
) -> Contour:
    start, end = interval
    selected = (times >= start - config.track_padding_seconds) & (
        times <= end + config.track_padding_seconds
    )
    if selected.sum() < 2:
        raise ValueError(f"Interval {index} contains fewer than two STFT frames.")
    selected_times = times[selected]
    score = enhanced[:, selected]
    interval_duration = end - start
    max_slope = config.slope_margin * abs(config.band_hz[1] - config.band_hz[0]) / interval_duration
    rows = _viterbi(score, frequencies, selected_times, config, max_slope)
    columns = np.arange(rows.size)
    path_scores = score[rows, columns]
    raw_frequency = frequencies[rows]
    frequency = _refine(score, frequencies, rows)
    dt = _spacing(selected_times)
    frequency = ndimage.median_filter(
        frequency, size=_odd(config.smoothing_seconds[0] / dt), mode="nearest"
    )
    frequency = ndimage.uniform_filter1d(
        frequency, size=_odd(config.smoothing_seconds[1] / dt), mode="nearest"
    )
    inside = (selected_times >= start) & (selected_times <= end)
    if inside.sum() < 2:
        raise ValueError(f"Interval {index} has no usable ridge frames.")
    confidence = float(path_scores[inside].mean())
    return Contour(
        index,
        interval,
        bool(np.isfinite(confidence) and confidence >= config.min_confidence),
        confidence,
        selected_times[inside],
        frequency[inside],
        raw_frequency[inside],
        path_scores[inside],
    )


def _viterbi(
    emission: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    config: ContourConfig,
    max_slope_hz_per_second: float,
) -> np.ndarray:
    rows, columns = emission.shape
    df, dt = _spacing(frequencies), _spacing(times)
    max_jump = max(1, min(rows - 1, math.ceil(max_slope_hz_per_second * dt / df)))
    previous = emission[:, 0].copy()
    back = np.zeros((rows, columns), dtype=np.int32)
    linear, quadratic = config.jump_penalty
    for column in range(1, columns):
        best = np.full(rows, -np.inf)
        parent = np.zeros(rows, dtype=np.int32)
        for jump in range(-max_jump, max_jump + 1):
            current = np.arange(max(0, jump), min(rows, rows + jump))
            prior = current - jump
            transition = previous[prior] - linear * abs(jump) - quadratic * jump * jump
            better = transition > best[current]
            best[current[better]] = transition[better]
            parent[current[better]] = prior[better]
        previous = emission[:, column] + best
        back[:, column] = parent
    path = np.empty(columns, dtype=int)
    path[-1] = int(np.argmax(previous))
    for column in range(columns - 1, 0, -1):
        path[column - 1] = back[path[column], column]
    return path


def _refine(emission: np.ndarray, frequencies: np.ndarray, rows: np.ndarray) -> np.ndarray:
    refined = frequencies[rows].astype(float, copy=True)
    df = _spacing(frequencies)
    for column, row in enumerate(rows):
        if row == 0 or row == frequencies.size - 1:
            continue
        below, center, above = emission[row - 1 : row + 2, column]
        denominator = below - 2 * center + above
        if abs(denominator) > np.finfo(float).eps:
            offset = np.clip(0.5 * (below - above) / denominator, -0.75, 0.75)
            refined[column] += offset * df
    return refined


def _save_npz(result: Extraction, config: ContourConfig) -> None:
    counts = np.array([item.time_seconds.size for item in result.contours], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)))

    def join(name: str) -> np.ndarray:
        return np.concatenate([getattr(item, name) for item in result.contours])

    np.savez_compressed(
        result.output_dir / "contours.npz",
        schema_version=np.array(1, dtype=np.int64),
        sample_rate=np.array(result.sample_rate, dtype=np.int64),
        input_path=np.array(str(result.input_path)),
        config_json=np.array(json.dumps(asdict(config), separators=(",", ":"))),
        intervals=np.array([item.interval for item in result.contours]),
        accepted=np.array([item.accepted for item in result.contours]),
        confidence=np.array([item.confidence for item in result.contours]),
        contour_offsets=offsets,
        time_seconds=join("time_seconds"),
        frequency_hz=join("frequency_hz"),
        raw_frequency_hz=join("raw_frequency_hz"),
        path_scores=join("path_scores"),
    )


def _plot_diagnostic(
    diagnostic: _Diagnostic,
    contour: Contour,
    sample_rate: int,
    path: Path,
    config: ContourConfig,
) -> None:
    import matplotlib.pyplot as plt

    sample_times = diagnostic.start_seconds + np.arange(diagnostic.samples.size) / sample_rate
    figure, axes = plt.subplots(3, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].plot(sample_times, diagnostic.samples, color="black", linewidth=0.6)
    axes[0].set(ylabel="Amplitude", title=f"Interval {contour.interval_index}")
    maximum = float(diagnostic.power_db.max())
    axes[1].pcolormesh(
        diagnostic.times_seconds,
        diagnostic.frequencies_hz / 1_000,
        diagnostic.power_db,
        shading="auto",
        vmin=maximum - config.figure_dynamic_range_db,
        vmax=maximum,
    )
    axes[1].set(ylabel="Frequency (kHz)", title="STFT power (dB)")
    axes[2].pcolormesh(
        diagnostic.times_seconds,
        diagnostic.frequencies_hz / 1_000,
        diagnostic.enhanced,
        shading="auto",
    )
    axes[2].plot(contour.time_seconds, contour.frequency_hz / 1_000, color="white", linewidth=1.5)
    axes[2].set(
        xlabel="Time (s)",
        ylabel="Frequency (kHz)",
        title=f"Enhanced ridge; confidence={contour.confidence:.2f}",
    )
    for axis in axes:
        axis.set_xlim(contour.interval)
    figure.savefig(path, dpi=config.figure_dpi)
    plt.close(figure)


def _spacing(values: np.ndarray) -> float:
    spacing = float(np.median(np.diff(values)))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Expected a strictly increasing coordinate vector.")
    return spacing


def _odd(value: float) -> int:
    size = max(1, round(value))
    return size if size % 2 else size + 1
