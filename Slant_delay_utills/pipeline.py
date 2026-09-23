"""Automatic whistle localization and recording-level slant-delay analysis."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .contour import Contour, ContourConfig, extract_contours_from_samples
from .signal import MultipathDelayEstimate, estimate_delays, template_from_contour


@dataclass(frozen=True)
class AnalysisConfig:
    """Physical settings for unknown-time whistle and delay analysis."""

    whistle_band_hz: tuple[float, float]
    min_delay_seconds: float = 0.002
    max_delay_seconds: float = 0.030
    min_path_separation_seconds: float = 0.002
    max_echoes: int = 4
    detector_window_seconds: float = 0.032
    detector_hop_seconds: float = 0.008
    min_whistle_seconds: float = 0.100
    max_whistle_seconds: float = 5.0
    merge_gap_seconds: float = 0.050
    min_peak_z: float = 3.0
    min_detection_confidence: float = 0.45
    max_frequency_jump_hz: float = 1_500.0
    pre_match_margin_seconds: float = 0.020
    post_match_margin_seconds: float = 0.010
    prominence_ratio: float = 0.05
    output_dir: Path = Path("whistle_analysis")

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        try:
            band = tuple(float(value) for value in self.whistle_band_hz)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("whistle_band_hz must contain two numeric values.") from error
        object.__setattr__(self, "whistle_band_hz", band)
        values = (
            *band,
            self.min_delay_seconds,
            self.max_delay_seconds,
            self.min_path_separation_seconds,
            self.detector_window_seconds,
            self.detector_hop_seconds,
            self.min_whistle_seconds,
            self.max_whistle_seconds,
            self.merge_gap_seconds,
            self.min_peak_z,
            self.min_detection_confidence,
            self.max_frequency_jump_hz,
            self.pre_match_margin_seconds,
            self.post_match_margin_seconds,
            self.prominence_ratio,
        )
        if len(band) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("Analysis settings must contain finite numeric values.")
        if not 0 < band[0] < band[1]:
            raise ValueError("whistle_band_hz must contain increasing positive frequencies.")
        if not 0 < self.min_delay_seconds < self.max_delay_seconds:
            raise ValueError("Require 0 < min_delay_seconds < max_delay_seconds.")
        if self.min_path_separation_seconds <= 0:
            raise ValueError("min_path_separation_seconds must be positive.")
        if isinstance(self.max_echoes, bool) or not isinstance(self.max_echoes, int) or self.max_echoes <= 0:
            raise ValueError("max_echoes must be a positive integer.")
        if not 0 < self.detector_hop_seconds <= self.detector_window_seconds:
            raise ValueError("Detector hop must be positive and no longer than its window.")
        if not 0 < self.min_whistle_seconds < self.max_whistle_seconds:
            raise ValueError("Require 0 < min_whistle_seconds < max_whistle_seconds.")
        if self.merge_gap_seconds < 0 or self.max_frequency_jump_hz <= 0:
            raise ValueError("Merge gap must be nonnegative and maximum jump must be positive.")
        if not 0 <= self.min_detection_confidence <= 1:
            raise ValueError("min_detection_confidence must lie between zero and one.")
        if self.pre_match_margin_seconds < 0 or self.post_match_margin_seconds < 0:
            raise ValueError("Matched-filter margins must be nonnegative.")
        if not 0 < self.prominence_ratio < 1:
            raise ValueError("prominence_ratio must lie strictly between zero and one.")

    def contour_config(self, **overrides) -> ContourConfig:
        values = {
            "band_hz": self.whistle_band_hz,
            "allow_inactive": True,
            "min_observed_fraction": 0.80,
            "max_inactive_seconds": self.merge_gap_seconds,
            **overrides,
        }
        return ContourConfig(**values)


@dataclass(frozen=True)
class WhistleDetection:
    start_seconds: float
    end_seconds: float
    time_seconds: np.ndarray
    frequency_hz: np.ndarray
    observed_mask: np.ndarray
    detection_confidence: float
    contour_confidence: float


@dataclass(frozen=True)
class WhistleResult:
    detection: WhistleDetection
    template: np.ndarray
    delay_samples: np.ndarray
    delay_seconds: np.ndarray
    delay_confidence: np.ndarray
    warnings: tuple[str, ...]
    response: np.ndarray
    residual_response: np.ndarray
    lags: np.ndarray
    direct_index: int
    echo_indices: np.ndarray


@dataclass(frozen=True)
class RecordingAnalysis:
    input_path: Path
    sample_rate: int
    whistles: tuple[WhistleResult, ...]
    output_dir: Path


@dataclass(frozen=True)
class _Candidate:
    interval: tuple[float, float]
    confidence: float


def detect_whistles(
    wav_path: str | Path,
    config: AnalysisConfig,
) -> tuple[WhistleDetection, ...]:
    """Detect and trace every separated, credible whistle in a WAV."""
    path, sample_rate, audio = _load_audio(wav_path, config)
    del path
    mono = _select_channel(audio)
    candidates = _candidate_intervals(mono, sample_rate, config)
    if not candidates:
        return ()
    contours = extract_contours_from_samples(
        audio,
        sample_rate,
        [candidate.interval for candidate in candidates],
        config.contour_config(),
    )
    detections: list[WhistleDetection] = []
    for candidate, contour in zip(candidates, contours):
        if not contour.accepted or not np.any(contour.observed_mask):
            continue
        observed_times = contour.time_seconds[contour.observed_mask]
        detection = WhistleDetection(
            start_seconds=float(observed_times[0]),
            end_seconds=float(observed_times[-1]),
            time_seconds=contour.time_seconds.copy(),
            frequency_hz=contour.frequency_hz.copy(),
            observed_mask=contour.observed_mask.copy(),
            detection_confidence=float(candidate.confidence),
            contour_confidence=float(contour.confidence),
        )
        detections.append(detection)
    return tuple(sorted(detections, key=lambda item: item.start_seconds))


def analyze_recording(
    wav_path: str | Path,
    config: AnalysisConfig,
) -> RecordingAnalysis:
    """Detect whistles, build contour templates, and estimate echo delays."""
    detections = detect_whistles(wav_path, config)
    return analyze_detections(wav_path, config, detections)


def analyze_detections(
    wav_path: str | Path,
    config: AnalysisConfig,
    detections: Sequence[WhistleDetection],
) -> RecordingAnalysis:
    """Analyze already detected whistles without running the detector again."""
    path, sample_rate, audio = _load_audio(wav_path, config)
    detections = _validate_detections(
        detections,
        duration_seconds=audio.shape[0] / sample_rate,
        sample_rate=sample_rate,
        config=config,
    )
    mono = _select_channel(audio)
    filtered = _bandpass(mono, sample_rate, config.whistle_band_hz)
    results: list[WhistleResult] = []
    for detection in detections:
        contour = _as_contour(detection)
        template, _ = template_from_contour(
            contour,
            sample_rate,
            frequency_band_hz=config.whistle_band_hz,
            reference_samples=filtered,
        )
        first = max(
            0,
            math.floor(
                (detection.start_seconds - config.pre_match_margin_seconds) * sample_rate
            ),
        )
        last = min(
            filtered.size,
            math.ceil(
                (
                    detection.end_seconds
                    + config.max_delay_seconds
                    + config.post_match_margin_seconds
                )
                * sample_rate
            ),
        )
        received = filtered[first:last]
        estimate = estimate_delays(
            received,
            template,
            sample_rate,
            num_echoes=None,
            max_echoes=config.max_echoes,
            min_delay_seconds=config.min_delay_seconds,
            max_delay_seconds=config.max_delay_seconds,
            min_path_separation_seconds=config.min_path_separation_seconds,
            prominence_ratio=config.prominence_ratio,
        )
        warnings: list[str] = []
        if _direct_is_ambiguous(estimate, sample_rate, config):
            warnings.append(
                "The dominant direct arrival is ambiguous; reflected-path delays were rejected."
            )
            estimate = MultipathDelayEstimate(
                delay_samples=np.array([], dtype=int),
                delay_seconds=np.array([], dtype=float),
                response=estimate.response,
                residual_response=estimate.residual_response,
                lags=estimate.lags,
                direct_index=estimate.direct_index,
                echo_indices=np.array([], dtype=int),
                echo_prominences=np.array([], dtype=float),
                delay_confidence=np.array([], dtype=float),
            )
        elif estimate.delay_samples.size == 0:
            warnings.append("No credible reflected path remained after direct-path cancellation.")
        results.append(_result_from_estimate(detection, template, estimate, warnings))

    destination = config.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    analysis = RecordingAnalysis(path, sample_rate, tuple(results), destination)
    _save_analysis(analysis, config)
    return analysis


def _validate_detections(
    detections: Sequence[WhistleDetection],
    *,
    duration_seconds: float,
    sample_rate: int,
    config: AnalysisConfig,
) -> tuple[WhistleDetection, ...]:
    try:
        values = tuple(detections)
    except TypeError as error:
        raise ValueError("detections must be a sequence of WhistleDetection values.") from error

    previous_end = -math.inf
    time_tolerance = 1 / sample_rate
    low_hz, high_hz = config.whistle_band_hz
    for index, detection in enumerate(values, start=1):
        if not isinstance(detection, WhistleDetection):
            raise ValueError(f"Detection {index} is not a WhistleDetection value.")
        scalars = (
            detection.start_seconds,
            detection.end_seconds,
            detection.detection_confidence,
            detection.contour_confidence,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError(f"Detection {index} contains non-finite metadata.")
        if not 0 <= detection.start_seconds < detection.end_seconds <= duration_seconds:
            raise ValueError(f"Detection {index} lies outside the recording.")
        if detection.start_seconds < previous_end:
            raise ValueError("Detections must be ordered and non-overlapping.")

        times = np.asarray(detection.time_seconds)
        frequencies = np.asarray(detection.frequency_hz)
        observed = np.asarray(detection.observed_mask)
        if (
            times.ndim != 1
            or times.size < 2
            or frequencies.shape != times.shape
            or observed.shape != times.shape
            or observed.dtype != np.bool_
        ):
            raise ValueError(
                f"Detection {index} contour arrays must be aligned one-dimensional values."
            )
        if (
            not np.all(np.isfinite(times))
            or not np.all(np.isfinite(frequencies))
            or np.any(np.diff(times) <= 0)
            or times[0] < 0
            or times[-1] > duration_seconds
            or np.count_nonzero(observed) < 2
        ):
            raise ValueError(f"Detection {index} contains an invalid contour path.")
        observed_times = times[observed]
        if (
            abs(observed_times[0] - detection.start_seconds) > time_tolerance
            or abs(observed_times[-1] - detection.end_seconds) > time_tolerance
        ):
            raise ValueError(
                f"Detection {index} bounds do not match its observed contour path."
            )
        observed_frequencies = frequencies[observed]
        if np.any(observed_frequencies < low_hz) or np.any(
            observed_frequencies > high_hz
        ):
            raise ValueError(f"Detection {index} lies outside whistle_band_hz.")
        previous_end = detection.end_seconds
    return values


def _load_audio(
    wav_path: str | Path, config: AnalysisConfig
) -> tuple[Path, int, np.ndarray]:
    path = Path(wav_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Recording not found: {path}")
    sample_rate, samples = wavfile.read(path)
    sample_rate = int(sample_rate)
    if config.whistle_band_hz[1] >= sample_rate / 2:
        raise ValueError(
            f"Frequency band {config.whistle_band_hz} Hz reaches Nyquist for {sample_rate} Hz."
        )
    return path, sample_rate, _as_float(samples)


def _as_float(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim not in (1, 2) or samples.shape[0] == 0:
        raise ValueError("Expected a nonempty mono or samples-by-channels WAV.")
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
    result = result.copy()
    result[~np.isfinite(result)] = 0.0
    return result


def _select_channel(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.astype(float, copy=False)
    return samples.mean(axis=1, dtype=float)


def _bandpass(
    samples: np.ndarray, sample_rate: int, band_hz: tuple[float, float]
) -> np.ndarray:
    sos = signal.butter(6, band_hz, btype="bandpass", fs=sample_rate, output="sos")
    try:
        return signal.sosfiltfilt(sos, samples - np.mean(samples))
    except ValueError as error:
        raise ValueError(f"Recording is too short for zero-phase filtering: {error}") from error


def _candidate_intervals(
    samples: np.ndarray, sample_rate: int, config: AnalysisConfig
) -> tuple[_Candidate, ...]:
    filtered = _bandpass(samples, sample_rate, config.whistle_band_hz)
    window = max(64, round(config.detector_window_seconds * sample_rate))
    hop = max(1, round(config.detector_hop_seconds * sample_rate))
    if filtered.size < window:
        return ()
    overlap = window - hop
    target_fft = max(window, 4 * window)
    nfft = 1 << math.ceil(math.log2(target_fft))
    frequencies, times, spectrum = signal.stft(
        filtered,
        fs=sample_rate,
        window=signal.windows.hann(window, sym=False),
        nperseg=window,
        noverlap=overlap,
        nfft=nfft,
        boundary=None,
        padded=False,
        scaling="spectrum",
    )
    band = (frequencies >= config.whistle_band_hz[0]) & (
        frequencies <= config.whistle_band_hz[1]
    )
    power = np.abs(spectrum[band]) ** 2 + np.finfo(float).eps
    if power.shape[0] < 3 or power.shape[1] < 2:
        return ()
    power_db = 10 * np.log10(power)
    column_median = np.median(power_db, axis=0, keepdims=True)
    column_mad = 1.4826 * np.median(
        np.abs(power_db - column_median), axis=0, keepdims=True
    )
    column_z = (power_db - column_median) / np.maximum(column_mad, 0.75)
    peak_z = np.max(column_z, axis=0)
    probabilities = power / np.sum(power, axis=0, keepdims=True)
    entropy = -np.sum(probabilities * np.log(probabilities), axis=0) / np.log(power.shape[0])
    entropy_evidence = _robust_z(-entropy)
    energy_db = 10 * np.log10(np.sum(power, axis=0))
    energy_evidence = _robust_z(energy_db)
    evidence = (
        0.60 * np.clip((peak_z - config.min_peak_z) / 6.0, 0.0, 1.0)
        + 0.25 * np.clip(entropy_evidence / 4.0, 0.0, 1.0)
        + 0.15 * np.clip(energy_evidence / 4.0, 0.0, 1.0)
    )
    dominant_rows = np.argmax(column_z, axis=0)
    dominant_frequency = frequencies[band][dominant_rows]
    active = (peak_z >= config.min_peak_z) & (evidence >= 0.35)
    maximum_gap = max(0, round(config.merge_gap_seconds / config.detector_hop_seconds))
    groups = _true_groups(active, maximum_gap)
    candidates: list[_Candidate] = []
    duration = samples.size / sample_rate
    for group in groups:
        first, last = int(group[0]), int(group[-1])
        active_duration = times[last] - times[first] + config.detector_window_seconds
        if not config.min_whistle_seconds <= active_duration <= config.max_whistle_seconds:
            continue
        path = dominant_frequency[group]
        continuity = 1.0 if path.size < 2 else float(
            np.mean(np.abs(np.diff(path)) <= config.max_frequency_jump_hz)
        )
        confidence = float(np.clip(0.75 * np.mean(evidence[group]) + 0.25 * continuity, 0, 1))
        if confidence < config.min_detection_confidence:
            continue
        # STFT timestamps are window centers.  Using the centers as conservative
        # boundaries avoids teaching the phase template from leading noise.
        start = max(0.0, float(times[first]))
        end = min(duration, float(times[last]))
        candidates.append(_Candidate((start, end), confidence))
    return tuple(candidates)


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    return (values - median) / max(mad, 0.25)


def _true_groups(mask: np.ndarray, maximum_gap: int = 0) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if not indices.size:
        return []
    sparse_groups = np.split(
        indices, np.flatnonzero(np.diff(indices) > maximum_gap + 1) + 1
    )
    return [np.arange(group[0], group[-1] + 1) for group in sparse_groups]


def _as_contour(detection: WhistleDetection) -> Contour:
    return Contour(
        interval_index=1,
        interval=(detection.start_seconds, detection.end_seconds),
        accepted=True,
        confidence=detection.contour_confidence,
        time_seconds=detection.time_seconds,
        frequency_hz=detection.frequency_hz,
        raw_frequency_hz=detection.frequency_hz,
        path_scores=np.full(detection.time_seconds.size, detection.contour_confidence),
        observed_mask=detection.observed_mask,
    )


def _result_from_estimate(
    detection: WhistleDetection,
    template: np.ndarray,
    estimate: MultipathDelayEstimate,
    warnings: list[str],
) -> WhistleResult:
    return WhistleResult(
        detection=detection,
        template=template,
        delay_samples=estimate.delay_samples,
        delay_seconds=estimate.delay_seconds,
        delay_confidence=estimate.delay_confidence,
        warnings=tuple(warnings),
        response=estimate.response,
        residual_response=estimate.residual_response,
        lags=estimate.lags,
        direct_index=estimate.direct_index,
        echo_indices=estimate.echo_indices,
    )


def _direct_is_ambiguous(
    estimate: MultipathDelayEstimate,
    sample_rate: int,
    config: AnalysisConfig,
) -> bool:
    magnitude = np.abs(estimate.response)
    direct_magnitude = float(magnitude[estimate.direct_index])
    if direct_magnitude <= np.finfo(float).eps:
        return True
    exclusion = max(1, round(config.min_path_separation_seconds * sample_rate))
    keep = np.ones(magnitude.size, dtype=bool)
    keep[
        max(0, estimate.direct_index - exclusion) : min(
            magnitude.size, estimate.direct_index + exclusion + 1
        )
    ] = False
    competitor = float(np.max(magnitude[keep], initial=0.0))
    return competitor >= 0.95 * direct_magnitude


def _save_analysis(analysis: RecordingAnalysis, config: AnalysisConfig) -> None:
    detections = [item.detection for item in analysis.whistles]
    contour_counts = np.array([item.time_seconds.size for item in detections], dtype=np.int64)
    delay_counts = np.array([item.delay_seconds.size for item in analysis.whistles], dtype=np.int64)
    response_counts = np.array([item.response.size for item in analysis.whistles], dtype=np.int64)
    template_counts = np.array([item.template.size for item in analysis.whistles], dtype=np.int64)

    def offsets(counts: np.ndarray) -> np.ndarray:
        return np.concatenate(([0], np.cumsum(counts)))

    def join(arrays: list[np.ndarray], dtype=float) -> np.ndarray:
        return np.concatenate(arrays) if arrays else np.array([], dtype=dtype)

    config_values = asdict(config)
    config_values["output_dir"] = str(config.output_dir)
    np.savez_compressed(
        analysis.output_dir / "analysis.npz",
        schema_version=np.array(1, dtype=np.int64),
        input_path=np.array(str(analysis.input_path)),
        sample_rate=np.array(analysis.sample_rate, dtype=np.int64),
        config_json=np.array(json.dumps(config_values, separators=(",", ":"))),
        intervals=np.array(
            [(item.start_seconds, item.end_seconds) for item in detections], dtype=float
        ).reshape(-1, 2),
        detection_confidence=np.array(
            [item.detection_confidence for item in detections], dtype=float
        ),
        contour_confidence=np.array(
            [item.contour_confidence for item in detections], dtype=float
        ),
        contour_offsets=offsets(contour_counts),
        time_seconds=join([item.time_seconds for item in detections]),
        frequency_hz=join([item.frequency_hz for item in detections]),
        observed_mask=join([item.observed_mask for item in detections], dtype=bool),
        template_offsets=offsets(template_counts),
        templates=join([item.template for item in analysis.whistles], dtype=complex),
        delay_offsets=offsets(delay_counts),
        delay_samples=join([item.delay_samples for item in analysis.whistles], dtype=np.int64),
        delay_seconds=join([item.delay_seconds for item in analysis.whistles]),
        delay_confidence=join([item.delay_confidence for item in analysis.whistles]),
        response_offsets=offsets(response_counts),
        lags=join([item.lags for item in analysis.whistles], dtype=np.int64),
        response=join([item.response for item in analysis.whistles], dtype=complex),
        residual_response=join(
            [item.residual_response for item in analysis.whistles], dtype=complex
        ),
        warnings_json=np.array(json.dumps([item.warnings for item in analysis.whistles])),
    )
