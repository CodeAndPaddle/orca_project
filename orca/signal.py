"""Synthetic whistle, analytic-template, and delay-estimation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from .contour import Contour, ContourConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated user settings and derived values for the synthetic experiment."""

    sample_rate: int = 24_001
    duration_seconds: float = 0.5
    true_delay_seconds: float = 0.02
    noise_std: float = 0.01
    wav_path: Path = Path("synthetic_dolphin_chirp.wav")
    output_dir: Path = Path("synthetic_dolphin_contours")
    whistle_band_hz: tuple[float, float] = (4_000.0, 10_000.0)
    analysis_margin_hz: float = 500.0
    min_delay_seconds: float = 0.002
    max_delay_seconds: float = 0.030
    window_seconds: float = 256 / 48_000
    delay_error_tolerance_seconds: float = 50e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "wav_path", Path(self.wav_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        try:
            integer_rate = int(self.sample_rate)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("sample_rate must be an integer number of hertz.") from error
        if isinstance(self.sample_rate, bool) or integer_rate != self.sample_rate:
            raise ValueError("sample_rate must be an integer number of hertz.")
        object.__setattr__(self, "sample_rate", integer_rate)
        numeric_settings = (
            self.duration_seconds,
            self.true_delay_seconds,
            self.noise_std,
            self.analysis_margin_hz,
            self.min_delay_seconds,
            self.max_delay_seconds,
            self.window_seconds,
            self.delay_error_tolerance_seconds,
            *self.whistle_band_hz,
        )
        if not all(math.isfinite(value) for value in numeric_settings):
            raise ValueError("Experiment settings must be finite numbers.")
        if not 24_001 <= self.sample_rate <= 96_000:
            raise ValueError("sample_rate must be between 24,001 and 96,000 Hz.")
        if not 0.1 <= self.duration_seconds <= 2.0:
            raise ValueError("duration_seconds must be between 0.1 and 2.0 seconds.")
        if self.noise_std < 0:
            raise ValueError("noise_std must be nonnegative.")
        start_hz, end_hz = self.whistle_band_hz
        if not 0 < start_hz < end_hz:
            raise ValueError("whistle_band_hz must contain increasing positive frequencies.")
        if self.analysis_margin_hz < 0 or start_hz - self.analysis_margin_hz <= 0:
            raise ValueError("analysis_margin_hz must keep the analysis band positive.")
        if end_hz + self.analysis_margin_hz >= self.sample_rate / 2:
            raise ValueError("The whistle and analysis margin must remain below Nyquist.")
        if not 0 < self.min_delay_seconds < self.max_delay_seconds:
            raise ValueError("Require 0 < min_delay_seconds < max_delay_seconds.")
        if not self.min_delay_seconds <= self.true_delay_seconds <= self.max_delay_seconds:
            raise ValueError("true_delay_seconds must lie inside the evaluation search window.")
        if self.true_delay_seconds >= self.duration_seconds:
            raise ValueError("true_delay_seconds must be shorter than the whistle duration.")
        if self.window_seconds <= 0 or self.delay_error_tolerance_seconds <= 0:
            raise ValueError("Window duration and delay tolerance must be positive.")

    @property
    def delay_samples(self) -> int:
        return round(self.true_delay_seconds * self.sample_rate)

    @property
    def delay_error_tolerance_samples(self) -> int:
        return math.ceil(self.delay_error_tolerance_seconds * self.sample_rate)

    @property
    def analysis_band_hz(self) -> tuple[float, float]:
        return (
            self.whistle_band_hz[0] - self.analysis_margin_hz,
            self.whistle_band_hz[1] + self.analysis_margin_hz,
        )

    def contour_config(self, **overrides) -> ContourConfig:
        """Build rate-independent contour settings for this experiment."""
        values = {
            "band_hz": self.analysis_band_hz,
            "window_seconds": self.window_seconds,
            **overrides,
        }
        return ContourConfig(**values)


@dataclass(frozen=True)
class DelayEstimate:
    delay_samples: int
    delay_seconds: float
    response: np.ndarray
    residual_response: np.ndarray
    lags: np.ndarray
    direct_index: int
    echo_index: int


def synthetic_whistle(
    duration_seconds: float = 1.0,
    sample_rate: int = 48_000,
    start_hz: float = 4_000.0,
    end_hz: float = 10_000.0,
    taper: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, instantaneous frequency, and pressure samples."""
    count = round(duration_seconds * sample_rate)
    if count < 2 or not 0 < start_hz < end_hz < sample_rate / 2:
        raise ValueError("Whistle duration and frequencies are incompatible with the sample rate.")
    time = np.arange(count) / sample_rate
    shape = _shape(np.linspace(-1.0, 1.0, count))
    frequency = start_hz + (end_hz - start_hz) * (shape - shape.min()) / np.ptp(shape)
    phase = 2 * np.pi * np.concatenate(([0.0], np.cumsum(frequency[:-1]) / sample_rate))
    samples = signal.windows.tukey(count, alpha=taper) * np.sin(phase)
    return time, frequency, samples


def template_from_contour(
    contour: Contour,
    sample_rate: int,
    frequency_band_hz: tuple[float, float] = (4_000.0, 10_000.0),
    taper: float = 0.25,
    max_contour_rmse_hz: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the ridge and return a band-calibrated analytic template."""
    if not contour.accepted:
        raise ValueError("Cannot build a template from a rejected contour.")
    start, end = contour.interval
    count = round((end - start) * sample_rate)
    time = start + np.arange(count) / sample_rate
    if count < 2:
        raise ValueError("Contour interval is too short for template reconstruction.")
    contour_time = np.asarray(contour.time_seconds, dtype=float)
    contour_frequency = np.asarray(contour.frequency_hz, dtype=float)
    if (
        contour_time.ndim != 1
        or contour_frequency.shape != contour_time.shape
        or contour_time.size < 2
        or not np.all(np.isfinite(contour_time))
        or not np.all(np.isfinite(contour_frequency))
        or np.any(np.diff(contour_time) <= 0)
        or contour_time[0] < start - 1 / sample_rate
        or contour_time[-1] > end + 1 / sample_rate
    ):
        raise ValueError("Contour points must be finite, ordered, and inside the interval.")
    start_hz, end_hz = frequency_band_hz
    if not 0 < start_hz < end_hz < sample_rate / 2:
        raise ValueError("frequency_band_hz is incompatible with the sample rate.")

    def calibrated_frequency(times: np.ndarray) -> np.ndarray:
        x = -1.0 + 2.0 * (times - start) / (time[-1] - start)
        shape = _shape(x)
        reference_shape = _shape(np.array([-1.0, 1.0]))
        normalized = (shape - reference_shape[0]) / np.ptp(reference_shape)
        return start_hz + (end_hz - start_hz) * normalized

    expected_ridge = calibrated_frequency(contour_time)
    contour_rmse = float(np.sqrt(np.mean((contour_frequency - expected_ridge) ** 2)))
    if not np.isfinite(contour_rmse) or contour_rmse > max_contour_rmse_hz:
        raise ValueError(
            f"Extracted contour RMSE ({contour_rmse:.1f} Hz) exceeds "
            f"the {max_contour_rmse_hz:.1f} Hz limit."
        )
    fitted = calibrated_frequency(time)
    if np.any(fitted <= 0) or np.any(fitted >= sample_rate / 2):
        raise ValueError("Fitted contour lies outside the usable frequency range.")
    phase = np.zeros(count)
    phase[1:] = 2 * np.pi * np.cumsum((fitted[:-1] + fitted[1:]) / (2 * sample_rate))
    template = signal.windows.tukey(count, alpha=taper) * np.exp(1j * phase)
    template /= np.linalg.norm(template)
    return template, fitted


def estimate_delay(
    received: np.ndarray,
    template: np.ndarray,
    sample_rate: int,
    min_delay_seconds: float = 0.002,
    max_delay_seconds: float = 0.030,
    prominence_ratio: float = 0.10,
) -> DelayEstimate:
    """Estimate echo separation after cancelling the dominant direct arrival."""
    received = np.asarray(received, dtype=float)
    template = np.asarray(template, dtype=complex)
    if received.ndim != 1 or template.ndim != 1 or not received.size or not template.size:
        raise ValueError("received and template must be nonempty one-dimensional arrays.")
    if not np.all(np.isfinite(received)) or not np.all(np.isfinite(template)):
        raise ValueError("received and template must contain only finite values.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if not 0 < min_delay_seconds < max_delay_seconds:
        raise ValueError("Require 0 < min_delay_seconds < max_delay_seconds.")
    if not 0 < prominence_ratio < 1:
        raise ValueError("prominence_ratio must lie strictly between 0 and 1.")

    kernel = np.conj(template[::-1]) / np.sum(abs(template) ** 2)
    analytic_received = signal.hilbert(received)
    response = signal.fftconvolve(analytic_received, kernel, mode="full")
    lags = np.arange(-(template.size - 1), received.size)
    magnitude = abs(response)
    direct = int(np.argmax(magnitude))

    direct_atom = _shifted_template(template, received.size, int(lags[direct]))
    atom_energy = np.vdot(direct_atom, direct_atom).real
    if atom_energy <= np.finfo(float).eps:
        raise RuntimeError("The direct template does not overlap the received signal.")
    direct_gain = np.vdot(direct_atom, analytic_received) / atom_energy
    residual = analytic_received - direct_gain * direct_atom
    residual_response = signal.fftconvolve(residual, kernel, mode="full")
    residual_magnitude = abs(residual_response)

    minimum = max(1, round(min_delay_seconds * sample_rate))
    maximum = round(max_delay_seconds * sample_rate)
    peaks, _ = signal.find_peaks(
        residual_magnitude,
        distance=minimum,
        prominence=prominence_ratio * magnitude[direct],
    )
    candidates = peaks[
        (lags[peaks] >= lags[direct] + minimum)
        & (lags[peaks] <= lags[direct] + maximum)
    ]
    if not candidates.size:
        raise RuntimeError("No credible delayed peak remained after direct-path cancellation.")
    echo = int(candidates[np.argmax(residual_magnitude[candidates])])
    delay = int(lags[echo] - lags[direct])
    return DelayEstimate(
        delay,
        delay / sample_rate,
        response,
        residual_response,
        lags,
        direct,
        echo,
    )


def _shifted_template(template: np.ndarray, length: int, lag: int) -> np.ndarray:
    """Place a template at a correlation lag, clipping it to the signal bounds."""
    atom = np.zeros(length, dtype=complex)
    source_start = max(0, -lag)
    destination_start = max(0, lag)
    count = min(template.size - source_start, length - destination_start)
    if count > 0:
        atom[destination_start : destination_start + count] = template[
            source_start : source_start + count
        ]
    return atom


def _shape(x: np.ndarray) -> np.ndarray:
    return np.tan(x) - np.sin(x) + 1
