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
    true_delay_seconds: tuple[float, ...] = (0.01, 0.02)
    direct_gain: float = 0.5
    echo_gains: tuple[float, ...] = (0.2, 0.15)
    noise_std: float = 0.01
    wav_path: Path = Path("synthetic_dolphin_chirp.wav")
    output_dir: Path = Path("synthetic_dolphin_contours")
    whistle_band_hz: tuple[float, float] = (4_000.0, 10_000.0)
    analysis_margin_hz: float = 500.0
    min_delay_seconds: float = 0.002
    max_delay_seconds: float = 0.030
    min_path_separation_seconds: float = 0.002
    window_seconds: float = 256 / 48_000
    delay_error_tolerance_seconds: float = 50e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "wav_path", Path(self.wav_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        try:
            true_delays = tuple(float(value) for value in self.true_delay_seconds)
            echo_gains = tuple(float(value) for value in self.echo_gains)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "true_delay_seconds and echo_gains must be numeric sequences."
            ) from error
        object.__setattr__(self, "true_delay_seconds", true_delays)
        object.__setattr__(self, "echo_gains", echo_gains)
        try:
            integer_rate = int(self.sample_rate)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("sample_rate must be an integer number of hertz.") from error
        if isinstance(self.sample_rate, bool) or integer_rate != self.sample_rate:
            raise ValueError("sample_rate must be an integer number of hertz.")
        object.__setattr__(self, "sample_rate", integer_rate)
        numeric_settings = (
            self.duration_seconds,
            self.direct_gain,
            self.noise_std,
            self.analysis_margin_hz,
            self.min_delay_seconds,
            self.max_delay_seconds,
            self.min_path_separation_seconds,
            self.window_seconds,
            self.delay_error_tolerance_seconds,
            *self.true_delay_seconds,
            *self.echo_gains,
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
        if not self.true_delay_seconds:
            raise ValueError("true_delay_seconds must contain at least one echo delay.")
        if len(self.echo_gains) != len(self.true_delay_seconds):
            raise ValueError("echo_gains must contain one gain per true delay.")
        if self.direct_gain <= 0 or any(
            gain <= 0 or gain >= self.direct_gain for gain in self.echo_gains
        ):
            raise ValueError("Echo gains must be positive and weaker than direct_gain.")
        start_hz, end_hz = self.whistle_band_hz
        if not 0 < start_hz < end_hz:
            raise ValueError("whistle_band_hz must contain increasing positive frequencies.")
        if self.analysis_margin_hz < 0 or start_hz - self.analysis_margin_hz <= 0:
            raise ValueError("analysis_margin_hz must keep the analysis band positive.")
        if end_hz + self.analysis_margin_hz >= self.sample_rate / 2:
            raise ValueError("The whistle and analysis margin must remain below Nyquist.")
        if not 0 < self.min_delay_seconds < self.max_delay_seconds:
            raise ValueError("Require 0 < min_delay_seconds < max_delay_seconds.")
        if self.min_path_separation_seconds <= 0:
            raise ValueError("min_path_separation_seconds must be positive.")
        if any(
            later <= earlier
            for earlier, later in zip(self.true_delay_seconds, self.true_delay_seconds[1:])
        ):
            raise ValueError("true_delay_seconds must be strictly increasing.")
        if any(
            delay < self.min_delay_seconds or delay > self.max_delay_seconds
            for delay in self.true_delay_seconds
        ):
            raise ValueError("Every true delay must lie inside the evaluation search window.")
        if any(delay >= self.duration_seconds for delay in self.true_delay_seconds):
            raise ValueError("Every true delay must be shorter than the whistle duration.")
        separation_samples = max(1, round(self.min_path_separation_seconds * self.sample_rate))
        if any(
            later - earlier < separation_samples
            for earlier, later in zip(self.delay_samples, self.delay_samples[1:])
        ):
            raise ValueError("True delays are too close at the configured sample rate.")
        if self.window_seconds <= 0 or self.delay_error_tolerance_seconds <= 0:
            raise ValueError("Window duration and delay tolerance must be positive.")

    @property
    def delay_samples(self) -> np.ndarray:
        return np.rint(np.asarray(self.true_delay_seconds) * self.sample_rate).astype(int)

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
class MultipathDelayEstimate:
    delay_samples: np.ndarray
    delay_seconds: np.ndarray
    response: np.ndarray
    residual_response: np.ndarray
    lags: np.ndarray
    direct_index: int
    echo_indices: np.ndarray


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


def synthesize_multipath(
    source: np.ndarray,
    delay_samples: np.ndarray | tuple[int, ...] | list[int],
    echo_gains: np.ndarray | tuple[float, ...] | list[float],
    *,
    direct_gain: float = 0.5,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a noisy multi-path signal and its sparse impulse response."""
    source = np.asarray(source, dtype=float)
    delays = np.asarray(delay_samples)
    gains = np.asarray(echo_gains, dtype=float)
    if source.ndim != 1 or not source.size or not np.all(np.isfinite(source)):
        raise ValueError("source must be a nonempty, finite one-dimensional array.")
    if delays.ndim != 1 or not delays.size:
        raise ValueError("delay_samples must be a nonempty one-dimensional sequence.")
    if not np.issubdtype(delays.dtype, np.integer):
        raise ValueError("delay_samples must contain integer sample offsets.")
    delays = delays.astype(int, copy=False)
    if np.any(delays <= 0) or np.any(np.diff(delays) <= 0):
        raise ValueError("delay_samples must contain strictly increasing positive offsets.")
    if gains.shape != delays.shape or not np.all(np.isfinite(gains)):
        raise ValueError("echo_gains must contain one finite value per delay.")
    if not math.isfinite(direct_gain) or direct_gain <= 0:
        raise ValueError("direct_gain must be a finite positive number.")
    if np.any(gains <= 0) or np.any(gains >= direct_gain):
        raise ValueError("Echo gains must be positive and weaker than direct_gain.")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be a finite nonnegative number.")

    impulse_response = np.zeros(int(delays[-1]) + 1)
    impulse_response[0] = direct_gain
    impulse_response[delays] = gains
    received = signal.convolve(source, impulse_response)
    if noise_std:
        received += noise_std * np.random.default_rng(seed).standard_normal(received.size)
    return received, impulse_response


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


def estimate_delays(
    received: np.ndarray,
    template: np.ndarray,
    sample_rate: int,
    *,
    num_echoes: int,
    min_delay_seconds: float = 0.002,
    max_delay_seconds: float = 0.030,
    min_path_separation_seconds: float = 0.002,
    prominence_ratio: float = 0.10,
) -> MultipathDelayEstimate:
    """Estimate delayed paths after cancelling the dominant direct arrival."""
    received = np.asarray(received, dtype=float)
    template = np.asarray(template, dtype=complex)
    if received.ndim != 1 or template.ndim != 1 or not received.size or not template.size:
        raise ValueError("received and template must be nonempty one-dimensional arrays.")
    if not np.all(np.isfinite(received)) or not np.all(np.isfinite(template)):
        raise ValueError("received and template must contain only finite values.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if isinstance(num_echoes, bool) or not isinstance(num_echoes, (int, np.integer)):
        raise ValueError("num_echoes must be a positive integer.")
    if num_echoes <= 0:
        raise ValueError("num_echoes must be a positive integer.")
    if not 0 < min_delay_seconds < max_delay_seconds:
        raise ValueError("Require 0 < min_delay_seconds < max_delay_seconds.")
    if min_path_separation_seconds <= 0:
        raise ValueError("min_path_separation_seconds must be positive.")
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
    minimum_separation = max(1, round(min_path_separation_seconds * sample_rate))
    peaks, _ = signal.find_peaks(
        residual_magnitude,
        distance=minimum_separation,
        prominence=prominence_ratio * magnitude[direct],
    )
    candidates = peaks[
        (lags[peaks] >= lags[direct] + minimum)
        & (lags[peaks] <= lags[direct] + maximum)
    ]
    if candidates.size < num_echoes:
        raise RuntimeError(
            f"Requested {num_echoes} echo(es), but only {candidates.size} credible delayed "
            "peak(s) remained after direct-path cancellation."
        )
    strongest = np.argsort(residual_magnitude[candidates])[-num_echoes:]
    echoes = np.sort(candidates[strongest])
    delays = (lags[echoes] - lags[direct]).astype(int)
    return MultipathDelayEstimate(
        delay_samples=delays,
        delay_seconds=delays / sample_rate,
        response=response,
        residual_response=residual_response,
        lags=lags,
        direct_index=direct,
        echo_indices=echoes,
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
