from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
from scipy import signal
from scipy.io import wavfile

from orca import (
    ContourConfig,
    ExperimentConfig,
    estimate_delay,
    extract_contours,
    synthetic_whistle,
    template_from_contour,
)


FS = 48_000


def write_whistle(path: Path, stereo: bool = False):
    time, frequency, samples = synthetic_whistle(sample_rate=FS)
    pcm = np.int16(samples / abs(samples).max() * np.iinfo(np.int16).max)
    if stereo:
        pcm = np.column_stack((pcm, pcm // 2))
    wavfile.write(path, FS, pcm)
    return time, frequency, samples


def make_received(source, delay_samples, echo_gain=0.2, seed=7):
    impulse = np.zeros(delay_samples + 1)
    impulse[[0, delay_samples]] = [0.5, echo_gain]
    received = signal.convolve(source, impulse)
    return received + 0.01 * np.random.default_rng(seed).standard_normal(received.size)


@pytest.fixture(scope="module")
def extraction(tmp_path_factory):
    root = tmp_path_factory.mktemp("orca")
    wav_path = root / "whistle.wav"
    time, frequency, samples = write_whistle(wav_path)
    result = extract_contours(
        wav_path,
        [(0.0, 1.0)],
        root / "output",
        ContourConfig(figure_dpi=72),
    )
    return result, time, frequency, samples


def test_contour_accuracy_and_minimal_artifacts(extraction):
    result, target_time, target_frequency, _ = extraction
    contour = result.contours[0]
    assert contour.accepted
    assert np.all(np.diff(contour.time_seconds) > 0)
    assert np.all(np.isfinite(contour.path_scores))
    expected = np.interp(contour.time_seconds, target_time, target_frequency)
    assert np.sqrt(np.mean((contour.frequency_hz - expected) ** 2)) < 300

    files = sorted(
        path.relative_to(result.output_dir).as_posix()
        for path in result.output_dir.rglob("*")
        if path.is_file()
    )
    assert files == ["contours.npz", "figures/interval_0001.png"]
    with np.load(result.output_dir / "contours.npz", allow_pickle=False) as archive:
        assert archive["accepted"].tolist() == [True]
        assert archive["path_scores"].dtype.kind == "f"
        np.testing.assert_allclose(archive["path_scores"], contour.path_scores)
        assert archive["config_json"].dtype.kind == "U"


def test_ten_millisecond_sidelobe_regression(extraction):
    result, _, _, source = extraction
    template, fitted_frequency = template_from_contour(result.contours[0], FS)
    assert 3_500 < fitted_frequency.min() < 4_500
    assert 9_500 < fitted_frequency.max() < 10_500

    true_delay = 480
    received = make_received(source, true_delay)
    estimate = estimate_delay(received, template, FS)
    assert abs(estimate.delay_samples - true_delay) <= 2

    assert abs(estimate.delay_samples / FS - 0.010) <= 50e-6


@pytest.mark.parametrize("delay_ms", [3, 5, 7, 10, 15, 20, 25, 30])
@pytest.mark.parametrize("echo_gain", [0.10, 0.20])
@pytest.mark.parametrize("seed", [1, 7])
def test_delay_range_noise_and_gain(extraction, delay_ms, echo_gain, seed):
    result, _, _, source = extraction
    template, _ = template_from_contour(result.contours[0], FS)
    true_delay = round(delay_ms / 1_000 * FS)
    estimate = estimate_delay(
        make_received(source, true_delay, echo_gain, seed),
        template,
        FS,
    )
    assert abs(estimate.delay_samples - true_delay) <= 2


@pytest.mark.parametrize(
    "sample_rate,min_delay,max_delay,prominence",
    [
        (0, 0.002, 0.030, 0.10),
        (FS, 0.030, 0.030, 0.10),
        (FS, 0.002, 0.030, 0.0),
        (FS, 0.002, 0.030, 1.0),
    ],
)
def test_delay_parameter_validation(
    extraction, sample_rate, min_delay, max_delay, prominence
):
    _, _, _, source = extraction
    template = signal.hilbert(source)
    template /= np.linalg.norm(template)
    with pytest.raises(ValueError):
        estimate_delay(
            source,
            template,
            sample_rate,
            min_delay_seconds=min_delay,
            max_delay_seconds=max_delay,
            prominence_ratio=prominence,
        )


def test_direct_only_signal_has_no_echo(extraction):
    _, _, _, source = extraction
    template = signal.hilbert(source)
    template /= np.linalg.norm(template)
    with pytest.raises(RuntimeError, match="direct-path cancellation"):
        estimate_delay(source, template, FS)


def test_multiple_intervals_use_npz_offsets(extraction, tmp_path):
    result, _, _, _ = extraction
    multi = extract_contours(
        result.input_path,
        [(0.0, 0.5), (0.5, 1.0)],
        tmp_path / "multi",
        ContourConfig(figure_dpi=50),
    )
    assert all(contour.accepted for contour in multi.contours)
    with np.load(multi.output_dir / "contours.npz", allow_pickle=False) as archive:
        offsets = archive["contour_offsets"]
        assert offsets.size == 3
        assert np.all(np.diff(offsets) > 0)
        assert offsets[-1] == archive["time_seconds"].size


@pytest.mark.parametrize(
    "intervals",
    [None, [], [(0.5, 0.5)], [(2.0, 3.0)], [(float("nan"), 0.5)]],
)
def test_invalid_intervals_fail(tmp_path, intervals):
    wav_path = tmp_path / "whistle.wav"
    write_whistle(wav_path)
    with pytest.raises(ValueError):
        extract_contours(wav_path, intervals, tmp_path / "output")


def test_low_sample_rate_fails(tmp_path):
    path = tmp_path / "low-rate.wav"
    wavfile.write(path, 16_000, np.zeros(1_600, dtype=np.int16))
    with pytest.raises(ValueError, match="invalid"):
        extract_contours(path, [(0.0, 0.1)], tmp_path / "output")


def test_stereo_and_low_confidence_inputs(tmp_path):
    stereo = tmp_path / "stereo.wav"
    write_whistle(stereo, stereo=True)
    result = extract_contours(stereo, [(0.0, 1.0)], tmp_path / "stereo-output")
    assert result.contours[0].accepted

    quiet = tmp_path / "quiet.wav"
    wavfile.write(quiet, FS, np.zeros(FS // 4, dtype=np.int16))
    rejected = extract_contours(quiet, [(0.0, 0.25)], tmp_path / "quiet-output")
    assert not rejected.contours[0].accepted
    assert list((rejected.output_dir / "figures").glob("*.png")) == []


def test_signed_scores_are_valid_for_template(extraction):
    contour = extraction[0].contours[0]
    scores = contour.path_scores.copy()
    scores[0] = -2.5
    template, _ = template_from_contour(replace(contour, path_scores=scores), FS)
    assert np.isclose(np.linalg.norm(template), 1.0)


@pytest.mark.parametrize("sample_rate", [24_001, 32_000, 48_000, 96_000])
def test_stft_resolution_tracks_sample_rate(sample_rate):
    experiment = ExperimentConfig(sample_rate=sample_rate)
    contour_config = experiment.contour_config()
    window, nfft = contour_config.resolve_stft(sample_rate)
    overlap = round(contour_config.overlap_fraction * window)
    hop_seconds = (window - overlap) / sample_rate
    reference_hop_seconds = (256 - round(0.85 * 256)) / 48_000
    assert abs(window / sample_rate - 256 / 48_000) <= 1 / sample_rate
    assert abs(hop_seconds - reference_hop_seconds) <= 1 / sample_rate + np.finfo(float).eps
    assert nfft >= 16 * window
    assert nfft & (nfft - 1) == 0


@pytest.mark.parametrize("sample_rate", [24_001, 32_000, 48_000, 96_000])
@pytest.mark.parametrize("duration_seconds", [0.1, 0.2, 0.5, 1.0, 2.0])
def test_adaptive_rate_and_duration_grid(tmp_path, sample_rate, duration_seconds):
    experiment = ExperimentConfig(
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        true_delay_seconds=0.02,
        wav_path=tmp_path / "whistle.wav",
        output_dir=tmp_path / "output",
    )
    target_time, target_frequency, source = synthetic_whistle(
        duration_seconds=experiment.duration_seconds,
        sample_rate=experiment.sample_rate,
        start_hz=experiment.whistle_band_hz[0],
        end_hz=experiment.whistle_band_hz[1],
    )
    wavfile.write(
        experiment.wav_path,
        experiment.sample_rate,
        np.int16(source / abs(source).max() * np.iinfo(np.int16).max),
    )
    extraction = extract_contours(
        experiment.wav_path,
        [(0.0, experiment.duration_seconds)],
        experiment.output_dir,
        experiment.contour_config(figure_dpi=30),
    )
    contour = extraction.contours[0]
    assert contour.accepted
    expected = np.interp(contour.time_seconds, target_time, target_frequency)
    assert np.sqrt(np.mean((contour.frequency_hz - expected) ** 2)) < 500

    template, fitted = template_from_contour(
        contour,
        experiment.sample_rate,
        experiment.whistle_band_hz,
    )
    assert fitted[0] == pytest.approx(experiment.whistle_band_hz[0])
    assert fitted[-1] == pytest.approx(experiment.whistle_band_hz[1])
    estimate = estimate_delay(
        make_received(source, experiment.delay_samples),
        template,
        experiment.sample_rate,
        experiment.min_delay_seconds,
        experiment.max_delay_seconds,
    )
    assert (
        abs(estimate.delay_seconds - experiment.true_delay_seconds)
        <= experiment.delay_error_tolerance_seconds
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"sample_rate": 24_000},
        {"sample_rate": 96_001},
        {"duration_seconds": 0.09},
        {"duration_seconds": 2.01},
        {"noise_std": -0.01},
        {"sample_rate": 48_000.5},
        {"analysis_margin_hz": -1.0},
        {"sample_rate": 24_001, "whistle_band_hz": (4_000.0, 12_000.0)},
        {"min_delay_seconds": 0.03, "max_delay_seconds": 0.03},
        {"true_delay_seconds": 0.05},
        {"duration_seconds": 0.1, "true_delay_seconds": 0.1, "max_delay_seconds": 0.1},
    ],
)
def test_experiment_configuration_validation(overrides):
    with pytest.raises(ValueError):
        ExperimentConfig(**overrides)
