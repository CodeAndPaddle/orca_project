from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
from scipy import signal
from scipy.io import wavfile

from Slant_delay_utills import (
    AnalysisConfig,
    ContourConfig,
    ExperimentConfig,
    estimate_delays,
    extract_contours,
    synthetic_whistle,
    synthesize_multipath,
    template_from_contour,
    analyze_detections,
    analyze_recording,
    detect_whistles,
)


FS = 48_000


def write_whistle(path: Path, stereo: bool = False):
    time, frequency, samples = synthetic_whistle(sample_rate=FS)
    pcm = np.int16(samples / abs(samples).max() * np.iinfo(np.int16).max)
    if stereo:
        pcm = np.column_stack((pcm, pcm // 2))
    wavfile.write(path, FS, pcm)
    return time, frequency, samples


def make_received(source, delay_samples, echo_gains=0.2, seed=7):
    delays = np.atleast_1d(delay_samples)
    gains = np.broadcast_to(echo_gains, delays.shape)
    received, _ = synthesize_multipath(
        source,
        delays,
        gains,
        noise_std=0.01,
        seed=seed,
    )
    return received


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
    template, fitted_frequency = template_from_contour(
        result.contours[0], FS, reference_samples=source
    )
    assert 3_500 < fitted_frequency.min() < 4_500
    assert 9_500 < fitted_frequency.max() < 10_500

    true_delay = 480
    received = make_received(source, true_delay)
    estimate = estimate_delays(received, template, FS, num_echoes=1)
    assert abs(estimate.delay_samples[0] - true_delay) <= 2

    assert abs(estimate.delay_samples[0] / FS - 0.010) <= 50e-6


def test_synthesize_multipath_impulse_and_deterministic_noise():
    source = np.array([1.0, -0.5, 0.25])
    delays = np.array([2, 4])
    gains = np.array([0.2, 0.1])
    noiseless, impulse = synthesize_multipath(source, delays, gains)

    np.testing.assert_allclose(impulse, [0.5, 0.0, 0.2, 0.0, 0.1])
    np.testing.assert_allclose(noiseless, signal.convolve(source, impulse))
    first, _ = synthesize_multipath(source, delays, gains, noise_std=0.01, seed=19)
    second, _ = synthesize_multipath(source, delays, gains, noise_std=0.01, seed=19)
    np.testing.assert_array_equal(first, second)
    assert first.size == source.size + delays[-1]


def test_experiment_normalizes_multipath_sequences():
    experiment = ExperimentConfig(
        sample_rate=48_000,
        true_delay_seconds=[0.010, 0.012],
        echo_gains=[0.20, 0.15],
    )
    assert experiment.true_delay_seconds == (0.010, 0.012)
    assert experiment.echo_gains == (0.20, 0.15)
    np.testing.assert_array_equal(experiment.delay_samples, [480, 576])


@pytest.mark.parametrize(
    "delays,gains",
    [
        ([], []),
        ([2.5], [0.2]),
        ([4, 2], [0.2, 0.1]),
        ([2, 2], [0.2, 0.1]),
        ([2, 4], [0.2]),
        ([2], [0.5]),
        ([2], [float("nan")]),
    ],
)
def test_synthesize_multipath_validation(delays, gains):
    with pytest.raises(ValueError):
        synthesize_multipath(np.ones(8), delays, gains)


@pytest.mark.parametrize(
    "delay_seconds,echo_gains",
    [
        ([0.010], [0.20]),
        ([0.010, 0.020], [0.20, 0.15]),
        ([0.005, 0.015, 0.025], [0.20, 0.15, 0.10]),
    ],
)
def test_detects_arbitrary_echo_count_in_arrival_order(
    extraction, delay_seconds, echo_gains
):
    result, _, _, source = extraction
    template, _ = template_from_contour(
        result.contours[0], FS, reference_samples=source
    )
    delays = np.rint(np.asarray(delay_seconds) * FS).astype(int)
    estimate = estimate_delays(
        make_received(source, delays, echo_gains),
        template,
        FS,
        num_echoes=len(delays),
    )

    np.testing.assert_allclose(estimate.delay_samples, delays, atol=2, rtol=0)
    assert np.all(np.diff(estimate.delay_samples) > 0)
    assert np.all(np.diff(estimate.echo_indices) > 0)


def test_arrival_order_is_independent_of_echo_strength(extraction):
    result, _, _, source = extraction
    template, _ = template_from_contour(
        result.contours[0], FS, reference_samples=source
    )
    delays = np.array([round(0.010 * FS), round(0.020 * FS)])
    estimate = estimate_delays(
        make_received(source, delays, [0.10, 0.20]),
        template,
        FS,
        num_echoes=2,
    )
    np.testing.assert_allclose(estimate.delay_samples, delays, atol=2, rtol=0)


def test_insufficient_echoes_raise(extraction):
    result, _, _, source = extraction
    template, _ = template_from_contour(
        result.contours[0], FS, reference_samples=source
    )
    with pytest.raises(RuntimeError, match="Requested 2 echo"):
        estimate_delays(
            make_received(source, round(0.010 * FS)),
            template,
            FS,
            num_echoes=2,
        )


@pytest.mark.parametrize("delay_ms", [3, 5, 7, 10, 15, 20, 25, 30])
@pytest.mark.parametrize("echo_gain", [0.10, 0.20])
@pytest.mark.parametrize("seed", [1, 7])
def test_delay_range_noise_and_gain(extraction, delay_ms, echo_gain, seed):
    result, _, _, source = extraction
    template, _ = template_from_contour(
        result.contours[0], FS, reference_samples=source
    )
    true_delay = round(delay_ms / 1_000 * FS)
    estimate = estimate_delays(
        make_received(source, true_delay, echo_gain, seed),
        template,
        FS,
        num_echoes=1,
    )
    assert abs(estimate.delay_samples[0] - true_delay) <= 2


@pytest.mark.parametrize(
    "sample_rate,num_echoes,min_delay,max_delay,min_separation,prominence",
    [
        (0, 1, 0.002, 0.030, 0.002, 0.10),
        (FS, 0, 0.002, 0.030, 0.002, 0.10),
        (FS, 1.5, 0.002, 0.030, 0.002, 0.10),
        (FS, 1, 0.030, 0.030, 0.002, 0.10),
        (FS, 1, 0.002, 0.030, 0.0, 0.10),
        (FS, 1, 0.002, 0.030, 0.002, 0.0),
        (FS, 1, 0.002, 0.030, 0.002, 1.0),
    ],
)
def test_delay_parameter_validation(
    extraction, sample_rate, num_echoes, min_delay, max_delay, min_separation, prominence
):
    _, _, _, source = extraction
    template = signal.hilbert(source)
    template /= np.linalg.norm(template)
    with pytest.raises(ValueError):
        estimate_delays(
            source,
            template,
            sample_rate,
            num_echoes=num_echoes,
            min_delay_seconds=min_delay,
            max_delay_seconds=max_delay,
            min_path_separation_seconds=min_separation,
            prominence_ratio=prominence,
        )


def test_direct_only_signal_has_no_echo(extraction):
    _, _, _, source = extraction
    template = signal.hilbert(source)
    template /= np.linalg.norm(template)
    with pytest.raises(RuntimeError, match="direct-path cancellation"):
        estimate_delays(source, template, FS, num_echoes=1)


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
        true_delay_seconds=[0.02],
        echo_gains=[0.2],
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
        reference_samples=source,
    )
    observed = contour.observed_mask
    fitted_time = contour.time_seconds[observed][0] + np.arange(fitted.size) / sample_rate
    expected_fitted = np.interp(fitted_time, target_time, target_frequency)
    assert np.sqrt(np.mean((fitted - expected_fitted) ** 2)) < 500
    estimate = estimate_delays(
        make_received(source, experiment.delay_samples),
        template,
        experiment.sample_rate,
        num_echoes=len(experiment.true_delay_seconds),
        min_delay_seconds=experiment.min_delay_seconds,
        max_delay_seconds=experiment.max_delay_seconds,
        min_path_separation_seconds=experiment.min_path_separation_seconds,
    )
    errors = abs(estimate.delay_seconds - np.asarray(experiment.true_delay_seconds))
    assert np.all(errors <= experiment.delay_error_tolerance_seconds)


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
        {"true_delay_seconds": []},
        {"true_delay_seconds": [0.01], "echo_gains": [0.2, 0.1]},
        {"true_delay_seconds": [0.02, 0.01]},
        {"true_delay_seconds": [0.01, 0.01]},
        {"true_delay_seconds": [0.01, 0.011]},
        {"true_delay_seconds": [0.01, float("nan")]},
        {"true_delay_seconds": [0.01, 0.05]},
        {"echo_gains": [0.2, 0.5]},
        {"direct_gain": 0.0},
        {"min_path_separation_seconds": 0.0},
        {
            "duration_seconds": 0.1,
            "true_delay_seconds": [0.1],
            "echo_gains": [0.2],
            "max_delay_seconds": 0.1,
        },
    ],
)
def test_experiment_configuration_validation(overrides):
    with pytest.raises(ValueError):
        ExperimentConfig(**overrides)


def write_unknown_time_recording(
    path: Path,
    *,
    delays: tuple[float, ...] = (0.010, 0.020),
    starts: tuple[float, ...] = (0.40,),
    seed: int = 11,
):
    rng = np.random.default_rng(seed)
    total = np.zeros(round(2.2 * FS))
    source_time, source_frequency, source = synthetic_whistle(
        duration_seconds=0.4,
        sample_rate=FS,
        start_hz=4_000,
        end_hz=10_000,
    )
    delay_samples = np.rint(np.asarray(delays) * FS).astype(int)
    gains = np.linspace(0.20, 0.10, len(delays))
    for start in starts:
        if delays:
            received, _ = synthesize_multipath(
                source,
                delay_samples,
                gains,
                noise_std=0,
            )
        else:
            received = 0.5 * source
        first = round(start * FS)
        total[first : first + received.size] += received
    total += 0.002 * rng.standard_normal(total.size)
    pcm = np.int16(total / max(abs(total).max(), 1e-12) * np.iinfo(np.int16).max)
    wavfile.write(path, FS, pcm)
    return source_time, source_frequency


def test_detects_all_unknown_time_whistles(tmp_path):
    wav_path = tmp_path / "unknown.wav"
    write_unknown_time_recording(wav_path, starts=(0.35, 1.25))
    config = AnalysisConfig(
        whistle_band_hz=(3_500, 10_500),
        output_dir=tmp_path / "analysis",
    )
    detections = detect_whistles(wav_path, config)
    assert len(detections) == 2
    np.testing.assert_allclose(
        [item.start_seconds for item in detections], [0.35, 1.25], atol=0.05
    )
    assert all(item.end_seconds > item.start_seconds for item in detections)
    assert all(item.observed_mask.mean() >= 0.80 for item in detections)
    staged = analyze_detections(wav_path, config, detections)
    assert len(staged.whistles) == 2
    for result in staged.whistles:
        np.testing.assert_allclose(
            result.delay_seconds, [0.010, 0.020], atol=2 / FS, rtol=0
        )


def test_recording_pipeline_estimates_unknown_time_delays(tmp_path):
    wav_path = tmp_path / "unknown-delay.wav"
    write_unknown_time_recording(wav_path)
    config = AnalysisConfig(
        whistle_band_hz=(3_500, 10_500),
        output_dir=tmp_path / "analysis",
    )
    result = analyze_recording(wav_path, config)
    assert len(result.whistles) == 1
    np.testing.assert_allclose(
        result.whistles[0].delay_seconds, [0.010, 0.020], atol=2 / FS, rtol=0
    )
    assert (result.output_dir / "analysis.npz").is_file()
    with np.load(result.output_dir / "analysis.npz", allow_pickle=False) as archive:
        assert archive["schema_version"] == 1
        assert archive["intervals"].shape == (1, 2)


def test_staged_detection_analysis_matches_convenience_pipeline(tmp_path):
    wav_path = tmp_path / "staged-delay.wav"
    write_unknown_time_recording(wav_path)
    detection_config = AnalysisConfig(
        whistle_band_hz=(3_500, 10_500),
        output_dir=tmp_path / "staged",
    )
    detections = detect_whistles(wav_path, detection_config)
    staged = analyze_detections(wav_path, detection_config, detections)
    automatic = analyze_recording(
        wav_path,
        replace(detection_config, output_dir=tmp_path / "automatic"),
    )
    assert len(staged.whistles) == len(automatic.whistles) == 1
    np.testing.assert_array_equal(
        staged.whistles[0].delay_samples,
        automatic.whistles[0].delay_samples,
    )
    np.testing.assert_allclose(
        staged.whistles[0].response,
        automatic.whistles[0].response,
    )
    np.testing.assert_array_equal(
        staged.whistles[0].echo_indices,
        automatic.whistles[0].echo_indices,
    )


def test_staged_analysis_supports_empty_detections(tmp_path):
    wav_path = tmp_path / "empty-detections.wav"
    wavfile.write(wav_path, FS, np.zeros(FS, dtype=np.int16))
    config = AnalysisConfig(
        whistle_band_hz=(3_500, 10_500),
        output_dir=tmp_path / "analysis",
    )
    result = analyze_detections(wav_path, config, ())
    assert result.whistles == ()
    assert (result.output_dir / "analysis.npz").is_file()


def test_staged_analysis_rejects_invalid_detections(tmp_path):
    wav_path = tmp_path / "invalid-detections.wav"
    write_unknown_time_recording(wav_path, starts=(0.35, 1.25))
    config = AnalysisConfig(
        whistle_band_hz=(3_500, 10_500),
        output_dir=tmp_path / "analysis",
    )
    detections = detect_whistles(wav_path, config)
    assert len(detections) == 2

    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        analyze_detections(wav_path, config, detections[::-1])
    with pytest.raises(ValueError, match="non-finite metadata"):
        analyze_detections(
            wav_path,
            config,
            (replace(detections[0], start_seconds=float("nan")),),
        )
    with pytest.raises(ValueError, match="aligned one-dimensional"):
        analyze_detections(
            wav_path,
            config,
            (replace(detections[0], frequency_hz=detections[0].frequency_hz[:-1]),),
        )
    with pytest.raises(ValueError, match="outside whistle_band_hz"):
        analyze_detections(
            wav_path,
            config,
            (replace(detections[0], frequency_hz=np.zeros_like(detections[0].frequency_hz)),),
        )


def test_unknown_time_pipeline_rejects_silence(tmp_path):
    wav_path = tmp_path / "silence.wav"
    wavfile.write(wav_path, FS, np.zeros(FS, dtype=np.int16))
    config = AnalysisConfig(
        whistle_band_hz=(3_500, 10_500),
        output_dir=tmp_path / "analysis",
    )
    assert detect_whistles(wav_path, config) == ()
    result = analyze_recording(wav_path, config)
    assert result.whistles == ()


def test_unknown_time_pipeline_direct_only_has_no_delay(tmp_path):
    wav_path = tmp_path / "direct-only.wav"
    write_unknown_time_recording(wav_path, delays=())
    config = AnalysisConfig(
        whistle_band_hz=(3_500, 10_500),
        output_dir=tmp_path / "analysis",
    )
    result = analyze_recording(wav_path, config)
    assert len(result.whistles) == 1
    assert result.whistles[0].delay_seconds.size == 0
    assert result.whistles[0].warnings


def test_unknown_time_pipeline_automatic_three_echoes(tmp_path):
    wav_path = tmp_path / "three-echoes.wav"
    write_unknown_time_recording(wav_path, delays=(0.005, 0.015, 0.025))
    config = AnalysisConfig(
        whistle_band_hz=(3_500, 10_500),
        output_dir=tmp_path / "analysis",
    )
    result = analyze_recording(wav_path, config)
    np.testing.assert_allclose(
        result.whistles[0].delay_seconds,
        [0.005, 0.015, 0.025],
        atol=2 / FS,
        rtol=0,
    )
