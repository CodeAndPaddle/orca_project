from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
from scipy import signal
from scipy.io import wavfile

from birdcall_contour_bundle import (
    Contour,
    birdcall_contour_default_config,
    extract_birdcall_contours,
    load_contours_npz,
)
from birdcall_contour_bundle.contour_extractor import _viterbi_dp


FS = 48_000
DURATION_S = 1.0


def make_synthetic_whistle(path: Path, stereo: bool = False) -> tuple[np.ndarray, np.ndarray]:
    sample_count = round(DURATION_S * FS)
    time_seconds = np.arange(sample_count) / FS
    x = np.linspace(-1.0, 1.0, sample_count)
    shape = np.tan(x) - np.sin(x) + 1
    normalized = (shape - shape.min()) / np.ptp(shape)
    frequency_hz = 4_000 + 6_000 * normalized
    phase = 2 * np.pi * np.concatenate(([0.0], np.cumsum(frequency_hz[:-1]) / FS))
    source = signal.windows.tukey(sample_count, alpha=0.25) * np.sin(phase)
    pcm = np.int16(source / np.max(np.abs(source)) * np.iinfo(np.int16).max)
    if stereo:
        pcm = np.column_stack((pcm, np.int16(pcm * 0.5)))
    wavfile.write(path, FS, pcm)
    return time_seconds, frequency_hz


@pytest.fixture(scope="module")
def synthetic_result(tmp_path_factory):
    folder = tmp_path_factory.mktemp("synthetic_contour")
    wav_path = folder / "synthetic.wav"
    target_time, target_frequency = make_synthetic_whistle(wav_path)
    cfg = birdcall_contour_default_config()
    cfg.output.dir = str(folder / "artifacts")
    cfg.output.prefix = "test"
    cfg.debug.verbose = False
    cfg.figures.resolution_dpi = 80
    results = extract_birdcall_contours(wav_path, [(0.0, DURATION_S)], cfg)
    return results, target_time, target_frequency


def test_synthetic_contour_and_artifacts(synthetic_result):
    results, target_time, target_frequency = synthetic_result
    assert len(results.contours) == 1
    assert results.interval_statuses[0].status == "accepted"
    contour = results.contours[0]
    assert contour.time_sec_abs.size > 100
    assert np.all(np.isfinite(contour.time_sec_abs))
    assert np.all(np.diff(contour.time_sec_abs) > 0)
    assert np.all(np.isfinite(contour.path_scores))
    assert np.all(contour.freq_hz > 0)
    assert np.all(contour.freq_hz < FS / 2)

    expected = np.interp(contour.time_sec_abs, target_time, target_frequency)
    rmse_hz = np.sqrt(np.mean((contour.freq_hz - expected) ** 2))
    assert rmse_hz < 300
    assert abs(contour.f_start_hz - expected[0]) < 600
    assert abs(contour.f_end_hz - expected[-1]) < 600

    paths = [
        results.summary_csv_path,
        results.points_csv_path,
        results.interval_csv_path,
        results.npz_path,
        results.metadata_json_path,
        Path(results.output_dir) / "figures" / "test_interval0001.png",
    ]
    assert all(Path(path).is_file() for path in paths)


def test_npz_round_trip_preserves_signed_scores(synthetic_result, tmp_path):
    results, _, _ = synthetic_result
    original = load_contours_npz(results.npz_path)
    modified_path = tmp_path / "signed_scores.npz"
    modified = dict(original)
    modified["path_scores"] = original["path_scores"].copy()
    modified["path_scores"][0] = -2.5
    np.savez_compressed(modified_path, **modified)
    loaded = load_contours_npz(modified_path)
    assert loaded["path_scores"][0] == pytest.approx(-2.5)
    assert loaded["contour_offsets"][-1] == loaded["time_sec_abs"].size


def test_npz_offsets_support_multiple_intervals(synthetic_result, tmp_path):
    source_results, _, _ = synthetic_result
    cfg = birdcall_contour_default_config()
    cfg.output.dir = str(tmp_path / "multi_artifacts")
    cfg.output.prefix = "multi"
    cfg.figures.save_png = False
    cfg.debug.verbose = False
    results = extract_birdcall_contours(
        source_results.wav_file,
        [(0.0, 0.5), (0.5, 1.0)],
        cfg,
    )
    assert len(results.contours) == 2
    archive = load_contours_npz(results.npz_path)
    assert archive["contour_offsets"].size == 3
    assert np.all(np.diff(archive["contour_offsets"]) > 0)
    assert archive["contour_interval_index"].tolist() == [1, 2]


@pytest.mark.parametrize(
    "intervals",
    [None, [], [(0.5, 0.5)], [(2.0, 3.0)], [(float("nan"), 0.5)]],
)
def test_invalid_intervals_fail(tmp_path, intervals):
    wav_path = tmp_path / "valid.wav"
    make_synthetic_whistle(wav_path)
    cfg = birdcall_contour_default_config()
    cfg.debug.verbose = False
    with pytest.raises(ValueError):
        extract_birdcall_contours(wav_path, intervals, cfg)


def test_low_sample_rate_is_rejected(tmp_path):
    wav_path = tmp_path / "low_rate.wav"
    wavfile.write(wav_path, 16_000, np.zeros(1_600, dtype=np.int16))
    cfg = birdcall_contour_default_config()
    cfg.debug.verbose = False
    with pytest.raises(ValueError, match="too low"):
        extract_birdcall_contours(wav_path, [(0.0, 0.1)], cfg)


def test_stereo_mean_channel_extracts(tmp_path):
    wav_path = tmp_path / "stereo.wav"
    make_synthetic_whistle(wav_path, stereo=True)
    cfg = birdcall_contour_default_config()
    cfg.output.dir = str(tmp_path / "stereo_artifacts")
    cfg.output.write_csv = False
    cfg.output.save_npz = False
    cfg.output.save_metadata_json = False
    cfg.figures.save_png = False
    cfg.debug.verbose = False
    results = extract_birdcall_contours(wav_path, [(0.0, DURATION_S)], cfg)
    assert len(results.contours) == 1


def test_low_confidence_interval_is_rejected(tmp_path):
    wav_path = tmp_path / "quiet.wav"
    wavfile.write(wav_path, FS, np.zeros(FS // 4, dtype=np.int16))
    cfg = birdcall_contour_default_config()
    cfg.output.dir = str(tmp_path / "quiet_artifacts")
    cfg.figures.save_png = False
    cfg.debug.verbose = False
    results = extract_birdcall_contours(wav_path, [(0.0, 0.25)], cfg)
    assert results.contours == []
    assert results.interval_statuses[0].status == "rejected_low_confidence"


def test_viterbi_respects_slope_limit():
    cfg = birdcall_contour_default_config()
    cfg.viterbi.max_slope_hz_per_sec = 1_000
    frequencies = np.arange(0, 1_000, 100.0)
    times = np.arange(5) * 0.01
    emission = np.zeros((frequencies.size, times.size))
    emission[1, 0] = 10
    emission[8, 1:] = 10
    rows, _ = _viterbi_dp(emission, frequencies, times, cfg)
    max_jump_bins = int(np.ceil(cfg.viterbi.max_slope_hz_per_sec * 0.01 / 100))
    assert np.all(np.abs(np.diff(rows)) <= max_jump_bins)


def test_matlab_golden_frequency_parity(synthetic_result):
    h5py = pytest.importorskip("h5py")
    golden_path = Path(__file__).parents[1] / "synthetic_dolphin_contours1.mat"
    if not golden_path.is_file():
        pytest.skip("MATLAB golden fixture is not present")
    with h5py.File(golden_path, "r") as matlab_file:
        matlab_contour = matlab_file["results"]["contours"]
        matlab_time = np.asarray(matlab_contour["timeSecAbs"], dtype=float).reshape(-1)
        matlab_frequency = np.asarray(matlab_contour["freqHz"], dtype=float).reshape(-1)
    python_contour: Contour = synthetic_result[0].contours[0]
    overlap = (
        (python_contour.time_sec_abs >= matlab_time.min())
        & (python_contour.time_sec_abs <= matlab_time.max())
    )
    assert np.count_nonzero(overlap) > 100
    golden_frequency = np.interp(python_contour.time_sec_abs[overlap], matlab_time, matlab_frequency)
    rmse_hz = np.sqrt(np.mean((python_contour.freq_hz[overlap] - golden_frequency) ** 2))
    # The archived MATLAB fixture used the earlier background-heavy weights,
    # which wander in the whistle middle. Keep it as a broad regression check;
    # the stricter ground-truth test above governs extraction accuracy.
    assert rmse_hz < 650
