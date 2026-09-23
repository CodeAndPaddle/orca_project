"""Compatibility entry point backed by :mod:`Slant_delay_utills`."""

from __future__ import annotations

import sys
from pathlib import Path

from Slant_delay_utills import AnalysisConfig, WhistleDetection, detect_whistles


def extract_contours(
    path_to_wav: str | Path,
    min_freq: float = 5_000.0,
    max_freq: float = 20_000.0,
) -> tuple[WhistleDetection, ...]:
    """Return every accepted whistle; intervals are detected automatically."""
    return detect_whistles(
        path_to_wav,
        AnalysisConfig(whistle_band_hz=(min_freq, max_freq)),
    )


def extract_contour(
    path_to_wav: str | Path,
    min_freq: float = 5_000.0,
    max_freq: float = 20_000.0,
    n_fft: int | None = None,
    hop_length: int | None = None,
) -> WhistleDetection | None:
    """Legacy singular API: return the strongest accepted whistle, if any."""
    del n_fft, hop_length
    detections = extract_contours(path_to_wav, min_freq, max_freq)
    return max(
        detections,
        key=lambda item: item.detection_confidence * item.contour_confidence,
        default=None,
    )


def process_single(path_to_wav: str | Path) -> tuple[WhistleDetection, ...]:
    detections = extract_contours(path_to_wav)
    if not detections:
        print(f"{Path(path_to_wav).name}: NO WHISTLE DETECTED")
        return ()
    for index, item in enumerate(detections, start=1):
        print(
            f"{Path(path_to_wav).name} whistle {index}: "
            f"{item.start_seconds:.3f}-{item.end_seconds:.3f} s, "
            f"detection={item.detection_confidence:.2f}, "
            f"contour={item.contour_confidence:.2f}"
        )
    return detections


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python main.py path/to/file.wav")
    process_single(sys.argv[1])
