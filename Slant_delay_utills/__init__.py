"""Whistle contour extraction and slant-delay analysis."""

from .contour import (
    Contour,
    ContourConfig,
    Extraction,
    extract_contours,
    extract_contours_from_samples,
)
from .pipeline import (
    AnalysisConfig,
    RecordingAnalysis,
    WhistleDetection,
    WhistleResult,
    analyze_detections,
    analyze_recording,
    detect_whistles,
)
from .signal import (
    ExperimentConfig,
    MultipathDelayEstimate,
    estimate_delays,
    synthetic_whistle,
    synthesize_multipath,
    template_from_contour,
)

__all__ = [
    "Contour",
    "ContourConfig",
    "AnalysisConfig",
    "ExperimentConfig",
    "Extraction",
    "MultipathDelayEstimate",
    "RecordingAnalysis",
    "WhistleDetection",
    "WhistleResult",
    "analyze_detections",
    "analyze_recording",
    "detect_whistles",
    "estimate_delays",
    "extract_contours",
    "extract_contours_from_samples",
    "synthetic_whistle",
    "synthesize_multipath",
    "template_from_contour",
]
