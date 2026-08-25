"""Small public API for the synthetic orca delay experiment."""

from .contour import Contour, ContourConfig, Extraction, extract_contours
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
    "ExperimentConfig",
    "Extraction",
    "MultipathDelayEstimate",
    "estimate_delays",
    "extract_contours",
    "synthetic_whistle",
    "synthesize_multipath",
    "template_from_contour",
]
