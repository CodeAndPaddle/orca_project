"""Small public API for the synthetic orca delay experiment."""

from .contour import Contour, ContourConfig, Extraction, extract_contours
from .signal import (
    DelayEstimate,
    ExperimentConfig,
    estimate_delay,
    synthetic_whistle,
    template_from_contour,
)

__all__ = [
    "Contour",
    "ContourConfig",
    "DelayEstimate",
    "ExperimentConfig",
    "Extraction",
    "estimate_delay",
    "extract_contours",
    "synthetic_whistle",
    "template_from_contour",
]
