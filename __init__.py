"""
Unified Ice Hockey Player Tracking Pipeline

A production-ready pipeline integrating detection, tracking, mask propagation,
jersey recognition, and team classification for ice hockey video analysis.
"""

from .config import PipelineConfig, load_config
from .data_types import Detection, Track, TrackState, FrameResult
from .pipeline import UnifiedPipeline

__version__ = "1.0.0"
__all__ = [
    "PipelineConfig",
    "load_config",
    "Detection",
    "Track",
    "TrackState",
    "FrameResult",
    "UnifiedPipeline",
]
