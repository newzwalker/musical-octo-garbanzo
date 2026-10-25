"""
shorts_loop_inspector - Deterministic audio/video seamless loop analyzer and beat-grid trimmer.
"""

from .analyzer import (
    LoopInspectionReport,
    analyze_audio_samples,
    analyze_media_file,
    detect_bpm_and_beats,
    find_optimal_loop_cut,
)

__version__ = "0.2.0"
__all__ = [
    "LoopInspectionReport",
    "analyze_audio_samples",
    "analyze_media_file",
    "detect_bpm_and_beats",
    "find_optimal_loop_cut",
]
