"""Unit tests for shorts-loop-inspector."""

import math
from pathlib import Path

import numpy as np

from shorts_loop_inspector.analyzer import (
    analyze_audio_samples,
    analyze_media_file,
    find_optimal_loop_cut,
)
from shorts_loop_inspector.cli import generate_demo_wav, main


def test_perfect_sine_loop_scores_grade_a():
    sr = 44100
    bpm = 120.0
    duration = 4.0  # Exactly 8 beats at 120 BPM
    n = int(sr * duration)
    t = np.arange(n) / float(sr)
    # Exact integer number of cycles (440 Hz * 4.0s = 1760 cycles)
    samples = 0.5 * np.sin(2.0 * math.pi * 440.0 * t)

    report = analyze_audio_samples(samples, sr, file_path="perfect.wav")
    assert report.loop_score >= 92.0
    assert report.grade == "A"
    assert report.is_seamless is True
    assert report.boundary_jump_db < -35.0


def test_glitched_loop_detects_boundary_discontinuity():
    sr = 44100
    n = sr * 2
    t = np.arange(n) / float(sr)
    samples = 0.2 * np.sin(2.0 * math.pi * 220.0 * t)
    # Create severe amplitude jump at seam and loud tail burst
    samples[0] = -0.95
    samples[-1] = 0.95
    samples[-2000:] = 0.9

    report = analyze_audio_samples(samples, sr, file_path="glitched.wav")
    assert report.loop_score < 75.0
    assert report.is_seamless is False
    assert any("click/pop" in d for d in report.diagnostics)


def test_optimal_zero_crossing_finder():
    sr = 44100
    t = np.arange(44100) / float(sr)
    # Phase-shifted sine wave that starts at non-zero value
    samples = 0.6 * np.sin(2.0 * math.pi * 100.0 * t + 1.1)
    start_idx, end_idx = find_optimal_loop_cut(samples, sr)
    assert 0 < start_idx < 1000
    assert 43100 < end_idx <= 44100
    assert abs(samples[start_idx]) < 0.05


def test_cli_demo_generation(tmp_path: Path):
    exit_code = main(["--generate-demo", str(tmp_path)])
    assert exit_code == 0
    seamless_wav = tmp_path / "seamless_128bpm.wav"
    glitched_wav = tmp_path / "glitched_cut.wav"
    assert seamless_wav.exists()
    assert glitched_wav.exists()

    rep_good = analyze_media_file(seamless_wav)
    rep_bad = analyze_media_file(glitched_wav)
    assert rep_good.loop_score > rep_bad.loop_score


if __name__ == "__main__":
    import tempfile

    test_perfect_sine_loop_scores_grade_a()
    test_glitched_loop_detects_boundary_discontinuity()
    test_optimal_zero_crossing_finder()
    with tempfile.TemporaryDirectory() as td:
        test_cli_demo_generation(Path(td))
    print("ALL 4 TEST SUITES PASSED DETERMINISTICALLY!")

