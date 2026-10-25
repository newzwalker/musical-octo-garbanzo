"""Command-line interface for shorts-loop-inspector."""

from __future__ import annotations

import argparse
import math
import sys
import wave
from pathlib import Path

import numpy as np

from .analyzer import analyze_media_file


def generate_demo_wav(output_path: Path, seamless: bool = True, bpm: float = 128.0) -> Path:
    """Generate a synthetic test loop WAV file (either seamless or with a boundary pop)."""
    sr = 44100
    beats = 8
    duration = (60.0 / bpm) * beats
    n_samples = int(sr * duration)
    t = np.arange(n_samples) / float(sr)

    # Fundamental bass + harmonic locked to exact loop cycle
    cycles = max(1, round(110.0 * duration))
    freq = cycles / duration
    signal = 0.45 * np.sin(2.0 * math.pi * freq * t)

    # Beat pulse envelope locked to BPM
    beat_freq = bpm / 60.0
    pulse = 0.5 + 0.5 * np.cos(2.0 * math.pi * beat_freq * t)
    signal = signal * (0.6 + 0.4 * pulse)

    if not seamless:
        # Inject boundary DC offset and tail spike to simulate a bad cut
        signal[-120:] = 0.85
        signal[:50] = -0.75

    pcm = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return output_path


def _print_pretty_report(report) -> None:
    status_badge = "PASS (SEAMLESS)" if report.is_seamless else "WARN (SEAM ARTIFACTS)"
    print("=" * 68)
    print(f" Shorts Loop Inspector v0.2.0 | {status_badge}")
    print("=" * 68)
    print(f" File              : {report.file_path}")
    print(f" Duration / SR     : {report.duration_sec:.4f}s @ {report.sample_rate} Hz ({report.channels} ch)")
    print(f" Loop Quality Score: {report.loop_score:.1f} / 100.0  [Grade: {report.grade}]")
    print(f" Boundary Jump     : {report.boundary_jump_db:.2f} dBFS")
    print(f" Tail/Head RMS Diff: {report.rms_delta_db:.2f} dB")
    print(f" Spectral Distance : {report.spectral_flux_delta:.4f}")
    print(f" Estimated Tempo   : {report.estimated_bpm:.1f} BPM (drift: {report.beat_alignment_error_ms:.1f} ms)")
    print(
        f" Optimal Zero-Cross: start={report.recommended_trim_start_sec:.4f}s, "
        f"end={report.recommended_trim_end_sec:.4f}s"
    )
    if report.video_frame_mse is not None:
        print(f" Video Frame Diff  : {report.video_frame_mse:.2f} RMSE (seamless={report.video_loop_seamless})")
    print("-" * 68)
    print(" Diagnostics:")
    for item in report.diagnostics:
        print(f"  - {item}")
    print("-" * 68)
    print(" Recommended FFmpeg Auto-Fix:")
    print(f"  {report.ffmpeg_fix_command}")
    print("=" * 68)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="shorts-loop-inspector",
        description="Inspect audio/video clips for seamless loop continuity, clicks/pops, and BPM grid alignment.",
    )
    parser.add_argument("input", nargs="?", help="Path to audio/video file (.wav, .mp3, .mp4, .mov)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON report")
    parser.add_argument(
        "--generate-demo",
        metavar="DIR",
        help="Generate sample seamless & glitched WAV files in DIR and inspect them",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help="Exit with code 1 if loop score is below this threshold (useful for CI/CD gates)",
    )

    args = parser.parse_args(argv)

    if args.generate_demo:
        demo_dir = Path(args.generate_demo)
        good_wav = generate_demo_wav(demo_dir / "seamless_128bpm.wav", seamless=True, bpm=128.0)
        bad_wav = generate_demo_wav(demo_dir / "glitched_cut.wav", seamless=False, bpm=128.0)
        for wav_file in (good_wav, bad_wav):
            rep = analyze_media_file(wav_file)
            if args.json:
                print(rep.to_json())
            else:
                _print_pretty_report(rep)
        return 0

    if not args.input:
        parser.print_help()
        return 1

    report = analyze_media_file(args.input)
    if args.json:
        print(report.to_json())
    else:
        _print_pretty_report(report)

    if args.fail_under > 0 and report.loop_score < args.fail_under:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
