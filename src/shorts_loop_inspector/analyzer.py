"""Core DSP and frame boundary analysis for seamless short-form loops."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LoopInspectionReport:
    """Structured QA report for a loop candidate."""

    file_path: str
    sample_rate: int
    duration_sec: float
    channels: int
    loop_score: float
    grade: str
    is_seamless: bool
    boundary_jump_db: float
    rms_delta_db: float
    spectral_flux_delta: float
    zero_crossing_offset_samples: int
    estimated_bpm: float
    beat_alignment_error_ms: float
    recommended_trim_start_sec: float
    recommended_trim_end_sec: float
    ffmpeg_fix_command: str
    video_frame_mse: Optional[float] = None
    video_loop_seamless: Optional[bool] = None
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _amp_to_db(val: float, floor_db: float = -96.0) -> float:
    if val <= 1e-9:
        return floor_db
    return max(floor_db, 20.0 * math.log10(val))


def detect_bpm_and_beats(samples: np.ndarray, sample_rate: int) -> Tuple[float, List[float]]:
    """
    Estimate tempo (BPM) and beat timestamps using onset envelope autocorrelation.
    Pure NumPy implementation with zero heavy native dependencies.
    """
    if samples.ndim > 1:
        mono = np.mean(samples, axis=1)
    else:
        mono = samples.astype(np.float64)

    hop = max(1, int(sample_rate * 0.01))  # 10ms hop
    n_frames = len(mono) // hop
    if n_frames < 20:
        return 120.0, []

    framed = mono[: n_frames * hop].reshape(n_frames, hop)
    energy = np.sqrt(np.mean(framed * framed, axis=1))
    onset_env = np.maximum(0.0, np.diff(energy, prepend=energy[0]))

    if np.max(onset_env) <= 1e-8 or float(np.std(energy)) / (float(np.mean(energy)) + 1e-9) < 0.05:
        return 120.0, []

    onset_env = (onset_env - np.mean(onset_env)) / (np.std(onset_env) + 1e-9)
    autocorr = np.correlate(onset_env, onset_env, mode="full")[len(onset_env) - 1 :]

    env_sr = sample_rate / hop
    min_lag = int(env_sr * 60.0 / 180.0)  # 180 BPM max
    max_lag = min(len(autocorr) - 1, int(env_sr * 60.0 / 70.0))  # 70 BPM min

    if max_lag <= min_lag:
        return 120.0, []

    search_region = autocorr[min_lag : max_lag + 1]
    best_lag_idx = int(np.argmax(search_region))
    best_lag = float(min_lag + best_lag_idx)
    if 0 < best_lag_idx < len(search_region) - 1:
        y0 = float(search_region[best_lag_idx - 1])
        y1 = float(search_region[best_lag_idx])
        y2 = float(search_region[best_lag_idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-9:
            best_lag += 0.5 * (y0 - y2) / denom

    bpm = round(float(60.0 * env_sr / best_lag), 1)

    beat_period_sec = 60.0 / bpm
    duration_sec = len(mono) / sample_rate
    # Find phase offset with strongest onset alignment
    phase_steps = max(1, int(round(best_lag)))
    best_phase = 0.0
    best_score = -1.0
    for step in range(phase_steps):
        phase_sec = (step / env_sr) % beat_period_sec
        beat_indices = np.arange(phase_sec * env_sr, len(onset_env), best_lag).astype(int)
        beat_indices = beat_indices[beat_indices < len(onset_env)]
        if len(beat_indices) > 0:
            score = float(np.mean(onset_env[beat_indices]))
            if score > best_score:
                best_score = score
                best_phase = phase_sec

    beats = [round(float(t), 4) for t in np.arange(best_phase, duration_sec, beat_period_sec)]
    return bpm, beats


def find_optimal_loop_cut(
    samples: np.ndarray,
    sample_rate: int,
    search_window_ms: float = 80.0,
) -> Tuple[int, int]:
    """
    Locate the nearest zero-crossing pair near the start and end of the audio buffer
    with matching slope polarity to eliminate boundary click/pop artifacts.
    """
    mono = np.mean(samples, axis=1) if samples.ndim > 1 else samples
    n = len(mono)
    window = min(n // 4, max(16, int(sample_rate * (search_window_ms / 1000.0))))
    if n < 32 or window < 4:
        return 0, n

    # Rising zero crossings near start
    start_region = mono[:window]
    rising_start = np.where((start_region[:-1] <= 0.0) & (start_region[1:] > 0.0))[0]
    start_idx = int(rising_start[0] + 1) if len(rising_start) > 0 else 0

    # Rising zero crossings near end
    end_region = mono[n - window :]
    rising_end = np.where((end_region[:-1] <= 0.0) & (end_region[1:] > 0.0))[0]
    end_idx = int((n - window) + rising_end[-1] + 1) if len(rising_end) > 0 else n

    if end_idx <= start_idx:
        return 0, n
    return start_idx, end_idx


def analyze_audio_samples(
    samples: np.ndarray,
    sample_rate: int,
    file_path: str = "<memory>",
    video_frame_mse: Optional[float] = None,
) -> LoopInspectionReport:
    """
    Evaluate loop seam quality across amplitude jump, RMS continuity, spectral continuity,
    and beat-grid alignment.
    """
    if samples.ndim == 1:
        channels = 1
        mono = samples.astype(np.float64)
    else:
        channels = samples.shape[1]
        mono = np.mean(samples.astype(np.float64), axis=1)

    n_samples = len(mono)
    duration_sec = round(n_samples / float(sample_rate), 4) if sample_rate > 0 else 0.0

    # 1. Sample-level discontinuity at seam (end -> start wrap-around)
    if n_samples > 2:
        expected_step = abs(mono[1] - mono[0])
        raw_jump = abs(mono[-1] - mono[0])
        boundary_diff = float(max(0.0, raw_jump - expected_step))
    else:
        boundary_diff = 0.0
    boundary_jump_db = round(_amp_to_db(boundary_diff), 2)

    # 2. RMS envelope continuity (first 50ms vs last 50ms)
    edge_win = min(n_samples // 4, max(64, int(sample_rate * 0.05)))
    head_rms = float(np.sqrt(np.mean(mono[:edge_win] ** 2))) + 1e-9
    tail_rms = float(np.sqrt(np.mean(mono[-edge_win:] ** 2))) + 1e-9
    rms_delta_db = round(abs(_amp_to_db(tail_rms) - _amp_to_db(head_rms)), 2)

    # 3. Spectral shape continuity at the boundary
    spec_win = min(n_samples // 2, max(128, int(sample_rate * 0.04)))
    window_fn = np.hanning(spec_win)
    head_spec = np.abs(np.fft.rfft(mono[:spec_win] * window_fn))
    tail_spec = np.abs(np.fft.rfft(mono[-spec_win:] * window_fn))
    head_norm = head_spec / (np.linalg.norm(head_spec) + 1e-9)
    tail_norm = tail_spec / (np.linalg.norm(tail_spec) + 1e-9)
    cosine_sim = float(np.clip(np.dot(head_norm, tail_norm), 0.0, 1.0))
    spectral_flux_delta = round(1.0 - cosine_sim, 4)

    # 4. Zero-crossing offset and optimal trim calculation
    opt_start, opt_end = find_optimal_loop_cut(mono, sample_rate)
    zc_offset = int(opt_start + (n_samples - opt_end))

    # 5. Tempo and beat-grid alignment
    bpm, _ = detect_bpm_and_beats(mono, sample_rate)
    beat_period = 60.0 / bpm if bpm > 0 else 0.5
    nearest_beats = max(1, round(duration_sec / beat_period))
    ideal_duration = nearest_beats * beat_period
    beat_alignment_error_ms = round(abs(duration_sec - ideal_duration) * 1000.0, 2)

    # Score calculation (0 - 100)
    score = 100.0
    diagnostics: List[str] = []

    if boundary_diff > 0.02:
        penalty = min(40.0, (boundary_diff / 0.25) * 40.0)
        score -= penalty
        diagnostics.append(
            f"Audible click/pop risk at seam: sample jump is {boundary_jump_db:.1f} dBFS."
        )

    if rms_delta_db > 1.5:
        penalty = min(25.0, (rms_delta_db - 1.5) * 3.0)
        score -= penalty
        diagnostics.append(
            f"Loudness jump across loop point: {rms_delta_db:.2f} dB difference between tail and head."
        )

    if spectral_flux_delta > 0.15:
        penalty = min(20.0, (spectral_flux_delta - 0.15) * 40.0)
        score -= penalty
        diagnostics.append(
            f"Timbral shift at loop boundary (spectral distance {spectral_flux_delta:.2f})."
        )

    if beat_alignment_error_ms > 25.0:
        penalty = min(15.0, (beat_alignment_error_ms - 25.0) * 0.15)
        score -= penalty
        diagnostics.append(
            f"Loop duration drifts off the {bpm} BPM grid by {beat_alignment_error_ms:.1f} ms."
        )

    video_seamless: Optional[bool] = None
    if video_frame_mse is not None:
        video_seamless = video_frame_mse < 18.0
        if not video_seamless:
            score -= min(20.0, (video_frame_mse - 18.0) * 0.4)
            diagnostics.append(
                f"Visual jump between last and first video frame (MSE={video_frame_mse:.1f})."
            )

    if not diagnostics:
        diagnostics.append("Clean zero-crossing boundary and steady loudness envelope.")

    loop_score = round(max(0.0, min(100.0, score)), 1)
    if loop_score >= 90.0:
        grade = "A"
    elif loop_score >= 78.0:
        grade = "B"
    elif loop_score >= 65.0:
        grade = "C"
    else:
        grade = "F"

    rec_start = round(opt_start / float(sample_rate), 4)
    rec_end = round(opt_end / float(sample_rate), 4)
    crossfade_ms = 15
    ffmpeg_cmd = (
        f'ffmpeg -y -i "{file_path}" '
        f'-af "atrim=start={rec_start}:end={rec_end},asetpts=PTS-STARTPTS,'
        f'afade=t=in:st=0:d=0.005,afade=t=out:st={max(0.0, rec_end - rec_start - (crossfade_ms / 1000.0)):.4f}:d={crossfade_ms / 1000.0:.3f}" '
        f'"fixed_loop.wav"'
    )

    return LoopInspectionReport(
        file_path=str(file_path),
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        channels=channels,
        loop_score=loop_score,
        grade=grade,
        is_seamless=loop_score >= 82.0,
        boundary_jump_db=boundary_jump_db,
        rms_delta_db=rms_delta_db,
        spectral_flux_delta=spectral_flux_delta,
        zero_crossing_offset_samples=zc_offset,
        estimated_bpm=bpm,
        beat_alignment_error_ms=beat_alignment_error_ms,
        recommended_trim_start_sec=rec_start,
        recommended_trim_end_sec=rec_end,
        ffmpeg_fix_command=ffmpeg_cmd,
        video_frame_mse=round(video_frame_mse, 2) if video_frame_mse is not None else None,
        video_loop_seamless=video_seamless,
        diagnostics=diagnostics,
    )


def _read_wav_pcm(wav_path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if channels > 1:
        data = data.reshape(-1, channels)
    return data, sample_rate


def analyze_media_file(path: str | Path) -> LoopInspectionReport:
    """
    Analyze a WAV, MP3, MP4, WebM, or MOV file for seamless loop quality.
    Uses standard library `wave` for `.wav` files and `ffmpeg` when decoding compressed/video containers.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Media file not found: {file_path}")

    if file_path.suffix.lower() == ".wav":
        samples, sr = _read_wav_pcm(file_path)
        return analyze_audio_samples(samples, sr, file_path=str(file_path))

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError(
            "ffmpeg is required in PATH to decode non-WAV media containers (.mp4, .mp3, .mov)."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_wav = Path(tmpdir) / "extracted.wav"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-i",
            str(file_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            str(tmp_wav),
        ]
        subprocess.run(cmd, check=True)
        samples, sr = _read_wav_pcm(tmp_wav)

        video_mse: Optional[float] = None
        if file_path.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}:
            first_raw = Path(tmpdir) / "first.raw"
            last_raw = Path(tmpdir) / "last.raw"
            try:
                subprocess.run(
                    [
                        ffmpeg_bin,
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(file_path),
                        "-vf",
                        "scale=64:64",
                        "-frames:v",
                        "1",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        str(first_raw),
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        ffmpeg_bin,
                        "-y",
                        "-v",
                        "error",
                        "-sseof",
                        "-0.15",
                        "-i",
                        str(file_path),
                        "-vf",
                        "scale=64:64",
                        "-frames:v",
                        "1",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        str(last_raw),
                    ],
                    check=True,
                )
                if first_raw.exists() and last_raw.exists():
                    b1 = np.frombuffer(first_raw.read_bytes(), dtype=np.uint8).astype(np.float64)
                    b2 = np.frombuffer(last_raw.read_bytes(), dtype=np.uint8).astype(np.float64)
                    if len(b1) == len(b2) and len(b1) > 0:
                        video_mse = float(np.mean((b1 - b2) ** 2) ** 0.5)
            except subprocess.CalledProcessError:
                video_mse = None

        return analyze_audio_samples(
            samples,
            sr,
            file_path=str(file_path),
            video_frame_mse=video_mse,
        )
