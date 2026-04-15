from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2

try:
    from moviepy import VideoFileClip
except ImportError:  # pragma: no cover - older MoviePy layouts
    from moviepy.video.io.VideoFileClip import VideoFileClip


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_audio(video_path: str, output_audio_path: str) -> str:
    video = VideoFileClip(video_path)
    try:
        if video.audio is None:
            raise ValueError("Input video has no audio track.")
        video.audio.write_audiofile(output_audio_path, logger=None)
    finally:
        video.close()
    return output_audio_path


def extract_frames(video_path: str, output_dir: str, frame_step: int = 3) -> tuple[str, float]:
    output_path = Path(output_dir)
    ensure_dir(output_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_step == 0:
            frame_path = output_path / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved += 1
        idx += 1

    cap.release()
    return str(output_path), float(fps)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def minmax_normalize(value: float, min_v: float, max_v: float) -> float:
    if max_v <= min_v:
        return 0.0
    return clamp01((value - min_v) / (max_v - min_v))


def weighted_fusion(scores: dict[str, float], weights: dict[str, float]) -> float:
    common_keys = [key for key in scores if key in weights]
    if not common_keys:
        raise ValueError("No overlapping keys between scores and weights.")

    weight_sum = sum(weights[key] for key in common_keys)
    if weight_sum <= 0:
        raise ValueError("Sum of weights must be positive.")

    weighted_sum = sum(clamp01(scores[key]) * weights[key] for key in common_keys)
    return clamp01(weighted_sum / weight_sum)


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else default
