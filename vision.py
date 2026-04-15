from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from utils import minmax_normalize


@dataclass
class VisionResult:
    blink_rate_per_min: float
    eye_movement_std: float
    blink_score: float
    eye_movement_score: float
    vision_score: float


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    vertical_1 = _distance(eye_pts[1], eye_pts[5])
    vertical_2 = _distance(eye_pts[2], eye_pts[4])
    horizontal = _distance(eye_pts[0], eye_pts[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def _summarize_visual_signals(openness_values: list[float], eye_center_x: list[float], video_fps: float, frame_step: int) -> VisionResult:
    if not openness_values:
        return VisionResult(0.0, 0.0, 0.0, 0.0, 0.0)

    openness_arr = np.array(openness_values)
    if openness_arr.size < 3:
        return VisionResult(0.0, 0.0, 0.0, 0.0, 0.0)

    baseline = float(np.median(openness_arr))
    if baseline <= 0:
        return VisionResult(0.0, 0.0, 0.0, 0.0, 0.0)

    blink_threshold = baseline * 0.75
    blinks = 0
    closed = False
    for value in openness_arr:
        if value < blink_threshold and not closed:
            closed = True
            blinks += 1
        elif value >= blink_threshold and closed:
            closed = False

    sampled_fps = video_fps / max(frame_step, 1)
    duration_sec = len(openness_arr) / sampled_fps if sampled_fps > 0 else 0.0
    blink_rate_per_min = (blinks / duration_sec) * 60.0 if duration_sec > 0 else 0.0
    movement_std = float(np.std(np.diff(eye_center_x))) if len(eye_center_x) > 1 else 0.0

    blink_score = minmax_normalize(blink_rate_per_min, 8.0, 35.0)
    eye_movement_score = minmax_normalize(movement_std, 1.5, 10.0)
    vision_score = float((blink_score + eye_movement_score) / 2.0)

    return VisionResult(
        blink_rate_per_min=blink_rate_per_min,
        eye_movement_std=movement_std,
        blink_score=blink_score,
        eye_movement_score=eye_movement_score,
        vision_score=vision_score,
    )


def _sample_frame_paths(frame_dir: str, max_frames: int) -> list[Path]:
    frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
    if max_frames <= 0 or len(frame_paths) <= max_frames:
        return frame_paths

    step = max(1, len(frame_paths) // max_frames)
    sampled = frame_paths[::step]
    return sampled[:max_frames]


def _analyze_vision_with_mediapipe(frame_dir: str, video_fps: float, frame_step: int, max_frames: int) -> VisionResult:
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    left_eye_idx = [33, 160, 158, 133, 153, 144]
    right_eye_idx = [362, 385, 387, 263, 373, 380]
    left_center_idx = [33, 133]
    right_center_idx = [362, 263]

    openness_values: list[float] = []
    eye_center_x: list[float] = []

    for frame_path in _sample_frame_paths(frame_dir, max_frames):
        image_bgr = cv2.imread(str(frame_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            continue

        face = results.multi_face_landmarks[0]
        h, w, _ = image_bgr.shape
        pts = np.array([(lm.x * w, lm.y * h) for lm in face.landmark], dtype=np.float32)
        left_eye = pts[left_eye_idx]
        right_eye = pts[right_eye_idx]
        ear = (_eye_aspect_ratio(left_eye) + _eye_aspect_ratio(right_eye)) / 2.0
        openness_values.append(ear)

        l_center = (pts[left_center_idx[0]] + pts[left_center_idx[1]]) / 2.0
        r_center = (pts[right_center_idx[0]] + pts[right_center_idx[1]]) / 2.0
        eye_center_x.append(float((l_center[0] + r_center[0]) / 2.0))

    face_mesh.close()
    return _summarize_visual_signals(openness_values, eye_center_x, video_fps, frame_step)


def _analyze_vision_with_opencv(frame_dir: str, video_fps: float, frame_step: int, max_frames: int) -> VisionResult:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    openness_values: list[float] = []
    eye_center_x: list[float] = []

    for frame_path in _sample_frame_paths(frame_dir, max_frames):
        image_bgr = cv2.imread(str(frame_path))
        if image_bgr is None:
            continue

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(faces) == 0:
            continue

        x, y, w, h = sorted(faces, key=lambda item: item[2] * item[3], reverse=True)[0]
        roi_gray = gray[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) < 2:
            continue

        eyes = sorted(eyes, key=lambda item: item[0])[:2]
        eye_ratios: list[float] = []
        centers: list[float] = []
        for ex, ey, ew, eh in eyes:
            if ew <= 0:
                continue
            eye_ratios.append(eh / ew)
            centers.append(x + ex + (ew / 2.0))

        if len(eye_ratios) == 2 and len(centers) == 2:
            openness_values.append(float(np.mean(eye_ratios)))
            eye_center_x.append(float(np.mean(centers)))

    return _summarize_visual_signals(openness_values, eye_center_x, video_fps, frame_step)


def analyze_vision(frame_dir: str, video_fps: float, frame_step: int = 3, max_frames: int = 180) -> VisionResult:
    if hasattr(mp, "solutions"):
        return _analyze_vision_with_mediapipe(frame_dir, video_fps, frame_step, max_frames)
    return _analyze_vision_with_opencv(frame_dir, video_fps, frame_step, max_frames)
