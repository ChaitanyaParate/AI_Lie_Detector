from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from text_audio import TextAudioResult, analyze_text_audio
from utils import extract_audio, extract_frames
from vision import analyze_vision
from model import DeceptionMLP


DEFAULT_FEATURE_COLS = [
    "text_score",
    "audio_score",
    "vision_score",
    "hesitation_score",
    "pitch_score",
    "silence_score",
    "blink_rate_per_min",
    "eye_movement_std",
    "blink_score",
    "eye_movement_score",
    "transcript_len",
    "has_transcript",
] + [f"emb_{i}" for i in range(384)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict lie/truth on a single video using model_full.pt")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--model-path", default="outputs/model_full.pt", help="Path to saved .pt model")
    parser.add_argument("--work-dir", default="inference_artifacts", help="Directory to store extracted audio/frames")
    parser.add_argument("--whisper-model", default="tiny", help="Whisper model size")
    parser.add_argument("--whisper-device", default="auto", choices=["auto", "cpu", "cuda"], help="Whisper inference device")
    parser.add_argument("--frame-step", default=8, type=int, help="Extract every Nth frame")
    parser.add_argument("--max-audio-seconds", default=30.0, type=float, help="Analyze only first N seconds of audio (<=0 means full audio)")
    parser.add_argument("--vision-max-frames", default=120, type=int, help="Maximum sampled frames for vision analysis")
    parser.add_argument("--threshold", default=0.5, type=float, help="Decision threshold for lie class")
    parser.add_argument("--output-json", default="", help="Optional path to save prediction JSON")
    return parser


def load_checkpoint(model_path: Path, device: torch.device) -> tuple[nn.Module, np.ndarray, np.ndarray, list[str]]:
    checkpoint: dict[str, Any] = torch.load(model_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise ValueError("Invalid checkpoint: missing model_state_dict")

    input_dim = int(checkpoint.get("input_dim", len(DEFAULT_FEATURE_COLS)))
    feature_cols = list(checkpoint.get("feature_cols", DEFAULT_FEATURE_COLS))

    scaler_mean = np.array(checkpoint.get("scaler_mean", [0.0] * input_dim), dtype=np.float32)
    scaler_scale = np.array(checkpoint.get("scaler_scale", [1.0] * input_dim), dtype=np.float32)

    if scaler_mean.shape[0] != input_dim or scaler_scale.shape[0] != input_dim:
        raise ValueError("Invalid checkpoint: scaler stats shape does not match input_dim")

    model = DeceptionMLP(input_dim=input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, scaler_mean, scaler_scale, feature_cols


def extract_single_video_features(
    video_path: Path,
    work_dir: Path,
    frame_step: int,
    whisper_model: str,
    whisper_device: str,
    max_audio_seconds: float,
    vision_max_frames: int,
) -> dict[str, float]:
    sample_dir = work_dir / video_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    audio_path = sample_dir / "audio.wav"
    frames_dir = sample_dir / "frames"

    has_audio = True
    try:
        extract_audio(str(video_path), str(audio_path))
    except ValueError:
        has_audio = False

    _, fps = extract_frames(str(video_path), str(frames_dir), frame_step=frame_step)

    if has_audio:
        ta = analyze_text_audio(
            str(audio_path),
            model_size=whisper_model,
            whisper_device=whisper_device,
            max_audio_seconds=max_audio_seconds,
        )
    else:
        ta = TextAudioResult(
            transcript="",
            hesitation_score=0.0,
            pitch_score=0.0,
            silence_score=0.0,
            text_score=0.0,
            audio_score=0.0,
        )

    vi = analyze_vision(str(frames_dir), video_fps=fps, frame_step=frame_step, max_frames=vision_max_frames)

    transcript_len = len(ta.transcript.split()) if ta.transcript else 0
    has_transcript = int(transcript_len > 0)

    result = {
        "text_score": float(ta.text_score),
        "audio_score": float(ta.audio_score),
        "vision_score": float(vi.vision_score),
        "hesitation_score": float(ta.hesitation_score),
        "pitch_score": float(ta.pitch_score),
        "silence_score": float(ta.silence_score),
        "blink_rate_per_min": float(vi.blink_rate_per_min),
        "eye_movement_std": float(vi.eye_movement_std),
        "blink_score": float(vi.blink_score),
        "eye_movement_score": float(vi.eye_movement_score),
        "transcript_len": float(transcript_len),
        "has_transcript": float(has_transcript),
        "transcript": ta.transcript,
    }
    
    for i, val in enumerate(ta.transcript_embedding):
        result[f"emb_{i}"] = float(val)
        
    return result


def main() -> None:
    args = build_parser().parse_args()

    video_path = Path(args.video_path)
    model_path = Path(args.model_path)
    work_dir = Path(args.work_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    work_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler_mean, scaler_scale, feature_cols = load_checkpoint(model_path, device)

    feats = extract_single_video_features(
        video_path=video_path,
        work_dir=work_dir,
        frame_step=args.frame_step,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        max_audio_seconds=args.max_audio_seconds,
        vision_max_frames=args.vision_max_frames,
    )

    x = np.array([feats.get(col, 0.0) for col in feature_cols], dtype=np.float32)
    safe_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
    x_scaled = (x - scaler_mean) / safe_scale

    with torch.no_grad():
        logit = model(torch.from_numpy(x_scaled).unsqueeze(0).to(device)).item()
    prob_lie = float(1.0 / (1.0 + np.exp(-logit)))
    pred_label = int(prob_lie >= args.threshold)

    result = {
        "video_path": str(video_path),
        "model_path": str(model_path),
        "prediction": {
            "label_id": pred_label,
            "label": "lie" if pred_label == 1 else "truth",
            "prob_lie": prob_lie,
            "threshold": float(args.threshold),
        },
        "features": {k: v for k, v in feats.items() if k != "transcript"},
        "transcript": feats.get("transcript", ""),
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
