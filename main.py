from __future__ import annotations

import argparse
import json
from pathlib import Path

from text_audio import TextAudioResult, analyze_text_audio
from utils import extract_audio, extract_frames, weighted_fusion
from vision import analyze_vision

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lightweight multimodal AI Lie Detector (text + audio + vision)."
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        help="Path to input video file",
    )
    parser.add_argument(
        "--work-dir",
        default="artifacts",
        help="Directory to store extracted audio/frames",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model size (tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "--whisper-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Whisper inference device",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        metavar=("TEXT", "AUDIO", "VISION"),
        default=[0.3, 0.3, 0.4],
        help="Fusion weights for text, audio, vision",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    inferred_video_path: Path | None = None
    if args.video_path:
        inferred_video_path = Path(args.video_path)
    else:
        work_dir_path = Path(args.work_dir)
        if work_dir_path.suffix.lower() in VIDEO_EXTENSIONS:
            inferred_video_path = work_dir_path

    if inferred_video_path is None:
        parser.error(
            "video_path is required. Example: python main.py /path/to/video.mp4 or python main.py --work-dir artifacts /path/to/video.mp4"
        )

    video_path = inferred_video_path
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if Path(args.work_dir).suffix.lower() in VIDEO_EXTENSIONS:
        work_dir = Path("artifacts")
    else:
        work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    audio_path = str(work_dir / "audio.wav")
    frames_dir = str(work_dir / "frames")

    print("[1/4] Extracting audio...")
    has_audio = True
    try:
        extract_audio(str(video_path), audio_path)
    except ValueError as exc:
        has_audio = False
        print(f"Warning: {exc} Falling back to text/audio scores = 0.0")

    print("[2/4] Extracting frames...")
    frame_step = 3
    _, fps = extract_frames(str(video_path), frames_dir, frame_step=frame_step)

    print("[3/4] Running text/audio analysis...")
    if has_audio:
        ta = analyze_text_audio(audio_path, model_size=args.whisper_model, whisper_device=args.whisper_device)
    else:
        ta = TextAudioResult(
            transcript="",
            hesitation_score=0.0,
            pitch_score=0.0,
            silence_score=0.0,
            text_score=0.0,
            audio_score=0.0,
        )

    print("[4/4] Running vision analysis...")
    vi = analyze_vision(frames_dir, video_fps=fps, frame_step=frame_step)

    modality_scores = {
        "text": ta.text_score,
        "audio": ta.audio_score,
        "vision": vi.vision_score,
    }
    weights = {
        "text": args.weights[0],
        "audio": args.weights[1],
        "vision": args.weights[2],
    }

    deception_score = weighted_fusion(modality_scores, weights)

    output = {
        "transcript": ta.transcript,
        "text_audio": {
            "hesitation_score": ta.hesitation_score,
            "pitch_score": ta.pitch_score,
            "silence_score": ta.silence_score,
            "text_score": ta.text_score,
            "audio_score": ta.audio_score,
        },
        "vision": {
            "blink_rate_per_min": vi.blink_rate_per_min,
            "eye_movement_std": vi.eye_movement_std,
            "blink_score": vi.blink_score,
            "eye_movement_score": vi.eye_movement_score,
            "vision_score": vi.vision_score,
        },
        "modality_scores": modality_scores,
        "weights": weights,
        "deception_score": deception_score,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
