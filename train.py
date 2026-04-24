from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from text_audio import TextAudioResult, analyze_text_audio
from utils import extract_audio, extract_frames
from vision import analyze_vision
from model import DeceptionMLP

DEFAULT_DATASET_ROOT = "/mnt/newvolume/Programming/Python/Deep_Learning/AI_Lie_Detector/Real-life Deception Detection Dataset With Train Test"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

LIE_TOKENS = {"lie", "liar", "deception", "deceptive", "fake", "false"}
TRUTH_TOKENS = {"truth", "honest", "real", "genuine", "nondeceptive", "non_deceptive", "true"}


def _default_output_path(name: str) -> str:
    kaggle_working_dir = os.environ.get("KAGGLE_WORKING_DIR")
    if kaggle_working_dir:
        return str(Path(kaggle_working_dir) / name)
    if Path("/kaggle/working").exists():
        return str(Path("/kaggle/working") / name)
    return str(Path("outputs") / name)


# Removed ExtractedSample dataclass to allow dynamic embedding columns

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a multimodal deception classifier on the Kaggle dataset.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Dataset root directory")
    parser.add_argument("--cache-dir", default=_default_output_path("dataset_artifacts"), help="Where extracted audio/frames/features are stored")
    parser.add_argument("--whisper-model", default="tiny", help="Whisper model for feature extraction")
    parser.add_argument("--whisper-device", default="auto", choices=["auto", "cpu", "cuda"], help="Whisper inference device")
    parser.add_argument("--max-videos", default=0, type=int, help="Limit total processed videos (0 = all)")
    parser.add_argument("--frame-step", default=6, type=int, help="Extract every Nth frame for faster vision processing")
    parser.add_argument("--max-audio-seconds", default=45.0, type=float, help="Analyze only first N seconds of audio (<=0 means full audio)")
    parser.add_argument("--vision-max-frames", default=180, type=int, help="Maximum number of sampled frames per video for vision features")
    parser.add_argument("--features-csv", default=_default_output_path("features_full.csv"), help="Output feature CSV path")
    parser.add_argument("--model-path", default=_default_output_path("model_full.pt"), help="Output trained model path")
    parser.add_argument("--metrics-path", default=_default_output_path("metrics_full.json"), help="Output metrics JSON path")
    parser.add_argument("--test-size", default=0.2, type=float, help="Fallback split if no usable Test set exists")
    return parser


def _video_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def infer_label(path: Path) -> int | None:
    stem_tokens = {t for t in re.split(r"[^a-zA-Z0-9]+", path.stem.lower()) if t}
    if stem_tokens & LIE_TOKENS:
        return 1
    if stem_tokens & TRUTH_TOKENS:
        return 0

    parent_tokens = {t for t in re.split(r"[^a-zA-Z0-9]+", path.parent.name.lower()) if t}
    if parent_tokens & LIE_TOKENS:
        return 1
    if parent_tokens & TRUTH_TOKENS:
        return 0
    return None


def discover_split_files(dataset_root: Path, split_name: str) -> list[Path]:
    split_dir = dataset_root / split_name
    if split_dir.exists():
        return _video_files(split_dir)
    return []


def balanced_subset(samples: list[tuple[str, Path]], max_videos: int, seed: int = 42) -> list[tuple[str, Path]]:
    if max_videos <= 0 or len(samples) <= max_videos:
        return samples

    by_label: dict[int, list[tuple[str, Path]]] = {0: [], 1: []}
    unlabeled: list[tuple[str, Path]] = []

    for split, path in samples:
        label = infer_label(path)
        if label is None:
            unlabeled.append((split, path))
        else:
            by_label[label].append((split, path))

    rng = random.Random(seed)
    for label in by_label:
        rng.shuffle(by_label[label])

    half = max_videos // 2
    selected: list[tuple[str, Path]] = []
    selected.extend(by_label[0][:half])
    selected.extend(by_label[1][:half])

    remaining_slots = max_videos - len(selected)
    if remaining_slots > 0:
        remaining_pool = by_label[0][half:] + by_label[1][half:]
        rng.shuffle(remaining_pool)
        selected.extend(remaining_pool[:remaining_slots])

    if len(selected) < max_videos and unlabeled:
        rng.shuffle(unlabeled)
        selected.extend(unlabeled[: max_videos - len(selected)])

    return selected[:max_videos]


def extract_features_for_video(
    video_path: Path,
    split: str,
    label: int,
    cache_dir: Path,
    frame_step: int,
    whisper_model: str,
    whisper_device: str,
    max_audio_seconds: float,
    vision_max_frames: int,
) -> dict[str, Any] | None:
    sample_id = video_path.stem
    sample_cache = cache_dir / split / sample_id
    sample_cache.mkdir(parents=True, exist_ok=True)

    audio_path = sample_cache / "audio.wav"
    frames_dir = sample_cache / "frames"

    try:
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
            ta = TextAudioResult(transcript="", hesitation_score=0.0, pitch_score=0.0, silence_score=0.0, text_score=0.0, audio_score=0.0)

        vi = analyze_vision(str(frames_dir), video_fps=fps, frame_step=frame_step, max_frames=vision_max_frames)

        transcript_len = len(ta.transcript.split()) if ta.transcript else 0
        has_transcript = int(transcript_len > 0)

        result = {
            "split": split,
            "video_path": str(video_path),
            "label": label,
            "text_score": ta.text_score,
            "audio_score": ta.audio_score,
            "vision_score": vi.vision_score,
            "hesitation_score": ta.hesitation_score,
            "pitch_score": ta.pitch_score,
            "silence_score": ta.silence_score,
            "blink_rate_per_min": vi.blink_rate_per_min,
            "eye_movement_std": vi.eye_movement_std,
            "blink_score": vi.blink_score,
            "eye_movement_score": vi.eye_movement_score,
            "transcript_len": transcript_len,
            "has_transcript": has_transcript,
        }
        
        for i, val in enumerate(ta.transcript_embedding):
            result[f"emb_{i}"] = val
            
        return result
    except Exception as exc:
        print(f"[WARN] Failed for {video_path.name}: {exc}")
        return None


def train_and_evaluate(df: pd.DataFrame, model_path: Path, metrics_path: Path, test_size: float) -> dict[str, Any]:
    feature_cols = [
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

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    if train_df.empty:
        raise ValueError("No train samples available after label inference.")

    if test_df.empty or train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2:
        if len(df) < 10:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        else:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df["label"])

    X_train = train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    X_test = test_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeceptionMLP(input_dim=X_train_scaled.shape[1]).to(device)

    X_train_t = torch.from_numpy(X_train_scaled)
    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    train_ds = TensorDataset(X_train_t, y_train_t)
    batch_size = min(32, len(train_ds))
    drop_last = (len(train_ds) > batch_size)  # Prevent BatchNorm crash if last batch is size 1
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    pos_count = float((y_train == 1).sum())
    neg_count = float((y_train == 0).sum())
    pos_weight_value = neg_count / max(pos_count, 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    for _ in range(80):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_test = model(torch.from_numpy(X_test_scaled).to(device)).cpu().numpy()
    y_prob = 1.0 / (1.0 + np.exp(-logits_test))
    y_pred = (y_prob >= 0.5).astype(np.int64)

    metrics: dict[str, Any] = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else None,
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X_train_scaled.shape[1],
            "feature_cols": feature_cols,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def main() -> None:
    args = build_parser().parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_files = discover_split_files(dataset_root, "Train")
    test_files = discover_split_files(dataset_root, "Test")
    all_samples: list[tuple[str, Path]] = [("train", p) for p in train_files] + [("test", p) for p in test_files]

    if not all_samples:
        all_videos = _video_files(dataset_root)
        all_samples = [("train", p) for p in all_videos]

    if args.max_videos > 0:
        all_samples = balanced_subset(all_samples, args.max_videos)

    extracted_rows: list[dict[str, Any]] = []
    skipped_no_label = 0

    for split, video_path in tqdm(all_samples, desc="Extracting features"):
        label = infer_label(video_path)
        if label is None:
            skipped_no_label += 1
            continue

        row = extract_features_for_video(
            video_path=video_path,
            split=split,
            label=label,
            cache_dir=cache_dir,
            frame_step=args.frame_step,
            whisper_model=args.whisper_model,
            whisper_device=args.whisper_device,
            max_audio_seconds=args.max_audio_seconds,
            vision_max_frames=args.vision_max_frames,
        )
        if row is not None:
            extracted_rows.append(row)

    if not extracted_rows:
        raise ValueError("No labeled samples extracted. Check that file names contain lie/truth tokens.")

    df = pd.DataFrame(extracted_rows)

    features_csv = Path(args.features_csv)
    features_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_csv, index=False)

    metrics = train_and_evaluate(df=df, model_path=Path(args.model_path), metrics_path=Path(args.metrics_path), test_size=args.test_size)

    print(json.dumps(
        {
            "saved_features": str(features_csv),
            "saved_model": str(Path(args.model_path)),
            "saved_metrics": str(Path(args.metrics_path)),
            "skipped_no_label": skipped_no_label,
            "metrics": metrics,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()