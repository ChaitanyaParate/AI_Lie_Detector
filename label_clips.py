"""
Interactive Box of Lies Clip Labeler
=====================================
Watch each raw video in your media player, then use this tool to cut
and save labeled clips into Lie/ and Truth/ folders.

Usage:
    python3 label_clips.py

The script will guide you through each video interactively.
Clips are saved to: ../DOLOS_videos/Lie/ and ../DOLOS_videos/Truth/

Controls:
    - Enter start/end timestamps as MM:SS or plain seconds (e.g. 1:23 or 83)
    - Enter label as 'l' (lie) or 't' (truth)
    - Type 'done' when finished with a video
    - Type 'skip' to skip to the next video
    - Type 'q' to quit and save progress
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path("bol_raw")
OUTPUT_DIR = Path("../DOLOS_videos")
PROGRESS_FILE = Path("bol_labeling_progress.json")

# Map video IDs to human-readable names
VIDEO_INFO = {
    "QhJIA8moL5s": "Jennifer Lawrence",
    "BvHCpWixZRA": "Taylor Swift",
    "LXG_-LGsAy8": "Jenna Ortega",
    "kY8stpY4rrc": "Millie Bobby Brown",
    "Md4QnipNYqM": "Chris Pratt",
}


def parse_time(s: str) -> float | None:
    """Parse MM:SS or plain seconds into float seconds."""
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        try:
            return int(parts[0]) * 60 + float(parts[1])
        except (ValueError, IndexError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def cut_clip(
    input_path: Path,
    start: float,
    end: float,
    label: str,
    clip_id: str,
    output_dir: Path,
) -> bool:
    """Cut a clip from the video using ffmpeg."""
    label_dir = output_dir / label.capitalize()
    label_dir.mkdir(parents=True, exist_ok=True)
    out_path = label_dir / f"{clip_id}_{label}.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(input_path),
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "128k",
        "-loglevel", "error",
        str(out_path),
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        size_kb = out_path.stat().st_size // 1024
        print(f"  ✓ Saved: {out_path.name}  ({size_kb} KB)")
        return True
    else:
        print(f"  ✗ ffmpeg failed for {clip_id}")
        return False


def open_video(path: Path) -> None:
    """Try to open the video in the system default player."""
    for player in ("xdg-open", "vlc", "mpv", "ffplay"):
        if subprocess.run(["which", player], capture_output=True).returncode == 0:
            print(f"  Opening with {player}...")
            subprocess.Popen([player, str(path)],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return
    print("  (Could not auto-open — open the file manually in your video player)")


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(progress: dict) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def label_video(yt_id: str, name: str, progress: dict) -> list[dict]:
    video_path = RAW_DIR / f"{yt_id}.mp4"
    if not video_path.exists():
        print(f"\n[!] Video not found: {video_path}")
        return []

    # Show duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0
    mins, secs = divmod(int(duration), 60)

    print(f"\n{'='*60}")
    print(f"  Video: {name}  ({yt_id})")
    print(f"  Duration: {mins}:{secs:02d}")
    print(f"  File: {video_path}")
    print(f"{'='*60}")
    print("  Opening video... Watch it and note the timestamps for each round.")
    print("  Typically: 3 rounds, each ~30-50 seconds of description.")
    print()

    open_video(video_path)

    clips: list[dict] = []
    round_num = 1
    already_done = progress.get(yt_id, {})
    clips = list(already_done.get("clips", []))

    print("  For each round, enter: start, end, label (l/t)")
    print("  Commands: 'done' = finished this video | 'skip' = skip | 'q' = quit all")
    print()

    while True:
        cmd = input(f"  Round {round_num} — Start time (MM:SS or seconds): ").strip().lower()
        if cmd == "q":
            return clips, True  # signal quit
        if cmd in ("done", "skip", ""):
            break

        start = parse_time(cmd)
        if start is None:
            print("  Invalid time format. Try '1:23' or '83'")
            continue

        end_raw = input(f"  Round {round_num} — End time:   ").strip().lower()
        if end_raw in ("q",):
            return clips, True
        end = parse_time(end_raw)
        if end is None:
            print("  Invalid time format.")
            continue
        if end <= start:
            print("  End must be after start.")
            continue

        label_raw = input(f"  Round {round_num} — Label (l=lie / t=truth): ").strip().lower()
        if label_raw in ("q",):
            return clips, True
        if label_raw not in ("l", "t", "lie", "truth"):
            print("  Invalid label. Use 'l' or 't'.")
            continue

        label = "lie" if label_raw.startswith("l") else "truth"
        guest_slug = name.lower().replace(" ", "_")
        clip_id = f"bol_{guest_slug}_r{round_num}"

        success = cut_clip(video_path, start, end, label, clip_id, OUTPUT_DIR)
        if success:
            clips.append({
                "yt_id": yt_id,
                "clip_id": clip_id,
                "start": start,
                "end": end,
                "label": label,
            })
            round_num += 1

    return clips, False  # no quit


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()

    print("\n" + "="*60)
    print("  BOX OF LIES — Interactive Clip Labeler")
    print("="*60)
    print(f"  Videos in: {RAW_DIR.resolve()}")
    print(f"  Clips out:  {OUTPUT_DIR.resolve()}")
    print()

    total_clips = 0
    for yt_id, name in VIDEO_INFO.items():
        if progress.get(yt_id, {}).get("done"):
            n = len(progress[yt_id].get("clips", []))
            print(f"  ✓ {name}: already labeled ({n} clips) — skipping")
            total_clips += n
            continue

        result = label_video(yt_id, name, progress)
        if isinstance(result, tuple):
            clips, quit_all = result
        else:
            clips, quit_all = result, False

        progress[yt_id] = {"done": True, "clips": clips}
        save_progress(progress)
        total_clips += len(clips)

        if quit_all:
            print("\n  Quitting. Progress saved.")
            break

    # Summary
    lie_dir = OUTPUT_DIR / "Lie"
    truth_dir = OUTPUT_DIR / "Truth"
    n_lie = len(list(lie_dir.glob("*.mp4"))) if lie_dir.exists() else 0
    n_truth = len(list(truth_dir.glob("*.mp4"))) if truth_dir.exists() else 0

    print(f"\n{'='*60}")
    print(f"  Done! Total clips saved: {total_clips}")
    print(f"    Lie/:   {n_lie} clips")
    print(f"    Truth/: {n_truth} clips")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print()
    print("  To train on these clips, run:")
    print(f"  python3 train.py --dataset-root {OUTPUT_DIR.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
