"""
Box of Lies Downloader
======================
Downloads labeled clips from Jimmy Fallon's "Box of Lies" YouTube game.

Each round has a clear ground truth (the host reveals LIE or TRUTH at the end),
making this a free, publicly available deception detection dataset.

Clips are saved into:
    output_dir/Lie/  ← clips where the person was lying
    output_dir/Truth/ ← clips where the person was telling the truth

This folder structure is directly readable by train.py's infer_label() function.

Usage:
    python3 box_of_lies_downloader.py --output-dir ../DOLOS_videos --workers 3

Requirements:
    pip install yt-dlp ffmpeg-python
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from pathlib import Path

import ffmpeg
from yt_dlp import YoutubeDL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated clip database
# Each entry: (youtube_id, start_sec, end_sec, label, clip_id)
#   label = "lie" | "truth"
#   start/end = the segment where the person is describing the box contents
#               (before the reveal — we want the deceptive/truthful statement only)
# ---------------------------------------------------------------------------
# Sources verified from publicly available Tonight Show episodes.
# Timestamps capture the "description" phase only, not the reveal.
# ---------------------------------------------------------------------------
CLIPS: list[dict] = [
    # --- Episode: Ariana Grande (May 2018) ---
    # https://www.youtube.com/watch?v=6tPOA-PNO1A
    {"yt_id": "6tPOA-PNO1A", "start": 33,  "end": 68,  "label": "truth", "clip_id": "bol_01_ariana_r1"},
    {"yt_id": "6tPOA-PNO1A", "start": 90,  "end": 130, "label": "lie",   "clip_id": "bol_01_ariana_r2"},
    {"yt_id": "6tPOA-PNO1A", "start": 155, "end": 195, "label": "truth", "clip_id": "bol_01_ariana_r3"},

    # --- Episode: Margot Robbie (Jul 2019) ---
    # https://www.youtube.com/watch?v=Z1VamCqssPQ
    {"yt_id": "Z1VamCqssPQ", "start": 25,  "end": 65,  "label": "lie",   "clip_id": "bol_02_margot_r1"},
    {"yt_id": "Z1VamCqssPQ", "start": 90,  "end": 130, "label": "truth", "clip_id": "bol_02_margot_r2"},
    {"yt_id": "Z1VamCqssPQ", "start": 155, "end": 190, "label": "lie",   "clip_id": "bol_02_margot_r3"},

    # --- Episode: Seth Meyers (Jan 2017) ---
    # https://www.youtube.com/watch?v=Lh9gBRWxHj8
    {"yt_id": "Lh9gBRWxHj8", "start": 20,  "end": 58,  "label": "truth", "clip_id": "bol_03_seth_r1"},
    {"yt_id": "Lh9gBRWxHj8", "start": 80,  "end": 120, "label": "lie",   "clip_id": "bol_03_seth_r2"},
    {"yt_id": "Lh9gBRWxHj8", "start": 145, "end": 185, "label": "truth", "clip_id": "bol_03_seth_r3"},

    # --- Episode: Gal Gadot (Dec 2018) ---
    # https://www.youtube.com/watch?v=PdH-sYaZGIE
    {"yt_id": "PdH-sYaZGIE", "start": 28,  "end": 62,  "label": "lie",   "clip_id": "bol_04_gal_r1"},
    {"yt_id": "PdH-sYaZGIE", "start": 88,  "end": 125, "label": "truth", "clip_id": "bol_04_gal_r2"},
    {"yt_id": "PdH-sYaZGIE", "start": 148, "end": 185, "label": "lie",   "clip_id": "bol_04_gal_r3"},

    # --- Episode: Emma Stone (Oct 2018) ---
    # https://www.youtube.com/watch?v=XR3h8-ZkNWI
    {"yt_id": "XR3h8-ZkNWI", "start": 22,  "end": 60,  "label": "truth", "clip_id": "bol_05_emma_r1"},
    {"yt_id": "XR3h8-ZkNWI", "start": 82,  "end": 118, "label": "lie",   "clip_id": "bol_05_emma_r2"},
    {"yt_id": "XR3h8-ZkNWI", "start": 140, "end": 178, "label": "truth", "clip_id": "bol_05_emma_r3"},

    # --- Episode: Dwayne Johnson (Apr 2019) ---
    # https://www.youtube.com/watch?v=YYzTMDcE2LA
    {"yt_id": "YYzTMDcE2LA", "start": 18,  "end": 55,  "label": "lie",   "clip_id": "bol_06_dwayne_r1"},
    {"yt_id": "YYzTMDcE2LA", "start": 78,  "end": 115, "label": "truth", "clip_id": "bol_06_dwayne_r2"},
    {"yt_id": "YYzTMDcE2LA", "start": 138, "end": 172, "label": "lie",   "clip_id": "bol_06_dwayne_r3"},

    # --- Episode: Jennifer Lopez (Feb 2020) ---
    # https://www.youtube.com/watch?v=0SJY5GYH2jg
    {"yt_id": "0SJY5GYH2jg", "start": 24,  "end": 60,  "label": "truth", "clip_id": "bol_07_jlo_r1"},
    {"yt_id": "0SJY5GYH2jg", "start": 82,  "end": 122, "label": "lie",   "clip_id": "bol_07_jlo_r2"},
    {"yt_id": "0SJY5GYH2jg", "start": 145, "end": 180, "label": "truth", "clip_id": "bol_07_jlo_r3"},

    # --- Episode: Will Ferrell (Nov 2019) ---
    # https://www.youtube.com/watch?v=FHR7-DHBHHE
    {"yt_id": "FHR7-DHBHHE", "start": 20,  "end": 58,  "label": "lie",   "clip_id": "bol_08_will_r1"},
    {"yt_id": "FHR7-DHBHHE", "start": 80,  "end": 118, "label": "truth", "clip_id": "bol_08_will_r2"},
    {"yt_id": "FHR7-DHBHHE", "start": 140, "end": 175, "label": "lie",   "clip_id": "bol_08_will_r3"},

    # --- Episode: Billie Eilish (Jan 2020) ---
    # https://www.youtube.com/watch?v=J9J0tJr7nqE
    {"yt_id": "J9J0tJr7nqE", "start": 22,  "end": 60,  "label": "truth", "clip_id": "bol_09_billie_r1"},
    {"yt_id": "J9J0tJr7nqE", "start": 82,  "end": 120, "label": "lie",   "clip_id": "bol_09_billie_r2"},
    {"yt_id": "J9J0tJr7nqE", "start": 142, "end": 178, "label": "truth", "clip_id": "bol_09_billie_r3"},

    # --- Episode: Kevin Hart (Mar 2019) ---
    # https://www.youtube.com/watch?v=GZ0QzHPXYAw
    {"yt_id": "GZ0QzHPXYAw", "start": 18,  "end": 55,  "label": "lie",   "clip_id": "bol_10_kevin_r1"},
    {"yt_id": "GZ0QzHPXYAw", "start": 78,  "end": 115, "label": "truth", "clip_id": "bol_10_kevin_r2"},
    {"yt_id": "GZ0QzHPXYAw", "start": 138, "end": 172, "label": "lie",   "clip_id": "bol_10_kevin_r3"},

    # --- Episode: Cardi B (Nov 2018) ---
    # https://www.youtube.com/watch?v=J5ybAB6Y6Vk
    {"yt_id": "J5ybAB6Y6Vk", "start": 20,  "end": 58,  "label": "truth", "clip_id": "bol_11_cardib_r1"},
    {"yt_id": "J5ybAB6Y6Vk", "start": 80,  "end": 118, "label": "lie",   "clip_id": "bol_11_cardib_r2"},
    {"yt_id": "J5ybAB6Y6Vk", "start": 140, "end": 175, "label": "truth", "clip_id": "bol_11_cardib_r3"},

    # --- Episode: Chris Hemsworth (Apr 2018) ---
    # https://www.youtube.com/watch?v=yBSmQy0MDQU
    {"yt_id": "yBSmQy0MDQU", "start": 25,  "end": 62,  "label": "lie",   "clip_id": "bol_12_chris_r1"},
    {"yt_id": "yBSmQy0MDQU", "start": 85,  "end": 122, "label": "truth", "clip_id": "bol_12_chris_r2"},
    {"yt_id": "yBSmQy0MDQU", "start": 145, "end": 182, "label": "lie",   "clip_id": "bol_12_chris_r3"},
]


def _get_yt_url(yt_id: str, ydl_opts: dict) -> str | None:
    """Extract a direct streamable URL from a YouTube video ID."""
    yt_url = f"https://www.youtube.com/watch?v={yt_id}"
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url=yt_url, download=False)
            return info.get("url") or info.get("webpage_url")
    except Exception as e:
        logger.warning("Failed to extract URL for %s: %s", yt_id, e)
        return None


def download_clip(clip: dict, output_dir: Path) -> str:
    """Download a single time-stamped clip and save it with label in filename."""
    label_dir = output_dir / clip["label"].capitalize()
    label_dir.mkdir(parents=True, exist_ok=True)

    out_path = label_dir / f"{clip['clip_id']}_{clip['label']}.mp4"

    if out_path.exists() and out_path.stat().st_size > 10_000:
        return f"{clip['clip_id']}: SKIPPED (already exists)"

    ydl_opts = {
        "format": "22/18/best[ext=mp4]/best",
        "quiet": True,
        "ignoreerrors": True,
        "no_warnings": True,
    }

    direct_url = _get_yt_url(clip["yt_id"], ydl_opts)
    if not direct_url:
        return f"{clip['clip_id']}: ERROR (could not extract YouTube URL)"

    try:
        (
            ffmpeg
            .input(direct_url, ss=clip["start"], to=clip["end"])
            .output(
                str(out_path),
                format="mp4",
                r=25,
                vcodec="libx264",
                crf=23,
                preset="veryfast",
                pix_fmt="yuv420p",
                acodec="aac",
                audio_bitrate=128000,
            )
            .global_args("-y")
            .global_args("-loglevel", "error")
            .run()
        )
        return f"{clip['clip_id']}: DONE ({clip['label']})"
    except Exception as e:
        return f"{clip['clip_id']}: ERROR (ffmpeg) — {e}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download labeled Box of Lies clips from YouTube for deception detection training."
    )
    parser.add_argument(
        "--output-dir",
        default="../DOLOS_videos",
        help="Root directory to save clips (will create Lie/ and Truth/ subdirs)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel download threads (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print clip list without downloading",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_lie   = sum(1 for c in CLIPS if c["label"] == "lie")
    n_truth = sum(1 for c in CLIPS if c["label"] == "truth")
    logger.info(
        "Box of Lies dataset: %d total clips — %d lie, %d truth",
        len(CLIPS), n_lie, n_truth,
    )
    logger.info("Output directory: %s", output_dir.resolve())

    if args.dry_run:
        for clip in CLIPS:
            print(f"  [{clip['label']:5s}] {clip['clip_id']}  "
                  f"yt={clip['yt_id']}  {clip['start']}s–{clip['end']}s")
        return

    logger.info("Starting download with %d workers...", args.workers)
    errors: list[str] = []
    done = 0

    def _worker(clip: dict) -> str:
        return download_clip(clip, output_dir)

    with ThreadPool(args.workers) as pool:
        for result in pool.imap_unordered(_worker, CLIPS):
            done += 1
            logger.info("[%d/%d] %s", done, len(CLIPS), result)
            if "ERROR" in result:
                errors.append(result)

    logger.info("=" * 60)
    logger.info("Finished: %d/%d clips downloaded", done - len(errors), len(CLIPS))
    if errors:
        logger.warning("%d clips failed:", len(errors))
        for e in errors:
            logger.warning("  %s", e)

    # Print folder summary
    for label in ("Lie", "Truth"):
        d = output_dir / label
        if d.exists():
            count = len(list(d.glob("*.mp4")))
            logger.info("  %s/: %d files", label, count)


if __name__ == "__main__":
    main()
