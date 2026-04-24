# 🧠 AI Lie Detector

> A lightweight, multimodal deception detection system that analyzes video input across three independent channels — **speech content**, **vocal acoustics**, and **facial behavior** — and fuses them into a single interpretable deception score.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-lightgrey?logo=openai)](https://github.com/openai/whisper)
[![MediaPipe](https://img.shields.io/badge/Google-MediaPipe-blueviolet)](https://mediapipe.dev/)

---

## Overview

AI Lie Detector processes a video file through three parallel analysis pipelines and combines their outputs using configurable weighted fusion to produce a final deception probability score.

```
Video Input
    ├── Audio Extraction → Whisper (transcription) + Librosa (pitch, silence, hesitation)
    ├── Frame Extraction → MediaPipe (facial landmarks, blink rate, eye movement)
    └── Weighted Fusion  → Deception Score [0.0 – 1.0]
```

Each modality produces its own independent score, making it easy to audit, debug, and extend individual components without affecting the others.

---

## Features

- **Automatic Speech Recognition** via OpenAI Whisper (tiny → large model selection)
- **Linguistic Analysis** — detects hesitation markers, filler words, and deceptive speech patterns
- **Acoustic Analysis** — models pitch variance, abnormal silence, and vocal stress using Librosa
- **Vision Analysis** — tracks blink rate and eye movement variance via MediaPipe face mesh
- **Weighted Fusion** — configurable per-modality weights (text / audio / vision)
- **Structured JSON Output** — full breakdown of intermediate scores alongside the final result
- **CPU & CUDA support** — Whisper device auto-detection or manual selection
- **Graceful degradation** — falls back to text-only analysis if audio extraction fails

---

## Repository Structure

```
AI_Lie_Detector/
├── main.py            # CLI entry point — orchestrates the full pipeline
├── text_audio.py      # Speech transcription + linguistic & acoustic feature extraction
├── vision.py          # Facial landmark analysis (blink rate, eye movement)
├── model.py           # Neural network model definition
├── train.py           # Model training script
├── predict.py         # Standalone inference script
├── loss.py            # Custom loss functions
├── utils.py           # Audio/frame extraction helpers + weighted fusion logic
├── model_full.pt      # Pre-trained model checkpoint
├── requirements.txt   # Python dependencies
├── outputs/           # Directory for storing analysis artifacts
└── .gitignore
```

---

## Installation

### Prerequisites

- Python 3.9+
- `ffmpeg` installed and available on your system PATH

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
```

### Clone & Install

```bash
git clone https://github.com/ChaitanyaParate/AI_Lie_Detector.git
cd AI_Lie_Detector

pip install -r requirements.txt
```

> **Note:** GPU acceleration for Whisper requires a CUDA-compatible PyTorch installation. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for platform-specific instructions.

---

## Usage

### Basic

```bash
python main.py path/to/video.mp4
```

### Full Options

```bash
python main.py path/to/video.mp4 \
  --work-dir artifacts \
  --whisper-model small \
  --whisper-device cuda \
  --weights 0.3 0.3 0.4
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `video_path` | *(required)* | Path to the input video file (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) |
| `--work-dir` | `artifacts` | Directory to store extracted audio and frames |
| `--whisper-model` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--whisper-device` | `auto` | Inference device: `auto`, `cpu`, `cuda` |
| `--weights` | `0.3 0.3 0.4` | Fusion weights for text, audio, and vision modalities (must sum to 1.0) |

---

## Output Format

The pipeline outputs a structured JSON object to stdout:

```json
{
  "transcript": "I was at home the entire evening.",
  "text_audio": {
    "hesitation_score": 0.42,
    "pitch_score": 0.61,
    "silence_score": 0.35,
    "text_score": 0.46,
    "audio_score": 0.52
  },
  "vision": {
    "blink_rate_per_min": 28.4,
    "eye_movement_std": 0.073,
    "blink_score": 0.58,
    "eye_movement_score": 0.49,
    "vision_score": 0.54
  },
  "modality_scores": {
    "text": 0.46,
    "audio": 0.52,
    "vision": 0.54
  },
  "weights": {
    "text": 0.3,
    "audio": 0.3,
    "vision": 0.4
  },
  "deception_score": 0.512
}
```

A `deception_score` closer to `1.0` indicates higher likelihood of deception; closer to `0.0` indicates lower likelihood.

---

## How It Works

### 1. Text / Linguistic Analysis
The audio track is transcribed using [OpenAI Whisper](https://github.com/openai/whisper). The resulting transcript is analyzed for hesitation markers (e.g., "um", "uh", "I mean"), unusual pauses, and other deceptive speech patterns that contribute to the `text_score`.

### 2. Acoustic Analysis
[Librosa](https://librosa.org/) extracts low-level acoustic features from the raw audio waveform — including pitch variance, silence ratio, and energy envelope characteristics — to compute the `audio_score`.

### 3. Vision Analysis
[MediaPipe](https://mediapipe.dev/) Face Mesh tracks 468 facial landmarks across extracted video frames. Blink rate per minute and the standard deviation of eye movement are computed and mapped to a `vision_score`.

### 4. Weighted Fusion
The three modality scores are combined using a configurable weighted average:

```
deception_score = w_text × text_score + w_audio × audio_score + w_vision × vision_score
```

Default weights are `0.3 / 0.3 / 0.4`, giving slightly more emphasis to visual cues.

---

## Dependencies

| Library | Purpose |
|---|---|
| `openai-whisper` | Speech-to-text transcription |
| `librosa` | Audio feature extraction |
| `mediapipe` | Facial landmark tracking |
| `opencv-python` | Video frame extraction |
| `moviepy` | Audio/video processing |
| `torch` + `torchvision` | Deep learning backend |
| `numpy`, `pandas` | Numerical computation |
| `scikit-learn`, `joblib` | ML utilities |
| `soundfile`, `ffmpeg-python` | Audio I/O |
| `albumentations`, `Pillow` | Image augmentation |

---

## Ethical Disclaimer

> ⚠️ This project is intended for **research and educational purposes only**.
>
> Automated deception detection is an active area of research and is **not reliable enough for real-world decision-making**. No AI system should be used as the sole basis for judgments about a person's honesty. Results from this tool should never be used in legal, forensic, employment, or any high-stakes context.

---

## License

This project is licensed under the [MIT License](LICENSE).

---


