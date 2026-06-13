from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import librosa
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from utils import minmax_normalize

logger = logging.getLogger(__name__)

_WHISPER_WARNING_EMITTED = False
_WHISPER_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_SENTENCE_MODEL = None

# Known deceptive vs. truthful anchor phrases used to compute a semantic deception score
# via cosine similarity. These are embedded once and cached.
_DECEPTIVE_ANCHORS = [
    "I didn't do it, I swear.",
    "You have to believe me, I'm telling the truth.",
    "I have no idea what you're talking about.",
    "I would never do something like that.",
    "To be honest with you, I wasn't even there.",
    "I promise I'm not lying.",
]
_TRUTHFUL_ANCHORS = [
    "I went to the store at around three o'clock.",
    "She called me and we talked for about twenty minutes.",
    "The meeting started late because of traffic.",
    "I left the office at five and drove straight home.",
    "We had dinner together and then watched a movie.",
    "I submitted the report before the deadline.",
]
_ANCHOR_EMBEDDINGS: dict[str, np.ndarray] | None = None

FILLER_WORDS = {
    "um",
    "uh",
    "erm",
    "hmm",
    "like",
    "you know",
    "i mean",
}


@dataclass
class TextAudioResult:
    transcript: str
    hesitation_score: float
    pitch_score: float
    silence_score: float
    text_score: float          # fused: hesitation + semantic deception
    audio_score: float
    transcript_embedding: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Clamp all score fields to [0.0, 1.0] and reject NaN values."""
        score_fields = ("hesitation_score", "pitch_score", "silence_score", "text_score", "audio_score")
        for fname in score_fields:
            val = getattr(self, fname)
            if math.isnan(val):
                raise ValueError(f"TextAudioResult.{fname} is NaN — check upstream feature computation.")
            setattr(self, fname, max(0.0, min(1.0, val)))


def _clean_transcript(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""

    alpha_count = sum(ch.isalpha() for ch in text)
    if alpha_count < 3:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def _get_cached_whisper_model(model_size: str, resolved_device: str, load_model: Any) -> Any:
    cache_key = (model_size, resolved_device)
    model = _WHISPER_MODEL_CACHE.get(cache_key)
    if model is None:
        model = load_model(model_size, device=resolved_device)
        _WHISPER_MODEL_CACHE[cache_key] = model
    return model


def _get_anchor_embeddings(sentence_model: SentenceTransformer) -> dict[str, np.ndarray]:
    """Lazily compute and cache anchor embeddings for semantic deception scoring."""
    global _ANCHOR_EMBEDDINGS
    if _ANCHOR_EMBEDDINGS is None:
        dec_embs = sentence_model.encode(_DECEPTIVE_ANCHORS, show_progress_bar=False)
        tru_embs = sentence_model.encode(_TRUTHFUL_ANCHORS, show_progress_bar=False)
        _ANCHOR_EMBEDDINGS = {
            "deceptive": np.mean(dec_embs, axis=0),   # centroid of deceptive anchor embeddings
            "truthful": np.mean(tru_embs, axis=0),    # centroid of truthful anchor embeddings
        }
    return _ANCHOR_EMBEDDINGS


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity clamped to [0, 1]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


def compute_semantic_deception_score(embedding: list[float], sentence_model: SentenceTransformer) -> float:
    """
    Compute a semantic deception score by comparing the transcript embedding to
    centroids of known deceptive vs. truthful anchor phrases.

    Returns a score in [0, 1] where 1.0 = maximally similar to deceptive language.
    """
    if not embedding or all(v == 0.0 for v in embedding):
        return 0.0

    emb = np.array(embedding, dtype=np.float32)
    anchors = _get_anchor_embeddings(sentence_model)

    sim_dec = _cosine_similarity(emb, anchors["deceptive"])
    sim_tru = _cosine_similarity(emb, anchors["truthful"])

    # Normalize: map relative similarity onto [0, 1]
    # sim_dec=1, sim_tru=0 → score=1.0 (deceptive)
    # sim_dec=0, sim_tru=1 → score=0.0 (truthful)
    total = sim_dec + sim_tru
    if total <= 0.0:
        return 0.5  # ambiguous — default to neutral
    return float(sim_dec / total)


def transcribe_with_whisper(audio_path: str, model_size: str = "base", device: str = "auto") -> str:
    global _WHISPER_WARNING_EMITTED

    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    try:
        whisper_module: Any = import_module("whisper")
        load_model = getattr(whisper_module, "load_model")
    except Exception:
        if not _WHISPER_WARNING_EMITTED:
            logger.warning(
                "Whisper ASR is unavailable. Install 'openai-whisper' (not 'whisper'). "
                "Continuing with empty transcript-based features."
            )
            _WHISPER_WARNING_EMITTED = True
        return ""

    model = _get_cached_whisper_model(model_size, resolved_device, load_model)

    try:
        result = model.transcribe(audio_path, fp16=(resolved_device == "cuda"), temperature=0.0)
        return _clean_transcript(result.get("text", ""))
    except Exception:
        if resolved_device == "cuda":
            model_cpu = _get_cached_whisper_model(model_size, "cpu", load_model)
            result = model_cpu.transcribe(audio_path, fp16=False, temperature=0.0)
            return _clean_transcript(result.get("text", ""))
        raise


def compute_hesitation_score(transcript: str) -> float:
    words = re.findall(r"\b\w+\b", transcript.lower())
    if not words:
        return 0.0

    filler_count = 0
    for i, word in enumerate(words):
        if word in FILLER_WORDS:
            filler_count += 1
        if i < len(words) - 1:
            if f"{word} {words[i + 1]}" in FILLER_WORDS:
                filler_count += 1

    filler_ratio = filler_count / len(words)
    return minmax_normalize(filler_ratio, 0.0, 0.15)


def compute_audio_scores(audio_path: str, max_audio_seconds: float | None = None) -> tuple[float, float, float]:
    duration = max_audio_seconds if max_audio_seconds and max_audio_seconds > 0 else None
    y, sr = librosa.load(audio_path, sr=16000, duration=duration)
    if y.size == 0:
        return 0.0, 0.0, 0.0

    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    voiced_f0 = f0[~np.isnan(f0)]
    pitch_std = float(np.std(voiced_f0)) if voiced_f0.size else 0.0
    voicing_ratio = float(voiced_f0.size / max(len(f0), 1))
    pitch_base = minmax_normalize(pitch_std, 15.0, 120.0)
    voicing_confidence = minmax_normalize(voicing_ratio, 0.2, 0.85)
    pitch_score = float(pitch_base * voicing_confidence)

    rms = librosa.feature.rms(y=y)[0]
    if rms.size:
        adaptive_floor = max(0.005, float(np.percentile(rms, 10)))
        silence_ratio = float(np.mean(rms < adaptive_floor))
    else:
        silence_ratio = 0.0
    silence_score = minmax_normalize(silence_ratio, 0.05, 0.35)

    audio_score = float((pitch_score + silence_score) / 2.0)
    return pitch_score, silence_score, audio_score


def get_sentence_model() -> SentenceTransformer | None:
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        try:
            _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning("Failed to load SentenceTransformer: %s", e)
            return None
    return _SENTENCE_MODEL


def analyze_text_audio(
    audio_path: str,
    model_size: str = "base",
    whisper_device: str = "auto",
    max_audio_seconds: float | None = None,
) -> TextAudioResult:
    transcript = transcribe_with_whisper(audio_path, model_size=model_size, device=whisper_device)
    hesitation_score = compute_hesitation_score(transcript)
    pitch_score, silence_score, audio_score = compute_audio_scores(audio_path, max_audio_seconds=max_audio_seconds)

    embed_model = get_sentence_model()
    if embed_model is not None and transcript:
        embedding: list[float] = embed_model.encode(transcript, show_progress_bar=False).tolist()
    else:
        embedding = [0.0] * 384

    # text_score fuses hesitation (lexical) + semantic deception (embedding-based)
    # This gives a richer text signal than hesitation alone.
    if embed_model is not None and transcript:
        semantic_score = compute_semantic_deception_score(embedding, embed_model)
        text_score = float((hesitation_score + semantic_score) / 2.0)
    else:
        # Fallback: text_score = hesitation only when embedding is unavailable
        text_score = hesitation_score

    return TextAudioResult(
        transcript=transcript,
        hesitation_score=hesitation_score,
        pitch_score=pitch_score,
        silence_score=silence_score,
        text_score=text_score,
        audio_score=audio_score,
        transcript_embedding=embedding,
    )
