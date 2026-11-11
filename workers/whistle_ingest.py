#!/usr/bin/env python3
"""
Phase 1 Ingestion Worker:
Uses Cactus Whistle (`cactus.transcribe`) with `word_timestamps=True` and AST keyword biasing,
paired with FFmpeg 256px scene-change keyframe extraction.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil
import soundfile as sf
import cactus

from ast_extractor import harvest_repo_keywords

SAMPLE_RATE = 16000
WINDOW_SECONDS = 30
SAMPLES_PER_WINDOW = SAMPLE_RATE * WINDOW_SECONDS
PTS_TIME_REGEX = re.compile(r"pts_time:([0-9]+\.?[0-9]*)")

def ensure_ffmpeg() -> str:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("FFmpeg not found in PATH. Run `winget install Gyan.FFmpeg` and restart PowerShell.")
    return ffmpeg_bin

def extract_audio_and_keyframes(video_path: Path, keyframe_dir: Path, scene_threshold: float = 0.25) -> List[Dict[str, Any]]:
    ffmpeg_bin = ensure_ffmpeg()
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    wav_path = keyframe_dir / "track_16k.wav"

    subprocess.run(
        [
            ffmpeg_bin, "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-acodec", "pcm_s16le",
            str(wav_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    frame_pattern = str(keyframe_dir / "frame_%04d.jpg")
    proc = subprocess.run(
        [
            ffmpeg_bin, "-y", "-i", str(video_path),
            "-vf", f"select='eq(n,0)+gt(scene,{scene_threshold})',scale=256:-1,showinfo",
            "-vsync", "vfr", "-q:v", "4",
            frame_pattern,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=True,
    )

    timestamps = [float(m.group(1)) for m in PTS_TIME_REGEX.finditer(proc.stderr)]
    frames = sorted(keyframe_dir.glob("frame_*.jpg"))
    return [
        {"timestamp_sec": round(timestamps[i] if i < len(timestamps) else 0.0, 3), "frame_path": str(f.resolve())}
        for i, f in enumerate(frames)
    ]

def match_keyframe(window_end_sec: float, keyframes: List[Dict[str, Any]]) -> str:
    if not keyframes:
        return ""
    active = keyframes[0]["frame_path"]
    for kf in keyframes:
        if kf["timestamp_sec"] <= window_end_sec:
            active = kf["frame_path"]
        else:
            break
    return active

def run_ingestion(media_path: str, repo_path: str, out_jsonl: str, keyframe_dir: str, threads: int = 8) -> Dict[str, Any]:
    process = psutil.Process(os.getpid())
    t0 = time.perf_counter()

    if hasattr(cactus, "set_threads"):
        cactus.set_threads(threads)

    keywords = harvest_repo_keywords(repo_path, max_keywords=150)
    media = Path(media_path).resolve()
    kf_dir = Path(keyframe_dir).resolve()

    keyframes = []
    if media.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm"}:
        keyframes = extract_audio_and_keyframes(media, kf_dir)
        wav_path = kf_dir / "track_16k.wav"
    else:
        wav_path = media

    audio_data, sr = sf.read(str(wav_path), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    total_samples = len(audio_data)
    segments = []
    ttft_samples = []
    tps_samples = []

    fd, tmp_wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        for start_idx in range(0, total_samples, SAMPLES_PER_WINDOW):
            end_idx = min(start_idx + SAMPLES_PER_WINDOW, total_samples)
            chunk = audio_data[start_idx:end_idx]
            if np.max(np.abs(chunk)) < 1e-3:
                continue

            sf.write(tmp_wav_path, chunk, SAMPLE_RATE)
            win_start_sec = round(start_idx / SAMPLE_RATE, 2)
            win_end_sec = round(end_idx / SAMPLE_RATE, 2)

            res = cactus.transcribe(tmp_wav_path, keywords=keywords, word_timestamps=True)
            if isinstance(res, str):
                res = {"text": res}

            text = str(res.get("text", "")).strip()
            if not text:
                continue

            if "ttft_ms" in res:
                ttft_samples.append(float(res["ttft_ms"]))
            if "decode_tps" in res:
                tps_samples.append(float(res["decode_tps"]))

            # Offset word timestamps relative to the full recording timeline
            words = []
            for w in res.get("words", []):
                words.append({
                    "word": w.get("word", ""),
                    "start": round(float(w.get("start", 0.0)) + win_start_sec, 3),
                    "end": round(float(w.get("end", 0.0)) + win_start_sec, 3),
                    "probability": round(float(w.get("probability", 1.0)), 4),
                })

            segments.append({
                "segment_id": len(segments),
                "timestamp_start_ms": int(win_start_sec * 1000),
                "timestamp_end_ms": int(win_end_sec * 1000),
                "frame_path": match_keyframe(win_end_sec, keyframes),
                "transcript": text,
                "language": res.get("language", "en"),
                "ttft_ms": res.get("ttft_ms", None),
                "decode_tps": res.get("decode_tps", None),
                "words": words,
            })
    finally:
        if os.path.exists(tmp_wav_path):
            try:
                os.remove(tmp_wav_path)
            except OSError:
                pass

    out_path = Path(out_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg) + "\n")

    summary = {
        "status": "completed",
        "segments_indexed": len(segments),
        "ast_keywords_injected": len(keywords),
        "keyframes_extracted": len(keyframes),
        "mean_ttft_ms": round(float(np.mean(ttft_samples)), 2) if ttft_samples else None,
        "mean_decode_tps": round(float(np.mean(tps_samples)), 2) if tps_samples else None,
        "output_jsonl": str(out_path),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "worker_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
    }
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="AST-Biased Whistle Ingestion Worker")
    parser.add_argument("--media", required=True, help="Path to input .mp4 or 16kHz .wav file")
    parser.add_argument("--repo", required=True, help="Path to target Git repository for AST biasing")
    parser.add_argument("--out", default="storage/transcripts/session_segments.jsonl", help="Output JSONL path")
    parser.add_argument("--keyframes", default="storage/keyframes", help="Directory to store extracted frames")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for Cactus SIMD kernels")
    args = parser.parse_args()

    run_ingestion(args.media, args.repo, args.out, args.keyframes, args.threads)