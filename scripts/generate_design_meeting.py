#!/usr/bin/env python3
"""
Generates a 2-Slide Engineering Architecture Review Multimodal Pack
using pure Python (Pillow + soundfile + numpy + Windows SAPI TTS).
Works on Windows without requiring external FFmpeg installation.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "storage" / "demo_media"
KEYFRAME_DIR = ROOT_DIR / "storage" / "keyframes"

SLIDES = [
    {
        "id": "slide_1",
        "title": "SLIDE 1 // BOTTLENECK: NATIVE TOOL EXECUTOR",
        "subtitle": "Legacy Agent-Tool-Presentation & Multi-Turn IPC Latency",
        "boxes": [
            ("Agent Scope", "#00f0ff"),
            ("agent-tool-presentation (DEPRECATED)", "#ff0055"),
            ("NativeToolExecutor (N Round-Trips)", "#fcee0a"),
        ],
        "notes": "Problem: Sequential tool calls block the Cordis event loop.",
        "speech": (
            "First, in agent tool presentation, mounting native tool executor causes high latency "
            "because the agent executes sequential tool batches over multiple turns."
        ),
    },
    {
        "id": "slide_2",
        "title": "SLIDE 2 // TARGET: PTC PROGRAMMATIC SANDBOX",
        "subtitle": "Migrate Cordis Context to @deepseek-ai/dsh-tool-executor-ptc",
        "boxes": [
            ("Cordis ctx.tools (defineTool)", "#00f0ff"),
            ("PtcToolExecutor (Single-Turn SDK)", "#00ff9d"),
            ("run_code Sandbox Execution", "#00ff9d"),
        ],
        "notes": "Architecture Decision: Route multi-step tool calls through PTC run_code.",
        "speech": (
            "Second, we should migrate the Cordis context to P T C tool executor and register define tool "
            "so models synthesize a single run code SDK program."
        ),
    },
]

def draw_whiteboard_slide(slide: dict, img_path: Path, thumb_path: Path):
    img = Image.new("RGB", (960, 540), color=(9, 13, 22))
    draw = ImageDraw.Draw(img)

    # Outer frame & header
    draw.rectangle([16, 16, 944, 524], outline=(30, 41, 59), width=2)
    draw.rectangle([16, 16, 944, 86], fill=(15, 23, 42), outline=(0, 240, 255), width=2)

    font_lg = ImageFont.load_default()
    draw.text((36, 32), slide["title"], fill=(0, 240, 255), font=font_lg)
    draw.text((36, 56), slide["subtitle"], fill=(148, 163, 184), font=font_lg)

    # Draw whiteboard boxes
    y = 140
    for idx, (label, color) in enumerate(slide["boxes"]):
        x1, y1, x2, y2 = 140, y, 820, y + 72
        draw.rectangle([x1, y1, x2, y2], fill=(17, 24, 39), outline=color, width=3)
        draw.text((x1 + 28, y1 + 28), f"[{idx + 1}]  {label}", fill=(241, 245, 249), font=font_lg)
        if idx < len(slide["boxes"]) - 1:
            draw.line([(480, y2), (480, y2 + 36)], fill=(0, 240, 255), width=3)
        y += 108

    draw.rectangle([40, 468, 920, 506], fill=(6, 40, 30), outline=(0, 255, 157), width=1)
    draw.text((56, 482), f"WHITEBOARD NOTE: {slide['notes']}", fill=(0, 255, 157), font=font_lg)
    img.save(img_path, quality=92)

    # Also save 256px keyframe thumbnail for the phase-1 keyframe store
    thumb = img.resize((256, 144))
    thumb.save(thumb_path, quality=90)

def synthesize_wav(text: str, wav_path: Path):
    safe = text.replace('"', "'")
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono); "
        f'$s.SetOutputToWaveFile("{wav_path}", $f); '
        f'$s.Speak("{safe}"); '
        "$s.Dispose();"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)

    audio_chunks = []
    manifest_slides = []
    cur_sec = 0.0

    for idx, s in enumerate(SLIDES, start=1):
        img_p = OUT_DIR / f"{s['id']}.jpg"
        thumb_p = KEYFRAME_DIR / f"frame_{idx:04d}.jpg"
        wav_p = OUT_DIR / f"{s['id']}.wav"

        draw_whiteboard_slide(s, img_p, thumb_p)
        synthesize_wav(s["speech"], wav_p)

        data, sr = sf.read(str(wav_p), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        dur = round(len(data) / sr, 2)
        audio_chunks.append(data)

        manifest_slides.append({
            "slide_index": idx,
            "id": s["id"],
            "title": s["title"],
            "start_sec": cur_sec,
            "end_sec": round(cur_sec + dur, 2),
            "image_path": str(img_p),
            "keyframe_path": str(thumb_p),
            "wav_path": str(wav_p),
        })
        cur_sec = round(cur_sec + dur, 2)

    # Stitch full 16kHz master audio track using soundfile + numpy
    master_wav = KEYFRAME_DIR / "track_16k.wav"
    merged_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(16000, dtype="float32")
    sf.write(str(master_wav), merged_audio, 16000, subtype="PCM_16")

    manifest_path = ROOT_DIR / "storage" / "design_review_manifest.json"
    manifest_path.write_text(json.dumps({
        "total_duration_sec": cur_sec,
        "master_wav": str(master_wav),
        "slides": manifest_slides,
    }, indent=2), encoding="utf-8")

    # Optional: If ffmpeg happens to be installed, also mux an .mp4 file
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        print("FFmpeg detected — muxing optional MP4 preview...")

    print(json.dumps({
        "status": "ready",
        "slides_generated": len(manifest_slides),
        "total_duration_sec": cur_sec,
        "keyframes_dir": str(KEYFRAME_DIR),
        "manifest": str(manifest_path),
    }, indent=2))

if __name__ == "__main__":
    main()