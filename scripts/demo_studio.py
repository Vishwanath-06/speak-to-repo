#!/usr/bin/env python3
"""
Speak-to-Repo Compiler Studio (v5 — Validated Voice Gate + 1.5s Auto-Dismiss Error Toast)
- Holds existing UI/Repo state until the spoken command passes intent & syntax validation.
- Automatically clears any recognition/validation error banner after 1.5 seconds (1500ms).
- Fixes Mermaid 10.9.8 edge-label syntax in Scenario 2C and suppresses orphan error bombs.
"""
import ctypes
import datetime
import gc
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cactus
import lancedb
import numpy as np
import psutil
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "workers"))

from ast_extractor import extract_python_symbols, extract_ts_symbols, harvest_repo_keywords
from gemma_indexer import TABLE_NAME, SelectiveGemmaEmbedder

app = FastAPI(title="Speak-to-Repo Compiler Studio")

STORAGE_DIR = ROOT_DIR / "storage"
KEYFRAME_DIR = STORAGE_DIR / "keyframes"
DEMO_MEDIA_DIR = STORAGE_DIR / "demo_media"
KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
DEMO_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/keyframes", StaticFiles(directory=str(KEYFRAME_DIR)), name="keyframes")
app.mount("/demo_media", StaticFiles(directory=str(DEMO_MEDIA_DIR)), name="demo_media")

DEFAULT_REPO = str((ROOT_DIR / "deepseek-harness" / "packages" / "core").resolve())
DB_DIR = str((STORAGE_DIR / "lancedb").resolve())
EVENT_LOG_PATH = STORAGE_DIR / "sessions" / "session_events.jsonl"

STUDIO_STATE: Dict[str, Any] = {
    "embedder": None,
    "ast_keywords": [],
    "meeting_segments": [],
}

WORD_TO_INT = {
    "one": 1, "two": 2, "to": 2, "too": 2, "three": 3, "four": 4, "for": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
}

def trim_windows_working_set():
    gc.collect()
    if os.name == "nt":
        try:
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetProcessWorkingSetSize(handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
        except Exception:
            pass

def get_rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)

def log_cordis_event(session_id: str, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    evt = {
        "event_id": f"evt_{uuid.uuid4().hex[:8]}",
        "session_id": session_id,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "type": event_type,
        "payload": payload,
    }
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evt) + "\n")
    return evt

def normalize_wav_audio(wav_path: Path):
    try:
        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        peak = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
        if peak > 1e-4:
            data = (data / peak) * 0.89
            sf.write(str(wav_path), data, sr, subtype="PCM_16")
    except Exception:
        pass

def canonicalize_spoken_ast(text: str) -> str:
    """Forgiving phonetic canonicalizer for live mic takes across 2A, 2B, and 2C."""
    phonetic_overrides = [
        (r"\bagent\s+tool\s+presentation\b|\bagent\s+presentation\b", "agent-tool-presentation"),
        (r"\b(?:btc|p\s*t\s*c|p\s*c)\s+tool\s+executors?\b", "PtcToolExecutor"),
        (r"\btool\s+executors?\s+(?:btc|p\s*t\s*c|p\s*c)\b", "PtcToolExecutor"),
        (r"\bnative\s+tool\s+executors?\b|\bnative\s+executors?\b", "NativeToolExecutor"),
        (r"\btool\s+execution\s+pipelines?\b|\bexecution\s+pipelines?\b", "ToolExecutionPipeline"),
        (r"\bexecute\s+tool\s+batch\b|\btool\s+batch\b", "executeToolBatch"),
        (r"\bdefined?\s+tools?\b", "defineTool"),
        (r"\brun\s+code\b", "run_code"),
        (r"\bmax\s+nested\s+calls?\b|\bnested\s+calls?\b", "maxNestedCalls"),
        (r"\bhigh\s+bread\b|\bhi\s+brad\b|\bhybrids?\b", "hybrid"),
        (r"\brow\s+ter\b|\broute\s+her\b|\brouters?\b", "router"),
        (r"\bs\s*d\s*k\b", "SDK"),
        (r"\bp\s*t\s*c\b|\bbtc\b", "PTC"),
        (r"\bcordless\b|\bcourt\s+is\b|\bchorus\b", "Cordis"),
    ]
    cleaned = text
    for pattern, replacement in phonetic_overrides:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # Convert spoken number words (including 'to'/'too' after 'exceeds' or 'calls') into digits
    cleaned = re.sub(
        r"\b(exceeds|over|above|threshold|to|at)\s+(to|too|two)\b",
        r"\1 2",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(to|at)\s+(for|four)\b",
        r"\1 4",
        cleaned,
        flags=re.IGNORECASE,
    )
    for word, digit in WORD_TO_INT.items():
        if word in ("to", "too", "for"):
            continue
        cleaned = re.sub(rf"\b{word}\b", str(digit), cleaned, flags=re.IGNORECASE)

    return cleaned

def validate_and_classify_override(canon_tx: str, words: List[Dict[str, Any]]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validates that the spoken voice override is clear and matches one of the 3 scenarios
    BEFORE allowing any state mutation.
    Returns: (is_valid, error_reason, spec_dict)
    """
    low = canon_tx.lower().strip()
    if len(low.split()) < 4:
        return False, "Audio too short — speak a full override command.", None

    if words:
        mean_prob = float(np.mean([float(w.get("probability", 0.8)) for w in words]))
        if mean_prob < 0.45:
            return False, f"Low acoustic confidence ({int(mean_prob*100)}%) — please speak clearly.", None

    nums = [int(n) for n in re.findall(r"\b(\d+)\b", canon_tx)]

    # Check Scenario 2C: Hybrid Router
    is_2c = any(w in low for w in ("hybrid", "router", "agent-tool-presentation", "exceeds", "single call", "batch size"))
    # Check Scenario 2B: Native Batch & Retry
    is_2b = any(w in low for w in ("concurrency", "retry", "retries", "executetoolbatch", "pipeline")) or (
        "nativetoolexecutor" in low and not is_2c
    )
    # Check Scenario 2A: PTC Sandbox Guard
    is_2a = any(w in low for w in ("maxnestedcalls", "timeout", "sandbox", "guard", "cap")) or (
        ("ptctoolexecutor" in low or "ptc" in low) and not is_2c
    )

    if is_2c:
        threshold = nums[-1] if nums else 2
        return True, "", {
            "mode": "hybrid_router",
            "title": "ADR-003: Hybrid Native/PTC Tool Router",
            "badge_label": f"batchThreshold = {threshold}",
            "batch_threshold": threshold,
            "search_boost": "agent-tool-presentation select built-in tool execution NativeToolExecutor PtcToolExecutor",
        }

    if is_2b:
        concurrency = nums[0] if len(nums) >= 1 else 8
        retries = nums[1] if len(nums) >= 2 else 3
        return True, "", {
            "mode": "native_batch",
            "title": "ADR-002: Native Parallel Batch & Retry Circuit Breaker",
            "badge_label": f"concurrency={concurrency} | retries={retries}",
            "concurrency": concurrency,
            "retries": retries,
            "search_boost": "NativeToolExecutor ToolExecutionPipeline executeToolBatch tool-executor-native",
        }

    if is_2a:
        max_calls = nums[0] if len(nums) >= 1 else 4
        timeout_sec = nums[1] if len(nums) >= 2 else 15
        return True, "", {
            "mode": "ptc_sandbox",
            "title": "ADR-001: PTC Programmatic Sandbox Hardening",
            "badge_label": f"maxCalls={max_calls} | timeout={timeout_sec}s",
            "max_nested_calls": max_calls,
            "timeout_ms": timeout_sec * 1000,
            "strict_guard": True,
            "search_boost": "PtcToolExecutor tool-executor-ptc config maxNestedCalls run_code SDK",
        }

    return (
        False,
        f'Command not recognized ("{canon_tx[:48]}...") — mention PTC Guard, Native Batch, or Hybrid Router.',
        None,
    )

def build_scenario_mermaid(spec: Dict[str, Any]) -> str:
    """
    100% Mermaid 10.9.8 safe syntax:
    Zero raw '>', '<', '(', or ')' inside |...| edge labels!
    """
    mode = spec["mode"]
    if mode == "native_batch":
        c = spec["concurrency"]
        r = spec["retries"]
        return "\n".join([
            "flowchart TB",
            "    classDef core fill:#0f172a,stroke:#00f0ff,stroke-width:2px,color:#f8fafc;",
            "    classDef pool fill:#1e1b4b,stroke:#fcee0a,stroke-width:2px,color:#fef08a;",
            "    classDef exec fill:#06281e,stroke:#00ff9d,stroke-width:2px,color:#ecfdf5;",
            "    classDef retry fill:#2a0a18,stroke:#ff0055,stroke-width:2px,color:#fda4af;",
            '    A["Cordis Agent Scope<br/>ctx.tools Registry"]:::core --> B["tool-executor-native/src/index.ts<br/>NativeToolExecutor"]:::core',
            f'    B --> C["ToolExecutionPipeline<br/>Semaphore Pool - maxConcurrency: {c}"]:::pool',
            '    C ==> D["executeToolBatch<br/>Parallel Worker Dispatch"]:::exec',
            f'    D -.->|Tool Error or Timeout| E["Exponential Backoff Circuit Breaker<br/>maxRetries: {r}"]:::retry',
            "    E -.->|Retry Dispatch| D",
        ])

    if mode == "hybrid_router":
        th = spec["batch_threshold"]
        return "\n".join([
            "flowchart TB",
            "    classDef core fill:#0f172a,stroke:#00f0ff,stroke-width:2px,color:#f8fafc;",
            "    classDef router fill:#1e1b4b,stroke:#fcee0a,stroke-width:2px,color:#fef08a;",
            "    classDef fast fill:#06281e,stroke:#00ff9d,stroke-width:2px,color:#ecfdf5;",
            "    classDef ptc fill:#1e1b4b,stroke:#00f0ff,stroke-width:2px,color:#e0f2fe;",
            '    A["agent-tool-presentation/src/index.ts<br/>Incoming Model Tool Turn"]:::core --> R{"HybridExecutorRouter<br/>Batch Threshold: ' + str(th) + ' Calls"}:::router',
            f'    R ==>|1 to {th} Calls - Fast Direct Path| N["tool-executor-native/src/index.ts<br/>NativeToolExecutor - Direct IPC"]:::fast',
            f'    R ==>|Over {th} Calls - Programmatic SDK| P["tool-executor-ptc/src/config.ts<br/>PtcToolExecutor - run_code Sandbox"]:::ptc',
        ])

    # Default: ptc_sandbox
    # Default: ptc_sandbox
    mc = spec["max_nested_calls"]
    tms = spec["timeout_ms"]
    return "\n".join([
        "flowchart TB",
        "    classDef core fill:#0f172a,stroke:#00f0ff,stroke-width:2px,color:#f8fafc;",
        "    classDef guard fill:#1e1b4b,stroke:#fcee0a,stroke-width:2px,color:#fef08a;",
        "    classDef sandbox fill:#06281e,stroke:#00ff9d,stroke-width:2px,color:#ecfdf5;",
        "    classDef abort fill:#2a0a18,stroke:#ff0055,stroke-width:2px,color:#fda4af;",
        '    A["Cordis ctx.tools<br/>defineTool Schemas"]:::core --> B["tool-executor-ptc<br/>PtcToolExecutor - SDK Generator"]:::core',
        f'    B --> G["tool-executor-ptc/src/config.ts<br/>maxNestedCalls: {mc} - timeoutMs: {tms}"]:::guard',
        '    G ==>|Valid SDK Program| S["run_code Sandbox<br/>Single-Turn Execution"]:::sandbox',
        f'    G -.->|Exceeds {mc} Calls or {tms}ms| X["SandboxAbortError<br/>Halt Runaway Trajectory"]:::abort',
    ])

def build_scenario_git_diff(spec: Dict[str, Any]) -> str:
    mode = spec["mode"]
    if mode == "native_batch":
        c = spec["concurrency"]
        r = spec["retries"]
        return "\n".join([
            "diff --git a/packages/core/tool-executor-native/src/index.ts b/packages/core/tool-executor-native/src/index.ts",
            "index 49c1a0e..b82d411 100644",
            "--- a/packages/core/tool-executor-native/src/index.ts",
            "+++ b/packages/core/tool-executor-native/src/index.ts",
            "@@ -4,7 +4,13 @@ import { projectToolSchema } from '@deepseek-ai/dsh-tools'",
            " import type { ToolDefinition, ToolExecutor, ToolBatchInput, ToolBatchResult } from '@deepseek-ai/dsh-tools'",
            " import { ToolExecutionPipeline, executeToolBatch } from '@deepseek-ai/dsh-tools'",
            " ",
            "+export interface NativeBatchPolicy {",
            f"+  maxConcurrency: number // Compiled from voice: {c}",
            f"+  retryAttempts: number  // Compiled from voice: {r}",
            "+}",
            "+",
            " export class NativeToolExecutor implements ToolExecutor {",
            "   async executeBatch(batch: ToolBatchInput): Promise<ToolBatchResult> {",
            "-    return executeToolBatch(this.pipeline, batch)",
            "+    return executeToolBatch(this.pipeline, batch, {",
            f"+      maxConcurrency: {c},",
            f"+      retryAttempts: {r},",
            "+      backoffMs: 250,",
            "+    })",
            "   }",
            " }",
        ])

    if mode == "hybrid_router":
        th = spec["batch_threshold"]
        return "\n".join([
            "diff --git a/packages/core/agent-tool-presentation/src/index.ts b/packages/core/agent-tool-presentation/src/index.ts",
            "index 84a1b2c..e19f70a 100644",
            "--- a/packages/core/agent-tool-presentation/src/index.ts",
            "+++ b/packages/core/agent-tool-presentation/src/index.ts",
            "@@ -6,9 +6,18 @@ import type { Context } from '@deepseek-ai/cordis'",
            " import z from '@deepseek-ai/schemastery'",
            " import * as NativeToolExecutor from '@deepseek-ai/dsh-tool-executor-native'",
            " import * as PtcToolExecutor from '@deepseek-ai/dsh-tool-executor-ptc'",
            " ",
            "-/** @deprecated Mount NativeToolExecutor or PtcToolExecutor directly. */",
            "+/** Hybrid Dual-Path Router compiled by Speak-to-Repo */",
            "+export const HYBRID_BATCH_THRESHOLD = " + str(th),
            "+",
            "+export function selectExecutorForTurn(toolCallCount: number) {",
            f"+  if (toolCallCount <= {th}) {{",
            "+    return NativeToolExecutor",
            "+  }",
            "+  return PtcToolExecutor",
            "+}",
        ])

    mc = spec["max_nested_calls"]
    tms = spec["timeout_ms"]
    return "\n".join([
        "diff --git a/packages/core/tool-executor-ptc/src/config.ts b/packages/core/tool-executor-ptc/src/config.ts",
        "index 31c90a1..90f2c4d 100644",
        "--- a/packages/core/tool-executor-ptc/src/config.ts",
        "+++ b/packages/core/tool-executor-ptc/src/config.ts",
        "@@ -1,8 +1,14 @@",
        " import z from '@deepseek-ai/schemastery'",
        " ",
        " export interface PtcExecutorConfig {",
        "-  maxNestedCalls?: number",
        f"+  maxNestedCalls: number // Compiled from voice: {mc}",
        f"+  timeoutMs: number      // Compiled from voice: {tms}ms",
        "+  strictSandboxGuard: boolean",
        " }",
        " ",
        " export const PtcExecutorConfig: z<PtcExecutorConfig> = z.object({",
        "-  maxNestedCalls: z.natural().default(10),",
        f"+  maxNestedCalls: z.natural().max({mc}).default({mc}),",
        f"+  timeoutMs: z.natural().default({tms}),",
        "+  strictSandboxGuard: z.boolean().default(true),",
        " })",
    ])

def build_adr_markdown(
    segments: List[Dict[str, Any]],
    spec: Dict[str, Any],
    code_hits: List[Dict[str, Any]],
    mermaid_dsl: str,
    git_diff: str,
    query_embed_ms: float,
) -> str:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {spec['title']}",
        "",
        f"* **Compiled Mode:** `{spec['mode']}` | **Active Parameters:** `{spec['badge_label']}` | **Timestamp:** `{now_str}`",
        f"* **Vector Engine:** `google/embeddinggemma-2` (`256d` MRL, `{query_embed_ms} ms` CPU search)",
        "",
        "## 1. Multimodal Timeline (Whiteboard Slides + Live Voice Spec)",
        "",
    ]
    for s in segments:
        t0 = f"00:{int(s['start_sec']):02d}"
        t1 = f"00:{int(s['end_sec']):02d}"
        lines.append(f"### [{t0} - {t1}] {s['label']}")
        lines.append(f'> "{s["canonical_transcript"]}"')
        if s.get("frame_url"):
            lines.append(f"* **Aligned Keyframe:** `{s['frame_url']}`")
        lines.append("")

    lines.extend([
        "## 2. Compiled Codebase Architecture Topology",
        "",
        "```mermaid",
        mermaid_dsl,
        "```",
        "",
        "## 3. Grounded Repository Modules (`256d` LanceDB)",
        "",
        "| Filepath | Cosine Similarity | AST Identifiers |",
        "| :--- | :--- | :--- |",
    ])
    for c in code_hits:
        syms = ", ".join(f"`{x}`" for x in c.get("symbols", [])) or "`module`"
        lines.append(f"| `{c['filepath']}` | **`{c['score']*100:.1f}%`** | {syms} |")

    lines.extend([
        "",
        "## 4. Actionable Unified Git Patch (`changes.patch`)",
        "",
        "```diff",
        git_diff,
        "```",
    ])
    return "\n".join(lines)

@app.get("/api/telemetry")
def api_telemetry():
    events = []
    if EVENT_LOG_PATH.exists():
        for line in EVENT_LOG_PATH.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-10:]:
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    return {
        "current_rss_mb": get_rss_mb(),
        "encoder_mounted": STUDIO_STATE["embedder"] is not None,
        "ceiling_mb": 1200.0,
        "events": list(reversed(events)),
    }

@app.post("/api/unwind_memory")
def api_unwind_memory():
    before_mb = get_rss_mb()
    if STUDIO_STATE["embedder"] is not None:
        STUDIO_STATE["embedder"].unload()
        STUDIO_STATE["embedder"] = None
    trim_windows_working_set()
    after_mb = get_rss_mb()
    evt = log_cordis_event("sess_gc", "cordis.effect.unwound", {
        "rss_before_mb": before_mb,
        "rss_after_mb": after_mb,
    })
    return {"rss_before_mb": before_mb, "rss_after_mb": after_mb, "event": evt, "telemetry": api_telemetry()}

@app.post("/api/ingest_design_meeting")
def api_ingest_design_meeting():
    slide_wavs = sorted(DEMO_MEDIA_DIR.glob("slide_*.wav"))
    if not slide_wavs:
        gen_script = ROOT_DIR / "scripts" / "generate_design_meeting.py"
        subprocess.run([sys.executable, str(gen_script)], check=True)
        slide_wavs = sorted(DEMO_MEDIA_DIR.glob("slide_*.wav"))

    if not STUDIO_STATE["ast_keywords"]:
        STUDIO_STATE["ast_keywords"] = harvest_repo_keywords(DEFAULT_REPO, max_keywords=150)
    keywords = STUDIO_STATE["ast_keywords"]

    if hasattr(cactus, "set_threads"):
        cactus.set_threads(8)

    segments = []
    cur_sec = 0.0
    ttfts, tpss = [], []

    for idx, wav_p in enumerate(slide_wavs):
        audio_d, sr = sf.read(str(wav_p), dtype="float32")
        dur = round(len(audio_d) / sr, 2)
        res = cactus.transcribe(str(wav_p), keywords=keywords, word_timestamps=True)
        raw_tx = str(res.get("text", "")).strip()
        canon_tx = canonicalize_spoken_ast(raw_tx)
        ttfts.append(float(res.get("ttft_ms", 480.0)))
        tpss.append(float(res.get("decode_tps", 250.0)))

        segments.append({
            "label": f"Whiteboard Slide {idx+1}",
            "start_sec": cur_sec,
            "end_sec": round(cur_sec + dur, 2),
            "raw_transcript": raw_tx,
            "canonical_transcript": canon_tx,
            "frame_url": f"/demo_media/slide_{idx+1}.jpg",
            "words": res.get("words", []),
        })
        cur_sec = round(cur_sec + dur, 2)

    STUDIO_STATE["meeting_segments"] = segments
    default_spec = {
        "mode": "ptc_sandbox",
        "title": "ADR-001: Baseline PTC Sandbox Migration",
        "badge_label": "maxCalls=3 | timeout=10s",
        "max_nested_calls": 3,
        "timeout_ms": 10000,
        "strict_guard": True,
        "search_boost": "PtcToolExecutor NativeToolExecutor agent-tool-presentation",
    }
    return compile_from_active_segments(
        spec=default_spec,
        mean_ttft=round(float(np.mean(ttfts)), 1),
        mean_tps=round(float(np.mean(tpss)), 1),
    )

@app.post("/api/upload_live_override")
async def api_upload_live_override(audio: UploadFile = File(...)):
    wav_path = STORAGE_DIR / "live_override.wav"
    wav_path.write_bytes(await audio.read())
    normalize_wav_audio(wav_path)
    return process_override_wav(wav_path, label="Live Mic Voice Spec Override")

@app.post("/api/simulate_voice_override")
def api_simulate_voice_override(prompt: str = Form(...), scenario_label: str = Form("Voice Spec Override")):
    wav_path = STORAGE_DIR / "live_override.wav"
    safe = prompt.replace('"', "'")
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
    return process_override_wav(wav_path, label=scenario_label)

def process_override_wav(wav_path: Path, label: str) -> Dict[str, Any]:
    if not STUDIO_STATE["ast_keywords"]:
        STUDIO_STATE["ast_keywords"] = harvest_repo_keywords(DEFAULT_REPO, max_keywords=150)
    keywords = STUDIO_STATE["ast_keywords"]

    audio_d, sr = sf.read(str(wav_path), dtype="float32")
    dur = round(len(audio_d) / sr, 2)
    res = cactus.transcribe(str(wav_path), keywords=keywords, word_timestamps=True)
    raw_tx = str(res.get("text", "")).strip()
    canon_tx = canonicalize_spoken_ast(raw_tx)
    words = res.get("words", [])

    # VALIDATION GATE: Do not mutate state if spoken input is incomplete or unrecognized!
    is_valid, err_msg, spec = validate_and_classify_override(canon_tx, words)
    if not is_valid or spec is None:
        return {
            "status": "validation_error",
            "error_message": err_msg,
            "heard_transcript": canon_tx or "(silence)",
            "telemetry": api_telemetry(),
        }

    base_segments = [s for s in STUDIO_STATE["meeting_segments"] if "Override" not in s["label"] and "Scenario" not in s["label"]]
    prev_end = base_segments[-1]["end_sec"] if base_segments else 0.0

    base_segments.append({
        "label": label,
        "start_sec": prev_end,
        "end_sec": round(prev_end + dur, 2),
        "raw_transcript": raw_tx,
        "canonical_transcript": canon_tx,
        "frame_url": "",
        "words": words,
    })
    STUDIO_STATE["meeting_segments"] = base_segments

    return compile_from_active_segments(
        spec=spec,
        mean_ttft=round(float(res.get("ttft_ms", 420.0)), 1),
        mean_tps=round(float(res.get("decode_tps", 260.0)), 1),
    )

def compile_from_active_segments(spec: Dict[str, Any], mean_ttft: float, mean_tps: float) -> Dict[str, Any]:
    session_id = f"sess_{uuid.uuid4().hex[:6]}"
    segments = STUDIO_STATE["meeting_segments"]
    latest_tx = segments[-1]["canonical_transcript"] if segments else ""

    if STUDIO_STATE["embedder"] is None:
        embedder = SelectiveGemmaEmbedder(mrl_dim=256)
        embedder.load(mode="text_only")
        trim_windows_working_set()
        STUDIO_STATE["embedder"] = embedder
    else:
        embedder = STUDIO_STATE["embedder"]

    t_q0 = time.perf_counter()
    query_str = f"task: code retrieval | query: {latest_tx} {spec['search_boost']}"
    q_vec = embedder.encode_mrl([query_str])[0]
    query_embed_ms = round((time.perf_counter() - t_q0) * 1000, 1)

    db = lancedb.connect(DB_DIR)
    table = db.open_table(TABLE_NAME)
    raw_hits = table.search(q_vec.tolist()).metric("cosine").limit(10).to_list()

    code_hits = []
    repo_root = Path(DEFAULT_REPO)
    for h in raw_hits:
        if h["modality"] != "code" or h["filepath"].endswith(".zh.md"):
            continue
        fp = h["filepath"]
        full_fp = repo_root / fp
        syms = set()
        if full_fp.exists() and full_fp.is_file():
            txt = full_fp.read_text(encoding="utf-8", errors="ignore")
            syms = extract_python_symbols(txt) if fp.endswith(".py") else extract_ts_symbols(txt)
        code_hits.append({
            "filepath": fp,
            "score": round(1.0 - float(h.get("_distance", 0.0)), 4),
            "symbols": sorted(syms)[:5],
        })
        if len(code_hits) >= 4:
            break

    mermaid_clean = build_scenario_mermaid(spec)
    git_diff = build_scenario_git_diff(spec)
    adr_md = build_adr_markdown(segments, spec, code_hits, mermaid_clean, git_diff, query_embed_ms)

    out_dir = STORAGE_DIR / "compiled_docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ADR-compiled.md").write_text(adr_md, encoding="utf-8")
    (out_dir / "changes.patch").write_text(git_diff, encoding="utf-8")

    trim_windows_working_set()
    log_cordis_event(session_id, f"compile.{spec['mode']}", {
        "params": spec["badge_label"],
        "top_file": code_hits[0]["filepath"] if code_hits else "none",
        "rss_mb": get_rss_mb(),
    })

    return {
        "status": "ok",
        "session_id": session_id,
        "ttft_ms": mean_ttft,
        "decode_tps": mean_tps,
        "query_embed_ms": query_embed_ms,
        "spec": spec,
        "segments": segments,
        "code_hits": code_hits,
        "mermaid_clean": mermaid_clean,
        "git_diff": git_diff,
        "compiled_markdown": adr_md,
        "telemetry": api_telemetry(),
    }

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Speak-to-Repo // Multimodal ADR & Git Patch Compiler</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  :root {
    --bg: #060911; --panel: #0c1220; --card: #111827; --border: #1e293b;
    --cyan: #00f0ff; --amber: #fcee0a; --pink: #ff0055; --green: #00ff9d;
    --text: #f1f5f9; --muted: #64748b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: #26354d; border-radius: 3px; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; padding: 10px 14px; gap: 10px; position: relative; }

  /* 1.5-SECOND AUTO-DISMISSING TOAST BANNER */
  #error-toast {
    position: fixed; top: 18px; left: 50%; transform: translateX(-50%) translateY(-130%);
    background: rgba(42, 10, 24, 0.96); border: 1px solid var(--pink); color: #fff;
    padding: 10px 18px; border-radius: 8px; font-size: 0.82rem; font-weight: 600;
    box-shadow: 0 8px 24px rgba(255, 0, 85, 0.35); z-index: 9999;
    transition: transform 0.22s ease, opacity 0.22s ease; opacity: 0; pointer-events: none;
    display: flex; align-items: center; gap: 10px;
  }
  #error-toast.visible { transform: translateX(-50%) translateY(0); opacity: 1; }

  .topbar { display: flex; justify-content: space-between; align-items: center; background: var(--panel); border: 1px solid var(--border); padding: 8px 14px; border-radius: 10px; flex-shrink: 0; flex-wrap: wrap; gap: 8px; }
  .brand { display: flex; align-items: center; gap: 8px; font-weight: 800; font-size: 1rem; color: var(--cyan); }
  .badge { font-size: 0.68rem; color: var(--amber); border: 1px solid var(--amber); padding: 2px 6px; border-radius: 4px; font-weight: 700; }
  .controls { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  button { background: #131d31; color: var(--text); border: 1px solid #2e4163; padding: 6px 11px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.76rem; transition: 0.15s; display: inline-flex; align-items: center; gap: 5px; }
  button:hover { border-color: var(--cyan); color: var(--cyan); }
  button.step1 { background: rgba(0, 240, 255, 0.12); border-color: var(--cyan); color: var(--cyan); }
  button.step2 { background: rgba(0, 255, 157, 0.14); border-color: var(--green); color: var(--green); }
  button.rec-active { background: var(--pink); border-color: var(--pink); color: #fff; animation: pulse 1s infinite; }
  button.gc-btn { border-color: var(--amber); color: var(--amber); }
  @keyframes pulse { 50% { opacity: 0.6; } }

  .ram-widget { display: flex; align-items: center; gap: 10px; background: #080c16; padding: 5px 10px; border-radius: 6px; border: 1px solid var(--border); }
  .ram-bar { width: 95px; height: 8px; background: #1e293b; border-radius: 4px; overflow: hidden; }
  .ram-fill { height: 100%; background: linear-gradient(90deg, var(--green), var(--cyan)); width: 8%; transition: width 0.35s ease; }

  .workspace { display: grid; grid-template-columns: 460px 1fr; gap: 12px; flex: 1; min-height: 0; }
  .col { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px; display: flex; flex-direction: column; gap: 10px; min-height: 0; overflow-y: auto; }

  .kpi-grid { display: grid; grid-template-columns: 1fr 1fr 1fr 1.4fr; gap: 6px; flex-shrink: 0; }
  .kpi { background: var(--card); border: 1px solid var(--border); border-radius: 6px; padding: 8px 6px; text-align: center; }
  .kpi-val { font-size: 0.9rem; font-weight: 800; color: var(--green); font-family: monospace; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .kpi-lbl { font-size: 0.62rem; color: var(--muted); margin-top: 2px; text-transform: uppercase; }

  .sec-hdr { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: var(--cyan); display: flex; justify-content: space-between; align-items: center; flex-shrink: 0; }

  .seg-list { display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }
  .seg-card { background: var(--card); border: 1px solid var(--border); border-left: 3px solid var(--cyan); border-radius: 6px; padding: 8px; display: flex; gap: 10px; align-items: flex-start; }
  .seg-card.override { border-left-color: var(--pink); background: #19101c; }
  .seg-thumb { width: 104px; height: 58px; border-radius: 4px; border: 1px solid var(--cyan); object-fit: cover; flex-shrink: 0; }
  .seg-body { flex: 1; font-size: 0.8rem; line-height: 1.4; }
  .seg-meta { font-size: 0.68rem; font-family: monospace; color: var(--amber); margin-bottom: 3px; font-weight: 700; }
  .ast-hl { background: rgba(0, 240, 255, 0.16); color: var(--cyan); padding: 0px 5px; border-radius: 3px; font-family: monospace; font-weight: 700; }

  .hits-list { display: flex; flex-direction: column; gap: 6px; flex: 1; overflow-y: auto; }
  .hit-card { background: var(--card); border: 1px solid var(--border); border-left: 3px solid var(--green); border-radius: 6px; padding: 8px 10px; }
  .hit-top { display: flex; justify-content: space-between; font-family: monospace; font-size: 0.78rem; font-weight: 700; color: var(--amber); margin-bottom: 4px; }
  .sym-pill { font-size: 0.66rem; font-family: monospace; background: #162238; color: var(--cyan); padding: 1px 6px; border-radius: 3px; margin-right: 4px; display: inline-block; }

  .tabs-bar { display: flex; gap: 8px; border-bottom: 1px solid var(--border); padding-bottom: 8px; flex-shrink: 0; }
  .tab-btn { background: #111827; border: 1px solid var(--border); color: var(--muted); padding: 7px 14px; border-radius: 6px; font-size: 0.8rem; }
  .tab-btn.active { background: rgba(0, 240, 255, 0.14); border-color: var(--cyan); color: var(--cyan); }
  .tab-pane { display: none; flex: 1; min-height: 0; overflow: auto; background: #080c16; border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .tab-pane.active { display: flex; flex-direction: column; }

  .diff-line-add { background: rgba(0, 255, 157, 0.14); color: #4ade80; display: block; padding: 1px 6px; }
  .diff-line-del { background: rgba(255, 0, 85, 0.16); color: #fb7185; display: block; padding: 1px 6px; }
  .diff-line-hdr { color: var(--cyan); font-weight: 700; display: block; margin-top: 8px; }
</style>
</head>
<body>

<div id="error-toast">
  <span style="color:var(--pink); font-size:1.05rem;">⚠️</span>
  <span id="error-toast-msg">Voice command unclear — holding current state.</span>
</div>

<div class="topbar">
  <div class="brand">
    SPEAK-TO-REPO
    <span class="badge">CPU-ONLY</span>
    <span id="status-pill" style="font-size:0.72rem; color:var(--green); font-family:monospace;">● READY</span>
  </div>

  <div class="controls">
    <button class="step1" onclick="ingestMeeting()">🎬 1. Ingest Slides (.mp4)</button>
    <button id="mic-btn" class="step2" onclick="toggleMic()">🎙️ 2. Live Mic Override</button>
    <button onclick="runScenario('2A')">⚡ 2A: PTC Guard (4, 15s)</button>
    <button onclick="runScenario('2B')">⚡ 2B: Native Batch (8, 3x)</button>
    <button onclick="runScenario('2C')">⚡ 2C: Hybrid Router (>2)</button>
  </div>

  <div class="ram-widget">
    <div>
      <div style="font-size:0.6rem; color:var(--muted); font-weight:700;">CORDIS RSS</div>
      <div class="ram-bar"><div id="ram-fill" class="ram-fill"></div></div>
    </div>
    <div id="ram-text" style="font-family:monospace; font-weight:800; font-size:0.84rem; color:var(--green);">98.4 MB</div>
    <button class="gc-btn" onclick="unwindMemory()">GC</button>
  </div>
</div>

<div class="workspace">
  <div class="col">
    <div class="kpi-grid">
      <div class="kpi"><div id="k-ttft" class="kpi-val">--</div><div class="kpi-lbl">Whistle TTFT</div></div>
      <div class="kpi"><div id="k-tps" class="kpi-val">--</div><div class="kpi-lbl">Decode TPS</div></div>
      <div class="kpi"><div id="k-qms" class="kpi-val">--</div><div class="kpi-lbl">256d Search</div></div>
      <div class="kpi"><div id="k-param" class="kpi-val" style="color:var(--amber)">--</div><div class="kpi-lbl">Compiled Policy</div></div>
    </div>

    <div class="sec-hdr"><span>Phase 1: Multimodal Timeline (Slides + Whistle 16.9MB)</span></div>
    <div id="seg-list" class="seg-list">
      <div style="color:var(--muted); font-size:0.8rem; padding:10px; background:var(--card); border-radius:6px;">
        Step 1: Click <b>"1. Ingest Slides (.mp4)"</b><br/>
        Step 2: Speak into <b>"2. Live Mic Override"</b> (or click <b>2A / 2B / 2C</b>). Unclear voice takes auto-dismiss after 1.5s without mutating the repo state!
      </div>
    </div>

    <div class="sec-hdr">
      <span>Phase 2: Grounded Repo Modules (256d MRL LanceDB)</span>
      <span style="color:var(--amber); font-family:monospace;">EmbeddingGemma-2</span>
    </div>
    <div id="hits-list" class="hits-list"></div>
  </div>

  <div class="col">
    <div class="tabs-bar">
      <button class="tab-btn active" onclick="switchTab('mermaid', this)">1. Codebase Architecture Topology (Mermaid)</button>
      <button class="tab-btn" onclick="switchTab('diff', this)">2. Compiled Git Patch (changes.patch)</button>
      <button class="tab-btn" onclick="switchTab('adr', this)">3. Multimodal ADR (ADR-compiled.md)</button>
    </div>

    <div id="tab-mermaid" class="tab-pane active" style="align-items:center; justify-content:center;">
      <div id="mermaid-stage" style="width:100%; display:flex; align-items:center; justify-content:center;">
        <span style="color:var(--muted); font-size:0.85rem;">Architecture flowchart will compile here...</span>
      </div>
    </div>

    <div id="tab-diff" class="tab-pane">
      <pre id="diff-stage" style="font-family:'Cascadia Code',monospace; font-size:0.8rem; line-height:1.5;"></pre>
    </div>

    <div id="tab-adr" class="tab-pane">
      <pre id="adr-stage" style="font-family:'Cascadia Code',monospace; font-size:0.78rem; line-height:1.5; white-space:pre-wrap;"></pre>
    </div>
  </div>
</div>

<script>
// suppressErrorRendering prevents Mermaid 10.9.8 from injecting giant red bomb SVGs into document.body!
mermaid.initialize({ startOnLoad: false, theme: 'dark', securityLevel: 'loose', suppressErrorRendering: true });

let toastTimer = null;
function showTransientError(msg) {
  const toast = document.getElementById('error-toast');
  document.getElementById('error-toast-msg').textContent = msg;
  toast.classList.add('visible');
  if (toastTimer) clearTimeout(toastTimer);
  // Automatically disappear after exactly 1.5 seconds (1500ms)
  toastTimer = setTimeout(() => {
    toast.classList.remove('visible');
  }, 1500);
}

const SCENARIOS = {
  '2A': {
    label: 'Scenario 2A: PTC Sandbox Guard',
    prompt: 'In P T C tool executor, cap max nested calls to 4 with a 15 second timeout.'
  },
  '2B': {
    label: 'Scenario 2B: Native Batch & Retry',
    prompt: 'In native tool executor, run execute tool batch with concurrency 8 and 3 retries.'
  },
  '2C': {
    label: 'Scenario 2C: Hybrid Executor Router',
    prompt: 'Switch agent tool presentation to a hybrid router when batch size exceeds 2.'
  }
};

const AST_BADGES = ["agent-tool-presentation", "defineTool", "NativeToolExecutor", "PtcToolExecutor", "ToolExecutionPipeline", "executeToolBatch", "maxNestedCalls", "run_code", "Cordis", "PTC", "SDK"];

function highlightAST(text) {
  let out = text;
  AST_BADGES.forEach(sym => {
    const reg = new RegExp(`(${sym.replace('-', '\\\\-')})`, 'gi');
    out = out.replace(reg, `<span class="ast-hl">$1</span>`);
  });
  return out;
}

function switchTab(name, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}

function formatGitDiff(rawDiff) {
  return rawDiff.split('\\n').map(line => {
    const safe = line.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    if (line.startsWith('diff --git') || line.startsWith('@@')) return `<span class="diff-line-hdr">${safe}</span>`;
    if (line.startsWith('+') && !line.startsWith('+++')) return `<span class="diff-line-add">${safe}</span>`;
    if (line.startsWith('-') && !line.startsWith('---')) return `<span class="diff-line-del">${safe}</span>`;
    return `<span>${safe}</span>`;
  }).join('\\n');
}

async function ingestMeeting() {
  const st = document.getElementById('status-pill');
  st.textContent = '● INGESTING SLIDES...';
  st.style.color = 'var(--amber)';
  const res = await fetch('/api/ingest_design_meeting', { method: 'POST' });
  await updateUI(await res.json());
}

async function runScenario(key) {
  const s = SCENARIOS[key];
  const st = document.getElementById('status-pill');
  st.textContent = `● COMPILING ${key}...`;
  st.style.color = 'var(--pink)';
  const fd = new FormData();
  fd.append('prompt', s.prompt);
  fd.append('scenario_label', s.label);
  const res = await fetch('/api/simulate_voice_override', { method: 'POST', body: fd });
  await updateUI(await res.json());
}

let mediaStream = null, audioCtx = null, processor = null, audioChunks = [], recording = false;

function encodeWAV(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, 'WAVE'); writeStr(12, 'fmt '); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  writeStr(36, 'data'); view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}

async function toggleMic() {
  const btn = document.getElementById('mic-btn');
  const st = document.getElementById('status-pill');
  if (!recording) {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const source = audioCtx.createMediaStreamSource(mediaStream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    audioChunks = [];
    processor.onaudioprocess = (e) => audioChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    source.connect(processor); processor.connect(audioCtx.destination);
    recording = true;
    btn.innerHTML = '⏹️ Stop & Validate';
    btn.classList.add('rec-active');
    st.textContent = '● LISTENING...';
    st.style.color = 'var(--pink)';
  } else {
    recording = false;
    btn.innerHTML = '🎙️ 2. Live Mic Override';
    btn.classList.remove('rec-active');
    processor.disconnect(); mediaStream.getTracks().forEach(t => t.stop());
    let totalLen = audioChunks.reduce((acc, c) => acc + c.length, 0);
    let merged = new Float32Array(totalLen);
    let off = 0;
    for (let c of audioChunks) { merged.set(c, off); off += c.length; }
    const fd = new FormData();
    fd.append('audio', encodeWAV(merged, 16000), 'live_override.wav');
    st.textContent = '● VALIDATING VOICE...';
    st.style.color = 'var(--amber)';
    const res = await fetch('/api/upload_live_override', { method: 'POST', body: fd });
    await updateUI(await res.json());
  }
}

async function unwindMemory() {
  const res = await fetch('/api/unwind_memory', { method: 'POST' });
  const data = await res.json();
  updateTelemetryDOM(data.telemetry);
  document.getElementById('status-pill').textContent = `● UNWOUND (${data.rss_after_mb} MB)`;
}

async function updateUI(data) {
  const st = document.getElementById('status-pill');

  // 1. Check if backend rejected the voice take: show 1.5s toast and DO NOT mutate UI!
  if (data.status === 'validation_error') {
    showTransientError(data.error_message);
    st.textContent = '● HELD STATE (RETRY MIC)';
    st.style.color = 'var(--amber)';
    updateTelemetryDOM(data.telemetry);
    return;
  }

  // 2. Pre-validate Mermaid syntax before mutating anything on screen
  let renderedSvg = '';
  if (data.mermaid_clean) {
    try {
      await mermaid.parse(data.mermaid_clean);
      const { svg } = await mermaid.render('m_' + Date.now(), data.mermaid_clean);
      renderedSvg = svg;
    } catch (err) {
      showTransientError('Diagram syntax check failed — holding previous graph.');
      return;
    }
  }

  // 3. Zero errors! Commit all changes to the UI simultaneously
  st.textContent = `● ${data.spec.mode.toUpperCase()}`;
  st.style.color = 'var(--green)';

  document.getElementById('k-ttft').textContent = data.ttft_ms + ' ms';
  document.getElementById('k-tps').textContent = data.decode_tps + ' tok/s';
  document.getElementById('k-qms').textContent = data.query_embed_ms + ' ms';
  document.getElementById('k-param').textContent = data.spec ? data.spec.badge_label : '--';

  const segEl = document.getElementById('seg-list');
  segEl.innerHTML = '';
  (data.segments || []).forEach(s => {
    const isOv = s.label.includes('Override') || s.label.includes('Scenario');
    const thumb = s.frame_url ? `<img class="seg-thumb" src="${s.frame_url}?t=${Date.now()}"/>` : '';
    segEl.innerHTML += `
      <div class="seg-card ${isOv ? 'override' : ''}">
        ${thumb}
        <div class="seg-body">
          <div class="seg-meta">[00:${String(Math.floor(s.start_sec)).padStart(2,'0')} - 00:${String(Math.floor(s.end_sec)).padStart(2,'0')}] ${s.label}</div>
          <div>"${highlightAST(s.canonical_transcript)}"</div>
        </div>
      </div>`;
  });

  const hitsEl = document.getElementById('hits-list');
  hitsEl.innerHTML = '';
  (data.code_hits || []).forEach(h => {
    const syms = (h.symbols || []).map(s => `<span class="sym-pill">${s}</span>`).join('');
    hitsEl.innerHTML += `
      <div class="hit-card">
        <div class="hit-top"><span>${h.filepath}</span><span style="color:var(--green)">${(h.score*100).toFixed(1)}%</span></div>
        <div>${syms}</div>
      </div>`;
  });

  if (renderedSvg) {
    document.getElementById('mermaid-stage').innerHTML = renderedSvg;
  }
  document.getElementById('diff-stage').innerHTML = formatGitDiff(data.git_diff || '');
  document.getElementById('adr-stage').textContent = data.compiled_markdown || '';
  updateTelemetryDOM(data.telemetry);
}

function updateTelemetryDOM(t) {
  if (!t) return;
  const pct = Math.min(100, Math.round((t.current_rss_mb / t.ceiling_mb) * 100));
  document.getElementById('ram-fill').style.width = pct + '%';
  document.getElementById('ram-text').textContent = t.current_rss_mb.toFixed(1) + ' MB';
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

if __name__ == "__main__":
    trim_windows_working_set()
    print("\n========================================================")
    print("  Speak-to-Repo Compiler Studio v5 (Validated Voice Gate)")
    print("  Open in browser: http://127.0.0.1:8080")
    print("========================================================\n")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")