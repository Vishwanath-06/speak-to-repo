#!/usr/bin/env python3
"""
Phase 4 Event-Sourced Session Orchestrator (Cordis Spatiotemporal Pattern).
Records append-only typed events for every pipeline stage and supports instant
session forking (--fork-session) to re-compile new documentation goals without re-indexing.
"""
import argparse
import datetime
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
PYTHON_BIN = sys.executable
EVENT_LOG_PATH = ROOT_DIR / "storage" / "sessions" / "session_events.jsonl"

def append_event(session_id: str, event_type: str, payload: Dict[str, Any], parent_session: str = "") -> Dict[str, Any]:
    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "event_id": f"evt_{uuid.uuid4().hex[:8]}",
        "session_id": session_id,
        "parent_session_id": parent_session or None,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "type": event_type,
        "payload": payload,
    }
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    return event

def run_worker_stage(session_id: str, stage_name: str, script_rel: str, args: list[str], parent_session: str = "") -> Dict[str, Any]:
    append_event(session_id, f"{stage_name}.started", {"script": script_rel, "args": args}, parent_session)
    proc = subprocess.run(
        [PYTHON_BIN, str(ROOT_DIR / script_rel), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    result = json.loads(proc.stdout.strip())
    append_event(session_id, f"{stage_name}.completed", result, parent_session)
    return result

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Event-Sourced Speak-to-Repo Session Runner")
    parser.add_argument("--goal", required=True, help="Architecture compilation goal")
    parser.add_argument("--repo", default="./deepseek-harness/packages/core", help="Target repository path")
    parser.add_argument("--media", default="storage/test_speech.wav", help="Input recording path")
    parser.add_argument("--fork-session", default="", help="Existing session_id to fork from (skips Ingest & Index)")
    parser.add_argument("--out-doc", default="storage/compiled_docs/forked_spec.md", help="Output Markdown path")
    args = parser.parse_args()

    session_id = f"sess_{uuid.uuid4().hex[:6]}"
    transcript_jsonl = "storage/transcripts/test_run.jsonl"
    db_dir = "storage/lancedb"

    if not args.fork_session:
        run_worker_stage(session_id, "phase1_ingest", "workers/whistle_ingest.py", [
            "--media", args.media, "--repo", args.repo, "--out", transcript_jsonl
        ])
        run_worker_stage(session_id, "phase2_index", "workers/gemma_indexer.py", [
            "index", "--repo", args.repo, "--transcript", transcript_jsonl, "--db", db_dir, "--max-files", "60", "--dim", "256"
        ])
    else:
        append_event(session_id, "session.forked", {
            "forked_from": args.fork_session,
            "reused_lancedb": db_dir,
            "new_goal": args.goal,
        }, parent_session=args.fork_session)

    compile_res = run_worker_stage(
        session_id,
        "phase3_compile",
        "workers/repo_compiler.py",
        ["--goal", args.goal, "--repo", args.repo, "--db", db_dir, "--out-doc", args.out_doc],
        parent_session=args.fork_session,
    )

    print(json.dumps({
        "session_id": session_id,
        "forked_from": args.fork_session or None,
        "event_log": str(EVENT_LOG_PATH),
        "compile_summary": compile_res,
    }, indent=2))

if __name__ == "__main__":
    main()