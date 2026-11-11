#!/usr/bin/env python3
"""
Phase 3 Worker: Speak-to-Repo Artifact Compiler.
Retrieves aligned meeting segments + source code from LanceDB (256d MRL),
synthesizes Markdown documentation with Mermaid architecture diagrams and
millisecond acoustic citations, and optionally stages a Git branch.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import psutil

from ast_extractor import extract_python_symbols, extract_ts_symbols
from gemma_indexer import query_index

def format_ms(ms: int) -> str:
    total_sec = max(0, int(ms // 1000))
    mins, secs = divmod(total_sec, 60)
    return f"{mins:02d}:{secs:02d}"

def build_mermaid_diagram(topic: str, code_hits: List[Dict[str, Any]], meeting_hits: List[Dict[str, Any]]) -> str:
    """Generates a deterministic Mermaid architecture flow connecting spoken intent to repo modules."""
    lines = [
        "```mermaid",
        "flowchart LR",
        '    subgraph Acoustic["Phase 1: Spoken Design Intent (Whistle 16.9MB)"]',
    ]
    if meeting_hits:
        for idx, m in enumerate(meeting_hits[:3]):
            ts_label = f"{format_ms(m['timestamp_start_ms'])}-{format_ms(m['timestamp_end_ms'])}"
            short_text = m["snippet"][:55].replace('"', "'") + "..."
            lines.append(f'        M{idx}["[{ts_label}] {short_text}"]')
    else:
        lines.append('        M0["Spoken Architecture Specification"]')

    lines.append("    end")
    lines.append('    subgraph VectorIndex["Phase 2: EmbeddingGemma-2 (256d MRL + LanceDB)"]')
    lines.append('        V0(("Cosine Alignment"))')
    lines.append("    end")
    lines.append('    subgraph RepoModules["Phase 3: Target Repository Modules"]')

    for idx, c in enumerate(code_hits[:4]):
        fp = c["filepath"] or f"module_{idx}"
        score = c["score"]
        lines.append(f'        C{idx}["{fp}<br/>(sim: {score:.3f})"]')

    lines.append("    end")

    for idx in range(max(1, len(meeting_hits[:3]))):
        lines.append(f"    M{idx} --> V0")
    for idx in range(len(code_hits[:4])):
        lines.append(f"    V0 --> C{idx}")

    lines.append("```")
    return "\n".join(lines)

def compile_markdown_artifact(
    goal: str,
    repo_path: str,
    retrieval: Dict[str, Any],
    out_doc_path: str,
    git_branch: str = "",
) -> Dict[str, Any]:
    process = psutil.Process(os.getpid())
    t0 = time.perf_counter()

    hits = retrieval.get("hits", [])
    meeting_hits = [h for h in hits if h["modality"] == "meeting_segment"]
    code_hits = [h for h in hits if h["modality"] == "code"]

    repo_root = Path(repo_path).resolve()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mermaid_block = build_mermaid_diagram(goal, code_hits, meeting_hits)

    md_lines = [
        f"# Architecture Spec: {goal}",
        "",
        f"> **Auto-Compiled by Speak-to-Repo** on `{now_str}` | **Vector Space:** `EmbeddingGemma-2 (256d MRL)` | **Query Latency:** `{retrieval.get('query_embed_ms', 0)} ms`",
        "",
        "## 1. Acoustic Design Decisions (Cactus Whistle Aligned Segments)",
        "",
    ]

    if meeting_hits:
        for m in meeting_hits:
            start_str = format_ms(m["timestamp_start_ms"])
            end_str = format_ms(m["timestamp_end_ms"])
            md_lines.append(f"* **[{start_str} - {end_str}]** *(Cosine Sim: `{m['score']:.4f}`)*")
            md_lines.append(f'  > "{m["snippet"]}"')
            if m.get("frame_path"):
                md_lines.append(f"  * **Aligned Visual Keyframe:** `{m['frame_path']}`")
    else:
        md_lines.append("* *No direct spoken transcript segments exceeded threshold.*")

    md_lines.extend([
        "",
        "## 2. Cross-Modal Architecture Topology",
        "",
        mermaid_block,
        "",
        "## 3. Grounded Repository Modules & AST Signatures",
        "",
        "| Filepath | Cosine Similarity (`256d`) | Extracted AST Symbols |",
        "| :--- | :--- | :--- |",
    ])

    for c in code_hits:
        fp = c["filepath"]
        full_file = repo_root / fp
        symbols = set()
        if full_file.exists() and full_file.is_file():
            txt = full_file.read_text(encoding="utf-8", errors="ignore")
            if fp.endswith(".py"):
                symbols = extract_python_symbols(txt)
            elif fp.endswith((".ts", ".tsx")):
                symbols = extract_ts_symbols(txt)
        sym_str = ", ".join(f"`{s}`" for s in sorted(symbols)[:6]) if symbols else "*(module entry)*"
        md_lines.append(f"| `{fp}` | `{c['score']:.4f}` | {sym_str} |")

    md_lines.extend([
        "",
        "## 4. Source Context Snippets",
        "",
    ])

    for c in code_hits[:3]:
        fp = c["filepath"]
        ext = Path(fp).suffix.lstrip(".") or "ts"
        md_lines.extend([
            f"### `{fp}`",
            f"```{ext}",
            c["snippet"].strip(),
            "```",
            "",
        ])

    out_file = Path(out_doc_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(md_lines), encoding="utf-8")

    branch_status = "skipped"
    if git_branch:
        try:
            subprocess.run(["git", "-C", str(repo_root), "checkout", "-B", git_branch], check=True, capture_output=True)
            branch_status = f"checked_out:{git_branch}"
        except Exception as e:
            branch_status = f"git_error:{e}"

    summary = {
        "status": "compiled",
        "goal": goal,
        "output_markdown": str(out_file),
        "meeting_citations_count": len(meeting_hits),
        "code_modules_linked": len(code_hits),
        "git_branch": branch_status,
        "query_embed_ms": retrieval.get("query_embed_ms"),
        "compile_elapsed_sec": round(time.perf_counter() - t0, 3),
        "peak_worker_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
    }
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Phase 3 Speak-to-Repo Artifact Compiler")
    parser.add_argument("--goal", required=True, help="Engineering specification or topic to compile")
    parser.add_argument("--repo", required=True, help="Target Git repository path")
    parser.add_argument("--db", default="storage/lancedb", help="LanceDB directory")
    parser.add_argument("--out-doc", default="storage/compiled_docs/architecture_spec.md", help="Output Markdown path")
    parser.add_argument("--top-k", type=int, default=6, help="Number of multimodal hits to retrieve")
    parser.add_argument("--branch", default="", help="Optional Git branch name to create/switch")
    args = parser.parse_args()

    # Run retrieval first (suppressing intermediate print)
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        ret = query_index(args.goal, args.db, top_k=args.top_k, mrl_dim=256)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    compile_markdown_artifact(args.goal, args.repo, ret, args.out_doc, args.branch)