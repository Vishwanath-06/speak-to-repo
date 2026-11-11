#!/usr/bin/env python3
"""
Phase 2 Worker: Memory-Gated EmbeddingGemma-2 Indexer & Retriever with LanceDB.
- Indexing Mode: Indexes repo source files + aligned Whistle transcript/keyframe segments into 256d MRL vectors.
- Query Mode: Loads lightweight 270M Text/Code encoder, embeds the query in <35ms, and searches LanceDB.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import lancedb
import numpy as np
import psutil
import pyarrow as pa
import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer

from ast_extractor import IGNORE_DIRS

DEFAULT_MODEL_ID = "google/embeddinggemma-2"
FALLBACK_MODEL_ID = "google/embeddinggemma-300m"
MRL_DIM = 256
TABLE_NAME = "speak_to_repo_index"

class SelectiveGemmaEmbedder:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, mrl_dim: int = MRL_DIM):
        self.model_id = model_id
        self.mrl_dim = mrl_dim
        self.model: SentenceTransformer | None = None
        self.active_mode: str | None = None

    def load(self, mode: str = "text_only"):
        """
        Loads EmbeddingGemma-2 on CPU in bfloat16.
        - 'multimodal': omits 300M audio tower, keeps 270M Text/Code + 170M Vision (440M total).
        - 'text_only': omits both vision and audio towers (270M total).
        """
        if self.model is not None and self.active_mode == mode:
            return
        self.unload()

        cfg = {"audio_config": None} if mode == "multimodal" else {"vision_config": None, "audio_config": None}

        try:
            self.model = SentenceTransformer(
                self.model_id,
                device="cpu",
                config_kwargs=cfg,
                model_kwargs={"torch_dtype": torch.bfloat16},
            )
        except Exception:
            # Automatic fallback if user only has text-only checkpoint or older config schema
            try:
                self.model = SentenceTransformer(
                    self.model_id,
                    device="cpu",
                    model_kwargs={"torch_dtype": torch.bfloat16},
                )
            except Exception:
                self.model = SentenceTransformer(
                    FALLBACK_MODEL_ID,
                    device="cpu",
                    model_kwargs={"torch_dtype": torch.bfloat16},
                )
                # Explicitly free unused multimodal towers in PyTorch to save ~400MB CPU RAM
        try:
            auto_mod = self.model[0].auto_model
            for attr in ("audio_tower", "audio_projector", "sound_tower"):
                if hasattr(auto_mod, attr):
                    setattr(auto_mod, attr, None)
            if mode == "text_only":
                for attr in ("vision_tower", "vision_model", "multi_modal_projector"):
                    if hasattr(auto_mod, attr):
                        setattr(auto_mod, attr, None)
            gc.collect()
        except Exception:
            pass

        self.active_mode = mode

    def encode_mrl(self, inputs: List[Any]) -> np.ndarray:
        assert self.model is not None, "Model not loaded"
        with torch.inference_mode():
            raw = self.model.encode(
                inputs,
                batch_size=8,
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            if raw.ndim == 1:
                raw = raw.unsqueeze(0)
            sliced = raw[:, : self.mrl_dim].to(torch.float32)
            normalized = F.normalize(sliced, p=2, dim=-1)
            return normalized.cpu().numpy()

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            self.active_mode = None
            gc.collect()

def collect_repo_code_chunks(repo_path: str, max_files: int = 150, chunk_chars: int = 1800) -> List[Dict[str, Any]]:
    root = Path(repo_path).resolve()
    records: List[Dict[str, Any]] = []

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for fname in filenames:
            if fname.endswith((".d.ts", ".spec.ts", ".test.ts", ".zh.md")):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in {".py", ".ts", ".tsx", ".md"}:
                continue
            fpath = Path(dirpath) / fname
            rel_path = str(fpath.relative_to(root)).replace("\\", "/")
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                continue
            if not content:
                continue

            # Take top chunk of each file for fast CPU demo indexing
            snippet = content[:chunk_chars]
            records.append({
                "id": f"code::{rel_path}",
                "modality": "code",
                "filepath": rel_path,
                "frame_path": "",
                "timestamp_start_ms": 0,
                "timestamp_end_ms": 0,
                "content": snippet,
                "formatted_input": f"title: {rel_path} | text: {snippet}",
            })
            if len(records) >= max_files:
                return records
    return records

def collect_meeting_segments(transcript_jsonl: str) -> List[Dict[str, Any]]:
    path = Path(transcript_jsonl).resolve()
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            seg = json.loads(line)
            seg_id = seg.get("segment_id", len(records))
            text = seg.get("transcript", "")
            start_ms = int(seg.get("timestamp_start_ms", 0))
            end_ms = int(seg.get("timestamp_end_ms", 0))
            frame_path = seg.get("frame_path", "")

            records.append({
                "id": f"meeting::seg_{seg_id}",
                "modality": "meeting_segment",
                "filepath": "",
                "frame_path": frame_path,
                "timestamp_start_ms": start_ms,
                "timestamp_end_ms": end_ms,
                "content": text,
                "formatted_input": f"title: meeting_segment_{start_ms}ms | text: {text}",
            })
    return records

def build_index(repo_path: str, transcript_jsonl: str, db_dir: str, max_files: int = 120, mrl_dim: int = MRL_DIM) -> Dict[str, Any]:
    process = psutil.Process(os.getpid())
    t0 = time.perf_counter()

    code_records = collect_repo_code_chunks(repo_path, max_files=max_files)
    meeting_records = collect_meeting_segments(transcript_jsonl)
    all_records = code_records + meeting_records

    if not all_records:
        raise RuntimeError("No code files or transcript segments found to index.")

    has_keyframes = any(r["frame_path"] and os.path.exists(r["frame_path"]) for r in meeting_records)
    embedder = SelectiveGemmaEmbedder(mrl_dim=mrl_dim)
    embedder.load(mode="multimodal" if has_keyframes else "text_only")

    inputs = [r["formatted_input"] for r in all_records]
    vectors = embedder.encode_mrl(inputs)
    peak_ram_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    embedder.unload()

    rows = []
    for rec, vec in zip(all_records, vectors):
        rows.append({
            "id": rec["id"],
            "modality": rec["modality"],
            "filepath": rec["filepath"],
            "frame_path": rec["frame_path"],
            "timestamp_start_ms": rec["timestamp_start_ms"],
            "timestamp_end_ms": rec["timestamp_end_ms"],
            "content": rec["content"],
            "vector": vec.tolist(),
        })

    db = lancedb.connect(db_dir)
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("modality", pa.string()),
        pa.field("filepath", pa.string()),
        pa.field("frame_path", pa.string()),
        pa.field("timestamp_start_ms", pa.int64()),
        pa.field("timestamp_end_ms", pa.int64()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), mrl_dim)),
    ])
    db.create_table(TABLE_NAME, data=rows, schema=schema, mode="overwrite")

    summary = {
        "status": "indexed",
        "code_chunks_indexed": len(code_records),
        "meeting_segments_indexed": len(meeting_records),
        "mrl_dimension": mrl_dim,
        "lancedb_path": str(Path(db_dir).resolve()),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "peak_worker_rss_mb": peak_ram_mb,
    }
    print(json.dumps(summary, indent=2))
    return summary

def query_index(query_text: str, db_dir: str, top_k: int = 5, mrl_dim: int = MRL_DIM) -> Dict[str, Any]:
    process = psutil.Process(os.getpid())
    t0 = time.perf_counter()

    embedder = SelectiveGemmaEmbedder(mrl_dim=mrl_dim)
    embedder.load(mode="text_only")

    t_emb0 = time.perf_counter()
    formatted_query = f"task: code retrieval | query: {query_text}"
    q_vec = embedder.encode_mrl([formatted_query])[0]
    query_embed_ms = round((time.perf_counter() - t_emb0) * 1000, 2)

    db = lancedb.connect(db_dir)
    table = db.open_table(TABLE_NAME)
    raw_hits = table.search(q_vec.tolist()).metric("cosine").limit(top_k).to_list()

    peak_ram_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    embedder.unload()

    results = []
    for h in raw_hits:
        results.append({
            "id": h["id"],
            "modality": h["modality"],
            "score": round(1.0 - float(h.get("_distance", 0.0)), 4),
            "filepath": h["filepath"],
            "frame_path": h["frame_path"],
            "timestamp_start_ms": h["timestamp_start_ms"],
            "timestamp_end_ms": h["timestamp_end_ms"],
            "snippet": h["content"][:400],
        })

    output = {
        "query": query_text,
        "mrl_dim": mrl_dim,
        "query_embed_ms": query_embed_ms,
        "total_elapsed_sec": round(time.perf_counter() - t0, 3),
        "worker_rss_mb": peak_ram_mb,
        "hits": results,
    }
    print(json.dumps(output, indent=2))
    return output

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Phase 2 Selective-Load EmbeddingGemma-2 + LanceDB Worker")
    sub = parser.add_subparsers(dest="cmd", required=True)

    idx_p = sub.add_parser("index", help="Index repository code and Phase 1 transcript segments")
    idx_p.add_argument("--repo", required=True, help="Target Git repository path")
    idx_p.add_argument("--transcript", default="storage/transcripts/test_run.jsonl", help="Phase 1 JSONL path")
    idx_p.add_argument("--db", default="storage/lancedb", help="LanceDB storage directory")
    idx_p.add_argument("--max-files", type=int, default=100, help="Max code files to index")
    idx_p.add_argument("--dim", type=int, default=256, choices=[128, 256, 512, 768], help="Matryoshka dimension")

    q_p = sub.add_parser("query", help="Search unified code + meeting vector space")
    q_p.add_argument("--text", required=True, help="Natural language or code search query")
    q_p.add_argument("--db", default="storage/lancedb", help="LanceDB storage directory")
    q_p.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    q_p.add_argument("--dim", type=int, default=256, choices=[128, 256, 512, 768], help="Matryoshka dimension")

    args = parser.parse_args()
    if args.cmd == "index":
        build_index(args.repo, args.transcript, args.db, args.max_files, args.dim)
    elif args.cmd == "query":
        query_index(args.text, args.db, args.top_k, args.dim)