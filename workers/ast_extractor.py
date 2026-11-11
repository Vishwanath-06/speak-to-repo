#!/usr/bin/env python3
"""
Fast AST Symbol Harvester using top-down os.walk directory pruning.
Scrapes Python/TS classes, functions, tool definitions, and named imports while
skipping compiled lib/ directories and .d.ts build artifacts.
"""
import argparse
import ast
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Set

IGNORE_DIRS = {
    ".git", ".venv", "node_modules", ".pnpm", "dist", "build", "lib", "out",
    "__pycache__", "vendor", "coverage", ".turbo", ".next", "target", "snapshots"
}
STOP_IDENTIFIERS = {
    "main", "init", "self", "args", "kwargs", "data", "item", "value", "result",
    "test", "none", "true", "false", "index", "types", "utils", "config", "options"
}

TS_SYMBOL_PATTERN = re.compile(
    r"(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|const)\s+([A-Za-z_][A-Za-z0-9_]{3,31})\b"
)
DEFINE_TOOL_PATTERN = re.compile(r"name:\s*['\"]([A-Za-z_][A-Za-z0-9_-]{3,31})['\"]")
IMPORT_SYMBOL_PATTERN = re.compile(r"import\s+(?:type\s+)?\{([^}]+)\}\s+from")


def to_spoken_phrase(identifier: str) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", identifier)
    return spaced.replace("_", " ").replace("-", " ").strip()


def is_clean_symbol(name: str) -> bool:
    if not (4 <= len(name) <= 32):
        return False
    low = name.lower()
    if low in STOP_IDENTIFIERS or low.startswith("test_") or "parameter_" in low:
        return False
    if name.count("_") > 3:
        return False
    return True


def extract_python_symbols(source_text: str) -> Set[str]:
    symbols: Set[str] = set()
    try:
        tree = ast.parse(source_text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name.strip("_")
                if is_clean_symbol(name):
                    symbols.add(name)
    except SyntaxError:
        pass
    return symbols


def extract_ts_symbols(source_text: str) -> Set[str]:
    symbols: Set[str] = set()
    for match in TS_SYMBOL_PATTERN.finditer(source_text):
        name = match.group(1).strip("_")
        if is_clean_symbol(name):
            symbols.add(name)
    for match in DEFINE_TOOL_PATTERN.finditer(source_text):
        name = match.group(1).strip("_")
        if is_clean_symbol(name):
            symbols.add(name)
    for match in IMPORT_SYMBOL_PATTERN.finditer(source_text):
        for part in match.group(1).split(","):
            clean_name = part.strip().split(" as ")[-1].strip()
            if clean_name.startswith("type "):
                clean_name = clean_name[5:].strip()
            if is_clean_symbol(clean_name):
                symbols.add(clean_name)
    return symbols


def harvest_repo_keywords(repo_path: str, max_keywords: int = 150) -> List[str]:
    root = Path(repo_path).resolve()
    collected: Set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]

        for fname in filenames:
            if fname.endswith((".d.ts", ".min.js", ".spec.ts", ".test.ts")):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in {".py", ".ts", ".tsx"}:
                continue

            stem = os.path.splitext(fname)[0].strip("_")
            if is_clean_symbol(stem):
                collected.add(stem)

            fpath = Path(dirpath) / fname
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
                if ext == ".py":
                    collected.update(extract_python_symbols(text))
                else:
                    collected.update(extract_ts_symbols(text))
            except OSError:
                continue

    ranked = sorted(
        collected,
        key=lambda s: (
            any(c.isupper() for c in s[1:]) and "_" not in s,
            8 <= len(s) <= 22,
            -abs(len(s) - 14),
        ),
        reverse=True,
    )

    final_keywords: List[str] = []
    seen: Set[str] = set()
    for sym in ranked:
        for variant in (sym, to_spoken_phrase(sym)):
            if variant.lower() not in seen:
                seen.add(variant.lower())
                final_keywords.append(variant)
        if len(final_keywords) >= max_keywords:
            break

    return final_keywords[:max_keywords]


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Harvest clean AST identifiers for Whistle keyword biasing")
    parser.add_argument("repo", nargs="?", default=".", help="Target repository path")
    parser.add_argument("--max-keywords", type=int, default=150, help="Maximum keywords to emit")
    parser.add_argument("--out", type=str, default="", help="Optional UTF-8 JSON output file path")
    args = parser.parse_args()

    t0 = time.perf_counter()
    keywords = harvest_repo_keywords(args.repo, max_keywords=args.max_keywords)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    payload = {
        "repo": str(Path(args.repo).resolve()),
        "elapsed_ms": elapsed_ms,
        "count": len(keywords),
        "keywords": keywords,
    }
    json_str = json.dumps(payload, indent=2)

    if args.out:
        out_file = Path(args.out).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json_str, encoding="utf-8")

    print(json_str)