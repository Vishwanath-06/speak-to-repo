# Speak-to-Repo Compiler

> **CPU-Native Multimodal Engineering Design Review & Codebase Compiler**  
> Built with **`Cactus-Compute/whistle` (16.9 MB)**, **`google/embeddinggemma-2` (256d Matryoshka MRL)**, **`LanceDB`**, and **`deepseek-ai/deepseek-harness` (Cordis Kernel)**

**Speak-to-Repo** converts recorded engineering design reviews (whiteboard slide decks + spoken discussions) and live microphone overrides into grounded repository deliverables:
1. **Codebase Architecture & Sequence Topologies (`Mermaid.js`)** mapped to real repository modules.
2. **Actionable Unified Git Diffs (`changes.patch`)** parameterized directly by spoken numbers and architectural rules.
3. **Multimodal Architecture Decision Records (`ADR-compiled.md`)** linking visual slide keyframes, `80 ms` word timestamps, and `256d` cosine-matched source files.

Designed from the ground up for **CPU-only developer laptops (16 GB RAM)**, the compiler runs within a **sub-500 MB query memory envelope** using PyTorch submodule pruning and Windows Working-Set reclamation.

---

## System Architecture & 4-Phase Memory Lifecycle

Running ASR, multimodal vision/code encoders, and an agent runtime simultaneously on CPU typically consumes 4–8 GB of RAM. **Speak-to-Repo** uses a **Sequential Modality Swap** pattern governed by reversible `ctx.effect()` hooks on the **DeepSeek Harness (`dsh`) Cordis kernel**:

```mermaid
flowchart TB
    subgraph Phase1["Phase 1: AST-Biased Multimodal Ingest (98.4 MB Peak RSS)"]
        A1["Target Git Repo"] -->|os.walk AST Harvester| A2["150 Domain Identifiers"]
        A3["Design Review Slides + 16kHz Audio"] -->|Keyframes + PCM Windows| A4["Cactus Whistle (16.9 MB, 2-bit/4-bit SIMD)"]
        A2 -->|Aho-Corasick Beam Bias + Phonetic Snap| A4
        A4 -->|277.4 tok/s + 80ms Word Timestamps| A5["Validated Voice Gate"]
    end

    subgraph Phase2["Phase 2: Matryoshka Multimodal Indexing (256d MRL)"]
        A5 --> B1["EmbeddingGemma-2 (bfloat16 CPU)"]
        A1 -->|Source Files (.ts, .py, .md)| B1
        B1 -->|MRL Slicing + L2 Norm (3x Compression)| B2[("LanceDB On-Disk Vector Store")]
    end

    subgraph Phase3["Phase 3: Pruned Query & Multi-Scenario Compiler (491.3 MB RSS)"]
        C1["Live Mic Voice Override (2A / 2B / 2C)"] --> A5
        A5 -->|Valid Command| C2["EmbeddingGemma-2 (Vision/Audio Towers Pruned)"]
        A5 -.->|Unclear Audio: Hold State + 1.5s Toast| C1
        C2 -->|237 ms Cosine Search| B2
        B2 -->|Top-K Grounded Modules| C3["Mermaid Topology + Git Diff + ADR Compiler"]
    end

    subgraph Phase4["Phase 4: Cordis Event Log & Working-Set GC"]
        C3 --> D1["Append-Only Session Log (session_events.jsonl)"]
        D1 -->|Win32 SetProcessWorkingSetSize / ctx.effect| D2["Reclaim Idle RAM (<100 MB)"]
    end
```

---

## Live Compilation Scenarios (Step 2 Voice Overrides)

After ingesting the baseline design review slides (**Step 1**), speaking any of the following architectural overrides into the **16 kHz Live Mic** (or triggering the presets) dynamically re-ranks the `LanceDB` modules, rewires the **Mermaid architecture diagram**, and compiles a new **Unified Git Patch (`changes.patch`)**:

| Scenario | Spoken Voice Command | AST Symbols Canonicalized | Compiled Deliverables (`changes.patch` & Mermaid) |
| :--- | :--- | :--- | :--- |
| **2A: PTC Sandbox Guard** | *"In P T C tool executor, cap max nested calls to 4 with a 15 second timeout."* | `PtcToolExecutor`, `maxNestedCalls`, `4`, `15000ms` | Patches `tool-executor-ptc/src/config.ts` with `.max(4)` and `timeoutMs: 15000`; renders `SandboxAbortError` guard graph. |
| **2B: Native Batch & Retry** | *"In native tool executor, run execute tool batch with concurrency 8 and 3 retries."* | `NativeToolExecutor`, `executeToolBatch`, `8`, `3` | Patches `tool-executor-native/src/index.ts` with `maxConcurrency: 8` and `retryAttempts: 3`; renders Semaphore + Backoff loop. |
| **2C: Hybrid Executor Router** | *"Switch agent tool presentation to a hybrid router when batch size exceeds 2."* | `agent-tool-presentation`, `NativeToolExecutor`, `PtcToolExecutor`, `2` | Patches `agent-tool-presentation/src/index.ts` with `selectExecutorForTurn(threshold = 2)`; renders dual-path decision tree. |

* **Transaction-Safe Voice Validation Gate:** If a spoken take is too short, noisy, or unrecognized, the backend holds the existing repository and UI state intact and displays a transient warning banner that auto-dismisses after **1.5 seconds**.

---

## Hardware Requirements & Prerequisites

* **Hardware:** Standard CPU-only laptop or desktop (x86_64 / ARM64) with **16 GB RAM** (no discrete GPU required).
* **Operating System:** Windows 10/11 (PowerShell) or Linux/macOS.
* **Runtime Requirements:**
  * **Python 3.10 – 3.12**
  * **Node.js 20+** and **pnpm** *(optional, required only if building the full `deepseek-harness` workspace)*
  * **Hugging Face Account:** Authenticated via `hf auth login` with access to `google/embeddinggemma-2`.

---

## Installation & Setup

### 1. Clone & Initialize Python Virtual Environment (Windows PowerShell)
```powershell
git clone [https://github.com/Vishwanath-06/speak-to-repo.git](https://github.com/Vishwanath-06/speak-to-repo.git)
cd speak-to-repo

python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
. .\.venv\Scripts\Activate.ps1

python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Authenticate Hugging Face & Prepare Target Repo
```powershell
# Log in to pull google/embeddinggemma-2 weights
hf auth login

# Clone deepseek-harness (used as the target codebase for AST biasing & indexing)
git clone [https://github.com/deepseek-ai/deepseek-harness.git](https://github.com/deepseek-ai/deepseek-harness.git)
```

### 3. Build the Initial `256d` Matryoshka Vector Index
```powershell
# Generate the 2-slide multimodal engineering review pack (whiteboard keyframes + 16kHz audio)
python scripts/generate_design_meeting.py

# Phase 1: Run AST-Biased Whistle Ingestion
python workers/whistle_ingest.py `
  --media storage/demo_media/slide_1.wav `
  --repo ./deepseek-harness/packages/core `
  --out storage/transcripts/test_run.jsonl

# Phase 2: Index repository source modules + transcripts into LanceDB (256d MRL)
python workers/gemma_indexer.py index `
  --repo ./deepseek-harness/packages/core `
  --transcript storage/transcripts/test_run.jsonl `
  --db storage/lancedb `
  --max-files 60 `
  --dim 256
```

---

## Running the Live Demo Studio, CLI Compiler & Tests

### 1. Launch the Interactive Web Studio (`http://127.0.0.1:8080`)
```powershell
python scripts/demo_studio.py
```
* Open **`http://127.0.0.1:8080`** in your browser.
* Click **`🎬 1. Ingest Slides (.mp4)`** to load the whiteboard keyframes and baseline architecture.
* Click **`🎙️ 2. Live Mic Override`** (or **`⚡ 2A` / `⚡ 2B` / `⚡ 2C`**) to compile new Mermaid topologies, Git diffs, and ADRs in real time.
* Click **`GC`** in the top-right memory bar to trigger `SetProcessWorkingSetSize` and watch idle RAM drop below `100 MB`.

### 2. Run Event-Sourced CLI Session Forking
```powershell
python scripts/cordis_session_runner.py `
  --fork-session sess_initial `
  --goal "Programmatic Tool Calling PTC run_code sandbox and SDK generation" `
  --repo ./deepseek-harness/packages/core `
  --out-doc storage/compiled_docs/ptc_sandbox_spec.md
```

### 3. Run the Automated Unit Test Suite & Hardware Benchmarks
```powershell
# Run fast unit tests (<2 seconds on CPU)
python -m unittest discover -s tests -v

# Run full MRL dimension sweep (128d-768d) & Whistle ablation benchmark
python scripts/benchmark_suite.py
```

---

## Measured CPU Hardware Benchmarks

Measured on a 16 GB RAM Windows x64 CPU host (`Python 3.12`, `torch 2.15.0+cpu`, `bfloat16`):

### 1. Cordis Subprocess Memory Lifecycle
| Pipeline Stage | Active Subsystem | Peak Worker RSS (MB) | Post-Task Retained RAM | Throughput / Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Acoustic Ingest** | `Cactus-Compute/whistle` (`16.9 MB` `.cact`) | **`98.43 MB`** | `0 MB` (`ctx.effect` reaped) | **`277.4 tok/s`** (`482.6 ms` TTFT) |
| **Phase 2: Batch Indexing** | `EmbeddingGemma-2` + `LanceDB` Write | **`1,269.60 MB`** | `0 MB` (`ctx.effect` reaped) | `61` cross-modal chunks indexed |
| **Phase 3: Pruned Query & Compile** | `EmbeddingGemma-2` (Vision/Audio Pruned) | **`491.35 MB`** *(60.1% reduction)* | `0 MB` (`ctx.effect` reaped) | **`237.27 ms`** query + **`2 ms`** compile |
| **Phase 4: Forked Session Compile** | `cordis_session_runner.py --fork-session` | **`494.64 MB`** | `0 MB` (`ctx.effect` reaped) | **`12.76 s`** total *(47x faster than cold index)* |

### 2. Matryoshka Representation Learning (MRL) Dimension Sweep (`EmbeddingGemma-2`)
| Vector Dim | Float32 Bytes/Vec | Storage Savings | Recall@5 (vs `768d`) | Mean Top-1 Cosine | Matrix Search Latency (CPU) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`128d`** | `512 B` | **`6.0x`** | `45.0%` | `0.8202` | **`107.27 µs`** |
| **`256d`** *(Default)* | `1024 B` | **`3.0x`** | `70.0%` *(100% Top-1 Hit)* | `0.7320` | `153.75 µs` |
| **`512d`** | `2048 B` | **`1.5x`** | **`85.0%`** | `0.7122` | `125.43 µs` |
| **`768d`** | `3072 B` | `1.0x` | **`100.0%`** | `0.7122` | `136.85 µs` |

---

## Repository Structure

```text
speak-to-repo/
├── requirements.txt              # CPU-pinned Python dependencies
├── README.md                     # Architecture, setup, and benchmark documentation
├── workers/
│   ├── ast_extractor.py          # Top-down os.walk AST harvester & import symbol scraper
│   ├── whistle_ingest.py         # Cactus Whistle (16.9MB) 30s windowed transcriber
│   ├── gemma_indexer.py          # Selective-load EmbeddingGemma-2 (tower pruning + 256d MRL)
│   └── repo_compiler.py          # CLI Markdown, Mermaid, and acoustic citation compiler
├── scripts/
│   ├── demo_studio.py            # Live WebAudio 16kHz Mic + Mermaid + Git Diff Studio (v5)
│   ├── generate_design_meeting.py# Pure-Python 2-slide whiteboard + 16kHz WAV generator
│   ├── cordis_session_runner.py  # Append-only JSONL event log runner with --fork-session
│   └── benchmark_suite.py        # MRL dimension sweep & Whistle ablation suite
├── cordis-plugin/
│   └── src/speak-ingest.ts       # DeepSeek Harness Cordis plugin (ctx.tools + ctx.effect)
├── tests/
│   └── test_speak_to_repo.py     # Unit tests for AST, voice gate, Mermaid safety & Git diffs
└── storage/
    ├── compiled_docs/            # Generated ADR-compiled.md & changes.patch artifacts
    ├── sessions/                 # Append-only session_events.jsonl trajectory logs
    └── BENCHMARK_RESULTS.md      # Reproducible hardware telemetry report
```