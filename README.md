# Speak-to-Repo Compiler

> **CPU-native multimodal engineering design review and codebase compiler**
>
> Turn recorded design reviews and live voice instructions into repository-grounded architecture diagrams, unified Git patches, and Architecture Decision Records.

<p align="center">
  <a href="#demo">Demo</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#benchmarks">Benchmarks</a>
</p>

<p align="center">
  <img alt="Python 3.10–3.12" src="https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white">
  <img alt="CPU only" src="https://img.shields.io/badge/Runtime-CPU--only-2E8B57">
  <img alt="EmbeddingGemma 256d" src="https://img.shields.io/badge/Embeddings-256d%20MRL-7B61FF">
  <img alt="LanceDB" src="https://img.shields.io/badge/Vector%20Store-LanceDB-orange">
</p>

## Overview

**Speak-to-Repo** is an offline-first engineering assistant that connects spoken design discussions to the source code they describe. It ingests whiteboard slides, recorded audio, and a target Git repository; identifies repository symbols; indexes source and transcript segments; and uses live voice overrides to compile grounded engineering deliverables.

The system is designed for CPU-only developer machines. Its query path uses selective model loading and modality pruning to stay within a measured **491.35 MB peak worker RSS** on the documented test host. The full indexing stage has a higher memory requirement and is benchmarked separately below.

### What it generates

| Deliverable | Description | Output |
|---|---|---|
| Architecture and sequence diagrams | Mermaid topologies mapped to repository modules | Mermaid `.md` content |
| Unified Git patch | Actionable source changes derived from spoken constraints | `changes.patch` |
| Architecture Decision Record | Design rationale grounded in slide keyframes, timestamped transcript segments, and matched source files | `ADR-compiled.md` |

## Demo

> **Demo link:** [Add the hosted demo URL here](#demo-link-placeholder)
>
> Replace this placeholder with a YouTube, Drive, or deployed-demo URL when it is available.

To run the local interactive studio, see [Running the Demo Studio](#running-the-demo-studio).

## Key Features

- **Multimodal review ingestion:** Processes design-review media and spoken discussion, including 16 kHz audio and slide keyframes.
- **Repository-aware speech recognition:** Uses AST-derived identifiers to bias recognition toward real project symbols and names.
- **Compact semantic indexing:** Embeds source files and transcript segments with `google/embeddinggemma-2` using 256-dimensional Matryoshka representations.
- **Grounded retrieval:** Uses LanceDB to retrieve relevant repository modules for a voice instruction.
- **Live voice overrides:** Supports preset or microphone-driven architectural changes across three compilation scenarios.
- **Multiple engineering outputs:** Produces Mermaid topology, a unified Git patch, and a compiled ADR.
- **Session traceability:** Appends session events to JSONL logs and supports session forking.
- **CPU-conscious execution:** Uses selective model loading, tower pruning, subprocess cleanup, and Windows working-set reclamation.
- **Safe voice validation:** Holds the current repository/UI state when an input is too short, noisy, or not confidently recognized.

## Architecture

Speak-to-Repo follows a **four-phase sequential modality-swap lifecycle**. Instead of keeping every modality model active together, the system loads the subsystem needed for the current phase and releases idle worker memory after the task.

<p align="center">
  <img src="assets\Speak-to-Repo System Architecture (1).png" alt="Speak-to-Repo dark system architecture showing inputs, four processing phases, outputs, technology stack, and CPU memory constraints" width="100%">
</p>

### Phase 1 — AST-biased multimodal ingest

1. The AST harvester scans the target repository and collects domain identifiers and source symbols.
2. Design-review media is prepared as slide keyframes and audio windows.
3. Cactus Whistle performs speech recognition, using repository vocabulary to help recognize technical names.
4. The voice-validation gate accepts a usable transcript or holds the current application state when the audio is unclear.

**Primary components:** `ast_extractor.py`, `whistle_ingest.py`, Cactus Whistle, validated voice gate.

### Phase 2 — Matryoshka multimodal indexing

1. Repository source files and transcript segments are prepared for embedding.
2. EmbeddingGemma-2 generates embeddings on CPU.
3. Matryoshka Representation Learning (MRL) reduces the vectors to the configured 256-dimensional representation, followed by normalization.
4. LanceDB stores the vectors and associated source metadata on disk.

**Primary components:** `gemma_indexer.py`, EmbeddingGemma-2, LanceDB.

### Phase 3 — Voice query and compilation

1. An engineer speaks a live override or selects one of the supported scenarios.
2. The validation gate checks the input before the query is processed.
3. A pruned EmbeddingGemma-2 query path embeds the instruction and searches the LanceDB index.
4. Retrieved source context is passed to the compiler to generate the updated Mermaid topology, Git patch, and ADR.

**Primary components:** Live Studio, pruned EmbeddingGemma-2, LanceDB retrieval, `repo_compiler.py`.

### Phase 4 — Session logging and memory cleanup

1. Session actions are appended to `session_events.jsonl`.
2. Cordis `ctx.effect()` hooks manage worker lifecycle and cleanup.
3. On Windows, working-set reclamation can release idle process memory after a task.

**Primary components:** `cordis_session_runner.py`, Cordis kernel, append-only JSONL event log, Windows working-set cleanup.

## Live Compilation Scenarios

After the baseline design review is ingested, the live microphone or preset controls can submit one of the following architectural overrides. The system retrieves relevant modules, updates the topology, and compiles a Git patch.

| Scenario | Example voice instruction | Main symbols / parameters | Expected compiled change |
|---|---|---|---|
| **2A — PTC Sandbox Guard** | “In PTC tool executor, cap max nested calls to 4 with a 15 second timeout.” | `PtcToolExecutor`, `maxNestedCalls=4`, `timeoutMs=15000` | Updates `tool-executor-ptc/src/config.ts` and renders a `SandboxAbortError` guard graph. |
| **2B — Native Batch & Retry** | “In native tool executor, run execute tool batch with concurrency 8 and 3 retries.” | `NativeToolExecutor`, `executeToolBatch`, concurrency `8`, retries `3` | Updates `tool-executor-native/src/index.ts` and renders a semaphore/backoff flow. |
| **2C — Hybrid Executor Router** | “Switch agent tool presentation to a hybrid router when batch size exceeds 2.” | `agent-tool-presentation`, `NativeToolExecutor`, `PtcToolExecutor`, threshold `2` | Updates `agent-tool-presentation/src/index.ts` and renders a dual-path decision tree. |

### Voice validation behavior

If a voice take is too short, noisy, or unrecognized, the backend keeps the existing repository and UI state unchanged. The studio displays a transient warning that automatically dismisses after **1.5 seconds**.

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Speech recognition | [Cactus Whistle](https://github.com/Cactus-Compute/whistle) | Compact speech transcription for recorded and live engineering discussions |
| Text embeddings | [EmbeddingGemma-2](https://huggingface.co/google/embeddinggemma-2) | CPU embedding generation with 256d Matryoshka representations |
| Vector database | [LanceDB](https://lancedb.com/) | On-disk vector storage and similarity retrieval |
| Agent/session runtime | [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) | Cordis kernel, tools, and effect-based worker lifecycle |
| Architecture output | Mermaid.js | Repository-grounded diagrams and topologies |
| Runtime | Python, PyTorch CPU | Ingestion, indexing, retrieval, and compilation |

## Hardware Requirements

| Requirement | Recommended / supported configuration |
|---|---|
| Processor | CPU-only x86_64 or ARM64; no discrete GPU required |
| Memory | 16 GB RAM developer machine |
| Operating system | Windows 10/11, Linux, or macOS |
| Python | 3.10–3.12 |
| Node.js | 20+ (only needed when building the full DeepSeek Harness workspace) |
| Package manager | `pnpm` (optional; required with the full harness workspace) |
| Hugging Face | Account authenticated with access to `google/embeddinggemma-2` |

> **Memory note:** The sub-500 MB target refers to the pruned query/compile path, not the full indexing pass. The benchmark table records indexing separately.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vishwanath-06/speak-to-repo.git
cd speak-to-repo
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Authenticate with Hugging Face

Authenticate so the indexer can download the EmbeddingGemma-2 model weights:

```bash
hf auth login
```

### 4. Clone the target repository

The following example uses DeepSeek Harness as the repository to inspect and index:

```bash
git clone https://github.com/deepseek-ai/deepseek-harness.git
```

The commands below assume the target source directory is `./deepseek-harness/packages/core`.

## Getting Started

### 1. Generate the sample design-review media

```bash
python scripts/generate_design_meeting.py
```

### 2. Run AST-biased speech ingestion

**Windows PowerShell**

```powershell
python workers/whistle_ingest.py `
  --media storage/demo_media/slide_1.wav `
  --repo ./deepseek-harness/packages/core `
  --out storage/transcripts/test_run.jsonl
```

### 3. Build the 256-dimensional vector index

```powershell
python workers/gemma_indexer.py index `
  --repo ./deepseek-harness/packages/core `
  --transcript storage/transcripts/test_run.jsonl `
  --db storage/lancedb `
  --max-files 60 `
  --dim 256
```

## Running the Demo Studio

Start the local interactive studio:

```bash
python scripts/demo_studio.py
```

Open **http://127.0.0.1:8080** in your browser.

1. Select **Ingest Slides** to load the review media and baseline architecture.
2. Select **Live Mic Override** or one of the `2A`, `2B`, or `2C` presets.
3. Review the generated Mermaid topology, Git diff, and ADR output.
4. Use the **GC** control in the memory bar to trigger the working-set cleanup path on supported Windows environments.

## CLI Session Forking

The event-sourced CLI runner can fork a prior session with a new goal and target repository:

```powershell
python scripts/cordis_session_runner.py `
  --fork-session sess_initial `
  --goal "Programmatic Tool Calling PTC run_code sandbox and SDK generation" `
  --repo ./deepseek-harness/packages/core `
  --out-doc storage/compiled_docs/ptc_sandbox_spec.md
```

## Tests and Benchmarks

Run the unit test suite:

```bash
python -m unittest discover -s tests -v
```

Run the benchmark suite, including the MRL dimension sweep and Whistle ablation benchmark:

```bash
python scripts/benchmark_suite.py
```

## Benchmarks

The following measurements are from the documented **16 GB RAM Windows x64** host using Python 3.12, CPU-only PyTorch, and bfloat16. Treat them as measurements from that environment, not universal performance guarantees.

### Runtime and memory measurements

| Pipeline stage | Active subsystem | Peak worker RSS | Throughput / latency | Notes |
|---|---|---:|---|---|
| Phase 1 — Acoustic ingest | Cactus Whistle (`16.9 MB` model file) | **98.43 MB** | **277.4 tok/s**, `482.6 ms` TTFT | Worker memory reclaimed after task |
| Phase 2 — Batch indexing | EmbeddingGemma-2 + LanceDB write | **1,269.60 MB** | `61` cross-modal chunks indexed | Higher-memory offline indexing stage |
| Phase 3 — Pruned query and compile | EmbeddingGemma-2 with vision/audio towers pruned | **491.35 MB** | `237.27 ms` query + `2 ms` compile | `60.1%` reduction reported for the pruned path |
| Phase 4 — Forked session compile | `cordis_session_runner.py --fork-session` | **494.64 MB** | `12.76 s` total | Reported as `47×` faster than a cold index in this benchmark |

### Embedding dimension sweep

| Vector dimension | Float32 storage / vector | Storage savings vs. 768d | Recall@5 vs. 768d | Mean top-1 cosine | CPU matrix-search latency |
|---:|---:|---:|---:|---:|---:|
| `128d` | `512 B` | `6.0×` | `45.0%` | `0.8202` | `107.27 µs` |
| **`256d` (default)** | `1,024 B` | **`3.0×`** | `70.0%` | `0.7320` | `153.75 µs` |
| `512d` | `2,048 B` | `1.5×` | `85.0%` | `0.7122` | `125.43 µs` |
| `768d` | `3,072 B` | `1.0×` | `100.0%` | `0.7122` | `136.85 µs` |

The 256d configuration is the default because it reduces vector storage by **3×** relative to 768d while retaining the reported **70% Recall@5** in this sweep.

## Repository Structure

```text
speak-to-repo/
├── .github/
│   └── workflows/                  # GitHub Actions workflows
├── workers/
│   ├── ast_extractor.py             # AST harvester and import-symbol scraper
│   ├── whistle_ingest.py            # Cactus Whistle windowed transcription
│   ├── gemma_indexer.py             # EmbeddingGemma-2 indexing and 256d MRL
│   └── repo_compiler.py             # Markdown, Mermaid, and citation compiler
├── scripts/
│   ├── demo_studio.py               # Live microphone, Mermaid, and Git diff studio
│   ├── generate_design_meeting.py   # Sample slide and audio generation
│   ├── cordis_session_runner.py     # Append-only session log and session forking
│   └── benchmark_suite.py           # MRL sweep and Whistle ablation benchmarks
├── cordis-plugin/
│   └── src/speak-ingest.ts          # Cordis plugin integration
├── tests/
│   └── test_speak_to_repo.py        # AST, voice gate, Mermaid, and Git diff tests
├── storage/
│   ├── compiled_docs/               # Generated ADRs and patches
│   ├── sessions/                    # session_events.jsonl trajectories
│   └── BENCHMARK_RESULTS.md         # Benchmark telemetry
├── requirements.txt
└── README.md
```

## Design Notes

- **Sequential modality swap:** Speech, embedding, and compilation workloads are staged rather than kept resident together.
- **AST-biased transcription:** Repository identifiers are used to improve recognition of technical symbols and names.
- **Retrieval-grounded compilation:** Generated outputs are linked to retrieved repository context rather than being based only on the spoken instruction.
- **Transaction-safe validation:** An unclear voice command does not replace the current repository/UI state.
- **Event-sourced sessions:** Session events are appended to JSONL so work can be inspected and forked.

## Limitations and Notes

- The documented measurements are hardware- and workload-dependent.
- Initial indexing has a higher memory footprint than the pruned query path.
- The Hugging Face model weights must be available to the authenticated account.
- The full DeepSeek Harness workspace requires the optional Node.js and `pnpm` toolchain.
- The demo URL is intentionally left as a placeholder until a hosted recording or deployment is available.

## Contributing

Contributions, bug reports, and suggestions are welcome. For substantial changes, open an issue first to discuss the proposed update. When submitting a pull request, include the motivation, the affected pipeline phase, and relevant test or benchmark output.

## License

No license is specified in the repository information used for this README. Add a `LICENSE` file before granting permissions for reuse or redistribution.

---

<p align="center">
  Built for CPU-native, repository-grounded engineering workflows.
</p>
