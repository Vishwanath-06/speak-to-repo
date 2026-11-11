# Speak-to-Repo Compiler: CPU Hardware Benchmark Report

## 1. Matryoshka Representation Learning (MRL) Dimension Sweep (`EmbeddingGemma-2`)

| Vector Dim | Float32 Bytes/Vec | Storage Savings | Recall@5 (vs `768d`) | Mean Top-1 Cosine | Matrix Search Latency (CPU) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`128d`** | `512 B` | `6.0x` | **`45.0%`** | `0.8202` | `107.27 µs` |
| **`256d`** | `1024 B` | `3.0x` | **`70.0%`** | `0.732` | `153.75 µs` |
| **`512d`** | `2048 B` | `1.5x` | **`85.0%`** | `0.7122` | `125.43 µs` |
| **`768d`** | `3072 B` | `1.0x` | **`100.0%`** | `0.7122` | `136.85 µs` |

## 2. Cactus Whistle (`16.9 MB`) AST Keyword Biasing Ablation

| Mode | AST Symbols Biased | TTFT (ms) | Decode Speed (TPS) | Mean Word Prob | Technical Term Prob | Transcript Output |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Unbiased Baseline** | `0` | `575.3 ms` | `248.9 tok/s` | `0.9311` | `0.8058` | *"We need to register defined tool inside the Cordis context and run tool executor PTC to patch the repository."* |
| **Repo AST-Biased** | `150` | `549.3 ms` | `246.3 tok/s` | `0.9311` | `0.8058` | *"We need to register defined tool inside the Cordis context and run tool executor PTC to patch the repository."* |

## 3. End-to-End Cordis Subprocess Memory Lifecycle (Measured on 16 GB CPU Host)

| Pipeline Stage | Active Subsystem | Peak Worker RSS (MB) | Post-Task Retained RAM |
| :--- | :--- | :--- | :--- |
| **Phase 1: Acoustic Ingest** | `Cactus-Compute/whistle` (`16.9 MB` 2-bit/4-bit) | **`98.43 MB`** | `0 MB` (Reaped via `ctx.effect`) |
| **Phase 2: Batch Indexing** | `EmbeddingGemma-2` + `LanceDB` Write | **`1,269.60 MB`** | `0 MB` (Reaped via `ctx.effect`) |
| **Phase 3: Pruned Query & Compile** | `EmbeddingGemma-2` (Vision/Audio Towers Pruned) | **`491.35 MB`** *(60.1% reduction)* | `0 MB` (Reaped via `ctx.effect`) |