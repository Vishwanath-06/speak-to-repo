import { spawn, type ChildProcess } from 'node:child_process'
import path from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'speak-to-repo-compiler'
export const inject = ['tools']

export function apply(ctx: Context) {
  const activeWorkers = new Set<ChildProcess>()
  const rootDir = path.resolve(import.meta.dirname, '../../..')
  const pythonRelPath = process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python'
  const pythonBin = path.join(rootDir, pythonRelPath)

  // Cordis temporal lifecycle hook: terminates any active CPU worker on plugin unload/reload
  ctx.effect(() => {
    return () => {
      for (const proc of activeWorkers) {
        if (!proc.killed) {
          proc.kill('SIGTERM')
        }
      }
      activeWorkers.clear()
    }
  })

  function runPythonWorker(scriptName: string, args: string[]): Promise<{ content: string }> {
    const scriptPath = path.join(rootDir, 'workers', scriptName)
    return new Promise((resolve, reject) => {
      const proc = spawn(pythonBin, [scriptPath, ...args])
      activeWorkers.add(proc)
      let stdoutData = ''
      let stderrData = ''

      proc.stdout.on('data', (chunk) => {
        stdoutData += chunk.toString()
      })
      proc.stderr.on('data', (chunk) => {
        stderrData += chunk.toString()
      })

      proc.on('close', (code) => {
        activeWorkers.delete(proc)
        if (code !== 0) {
          reject(new Error(`Worker ${scriptName} failed (exit ${code}): ${stderrData}`))
          return
        }
        resolve({ content: stdoutData.trim() })
      })
    })
  }

  // Tool 1: Phase 1 AST-Biased Whistle Ingestion
  ctx.tools.register(
    defineTool({
      name: 'ingest_meeting_recording',
      description:
        'Phase 1: Transcribes a meeting recording on CPU using Cactus Whistle (16.9MB) with AST keyword biasing and 80ms word timestamps.',
      parameters: {
        type: 'object',
        properties: {
          mediaPath: { type: 'string', description: 'Path to .mp4 or .wav file' },
          repoPath: { type: 'string', description: 'Path to target Git repository' },
        },
        required: ['mediaPath', 'repoPath'],
      },
      async execute({ mediaPath, repoPath }) {
        const outJsonl = path.join(rootDir, 'storage/transcripts/latest_ingest.jsonl')
        const keyframeDir = path.join(rootDir, 'storage/keyframes')
        return runPythonWorker('whistle_ingest.py', [
          '--media', mediaPath,
          '--repo', repoPath,
          '--out', outJsonl,
          '--keyframes', keyframeDir,
        ])
      },
    }),
  )

  // Tool 2: Phase 2 EmbeddingGemma-2 256d MRL Indexer
  ctx.tools.register(
    defineTool({
      name: 'index_multimodal_workspace',
      description:
        'Phase 2: Encodes repository source files and Whistle transcript segments into 256d Matryoshka vectors in LanceDB.',
      parameters: {
        type: 'object',
        properties: {
          repoPath: { type: 'string', description: 'Path to target Git repository' },
          transcriptPath: { type: 'string', description: 'Path to Phase 1 JSONL transcript' },
          mrlDim: { type: 'number', description: 'Matryoshka dimension (128, 256, 512, 768)', default: 256 },
        },
        required: ['repoPath'],
      },
      async execute({ repoPath, transcriptPath, mrlDim = 256 }) {
        const tPath = transcriptPath || path.join(rootDir, 'storage/transcripts/latest_ingest.jsonl')
        const dbPath = path.join(rootDir, 'storage/lancedb')
        return runPythonWorker('gemma_indexer.py', [
          'index',
          '--repo', repoPath,
          '--transcript', tPath,
          '--db', dbPath,
          '--dim', String(mrlDim),
        ])
      },
    }),
  )

  // Tool 3: Phase 2 Fast 256d Cross-Modal Search
  ctx.tools.register(
    defineTool({
      name: 'search_multimodal_timeline',
      description:
        'Searches the unified 256d LanceDB index across spoken meeting timestamps, visual keyframes, and repository code.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Technical search query' },
          topK: { type: 'number', description: 'Number of results to return', default: 5 },
        },
        required: ['query'],
      },
      async execute({ query, topK = 5 }) {
        const dbPath = path.join(rootDir, 'storage/lancedb')
        return runPythonWorker('gemma_indexer.py', [
          'query',
          '--text', query,
          '--db', dbPath,
          '--top-k', String(topK),
        ])
      },
    }),
  )

  // Tool 4: Phase 3 Repo Artifact & Mermaid Compiler
  ctx.tools.register(
    defineTool({
      name: 'compile_repo_documentation',
      description:
        'Phase 3: Synthesizes a Markdown architecture specification with Mermaid diagrams and timestamped audio citations from the LanceDB index.',
      parameters: {
        type: 'object',
        properties: {
          goal: { type: 'string', description: 'Architecture topic or feature to document' },
          repoPath: { type: 'string', description: 'Target Git repository path' },
          outputDocPath: { type: 'string', description: 'Output Markdown file path' },
        },
        required: ['goal', 'repoPath'],
      },
      async execute({ goal, repoPath, outputDocPath }) {
        const dbPath = path.join(rootDir, 'storage/lancedb')
        const outDoc = outputDocPath || path.join(rootDir, 'storage/compiled_docs/architecture_spec.md')
        return runPythonWorker('repo_compiler.py', [
          '--goal', goal,
          '--repo', repoPath,
          '--db', dbPath,
          '--out-doc', outDoc,
        ])
      },
    }),
  )

  console.log('[speak-to-repo-compiler] Mounted 4-stage CPU pipeline tools on Cordis ctx.tools')
}