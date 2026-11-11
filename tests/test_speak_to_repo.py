#!/usr/bin/env python3
"""
Unit Test Suite for Speak-to-Repo Compiler.
Run with: python -m unittest discover -s tests -v
"""
import re
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "workers"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from ast_extractor import (
    extract_python_symbols,
    extract_ts_symbols,
    is_clean_symbol,
    to_spoken_phrase,
)
from demo_studio import (
    build_scenario_git_diff,
    build_scenario_mermaid,
    canonicalize_spoken_ast,
    validate_and_classify_override,
)


class TestSpeakToRepoCompiler(unittest.TestCase):
    def test_01_ast_extractor_filters_noise_and_scrapes_imports(self):
        """Verifies clean TS/Python symbol extraction and rejection of long build artifacts."""
        ts_sample = """
        import { defineTool, executeToolBatch } from '@deepseek-ai/dsh-tools'
        export class NativeToolExecutor {}
        export const deepseek_ai_dsh_cordis_host_runner_dynamicCordisRunner_parameter_0_strict = 1
        """
        symbols = extract_ts_symbols(ts_sample)
        self.assertIn("NativeToolExecutor", symbols)
        self.assertIn("defineTool", symbols)
        self.assertIn("executeToolBatch", symbols)
        self.assertFalse(
            any("parameter_0_strict" in s for s in symbols),
            "Should reject auto-generated long schema validator identifiers",
        )

        py_sample = "class SelectiveGemmaEmbedder:\n    def encode_mrl(self):\n        pass\n"
        py_syms = extract_python_symbols(py_sample)
        self.assertIn("SelectiveGemmaEmbedder", py_syms)
        self.assertIn("encode_mrl", py_syms)
        self.assertEqual(to_spoken_phrase("NativeToolExecutor"), "Native Tool Executor")

    def test_02_phonetic_ast_canonicalizer(self):
        """Verifies spoken multi-word phrases snap to exact AST identifiers and digits."""
        raw_speech = (
            "Switch agent tool presentation to a high bread router when batch size exceeds two "
            "and mount P T C tool executor with defined tool."
        )
        canon = canonicalize_spoken_ast(raw_speech)
        self.assertIn("agent-tool-presentation", canon)
        self.assertIn("hybrid", canon)
        self.assertIn("router", canon)
        self.assertIn("exceeds 2", canon)
        self.assertIn("PtcToolExecutor", canon)
        self.assertIn("defineTool", canon)

    def test_03_voice_validation_gate_rejects_invalid_or_noisy_audio(self):
        """Verifies incomplete or low-confidence speech is rejected without mutating state."""
        # Case A: Too short
        ok_short, msg_short, spec_short = validate_and_classify_override("hello test", [])
        self.assertFalse(ok_short)
        self.assertIsNone(spec_short)
        self.assertIn("too short", msg_short.lower())

        # Case B: Low acoustic confidence (<45%)
        low_prob_words = [
            {"word": "random", "probability": 0.25},
            {"word": "mumble", "probability": 0.30},
            {"word": "noise", "probability": 0.20},
            {"word": "here", "probability": 0.35},
        ]
        ok_low, msg_low, spec_low = validate_and_classify_override(
            "random mumble noise here today", low_prob_words
        )
        self.assertFalse(ok_low)
        self.assertIsNone(spec_low)
        self.assertIn("confidence", msg_low.lower())

        # Case C: Unrelated sentence with no architectural keywords
        ok_unrec, msg_unrec, _ = validate_and_classify_override(
            "the weather outside is very sunny today", [{"word": "weather", "probability": 0.95}]
        )
        self.assertFalse(ok_unrec)
        self.assertIn("not recognized", msg_unrec.lower())

    def test_04_multi_scenario_classification_2a_2b_2c(self):
        """Verifies all 3 Step-2 scenarios classify accurately and extract spoken integers."""
        good_words = [{"word": "ok", "probability": 0.95}]

        # Scenario 2A: PTC Sandbox Guard
        tx_2a = canonicalize_spoken_ast(
            "In P T C tool executor, cap max nested calls to four with a fifteen second timeout."
        )
        ok_a, _, spec_a = validate_and_classify_override(tx_2a, good_words)
        self.assertTrue(ok_a)
        self.assertEqual(spec_a["mode"], "ptc_sandbox")
        self.assertEqual(spec_a["max_nested_calls"], 4)
        self.assertEqual(spec_a["timeout_ms"], 15000)

        # Scenario 2B: Native Batch & Retry
        tx_2b = canonicalize_spoken_ast(
            "In native tool executor, run execute tool batch with concurrency eight and three retries."
        )
        ok_b, _, spec_b = validate_and_classify_override(tx_2b, good_words)
        self.assertTrue(ok_b)
        self.assertEqual(spec_b["mode"], "native_batch")
        self.assertEqual(spec_b["concurrency"], 8)
        self.assertEqual(spec_b["retries"], 3)

        # Scenario 2C: Hybrid Executor Router
        tx_2c = canonicalize_spoken_ast(
            "Switch agent tool presentation to a hybrid router when batch size exceeds two."
        )
        ok_c, _, spec_c = validate_and_classify_override(tx_2c, good_words)
        self.assertTrue(ok_c)
        self.assertEqual(spec_c["mode"], "hybrid_router")
        self.assertEqual(spec_c["batch_threshold"], 2)

    def test_05_mermaid_10_9_8_edge_label_syntax_safety(self):
        """Ensures Mermaid edge labels |...| never contain raw '>', '<', '(', or ')' characters."""
        specs = [
            {"mode": "ptc_sandbox", "max_nested_calls": 4, "timeout_ms": 15000},
            {"mode": "native_batch", "concurrency": 8, "retries": 3},
            {"mode": "hybrid_router", "batch_threshold": 2},
        ]
        edge_label_pattern = re.compile(r"\|([^|]+)\|")
        for sp in specs:
            dsl = build_scenario_mermaid(sp)
            self.assertTrue(dsl.startswith("flowchart TB"))
            for match in edge_label_pattern.finditer(dsl):
                label_text = match.group(1)
                for forbidden in (">", "<", "(", ")"):
                    self.assertNotIn(
                        forbidden,
                        label_text,
                        f"Forbidden character '{forbidden}' found inside Mermaid edge label '|{label_text}|' for mode {sp['mode']}",
                    )

    def test_06_unified_git_diff_generation(self):
        """Verifies each scenario compiles a valid unified git diff with spoken parameters."""
        spec_2c = {"mode": "hybrid_router", "batch_threshold": 2}
        diff_2c = build_scenario_git_diff(spec_2c)
        self.assertIn("diff --git a/packages/core/agent-tool-presentation/src/index.ts", diff_2c)
        self.assertIn("+export const HYBRID_BATCH_THRESHOLD = 2", diff_2c)
        self.assertIn("+  if (toolCallCount <= 2) {", diff_2c)

    def test_07_matryoshka_256d_truncation_and_l2_normalization(self):
        """Verifies 768d embeddings slice cleanly to 256d unit-norm vectors."""
        fake_768d = torch.randn(4, 768, dtype=torch.bfloat16)
        sliced_256d = F.normalize(fake_768d[:, :256].to(torch.float32), p=2, dim=-1)
        self.assertEqual(tuple(sliced_256d.shape), (4, 256))
        norms = torch.norm(sliced_256d, p=2, dim=-1)
        for n in norms.tolist():
            self.assertAlmostEqual(n, 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)