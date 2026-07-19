#!/usr/bin/env python3
"""Regression tests for UniFlow GGUF converters.

Requires converted files under models/ (run `make convert-small` first for
full suite). Always runs static/convention checks; skips load checks if GGUFs
are absent.
"""
from __future__ import annotations

import ast
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONVERT = ROOT / "convert"
MODELS = ROOT / "models"


class TestConvertConventions(unittest.TestCase):
    def test_t5_arch_is_uniflow(self):
        src = (CONVERT / "convert_t5_encoder.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        arch = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "ARCH":
                        arch = ast.literal_eval(node.value)
        self.assertEqual(arch, "uniflow_t5enc")

    def test_dit_arch_and_no_dasheng(self):
        text = (CONVERT / "convert_dit.py").read_text(encoding="utf-8")
        self.assertIn('ARCH = "uniflow_dit"', text)
        self.assertIn("UniFlow", text)
        self.assertNotIn("mispeech/Dasheng", text)
        self.assertNotIn("dasheng_audiogen", text)

    def test_vae_fuses_weight_norm(self):
        text = (CONVERT / "convert_vae.py").read_text(encoding="utf-8")
        self.assertIn('ARCH = "uniflow_vae"', text)
        self.assertIn("fuse_weight_norm", text)
        self.assertIn("weight_norm_fused", text)

    def test_instructions_converter_exists(self):
        text = (CONVERT / "convert_instructions.py").read_text(encoding="utf-8")
        self.assertIn('ARCH = "uniflow_instructions"', text)


@unittest.skipUnless(
    (MODELS / "dit.gguf").is_file(), "dit.gguf not built yet"
)
class TestDitGgufLoad(unittest.TestCase):
    def test_dit_metadata_and_counts(self):
        import gguf

        reader = gguf.GGUFReader(str(MODELS / "dit.gguf"))
        # tensor count: backbone+adapter+text proj+dummies for Small ≈ 800+
        self.assertGreaterEqual(len(reader.tensors), 800)
        names = {t.name for t in reader.tensors}
        self.assertIn("backbone.patch_embed.proj.weight", names)
        self.assertIn("content_adapter.cross_attn.in_proj_weight", names)
        self.assertIn("content_encoder.text_encoder.proj.weight", names)
        self.assertIn("dummy_nta_embed", names)
        # non-T2A encoders must be excluded
        self.assertFalse(any("phoneme_encoder" in n for n in names))
        self.assertFalse(any("midi_encoder" in n for n in names))

        fields = {k: v for k, v in reader.fields.items()}
        # gguf field access varies by version — check via get_field helper
        def field_int(key: str) -> int:
            f = reader.get_field(key)
            if f is None:
                self.fail(f"missing field {key}")
            return int(f.parts[-1] if hasattr(f, "parts") else f.data[0])

        # Prefer parts decoding used by gguf python
        emb = None
        for k, field in reader.fields.items():
            if k.endswith("dit_embed_dim") or k == "uniflow.dit_embed_dim":
                part = field.parts[-1]
                emb = int(part.item() if hasattr(part, "item") else part)
                break
        self.assertEqual(emb, 512, "Small variant embed_dim")


@unittest.skipUnless(
    (MODELS / "vae.gguf").is_file(), "vae.gguf not built yet"
)
class TestVaeGgufLoad(unittest.TestCase):
    def test_vae_has_decoder_no_weight_g(self):
        import gguf

        reader = gguf.GGUFReader(str(MODELS / "vae.gguf"))
        names = {t.name for t in reader.tensors}
        self.assertTrue(any(n.startswith("decoder.") for n in names))
        self.assertFalse(any(n.endswith("weight_g") for n in names))
        self.assertFalse(any(n.endswith("weight_v") for n in names))
        self.assertTrue(any(n.endswith(".weight") and n.startswith("decoder.") for n in names))


@unittest.skipUnless(
    (MODELS / "instructions.gguf").is_file(), "instructions.gguf not built yet"
)
class TestInstructionsGgufLoad(unittest.TestCase):
    def test_t2a_instruction_present(self):
        import gguf

        reader = gguf.GGUFReader(str(MODELS / "instructions.gguf"))
        names = {t.name for t in reader.tensors}
        self.assertIn("instr.text_to_audio_0", names)
        self.assertIn("instr.text_to_music_0", names)
        t = next(t for t in reader.tensors if t.name == "instr.text_to_audio_0")
        # GGUFReader reports shape reversed vs numpy row-major writer input
        shape = tuple(int(x) for x in t.shape)
        self.assertIn(shape, {(14, 1024), (1024, 14)})


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
