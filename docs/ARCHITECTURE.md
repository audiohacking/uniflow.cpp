# Architecture (uniflow.cpp)

See [DEVELOPMENT.md](../DEVELOPMENT.md) for live status.

## Pipeline (target T2A)

1. **Text encoder** — `google/flan-t5-large` (`src/t5_encoder.*`, recycled from audiogen.cpp).
2. **Content adapter + duration** — UniFlow `CrossAttentionAdapter` + instruction embeddings (WIP).
3. **DiT** — `LayerFusionAudioDiT` @ latent 128 (WIP; audiogen.cpp DiT is same family but wrong dims/weights).
4. **Flow-matching Euler** — `src/scheduler.*` with sway sampling (ported; under test).
5. **StableVAE decode** — 24 kHz waveform (replaces Dasheng Vocos; WIP).

## Layout

```
third_party/   ggml + pocketfft
convert/       HF → GGUF (offline only)
src/           C++ inference
reference/     UniFlow Python ground truth (from HF Space)
tests/         regression tests (TDD)
```
