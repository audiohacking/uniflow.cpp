# DEVELOPMENT.md — living project brain for uniflow.cpp

> **Read this first.** Update this file in every session that changes status,
> adds a conquer, hits a blocker, or lands a regression test.

## Goal

Portable **C++17 GGML/GGUF** inference for
[UniFlow-Audio](https://github.com/wsntxxn/UniFlow-Audio) with **0% Python at
runtime**. Scaffold recycled from
[audiogen.cpp](https://github.com/audiohacking/audiogen.cpp).

Human listening is required for all audio quality claims.

## Current status (2026-07-20)

| Item | State |
|------|--------|
| Phase | **1 — Conversion (COMPLETE for Small T2A subset)** → next: C++ graphs |
| Binary | `uniflow-audio` builds Metal + CPU; `--smoke-test` OK on Apple M4 |
| Inference | **Not wired** (GGUFs ready; DiT/VAE C++ graphs still TODO) |
| GGUF (Small) | `models/{t5_encoder,dit,vae,instructions}.gguf` + `spiece.model` |
| Reference Python | Vendored from HF Space under `reference/python/` |
| Tests | `make test` → **3/3 C++ + 7/7 Python PASS** |

### Conquered (with regression tests)

| Conquer | Test |
|---------|------|
| ggml backend links + runs (Metal M4) | `tests/cpp/test_smoke.cpp` |
| Flow-match scheduler (25 steps, sway=-1, N+1 knots) | `tests/cpp/test_scheduler.cpp` |
| 24 kHz mono WAV I/O | `tests/cpp/test_wav_io.cpp` |
| Converter ARCH strings + no Dasheng Dit masquerade | `tests/python/test_convert_conventions.py` |
| `dit.gguf` load: ≥800 tensors, embed_dim=512, T2A subset | same (TestDitGgufLoad) |
| `vae.gguf` fused weight_norm (no weight_g/v) | same (TestVaeGgufLoad) |
| `instructions.gguf` has `instr.text_to_audio_0` | same (TestInstructionsGgufLoad) |

### Blockers / next actions

1. **Port C++ DiT + content adapter + StableVAE decode** against Small GGUFs (Phase 2).
2. **Numerical parity dumps** from Python `InferenceCLI` (fixed seed) before claiming audio quality.
3. Homebrew `sentencepiece` requires linking **`absl_status`** (handled in `CMakeLists.txt`).
4. Base/Large/XLarge conversion: same scripts with `--variant`; not run yet.
5. HF Space `data/egs/se_*.wav` / `sr_*.wav` / `v2a_*.mp4` may be LFS stubs.

## Weight inventory (UniFlow-Audio-v1.1-Small)

Source dir: `models/uniflow-small/` (from `wsntxxn/UniFlow-Audio-v1.1-Small`).

### Config highlights (`config.yaml`)

| Key | Value |
|-----|-------|
| sample_rate | **24000** |
| latent_dim / in_chans / out_chans | **128** |
| downsampling_ratio (hop) | **480** → **50 Hz** latent rate |
| content_dim | **1024** |
| backbone | `LayerFusionAudioDiT` embed **512**, depth **20**, heads **8** |
| U-Net blocks | **10 in + 1 mid + 10 out** (`skip=True`) |
| content_adapter | `CrossAttentionAdapter` heads **16** |
| autoencoder | `StableVAE` Oobleck, SnakeBeta, VAE bottleneck |
| instructions | external H5 (not in safetensors) |

### `model.safetensors` (1056 tensors)

| Prefix | Count | ~MB | In `dit.gguf`? |
|--------|------:|----:|----------------|
| backbone.* | 814 | 646 | **yes** |
| content_adapter.* | 48 | 82 | **yes** |
| content_encoder.text_encoder.proj.* | 2 | 4 | **yes** |
| dummy_nta/ta_embed | 2 | ~0 | **yes** |
| content_encoder.{phoneme,midi,speaker,video,audio,singer}.* | ~189 | ~57 | **no** (Phase 4+) |
| dummy_param | 1 | 0 | skipped |

Critical shapes:

- `backbone.patch_embed.proj.weight` **[512, 128, 1]**
- `backbone.final_block.final_layer.weight` **[128, 128, 3]**
- `content_encoder.text_encoder.proj.weight` **[1024, 1024]**
- `dummy_*_embed` **[1024]**

### VAE ckpt (`vae/speech_audio_sound_step=1000000.ckpt`)

- Lightning `state_dict` with `autoencoder.encoder.*` / `autoencoder.decoder.*`
- Oobleck strides **[2,4,6,10]** (=480), channels 128, c_mults [1,2,4,8]
- Encoder latent_dim **256** (μ‖σ) → bottleneck → **128**
- Decoder latent_dim **128**, `final_tanh=false`, `use_snake=true`
- **Conversion fuses** `weight_g`/`weight_v` → `weight` (`uniflow.vae_weight_norm_fused=true`)
- Discriminator / losses dropped

### Instructions H5 (70 keys)

Keys `{task}_{0..9}` for: text_to_audio, text_to_music, text_to_speech,
speech_enhancement, audio_super_resolution, singing_voice_synthesis, video_to_audio.
Each `[seq, 1024]` float32 (seq varies; e.g. `text_to_audio_0` = 14).

### Converted artifacts

```
models/t5_encoder.gguf      ~1.3G   ARCH uniflow_t5enc
models/dit.gguf             ~350M   ARCH uniflow_dit   (866 tensors F16)
models/vae.gguf             ~84M    ARCH uniflow_vae   (235 tensors F16, fused)
models/instructions.gguf    ~8.2M   ARCH uniflow_instructions (70 tensors)
models/spiece.model         ~772K
```

```bash
# re-convert Small T2A pack
make convert-small
```


## Architecture (target)

```
Caption → SentencePiece → T5Encoder (flan-t5-large)
       → ContentAdapter + instruction emb (from instructions H5)
       → duration → latent length @ 50 Hz (24k / hop 480)
       → LayerFusionAudioDiT flow-matching (CFG=5, steps=25, sway=-1)
       → StableVAE decode → WAV @ 24 kHz
```

Shared with audiogen.cpp: T5, ggml backends, GGUF util, scheduler, tokenizer.
**Different:** StableVAE (not Vocos), latent dim **128** (not 1280), **24 kHz**
(not 16 kHz), task instructions H5, multi-task content encoders later.

## Python ground truth

| Source | Path / URL |
|--------|------------|
| HF Space (demo + examples) | https://huggingface.co/spaces/wsntxxn/UniFlow-Audio |
| Vendored Space code | `reference/python/` (`inference_cli.py`, `models/`, `app.py`) |
| Upstream training repo | https://github.com/wsntxxn/UniFlow-Audio |
| Model weights | `wsntxxn/UniFlow-Audio-v1.1-{Small,Base,Large,XLarge}` |

Demo defaults (must match C++ CLI when wired): CFG **5.0**, steps **25** for
T2A/T2M; CFG **1.0** for SE/SR.

Example captions for human listen (from Space):

- T2A: `"a man is speaking while a dog barks"`, `"footsteps on wooden floor"`
- T2M: see `reference/python/app.py` Gradio examples

## Build

```bash
git submodule update --init --recursive
# deps: cmake, pkg-config, sentencepiece (brew install sentencepiece)
make metal          # or: make cpu
./build-metal/uniflow-audio --smoke-test
make test           # regression suite
```

Env: `GGML_BACKEND=Metal` / `CPU` / `CUDA` to force backend (see `src/backend.cpp`).

## Conversion policy (do not sabotage)

1. **Inspect first** — `python convert/inspect_safetensors.py <path>` → paste
   summary into Weight inventory above.
2. **Dump Python tensors** (fixed seed) before claiming graph parity.
3. **One component per session** (T5 → adapter → DiT block → VAE decode).
4. **Never** copy Dasheng `convert_dit.py` / `convert_vocoder.py` and rename.
5. ARCH strings: `uniflow_t5enc`, `uniflow_dit`, `uniflow_vae`, `uniflow_instructions`.
6. VAE: always fuse weight_norm at convert time; C++ consumes plain `weight`.

Safe commands:

```bash
make convert-t5
make convert-small   # DiT + VAE + instructions (+ T5 if missing)
```

## Test policy (TDD)

- Every conquer lands a test under `tests/cpp` or `tests/python`.
- `make test` must stay green.
- Audio WAVs are **not** asserted by CI as “sounds good”; agent produces files,
  **human listens**, then mark gate G1/G2 below.

### Human listen gates

| Gate | Status | Notes |
|------|--------|-------|
| G0 smoke binary | **ready for human confirm** — `make test-smoke-metal` passed on M4 | |
| G1 Python T2A ref WAV | not started | |
| G2 C++ ≈ Python T2A | not started | |
| G3 T2M | not started | |
| G4 Q8 quality | not started | |

## Repo layout

```
src/           C++ inference (backend, t5, tokenizer, scheduler, wav_io, …)
convert/       Offline HF → GGUF (Python only)
tests/cpp      C++ regression binaries (ctest)
tests/python   Converter / convention unit tests
reference/     Vendored UniFlow Python + sample assets
docs/          Architecture notes
DEVELOPMENT.md This file
```

## Session log

| Date | What | Result |
|------|------|--------|
| 2026-07-19 | Phase 0 scaffold + TDD + Metal smoke | Phase 0 complete |
| 2026-07-20 | Download Small; inspect; convert_dit/vae/instructions/t5; GGUF load tests | Phase 1 convert complete |

## Contacts / links

- Plan: Cursor plan `UniFlow GGUF C++ Port`
- Paper: https://arxiv.org/abs/2509.24391
- License: follow UniFlow MIT + audiogen Apache-2.0 for recycled scaffolding; clarify in README before release
