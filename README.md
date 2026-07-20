# uniflow.cpp

C++17 [GGML](https://github.com/ggerganov/ggml)/GGUF inference for [UniFlow-Audio](https://github.com/wsntxxn/UniFlow-Audio) — text-to-audio and text-to-music via flow matching.

> Converted/Quantized GGUF models: [audiohacking/uniflow-audio-gguf](https://huggingface.co/audiohacking/uniflow-audio-gguf)

## Build

```bash
git submodule update --init --recursive

# Apple Silicon / macOS
make metal

# Linux / CPU-only
make cpu

# NVIDIA
make cuda
```

Binary: `./build-metal/uniflow-audio` (or `build-cpu` / `build-cuda`).

Requires CMake, a C++17 compiler, and (for Metal) Xcode Command Line Tools.

## Download models

Packs live under `uniflow-audio-v1.1-{small,base,large}/`.

- **Size** = DiT capacity (`small` / `base` / `large`)
- **Quant** = DiT encoding in the filename (`dit-F16.gguf` / `dit-Q8_0.gguf` / `dit-Q4_0.gguf`)

Every size publishes all three quants. Shared T5 / VAE / instructions / tokenizer
(~1.4 GB) are included once per pack.

**Default: Small + F16.**

```bash
# requires: pip install -U "huggingface_hub[cli]"
./scripts/download_gguf.sh                    # small F16
./scripts/download_gguf.sh small Q8_0
./scripts/download_gguf.sh small Q4_0
./scripts/download_gguf.sh base F16
./scripts/download_gguf.sh base Q8_0
./scripts/download_gguf.sh base Q4_0
./scripts/download_gguf.sh large F16
./scripts/download_gguf.sh large Q8_0         # recommended for Large
./scripts/download_gguf.sh large Q4_0
./scripts/download_gguf.sh large all          # every DiT quant for that size

# Make helpers (override with QUANT=...)
make download-gguf              # small F16
make download-gguf-base         # base F16
make download-gguf-large        # large Q8_0
QUANT=Q4_0 make download-gguf-base
QUANT=all make download-gguf-large
```

Files land in `models/uniflow-audio-v1.1-<size>/`.

| Size  | `dit-F16` | `dit-Q8_0` | `dit-Q4_0` |
|-------|-----------|------------|------------|
| small | ~350 MB   | ~247 MB    | ~170 MB    |
| base  | ~1.4 GB   | ~834 MB    | ~482 MB    |
| large | ~2.5 GB   | ~1.4 GB    | ~799 MB    |

## Generate

```bash
# Text-to-audio (default: auto-picks dit-F16.gguf if present)
./build-metal/uniflow-audio --model small \
  --caption "a man is speaking while a dog barks" \
  --output output/out.wav \
  --duration 5 --steps 25 --cfg 5.0 --sway -1 --seed 42

# Explicit size + quant
./build-metal/uniflow-audio --model base --quant Q8_0 \
  --caption "a man is speaking while a dog barks" \
  --output output/base_q8.wav --duration 5

./build-metal/uniflow-audio --model large --quant Q4_0 \
  --caption "a man is speaking while a dog barks" \
  --output output/large_q4.wav --duration 5

# Text-to-music
./build-metal/uniflow-audio --model small --quant F16 \
  --task t2m \
  --caption "lo-fi hip hop beat" \
  --output output/music.wav \
  --duration 10

# Batch: one caption per line
./build-metal/uniflow-audio --model small --quant F16 \
  --batch prompts.txt --output-dir output/
```

`--model small|base|large` → `models/uniflow-audio-v1.1-<size>/`  
`--quant F16|Q8_0|Q4_0` → `dit-<QUANT>.gguf` (if omitted: F16 → Q8_0 → Q4_0)  
`--models-dir DIR` if weights live elsewhere.

### Useful options

| Flag | Default | Description |
|------|---------|-------------|
| `--caption` | — | Prompt text (T2A / T2M) |
| `--task` | `t2a` | `t2a` or `t2m` |
| `--quant` | auto | DiT file: `F16`, `Q8_0`, or `Q4_0` |
| `--duration` | model-predicted | Length in seconds |
| `--steps` | `25` | Flow-matching steps |
| `--cfg` | `5.0` | Classifier-free guidance |
| `--sway` | `-1.0` | Sway sampling coefficient |
| `--seed` | random | Reproducible RNG |
| `--threads` | `4` | CPU threads for T5 |

Full list: `./build-metal/uniflow-audio --help`

On Apple Silicon, Metal is used by default for DiT and VAE. Force a backend with
`GGML_BACKEND=CPU` (or `Metal` / `GPU`).

## License

Upstream UniFlow-Audio is MIT. This project follows that license for model use;
see the repository for third-party notices.
