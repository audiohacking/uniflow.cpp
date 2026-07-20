# uniflow.cpp

C++17 [GGML](https://github.com/ggerganov/ggml)/GGUF inference for
[UniFlow-Audio](https://github.com/wsntxxn/UniFlow-Audio) — text-to-audio and
text-to-music via flow matching. No Python at runtime.

GGUF packs: [audiohacking/uniflow-audio-gguf](https://huggingface.co/audiohacking/uniflow-audio-gguf)

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

Packs live under `uniflow-audio-v1.1-{small,base,large}/`. DiT files use HF GGUF
quant tags in the filename (`dit-F16.gguf`, `dit-Q8_0.gguf`, `dit-Q4_0.gguf`).
**Small + F16 is the default.**

```bash
# requires: pip install -U "huggingface_hub[cli]"
./scripts/download_gguf.sh                 # small F16
./scripts/download_gguf.sh base            # base F16
./scripts/download_gguf.sh large Q8_0      # large Q8 (recommended)
./scripts/download_gguf.sh large F16
./scripts/download_gguf.sh large Q4_0
./scripts/download_gguf.sh large all       # every DiT quant

# or via Make
make download-gguf         # small F16
make download-gguf-base
make download-gguf-large   # large Q8_0
```

Files land in `models/uniflow-audio-v1.1-<size>/`.

| Size  | DiT quants on HF | Notes |
|-------|------------------|-------|
| small | F16 / Q8_0 / Q4_0 | Default size; F16 default quant |
| base  | F16 / Q8_0 / Q4_0 | Higher capacity |
| large | F16 / Q8_0 / Q4_0 | Q8_0 recommended for Large |

Each pack also includes shared T5, VAE, instructions, and tokenizer (~1.4 GB).

## Generate

```bash
# Text-to-audio (default task)
./build-metal/uniflow-audio --model small \
  --caption "a man is speaking while a dog barks" \
  --output output/out.wav \
  --duration 5 --steps 25 --cfg 5.0 --sway -1 --seed 42

# Large with Q8 DiT
./build-metal/uniflow-audio --model large --quant Q8_0 \
  --caption "a man is speaking while a dog barks" \
  --output output/out.wav --duration 5

# Text-to-music
./build-metal/uniflow-audio --model small \
  --task t2m \
  --caption "lo-fi hip hop beat" \
  --output output/music.wav \
  --duration 10

# Batch: one caption per line
./build-metal/uniflow-audio --model small \
  --batch prompts.txt --output-dir output/
```

`--model small|base|large` looks under `models/uniflow-audio-v1.1-<size>/`.
`--quant F16|Q8_0|Q4_0` selects `dit-<QUANT>.gguf` (auto-picks F16 → Q8_0 → Q4_0 if omitted).
Use `--models-dir DIR` if you keep weights elsewhere.

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
