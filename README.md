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

Packs are published as `uniflow-audio-v1.1-{small,base,large}` on Hugging Face.
**Small is the default** (fastest to download and run).

```bash
# requires: pip install -U "huggingface_hub[cli]"
./scripts/download_gguf.sh              # small
./scripts/download_gguf.sh base
./scripts/download_gguf.sh large

# or via Make
make download-gguf         # small (default)
make download-gguf-base
make download-gguf-large
```

Files land in `models/uniflow-audio-v1.1-<size>/`.

| Size  | DiT (approx.) | Notes                          |
|-------|---------------|--------------------------------|
| small | ~1.1 GB       | Default; good starting point   |
| base  | ~1.5 GB       | Higher capacity                |
| large | ~2.5 GB       | Highest quality, more VRAM/RAM |

Each pack also includes shared T5, VAE, instructions, and tokenizer (~1.4 GB).

## Generate

```bash
# Text-to-audio (default task)
./build-metal/uniflow-audio --model small \
  --caption "a man is speaking while a dog barks" \
  --output output/out.wav \
  --duration 5 --steps 25 --cfg 5.0 --sway -1 --seed 42

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
Use `--models-dir DIR` if you keep weights elsewhere.

### Useful options

| Flag | Default | Description |
|------|---------|-------------|
| `--caption` | — | Prompt text (T2A / T2M) |
| `--task` | `t2a` | `t2a` or `t2m` |
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
