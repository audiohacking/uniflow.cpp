# uniflow.cpp

C++17 [GGML](https://github.com/ggerganov/ggml)/GGUF inference for [UniFlow-Audio](https://github.com/wsntxxn/UniFlow-Audio) — text-to-audio and text-to-music via flow matching. Converted/Quantized GGUF models: [audiohacking/uniflow-audio-gguf](https://huggingface.co/audiohacking/uniflow-audio-gguf)

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

Browser demo (WebGPU, no install): `make web` — see [web/README.md](web/README.md).

## Download models

Packs live under `uniflow-audio-v1.1-{small,base,large}/`.

- **Size** = DiT capacity (`small` / `base` / `large`)
- **DiT quant** = `dit-F16.gguf` / `dit-Q8_0.gguf` / `dit-Q4_0.gguf`
- **T5** = per pack (not shared): **small → Q4_0**, **base/large → Q8_0**
  (F16 T5 is unsupported)

**Default: Base + DiT Q8_0** (with T5 Q8_0 in that pack).

```bash
# requires: pip install -U "huggingface_hub[cli]"
./scripts/download_gguf.sh                 # base Q8_0
./scripts/download_gguf.sh small F16
./scripts/download_gguf.sh small Q8_0
./scripts/download_gguf.sh base F16
./scripts/download_gguf.sh large Q4_0
./scripts/download_gguf.sh base all        # every DiT quant for base

make download-gguf                         # base Q8_0
QUANT=F16 make download-gguf-small
```

Files land in `models/uniflow-audio-v1.1-<size>/`.

| Size  | T5 in pack | `dit-F16` | `dit-Q8_0` | `dit-Q4_0` |
|-------|------------|-----------|------------|------------|
| small | Q4_0 (~183 MB) | ~350 MB | ~247 MB | ~170 MB |
| base  | Q8_0 (~346 MB) | ~1.4 GB | ~834 MB | ~482 MB |
| large | Q8_0 (~346 MB) | ~2.5 GB | ~1.4 GB | ~799 MB |

Each pack also includes VAE + instructions + tokenizer (~93 MB).

## Performance (Q8)

Benchmark: DiT **Q8_0** + pack T5, **5 s** of audio, **25** steps, CFG **5.0**, sway **-1**. 

| Platform | Wall time | vs audio | DiT |
|-------|-----------|----------|-----|
| GB10 (CUDA)      |      ~1.7 s       |  ~2.9x   |  ~61 ms  |
| M3 Ultra (Metal) |      ~2.2 s       |  ~2.2x   |  ~77 ms  |                                                                                       

## Generate

```bash
# Default experience (after download_gguf.sh): base + Q8 DiT
./build-metal/uniflow-audio --model base --quant Q8_0 \
  --caption "a man is speaking while a dog barks" \
  --output output/out.wav \
  --duration 5 --steps 25 --cfg 5.0 --sway -1 --seed 42

./build-metal/uniflow-audio --model small --quant F16 \
  --caption "a man is speaking while a dog barks" \
  --output output/small.wav --duration 5

./build-metal/uniflow-audio --model large --quant Q4_0 \
  --caption "a man is speaking while a dog barks" \
  --output output/large_q4.wav --duration 5

# Text-to-music
./build-metal/uniflow-audio --model base --quant Q8_0 \
  --task t2m \
  --caption "lo-fi hip hop beat" \
  --output output/music.wav \
  --duration 10
```

`--model small|base|large` → `models/uniflow-audio-v1.1-<size>/`  
`--quant F16|Q8_0|Q4_0` → `dit-<QUANT>.gguf` (if omitted: Q8_0 → F16 → Q4_0)  
`--models-dir DIR` if weights live elsewhere.

### Useful options

| Flag | Default | Description |
|------|---------|-------------|
| `--caption` | — | Prompt text (T2A / T2M) |
| `--task` | `t2a` | `t2a` or `t2m` |
| `--quant` | auto | DiT: `F16`, `Q8_0`, or `Q4_0` |
| `--duration` | model-predicted | Length in seconds |
| `--steps` | `25` | Flow-matching steps |
| `--cfg` | `5.0` | Classifier-free guidance |
| `--sway` | `-1.0` | Sway sampling coefficient |
| `--seed` | random | Reproducible RNG |
| `--threads` | `4` | CPU threads for T5 |
| `--batch FILE` | — | One caption per line → multiple outputs |
| `--output-dir DIR` | `.` | Output directory for `--batch` |
| `--instruction-idx N` | `0` | Instruction embedding index (0-9) |

Full list: `./build-metal/uniflow-audio --help`

Metal is used by default on Apple Silicon, CUDA on `build-cuda`, else CPU.
Force a backend with `GGML_BACKEND=CPU` (or `Metal` / `CUDA` / `GPU`).

## Citation

If you use this in research, please cite the original UniFlow-Audio paper:

```bibtex
@article{xu2025uniflow,
  title={UniFlow-Audio: Unified Flow Matching for Audio Generation from Omni-Modalities},
  author={Xu, Xuenan and Mei, Jiahao and Zheng, Zihao and Tao, Ye and Xie, Zeyu and Zhang, Yaoyun and Liu, Haohe and Wu, Yuning and Yan, Ming and Wu, Wen and Zhang, Chao and Wu, Mengyue},
  journal={arXiv preprint arXiv:2509.24391},
  year={2025}
}
```

## License

Upstream UniFlow-Audio is MIT. This project follows that license for model use;
see the repository for third-party notices.
