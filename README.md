# uniflow.cpp

Portable **C++17 GGML/GGUF** inference for
[UniFlow-Audio](https://github.com/wsntxxn/UniFlow-Audio)
(unified flow-matching audio generation from omni-modalities).

> Experimental WIP. Scaffolded from
> [audiogen.cpp](https://github.com/audiohacking/audiogen.cpp).
> **Read [DEVELOPMENT.md](DEVELOPMENT.md)** for status, blockers, and how to
> work on this repo.

## Status

**T2A works on Metal** (Small GGUF). DiT + StableVAE on GPU; T5/adapter on CPU.
Human listen required for audio QA — see [DEVELOPMENT.md](DEVELOPMENT.md).

GGUF weights: [audiohacking/uniflow-audio-gguf](https://huggingface.co/audiohacking/uniflow-audio-gguf)
(`uniflow-audio-v1.1-small/`).

```bash
git submodule update --init --recursive
make metal
make test
# download Small GGUF pack (or make convert-small from upstream safetensors):
hf download audiohacking/uniflow-audio-gguf \
  --include "uniflow-audio-v1.1-small/*" \
  --local-dir models
./build-metal/uniflow-audio --models-dir models/uniflow-audio-v1.1-small \
  --caption "a man is speaking while a dog barks" \
  --output output/out.wav --steps 25 --cfg 5.0 --sway -1 --seed 42 --duration 5
```

CLI mirrors [audiogen.cpp](https://github.com/audiohacking/audiogen.cpp) where it fits,
with UniFlow-specific flags (`--task`, `--instruction-idx`, `--models-dir`). Run
`./build-metal/uniflow-audio --help` for the full list.

## Goal

0% Python at runtime. Cross-platform via ggml (Metal / CPU / CUDA).

## License

TBD before first release (UniFlow MIT upstream + Apache-2.0 scaffold heritage).
