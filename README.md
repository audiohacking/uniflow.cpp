# uniflow.cpp

Portable **C++17 GGML/GGUF** inference for
[UniFlow-Audio](https://github.com/wsntxxn/UniFlow-Audio)
(unified flow-matching audio generation from omni-modalities).

> Experimental WIP. Scaffolded from
> [audiogen.cpp](https://github.com/audiohacking/audiogen.cpp).
> **Read [DEVELOPMENT.md](DEVELOPMENT.md)** for status, blockers, and how to
> work on this repo.

## Status

Phase 1: **Small GGUFs converted** (T5 + DiT + VAE + instructions).  
Full T2A C++ inference graphs still WIP — see [DEVELOPMENT.md](DEVELOPMENT.md).

```bash
git submodule update --init --recursive
make metal
./build-metal/uniflow-audio --smoke-test
make test
# optional: make convert-small  # if models/uniflow-small/ is present
```

## Goal

0% Python at runtime. Cross-platform via ggml (Metal / CPU / CUDA).

## License

TBD before first release (UniFlow MIT upstream + Apache-2.0 scaffold heritage).
