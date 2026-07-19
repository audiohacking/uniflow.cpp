# uniflow.cpp

Portable **C++17 GGML/GGUF** inference for
[UniFlow-Audio](https://github.com/wsntxxn/UniFlow-Audio)
(unified flow-matching audio generation from omni-modalities).

> Experimental WIP. Scaffolded from
> [audiogen.cpp](https://github.com/audiohacking/audiogen.cpp).
> **Read [DEVELOPMENT.md](DEVELOPMENT.md)** for status, blockers, and how to
> work on this repo.

## Status

**T2A works on Metal** (Small GGUF). Quality is early — hear `output/human_t2a_dog.wav` after generating. See [DEVELOPMENT.md](DEVELOPMENT.md).

```bash
git submodule update --init --recursive
make metal
make test
# after make convert-small (or with existing models/*.gguf):
./build-metal/uniflow-audio \
  models/t5_encoder.gguf models/dit.gguf models/vae.gguf models/instructions.gguf models/spiece.model \
  --caption "a man is speaking while a dog barks" \
  --output output/out.wav --steps 25 --cfg 5.0 --seed 42 --duration 5
```

## Goal

0% Python at runtime. Cross-platform via ggml (Metal / CPU / CUDA).

## License

TBD before first release (UniFlow MIT upstream + Apache-2.0 scaffold heritage).
