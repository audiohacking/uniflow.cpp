# Browser WebGPU demo (Small Q8)

Experimental static site for running UniFlow-Audio **Small** (`dit-Q8_0` + pack T5 Q4_0)
in Chrome/Edge via **ggml WebGPU** (Emscripten).

Weights are **not** stored in git. The page fetches them from
[audiohacking/uniflow-audio-gguf](https://huggingface.co/audiohacking/uniflow-audio-gguf).

## Build

```bash
# Homebrew emscripten, or source emsdk_env.sh
export PATH="/opt/homebrew/opt/emscripten/bin:$PATH"

cd /path/to/uniflow.cpp
git submodule update --init --recursive

rm -rf build-web
emcmake cmake -B build-web -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DUNIFLOW_METAL=OFF \
  -DUNIFLOW_CUDA=OFF \
  -DUNIFLOW_BUILD_TESTS=OFF \
  -DGGML_WEBGPU=ON \
  -DGGML_WEBGPU_JSPI=ON \
  -DGGML_OPENMP=OFF

emmake cmake --build build-web --target uniflow-web --parallel
# outputs: web/uniflow-web.js + web/uniflow-web.wasm
```

## Local serve

```bash
npx --yes serve web -p 8080
# open http://localhost:8080  (Chrome with WebGPU)
```

## GitHub Pages

Workflow: `.github/workflows/pages-webgpu.yml`

1. **Settings → Pages → Build and deployment → Source = GitHub Actions** (once).
2. On push to `main` (or **Actions → Deploy WebGPU demo → Run workflow**):
   - **build** job: recursive submodule checkout → Emscripten + WebGPU compile → upload artifact
   - **deploy** job: `actions/deploy-pages` only
3. Site URL is printed on the deploy job (`environment: github-pages`).

Users need a WebGPU browser and enough RAM for ~520 MB weights + WASM heap.

## API (WASM)

| Export | Role |
|--------|------|
| `initBackend()` | Init WebGPU + CPU sched |
| `loadModels(t5, dit, vae, instructions)` | MEMFS paths |
| `generateFromTokens(ids, duration, steps, cfg, sway, seed)` | PCM length |
| `pcmPointer()` / `pcmSampleRate()` / `freePcm()` | Read float32 PCM |

Tokenization is done in JS (`@huggingface/transformers`).
