# Browser WebGPU demo (Small Q8)

Experimental static site for running UniFlow-Audio **Small** (`dit-Q4_0` + pack T5 Q4_0)
in Chrome/Edge via **ggml WebGPU** (Emscripten). Q4 keeps the download smaller for demos.

Weights are **not** stored in git. The page fetches them from
[audiohacking/uniflow-audio-gguf](https://huggingface.co/audiohacking/uniflow-audio-gguf).

## Build

Needs **Emscripten ≥ 4.0.10** (CI uses 4.0.23). Current ggml-webgpu needs a newer
Dawn `emdawnwebgpu` package than Emscripten 4.0.10–4.0.12 ship by default —
`make web` / the Pages workflow fetch a pinned package via
`scripts/fetch_emdawnwebgpu.sh`.

Built with **ASYNCIFY** (not JSPI) so WebGPU `WaitAny` works without
`WebAssembly.promising` / Chrome experimental flags.

```bash
# Homebrew emscripten, or source emsdk_env.sh
export PATH="/opt/homebrew/opt/emscripten/bin:$PATH"

cd /path/to/uniflow.cpp
git submodule update --init --recursive

make web
# outputs: web/uniflow-web.{js,wasm}
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
| `getPcm()` / `pcmSampleRate()` / `freePcm()` | Copied float32 PCM |

Tokenization is done in JS (`@huggingface/transformers`).
