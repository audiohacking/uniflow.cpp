// Browser WASM API for UniFlow Small Q8 demo (Emscripten + WebGPU).
// Tokenization stays in JS; C++ runs T5→DiT→VAE.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "backend.h"
#include "pipeline.h"

namespace {

std::unique_ptr<uniflow::Pipeline> g_pipeline;
std::string g_last_error;
std::vector<float> g_pcm;

void set_error(const std::string &msg) {
    g_last_error = msg;
    std::fprintf(stderr, "[uniflow-web] error: %s\n", msg.c_str());
}

}  // namespace

std::string initBackend() {
    g_last_error.clear();
    try {
        auto &bp = uniflow::global_backend_pair();
        std::string name = ggml_backend_name(bp.backend);
        if (bp.has_gpu) {
            name += " (GPU)";
        }
        return name;
    } catch (const std::exception &e) {
        set_error(e.what());
        return std::string("error: ") + e.what();
    }
}

bool loadModels(const std::string &t5, const std::string &dit, const std::string &vae,
                const std::string &instructions) {
    g_last_error.clear();
    try {
        uniflow::PipelineConfig cfg;
        cfg.t5_gguf_path = t5;
        cfg.dit_gguf_path = dit;
        cfg.vae_gguf_path = vae;
        cfg.instructions_gguf_path = instructions;
        cfg.spiece_model_path.clear();
        cfg.n_threads = 4;
        cfg.num_steps = 25;
        cfg.guidance_scale = 5.0f;
        cfg.sway_sampling_coef = -1.0f;
        g_pipeline = std::make_unique<uniflow::Pipeline>(cfg);
        return true;
    } catch (const std::exception &e) {
        g_pipeline.reset();
        set_error(e.what());
        return false;
    }
}

// Returns PCM sample count @ 24 kHz. Pointer via pcmPointer(); call freePcm() when done.
int generateFromTokens(const emscripten::val &token_ids_js, float duration_sec, int steps,
                       float cfg, float sway, unsigned int seed) {
    g_last_error.clear();
    g_pcm.clear();
    if (!g_pipeline) {
        set_error("models not loaded");
        return 0;
    }
    try {
        const unsigned len = token_ids_js["length"].as<unsigned>();
        if (len == 0) {
            set_error("empty token ids");
            return 0;
        }
        std::vector<int32_t> ids(len);
        for (unsigned i = 0; i < len; ++i) {
            // transformers.js may put BigInt in the array; embind as<int32_t> rejects it.
            emscripten::val item = token_ids_js[i];
            if (item.typeOf().as<std::string>() == "bigint") {
                ids[i] = static_cast<int32_t>(item.as<long long>());
            } else {
                ids[i] = item.as<int32_t>();
            }
        }
        g_pipeline->set_generation_params(steps, cfg, sway, duration_sec);
        g_pipeline->set_seed(seed == 0 ? 42u : seed);
        g_pcm = g_pipeline->generate_from_tokens(ids);
        return static_cast<int>(g_pcm.size());
    } catch (const std::exception &e) {
        set_error(e.what());
        g_pcm.clear();
        return 0;
    }
}

std::string lastError() { return g_last_error; }

// Return as int so embind never hands JS a BigInt (uintptr_t → BigInt on some builds).
int pcmPointer() {
    return static_cast<int>(reinterpret_cast<uintptr_t>(g_pcm.data()));
}

int pcmSampleRate() { return 24000; }

void freePcm() {
    g_pcm.clear();
    g_pcm.shrink_to_fit();
}

EMSCRIPTEN_BINDINGS(uniflow_web) {
    emscripten::function("initBackend", &initBackend);
    emscripten::function("loadModels", &loadModels);
    emscripten::function("generateFromTokens", &generateFromTokens);
    emscripten::function("lastError", &lastError);
    emscripten::function("pcmPointer", &pcmPointer);
    emscripten::function("pcmSampleRate", &pcmSampleRate);
    emscripten::function("freePcm", &freePcm);
}
