#include "pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>

#include "backend.h"
#include "content_adapter.h"
#include "dit.h"
#include "gguf_util.h"
#include "instructions.h"
#include "scheduler.h"
#include "t5_encoder.h"
#ifndef UNIFLOW_WEB_BUILD
#include "tokenizer.h"
#endif
#include "vae.h"

namespace uniflow {

namespace {

constexpr int kContentDim = 1024;
constexpr int kLatentDim = 128;
constexpr int kLatentTokenRate = 50;

using clock = std::chrono::steady_clock;

double ms_since(clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

}  // namespace

struct Pipeline::Impl {
    explicit Impl(const PipelineConfig &config)
        : config(config),
          bp(global_backend_pair()),
          t5(config.t5_gguf_path, config.n_threads),
          dit_model(config.dit_gguf_path),
          instructions(config.instructions_gguf_path),
          adapter(dit_model, config.n_threads),
          dit(config.dit_gguf_path, config.n_threads),
          vae(config.vae_gguf_path, config.n_threads) {
#ifndef UNIFLOW_WEB_BUILD
        if (!config.spiece_model_path.empty()) {
            tokenizer = std::make_unique<T5Tokenizer>(config.spiece_model_path);
        }
#endif
        std::fprintf(stderr,
                     "[pipeline] backend=%s has_gpu=%d | T5+adapter=CPU | DiT+VAE=%s\n",
                     ggml_backend_name(bp.backend), bp.has_gpu ? 1 : 0,
                     ggml_backend_name(bp.backend));
        if (!bp.has_gpu) {
            std::fprintf(stderr,
                         "[pipeline] WARNING: no GPU backend — DiT/VAE will run on CPU\n");
        }
    }

    std::vector<float> generate_from_tokens(const std::vector<int32_t> &token_ids);

    PipelineConfig config;
    BackendPair &bp;
#ifndef UNIFLOW_WEB_BUILD
    std::unique_ptr<T5Tokenizer> tokenizer;
#endif
    T5Encoder t5;
    GgufModel dit_model;
    InstructionBank instructions;
    ContentAdapter adapter;
    DiT dit;
    VaeDecoder vae;
};

Pipeline::Pipeline(const PipelineConfig &config) : impl_(new Impl(config)) {}
Pipeline::~Pipeline() { delete impl_; }

void Pipeline::set_seed(unsigned int seed) { impl_->config.seed = seed; }

void Pipeline::set_generation_params(int num_steps, float guidance_scale, float sway,
                                     float duration_seconds) {
    impl_->config.num_steps = num_steps;
    impl_->config.guidance_scale = guidance_scale;
    impl_->config.sway_sampling_coef = sway;
    impl_->config.duration_seconds = duration_seconds;
}

std::vector<float> Pipeline::generate(const std::string &caption) {
#ifdef UNIFLOW_WEB_BUILD
    (void)caption;
    throw std::runtime_error("pipeline: caption generate unavailable in web build; use tokens");
#else
    if (!impl_->tokenizer) {
        throw std::runtime_error("pipeline: no tokenizer (empty spiece path); use generate_from_tokens");
    }
    auto t0 = clock::now();
    std::vector<int32_t> token_ids = impl_->tokenizer->encode(caption);
    std::fprintf(stderr, "[pipeline] tokens=%d (%.1f ms)\n", static_cast<int>(token_ids.size()),
                 ms_since(t0));
    return impl_->generate_from_tokens(token_ids);
#endif
}

std::vector<float> Pipeline::generate_from_tokens(const std::vector<int32_t> &token_ids) {
    return impl_->generate_from_tokens(token_ids);
}

std::vector<float> Pipeline::Impl::generate_from_tokens(const std::vector<int32_t> &token_ids) {
    auto t_all = clock::now();
    const int seq_len = static_cast<int>(token_ids.size());
    if (seq_len <= 0) {
        throw std::runtime_error("pipeline: empty token sequence");
    }

    auto t0 = clock::now();
    std::vector<float> t5_hidden = t5.encode(token_ids);
    std::fprintf(stderr, "[pipeline] T5(CPU) %d x %d (%.1f ms)\n", seq_len, t5.d_model(),
                 ms_since(t0));

    t0 = clock::now();
    const std::string task_key =
        (config.task == "text_to_music" ? "text_to_music_" : "text_to_audio_") +
        std::to_string(config.instruction_idx);
    int instr_len = 0;
    std::vector<float> instruction = instructions.get(task_key, &instr_len);
    std::fprintf(stderr, "[pipeline] instruction %s len=%d (%.1f ms)\n", task_key.c_str(),
                 instr_len, ms_since(t0));

    t0 = clock::now();
    ContentAdapterOutput ca_out = adapter.run(t5_hidden, seq_len, instruction, instr_len);

    int T = ca_out.global_latent_length;
    if (config.duration_seconds > 0.0f) {
        T = std::max(1, static_cast<int>(std::round(config.duration_seconds * kLatentTokenRate)));
        ca_out.time_aligned_content.resize(static_cast<size_t>(T) * kContentDim);
        std::vector<float> dummy(kContentDim);
        if (!ca_out.time_aligned_content.empty()) {
            std::memcpy(dummy.data(), ca_out.time_aligned_content.data(), kContentDim * sizeof(float));
        }
        for (int t = 0; t < T; ++t) {
            std::memcpy(ca_out.time_aligned_content.data() + static_cast<size_t>(t) * kContentDim,
                        dummy.data(), kContentDim * sizeof(float));
        }
    }
    std::fprintf(stderr, "[pipeline] adapter(CPU) context_len=%d T=%d (%.2fs) (%.1f ms)\n",
                 ca_out.context_len, T, static_cast<float>(T) / kLatentTokenRate, ms_since(t0));

    std::vector<float> latent(static_cast<size_t>(T) * kLatentDim);
    unsigned int seed = config.seed;
    if (seed == 0) {
#ifdef __EMSCRIPTEN__
        seed = 42;  // random_device is weak/unavailable in some browsers
#else
        seed = static_cast<unsigned int>(std::random_device{}());
#endif
    }
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (float &v : latent) v = dist(rng);
    }
    std::fprintf(stderr, "[pipeline] seed=%u\n", seed);

    FlowMatchScheduler scheduler(config.num_steps, config.sway_sampling_coef);
    std::vector<float> sigmas = scheduler.sigmas();
    std::vector<float> timesteps = scheduler.timesteps();

    const bool use_cfg = config.guidance_scale > 1.0f;
    const float cfg_scale = config.guidance_scale;
    std::vector<float> uncond_context(ca_out.context.size(), 0.0f);
    std::vector<float> uncond_ta(ca_out.time_aligned_content.size(), 0.0f);

    std::fprintf(stderr, "[pipeline] DiT loop steps=%d cfg=%.1f on %s\n", config.num_steps,
                 cfg_scale, ggml_backend_name(bp.backend));

    t0 = clock::now();
    for (int step = 0; step < config.num_steps; ++step) {
        float sigma = sigmas[step];
        float sigma_next = sigmas[step + 1];
        float timestep = timesteps[step];

        std::vector<float> velocity;
        if (use_cfg) {
            auto vel_u = dit.forward(latent, T, timestep, uncond_context, ca_out.context_len,
                                     uncond_ta);
            auto vel_c = dit.forward(latent, T, timestep, ca_out.context, ca_out.context_len,
                                     ca_out.time_aligned_content);
            velocity.resize(vel_c.size());
            for (size_t i = 0; i < velocity.size(); ++i) {
                velocity[i] = vel_u[i] + cfg_scale * (vel_c[i] - vel_u[i]);
            }
        } else {
            velocity = dit.forward(latent, T, timestep, ca_out.context, ca_out.context_len,
                                   ca_out.time_aligned_content);
        }

        const float dt = sigma_next - sigma;
        for (size_t i = 0; i < latent.size(); ++i) {
            latent[i] += dt * velocity[i];
        }
        if ((step + 1) % 5 == 0 || step == 0) {
            std::fprintf(stderr, "[pipeline] step %d/%d sigma=%.4f\n", step + 1, config.num_steps,
                         sigma);
        }
    }
    const double dit_ms = ms_since(t0);
    std::fprintf(stderr, "[pipeline] DiT done (%.1f ms, %.1f ms/step)\n", dit_ms,
                 dit_ms / std::max(1, config.num_steps));

    t0 = clock::now();
    std::fprintf(stderr, "[pipeline] VAE decode on %s...\n", ggml_backend_name(bp.backend));
    std::vector<float> waveform = vae.decode(latent, T);
    std::fprintf(stderr, "[pipeline] VAE %zu samples (%.1f ms)\n", waveform.size(), ms_since(t0));
    std::fprintf(stderr, "[pipeline] TOTAL %.1f ms (%.2fs audio @ 24kHz)\n", ms_since(t_all),
                 static_cast<float>(waveform.size()) / 24000.0f);
    return waveform;
}

}  // namespace uniflow
