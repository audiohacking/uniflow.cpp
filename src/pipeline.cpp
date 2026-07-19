#include "pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

#include "content_adapter.h"
#include "dit.h"
#include "gguf_util.h"
#include "instructions.h"
#include "scheduler.h"
#include "t5_encoder.h"
#include "tokenizer.h"
#include "vae.h"

namespace uniflow {

namespace {

constexpr int kContentDim = 1024;
constexpr int kLatentDim = 128;
constexpr int kLatentTokenRate = 50;  // 24000 / 480

}  // namespace

struct Pipeline::Impl {
    explicit Impl(const PipelineConfig &config)
        : config(config),
          tokenizer(config.spiece_model_path),
          t5(config.t5_gguf_path, config.n_threads),
          dit_model(config.dit_gguf_path),
          instructions(config.instructions_gguf_path),
          adapter(dit_model, config.n_threads),
          dit(config.dit_gguf_path, config.n_threads),
          vae(config.vae_gguf_path, config.n_threads) {}

    PipelineConfig config;
    T5Tokenizer tokenizer;
    T5Encoder t5;
    GgufModel dit_model;
    InstructionBank instructions;
    ContentAdapter adapter;
    DiT dit;
    VaeDecoder vae;
};

Pipeline::Pipeline(const PipelineConfig &config) : impl_(new Impl(config)) {}
Pipeline::~Pipeline() { delete impl_; }

std::vector<float> Pipeline::generate(const std::string &caption) {
    // 1. Tokenize
    std::vector<int32_t> token_ids = impl_->tokenizer.encode(caption);
    const int seq_len = static_cast<int>(token_ids.size());
    std::fprintf(stderr, "[pipeline] tokens=%d\n", seq_len);

    // 2. T5 encode
    std::vector<float> t5_hidden = impl_->t5.encode(token_ids);
    std::fprintf(stderr, "[pipeline] T5 %d x %d\n", seq_len, impl_->t5.d_model());

    // 3. Instruction embedding
    const std::string task_key =
        (impl_->config.task == "text_to_music" ? "text_to_music_" : "text_to_audio_") +
        std::to_string(impl_->config.instruction_idx);
    int instr_len = 0;
    std::vector<float> instruction = impl_->instructions.get(task_key, &instr_len);
    std::fprintf(stderr, "[pipeline] instruction %s len=%d\n", task_key.c_str(), instr_len);

    // 4. Content adapter
    ContentAdapterOutput ca_out =
        impl_->adapter.run(t5_hidden, seq_len, instruction, instr_len);

    int T = ca_out.global_latent_length;
    if (impl_->config.duration_seconds > 0.0f) {
        T = std::max(1, static_cast<int>(std::round(impl_->config.duration_seconds * kLatentTokenRate)));
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
    std::fprintf(stderr, "[pipeline] context_len=%d T=%d (%.2fs)\n", ca_out.context_len, T,
                 static_cast<float>(T) / kLatentTokenRate);

    // 5. Noise latent
    std::vector<float> latent(static_cast<size_t>(T) * kLatentDim);
    unsigned int seed = impl_->config.seed;
    if (seed == 0) {
        seed = static_cast<unsigned int>(std::random_device{}());
    }
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (float &v : latent) v = dist(rng);
    }
    std::fprintf(stderr, "[pipeline] seed=%u\n", seed);

    // 6. Flow-matching loop
    FlowMatchScheduler scheduler(impl_->config.num_steps, impl_->config.sway_sampling_coef);
    std::vector<float> sigmas = scheduler.sigmas();
    std::vector<float> timesteps = scheduler.timesteps();

    const bool use_cfg = impl_->config.guidance_scale > 1.0f;
    const float cfg_scale = impl_->config.guidance_scale;
    std::vector<float> uncond_context(ca_out.context.size(), 0.0f);
    std::vector<float> uncond_ta(ca_out.time_aligned_content.size(), 0.0f);

    std::fprintf(stderr, "[pipeline] steps=%d cfg=%.1f\n", impl_->config.num_steps, cfg_scale);

    for (int step = 0; step < impl_->config.num_steps; ++step) {
        float sigma = sigmas[step];
        float sigma_next = sigmas[step + 1];
        float timestep = timesteps[step];

        std::vector<float> velocity;
        if (use_cfg) {
            auto vel_u = impl_->dit.forward(latent, T, timestep, uncond_context, ca_out.context_len, uncond_ta);
            auto vel_c = impl_->dit.forward(latent, T, timestep, ca_out.context, ca_out.context_len,
                                            ca_out.time_aligned_content);
            velocity.resize(vel_c.size());
            for (size_t i = 0; i < velocity.size(); ++i) {
                velocity[i] = vel_u[i] + cfg_scale * (vel_c[i] - vel_u[i]);
            }
        } else {
            velocity = impl_->dit.forward(latent, T, timestep, ca_out.context, ca_out.context_len,
                                          ca_out.time_aligned_content);
        }

        const float dt = sigma_next - sigma;
        for (size_t i = 0; i < latent.size(); ++i) {
            latent[i] += dt * velocity[i];
        }
        if ((step + 1) % 5 == 0 || step == 0) {
            std::fprintf(stderr, "[pipeline] step %d/%d sigma=%.4f\n", step + 1, impl_->config.num_steps,
                         sigma);
        }
    }

    std::fprintf(stderr, "[pipeline] VAE decode...\n");
    std::vector<float> waveform = impl_->vae.decode(latent, T);
    std::fprintf(stderr, "[pipeline] %zu samples (%.2fs @ 24kHz)\n", waveform.size(),
                 static_cast<float>(waveform.size()) / 24000.0f);
    return waveform;
}

}  // namespace uniflow
