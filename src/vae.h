#pragma once

#include <string>
#include <vector>

#include "gguf_util.h"

namespace uniflow {

// StableVAE Oobleck decoder (24 kHz). weight_norm already fused in GGUF.
class VaeDecoder {
public:
    explicit VaeDecoder(const std::string &gguf_path, int n_threads = 4);
    ~VaeDecoder();

    int latent_dim() const { return 128; }
    int hop() const { return 480; }
    int sample_rate() const { return 24000; }

    // latent: [T, 128] row-major DiT output → mono PCM length ≈ T * 480
    std::vector<float> decode(const std::vector<float> &latent, int T);

private:
    struct Impl;
    Impl *impl_;
};

}  // namespace uniflow
