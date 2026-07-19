#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "vae.h"

// Constructs the StableVAE Oobleck decoder from models/vae.gguf (if present) and
// decodes a tiny random latent. Skips gracefully when the model file is missing
// so the test suite can run without large model assets.
int main() {
    const std::string path = "models/vae.gguf";

    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::printf("SKIP test_vae_load (%s not found)\n", path.c_str());
        return 0;
    }
    std::fclose(f);

    uniflow::VaeDecoder vae(path);

    const int T = 4;
    const int latent_dim = vae.latent_dim();  // 128
    const int hop = vae.hop();                // 480

    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> latent(static_cast<size_t>(T) * latent_dim);
    for (auto &v : latent) {
        v = dist(rng);
    }

    std::vector<float> audio = vae.decode(latent, T);

    const size_t expected = static_cast<size_t>(T) * hop;
    if (audio.size() != expected) {
        std::fprintf(stderr, "FAIL: output length %zu != expected %zu\n",
                     audio.size(), expected);
        return 1;
    }

    for (size_t i = 0; i < audio.size(); ++i) {
        if (!std::isfinite(audio[i])) {
            std::fprintf(stderr, "FAIL: non-finite sample at %zu\n", i);
            return 1;
        }
    }

    std::printf("PASS test_vae_load (n=%zu)\n", audio.size());
    return 0;
}
