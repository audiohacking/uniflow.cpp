#pragma once

#include <string>
#include <vector>

namespace uniflow {

struct PipelineConfig {
    std::string t5_gguf_path;
    std::string dit_gguf_path;
    std::string vae_gguf_path;
    std::string instructions_gguf_path;
    std::string spiece_model_path;
    std::string task = "text_to_audio";  // or text_to_music
    int instruction_idx = 0;
    int num_steps = 25;
    float guidance_scale = 5.0f;
    float sway_sampling_coef = -1.0f;
    float duration_seconds = 0.0f;  // 0 = model prediction
    int n_threads = 4;
    unsigned int seed = 0;  // 0 = random
};

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig &config);
    ~Pipeline();

    // Generates 24 kHz mono PCM for the caption.
    std::vector<float> generate(const std::string &caption);

private:
    struct Impl;
    Impl *impl_;
};

}  // namespace uniflow
