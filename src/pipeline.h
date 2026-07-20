#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace uniflow {

struct PipelineConfig {
    std::string t5_gguf_path;
    std::string dit_gguf_path;
    std::string vae_gguf_path;
    std::string instructions_gguf_path;
    std::string spiece_model_path;  // empty → caption tokenize unavailable; use token ids
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

    // Tokenize caption (requires spiece_model_path) then generate 24 kHz mono PCM.
    std::vector<float> generate(const std::string &caption);

    // Browser / external tokenizer path — skip SentencePiece.
    std::vector<float> generate_from_tokens(const std::vector<int32_t> &token_ids);

    void set_seed(unsigned int seed);

    // Update sampling knobs without reloading weights (web demo).
    void set_generation_params(int num_steps, float guidance_scale, float sway,
                               float duration_seconds);

private:
    struct Impl;
    Impl *impl_;
};

}  // namespace uniflow
