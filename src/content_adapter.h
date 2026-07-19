#pragma once

#include <vector>

#include "gguf_util.h"

namespace uniflow {

struct ContentAdapterOutput {
    std::vector<float> context;  // [context_len, 1024] NTA (caption) after adapter
    int context_len = 0;
    int global_latent_length = 0;
    std::vector<float> time_aligned_content;  // [T, 1024] dummy_ta_embed for T2A
};

// UniFlow CrossAttentionAdapter for text_to_audio / text_to_music:
//   ta = zeros[1,1024], nta = T5-proj(caption), prefix = instruction emb
//   nta_attn / ta_attn / cross_attn → duration from TA path; context = NTA
class ContentAdapter {
public:
    explicit ContentAdapter(GgufModel &dit_model, int n_threads = 4);
    ~ContentAdapter();

    // t5_hidden: [seq_len, 1024] raw T5 encoder output (pre-proj)
    // instruction: [instr_len, 1024] from instructions.gguf
    ContentAdapterOutput run(const std::vector<float> &t5_hidden, int seq_len,
                             const std::vector<float> &instruction, int instr_len);

private:
    struct Impl;
    Impl *impl_;
};

}  // namespace uniflow
