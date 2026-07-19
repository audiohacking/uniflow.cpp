#pragma once

#include <string>
#include <vector>

#include "gguf_util.h"

// LayerFusionAudioDiT: architecture read from GGUF metadata
// (uniflow.dit_embed_dim / dit_num_heads / dit_n_in_blocks / …).
// Small=512/8/10, Base=1024/16/12, Large=1280/20/14.
namespace uniflow {

class DiT {
public:
    explicit DiT(const std::string &gguf_path, int n_threads = 4);
    ~DiT();

    int latent_dim() const;  // from GGUF (128)
    int embed_dim() const;   // from GGUF (512/1024/1280)

    // x: [T, latent_dim] row-major -- the noisy latent.
    // timestep: scalar diffusion timestep, same convention as the reference
    // TimestepEmbedder (raw timestep value, not normalized to [0,1]).
    // context: [context_len, content_dim] row-major
    // (ContentAdapterOutput::context).
    // time_aligned_content: [T, content_dim] row-major
    // (ContentAdapterOutput::time_aligned_content; length must equal T).
    // Returns the predicted velocity field, [T, latent_dim] row-major.
    std::vector<float> forward(const std::vector<float> &x, int T, float timestep,
                                const std::vector<float> &context, int context_len,
                                const std::vector<float> &time_aligned_content);

private:
    struct Impl;
    Impl *impl_;
};

}  // namespace uniflow
