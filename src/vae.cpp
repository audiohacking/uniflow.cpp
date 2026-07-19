#include "vae.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"

namespace uniflow {

namespace {

// ---------------------------------------------------------------------------
// Tensor -> f32 helper (handles F32 / F16 GGUF tensors).
// ---------------------------------------------------------------------------
std::vector<float> tensor_to_f32(const struct ggml_tensor *t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(static_cast<size_t>(n));
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, static_cast<size_t>(n) * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t *>(t->data), out.data(), n);
    } else {
        throw std::runtime_error("vae: unsupported tensor dtype");
    }
    return out;
}

// A 1-D activation buffer in channels-first layout: data[c * T + t].
struct Tensor1d {
    std::vector<float> data;
    int C = 0;
    int T = 0;

    Tensor1d() = default;
    Tensor1d(int c, int t) : data(static_cast<size_t>(c) * t, 0.0f), C(c), T(t) {}

    float &at(int c, int t) { return data[static_cast<size_t>(c) * T + t]; }
    float at(int c, int t) const { return data[static_cast<size_t>(c) * T + t]; }
};

// ---------------------------------------------------------------------------
// Weight containers.
// ---------------------------------------------------------------------------

// Regular Conv1d. Weight flat layout matches torch [Cout, Cin, K] (row-major),
// which is identical to the GGUF/ggml element order (ne = [K, Cin, Cout]).
struct Conv1d {
    std::vector<float> w;   // [Cout * Cin * K]
    std::vector<float> b;   // [Cout] (empty if no bias)
    int Cout = 0;
    int Cin = 0;
    int K = 0;
    int pad = 0;
    int dilation = 1;
    bool has_bias = false;

    float weight(int o, int i, int k) const {
        return w[(static_cast<size_t>(o) * Cin + i) * K + k];
    }
};

// ConvTranspose1d. Weight flat layout matches torch [Cin, Cout, K] (row-major),
// identical to GGUF element order (ne = [K, Cout, Cin]).
struct ConvT1d {
    std::vector<float> w;   // [Cin * Cout * K]
    std::vector<float> b;   // [Cout]
    int Cin = 0;
    int Cout = 0;
    int K = 0;
    int stride = 1;
    int pad = 0;
    bool has_bias = false;

    float weight(int i, int o, int k) const {
        return w[(static_cast<size_t>(i) * Cout + o) * K + k];
    }
};

// SnakeBeta activation. alpha/beta stored in LOG scale in the GGUF; exp() applied
// at runtime.
struct Snake {
    std::vector<float> alpha;  // [C], log scale
    std::vector<float> beta;   // [C], log scale
    int C = 0;
};

struct ResidualUnit {
    Snake s1;
    Conv1d c7;  // k=7, dilated
    Snake s2;
    Conv1d c1;  // k=1
};

struct DecoderBlock {
    Snake snake;      // pre-activation
    ConvT1d up;       // upsampling transpose conv
    ResidualUnit res[3];  // dilations 1, 3, 9
    int stride = 1;
};

// ---------------------------------------------------------------------------
// CPU kernels.
// ---------------------------------------------------------------------------

Tensor1d conv1d_forward(const Tensor1d &x, const Conv1d &c) {
    // stride = 1 for all Conv1d layers in this decoder.
    const int Tin = x.T;
    const int Tout = Tin + 2 * c.pad - c.dilation * (c.K - 1);
    Tensor1d y(c.Cout, Tout);
    for (int o = 0; o < c.Cout; ++o) {
        const float bias = c.has_bias ? c.b[o] : 0.0f;
        for (int t = 0; t < Tout; ++t) {
            float acc = bias;
            for (int i = 0; i < c.Cin; ++i) {
                for (int k = 0; k < c.K; ++k) {
                    const int in_t = t - c.pad + k * c.dilation;
                    if (in_t >= 0 && in_t < Tin) {
                        acc += c.weight(o, i, k) * x.at(i, in_t);
                    }
                }
            }
            y.at(o, t) = acc;
        }
    }
    return y;
}

Tensor1d conv_transpose_1d_forward(const Tensor1d &x, const ConvT1d &c) {
    const int Tin = x.T;
    const int Tout = (Tin - 1) * c.stride - 2 * c.pad + c.K;
    Tensor1d y(c.Cout, Tout);
    // Initialize with bias.
    if (c.has_bias) {
        for (int o = 0; o < c.Cout; ++o) {
            for (int t = 0; t < Tout; ++t) {
                y.at(o, t) = c.b[o];
            }
        }
    }
    for (int i = 0; i < c.Cin; ++i) {
        for (int t = 0; t < Tin; ++t) {
            const float xv = x.at(i, t);
            const int base = t * c.stride - c.pad;
            for (int k = 0; k < c.K; ++k) {
                const int out_t = base + k;
                if (out_t >= 0 && out_t < Tout) {
                    for (int o = 0; o < c.Cout; ++o) {
                        y.at(o, out_t) += c.weight(i, o, k) * xv;
                    }
                }
            }
        }
    }
    return y;
}

// snake: y = x + (1 / (beta + 1e-9)) * sin(x * alpha)^2, with alpha/beta in log
// scale (exp applied here).
void snake_forward(Tensor1d &x, const Snake &s) {
    for (int c = 0; c < s.C; ++c) {
        const float alpha = std::exp(s.alpha[c]);
        const float beta = std::exp(s.beta[c]);
        const float inv_beta = 1.0f / (beta + 1e-9f);
        for (int t = 0; t < x.T; ++t) {
            const float v = x.at(c, t);
            const float sn = std::sin(v * alpha);
            x.at(c, t) = v + inv_beta * sn * sn;
        }
    }
}

Tensor1d residual_forward(const Tensor1d &x, const ResidualUnit &r) {
    Tensor1d h = x;
    snake_forward(h, r.s1);
    h = conv1d_forward(h, r.c7);
    snake_forward(h, r.s2);
    h = conv1d_forward(h, r.c1);
    // Residual add (shapes match: same channels, same T).
    for (size_t i = 0; i < h.data.size(); ++i) {
        h.data[i] += x.data[i];
    }
    return h;
}

Tensor1d decoder_block_forward(const Tensor1d &x, const DecoderBlock &blk) {
    Tensor1d h = x;
    snake_forward(h, blk.snake);
    h = conv_transpose_1d_forward(h, blk.up);
    h = residual_forward(h, blk.res[0]);
    h = residual_forward(h, blk.res[1]);
    h = residual_forward(h, blk.res[2]);
    return h;
}

}  // namespace

// ---------------------------------------------------------------------------
// Impl.
// ---------------------------------------------------------------------------

struct VaeDecoder::Impl {
    Impl(const std::string &path, int nt) : model(path), n_threads(nt) { load(); }

    GgufModel model;
    int n_threads;

    Conv1d conv_in;                 // decoder.layers.0
    std::vector<DecoderBlock> blocks;  // decoder.layers.1..4
    Snake snake_out;                // decoder.layers.5
    Conv1d conv_out;                // decoder.layers.6 (no bias)

    // Load a Conv1d (torch weight [Cout, Cin, K], GGUF ne = [K, Cin, Cout]).
    Conv1d load_conv1d(const std::string &prefix, int pad, int dilation, bool expect_bias) {
        Conv1d c;
        const struct ggml_tensor *w = model.tensor(prefix + ".weight");
        c.K = static_cast<int>(w->ne[0]);
        c.Cin = static_cast<int>(w->ne[1]);
        c.Cout = static_cast<int>(w->ne[2]);
        c.w = tensor_to_f32(w);
        c.pad = pad;
        c.dilation = dilation;
        if (expect_bias && model.has_tensor(prefix + ".bias")) {
            c.b = tensor_to_f32(model.tensor(prefix + ".bias"));
            c.has_bias = true;
        }
        return c;
    }

    // Load a ConvTranspose1d (torch weight [Cin, Cout, K], GGUF ne = [K, Cout, Cin]).
    ConvT1d load_convt1d(const std::string &prefix, int stride, int pad) {
        ConvT1d c;
        const struct ggml_tensor *w = model.tensor(prefix + ".weight");
        c.K = static_cast<int>(w->ne[0]);
        c.Cout = static_cast<int>(w->ne[1]);
        c.Cin = static_cast<int>(w->ne[2]);
        c.w = tensor_to_f32(w);
        c.stride = stride;
        c.pad = pad;
        if (model.has_tensor(prefix + ".bias")) {
            c.b = tensor_to_f32(model.tensor(prefix + ".bias"));
            c.has_bias = true;
        }
        return c;
    }

    Snake load_snake(const std::string &prefix) {
        Snake s;
        s.alpha = tensor_to_f32(model.tensor(prefix + ".alpha"));
        s.beta = tensor_to_f32(model.tensor(prefix + ".beta"));
        s.C = static_cast<int>(s.alpha.size());
        return s;
    }

    ResidualUnit load_residual(const std::string &prefix, int dilation) {
        ResidualUnit r;
        r.s1 = load_snake(prefix + ".layers.0");
        r.c7 = load_conv1d(prefix + ".layers.1", (dilation * (7 - 1)) / 2, dilation, true);
        r.s2 = load_snake(prefix + ".layers.2");
        r.c1 = load_conv1d(prefix + ".layers.3", 0, 1, true);
        return r;
    }

    DecoderBlock load_block(const std::string &prefix, int stride) {
        DecoderBlock blk;
        blk.stride = stride;
        blk.snake = load_snake(prefix + ".layers.0");
        blk.up = load_convt1d(prefix + ".layers.1", stride,
                              static_cast<int>(std::ceil(stride / 2.0)));
        blk.res[0] = load_residual(prefix + ".layers.2", 1);
        blk.res[1] = load_residual(prefix + ".layers.3", 3);
        blk.res[2] = load_residual(prefix + ".layers.4", 9);
        return blk;
    }

    void load() {
        // decoder.layers.0: Conv1d 128 -> 1024, k=7, pad=3.
        conv_in = load_conv1d("decoder.layers.0", 3, 1, true);

        // decoder.layers.1..4: DecoderBlocks. Strides are reversed encoder strides.
        const int strides[4] = {10, 6, 4, 2};
        blocks.clear();
        for (int i = 0; i < 4; ++i) {
            blocks.push_back(load_block("decoder.layers." + std::to_string(i + 1), strides[i]));
        }

        // decoder.layers.5: final SnakeBeta on 128 channels.
        snake_out = load_snake("decoder.layers.5");

        // decoder.layers.6: Conv1d 128 -> 1, k=7, pad=3, bias=False.
        conv_out = load_conv1d("decoder.layers.6", 3, 1, false);
    }

    std::vector<float> decode(const std::vector<float> &latent, int T) {
        const int latent_dim = 128;
        if (static_cast<int>(latent.size()) < T * latent_dim) {
            throw std::runtime_error("vae: latent buffer too small for T");
        }

        // Convert [T, 128] row-major -> channels-first [128, T].
        Tensor1d x(latent_dim, T);
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < latent_dim; ++c) {
                x.at(c, t) = latent[static_cast<size_t>(t) * latent_dim + c];
            }
        }

        x = conv1d_forward(x, conv_in);
        for (const auto &blk : blocks) {
            x = decoder_block_forward(x, blk);
        }
        snake_forward(x, snake_out);
        x = conv1d_forward(x, conv_out);

        // final_tanh = false -> Identity. Output is mono: [1, T*480].
        // Return the single channel as a flat vector.
        std::vector<float> out(x.data.begin(), x.data.begin() + x.T);
        return out;
    }
};

VaeDecoder::VaeDecoder(const std::string &gguf_path, int n_threads)
    : impl_(new Impl(gguf_path, n_threads)) {}

VaeDecoder::~VaeDecoder() { delete impl_; }

std::vector<float> VaeDecoder::decode(const std::vector<float> &latent, int T) {
    return impl_->decode(latent, T);
}

}  // namespace uniflow
