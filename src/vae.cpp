#include "vae.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.h"
#include "ggml.h"

namespace uniflow {

namespace {

constexpr int kLatentDim = 128;
constexpr int kHop = 480;
constexpr size_t kMaxGraphSize = 1u << 14;

// Snake: y = x + sin(x * exp(a))^2 * inv_exp(b); a/b stored log-scale in GGUF.
// x: [T, C], exp_a/inv_b: [1, C]
struct ggml_tensor *snake(struct ggml_context *ctx, struct ggml_tensor *x,
                          struct ggml_tensor *exp_a, struct ggml_tensor *inv_b) {
    struct ggml_tensor *xa = ggml_mul(ctx, x, exp_a);
    struct ggml_tensor *s = ggml_sin(ctx, xa);
    struct ggml_tensor *s2 = ggml_mul(ctx, s, s);
    return ggml_add(ctx, x, ggml_mul(ctx, s2, inv_b));
}

// Conv1d + bias. w: [K, IC, OC], x: [T, IC] → [Tout, OC]
struct ggml_tensor *conv1d(struct ggml_context *ctx, struct ggml_tensor *w, struct ggml_tensor *b,
                           struct ggml_tensor *x, int stride, int pad, int dil) {
    struct ggml_tensor *y = ggml_conv_1d(ctx, w, x, stride, pad, dil);
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    if (b) {
        struct ggml_tensor *b2d = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        y = ggml_add(ctx, y, b2d);
    }
    return y;
}

// ConvTranspose via GEMM + col2im (Metal-friendly; ggml_conv_transpose_1d asserts p0==0).
// w_perm: [IC, K*OC], x: [T, IC] → [Tout, OC]
struct ggml_tensor *conv_t1d(struct ggml_context *ctx, struct ggml_tensor *w_perm,
                             struct ggml_tensor *b, struct ggml_tensor *x, int stride, int pad,
                             int oc) {
    struct ggml_tensor *xt = ggml_cont(ctx, ggml_transpose(ctx, x));  // [IC, T]
    struct ggml_tensor *col = ggml_mul_mat(ctx, w_perm, xt);          // [K*OC, T]
    struct ggml_tensor *y = ggml_col2im_1d(ctx, col, stride, oc, pad);
    if (b) {
        struct ggml_tensor *b2d = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        y = ggml_add(ctx, y, b2d);
    }
    return y;
}

void tensor_to_f32(struct ggml_tensor *t, std::vector<float> &out) {
    const int64_t n = ggml_nelements(t);
    out.resize(static_cast<size_t>(n));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, static_cast<size_t>(n) * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(static_cast<size_t>(n));
        ggml_backend_tensor_get(t, tmp.data(), 0, static_cast<size_t>(n) * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), out.data(), n);
    } else {
        throw std::runtime_error("vae: unsupported tensor dtype");
    }
}

}  // namespace

struct VaeDecoder::Impl {
    explicit Impl(const std::string &path, int /*n_threads*/)
        : bp(global_backend_pair()),
          model(path, bp.backend),
          sched(backend_sched_new(bp, kMaxGraphSize)),
          derived_ctx(nullptr),
          derived_buf(nullptr),
          graph_ctx(nullptr),
          graph_buf(nullptr),
          graph(nullptr),
          graph_input(nullptr),
          graph_output(nullptr),
          graph_T(0) {
        build_derived_weights();
        std::fprintf(stderr, "[vae] StableVAE decode via sched on %s (has_gpu=%d)\n",
                     ggml_backend_name(bp.backend), bp.has_gpu ? 1 : 0);
    }

    ~Impl() {
        if (graph_ctx) {
            ggml_backend_sched_reset(sched);
            ggml_free(graph_ctx);
            free(graph_buf);
        }
        if (sched) ggml_backend_sched_free(sched);
        if (derived_buf) ggml_backend_buffer_free(derived_buf);
        if (derived_ctx) ggml_free(derived_ctx);
    }

    BackendPair &bp;
    GgufModelGPU model;
    ggml_backend_sched_t sched;

    struct ggml_context *derived_ctx;
    ggml_backend_buffer_t derived_buf;

    // Precomputed snake: exp(alpha), 1/(exp(beta)+eps) as [1, C]
    struct ggml_tensor *blk_sa[4]{};
    struct ggml_tensor *blk_sb[4]{};
    struct ggml_tensor *res_s1a[4][3]{};
    struct ggml_tensor *res_s1b[4][3]{};
    struct ggml_tensor *res_s2a[4][3]{};
    struct ggml_tensor *res_s2b[4][3]{};
    struct ggml_tensor *snake_out_a = nullptr;
    struct ggml_tensor *snake_out_b = nullptr;

    // Permuted conv-transpose weights [IC, K*OC] F16
    struct ggml_tensor *ct_w[4]{};
    int ct_stride[4]{};
    int ct_pad[4]{};
    int ct_oc[4]{};
    int ct_ic[4]{};
    int ct_k[4]{};

    // Graph cache
    struct ggml_context *graph_ctx;
    uint8_t *graph_buf;
    struct ggml_cgraph *graph;
    struct ggml_tensor *graph_input;
    struct ggml_tensor *graph_output;
    int graph_T;
    std::vector<float> scratch_in;

    void alloc_snake_pair(const std::string &prefix, struct ggml_tensor **a_out,
                          struct ggml_tensor **b_out) {
        std::vector<float> raw_a;
        tensor_to_f32(model.tensor(prefix + ".alpha"), raw_a);
        const int C = static_cast<int>(raw_a.size());
        *a_out = ggml_new_tensor_2d(derived_ctx, GGML_TYPE_F32, 1, C);
        *b_out = ggml_new_tensor_2d(derived_ctx, GGML_TYPE_F32, 1, C);
        ggml_set_name(*a_out, (prefix + ".exp_a").c_str());
        ggml_set_name(*b_out, (prefix + ".inv_b").c_str());
    }

    void fill_snake(struct ggml_tensor *a, struct ggml_tensor *b, const std::string &prefix) {
        std::vector<float> raw_a, raw_b;
        tensor_to_f32(model.tensor(prefix + ".alpha"), raw_a);
        tensor_to_f32(model.tensor(prefix + ".beta"), raw_b);
        const int C = static_cast<int>(raw_a.size());
        std::vector<float> ea(static_cast<size_t>(C)), ib(static_cast<size_t>(C));
        for (int i = 0; i < C; ++i) {
            ea[static_cast<size_t>(i)] = std::exp(raw_a[static_cast<size_t>(i)]);
            ib[static_cast<size_t>(i)] =
                1.0f / (std::exp(raw_b[static_cast<size_t>(i)]) + 1e-9f);
        }
        ggml_backend_tensor_set(a, ea.data(), 0, ea.size() * sizeof(float));
        ggml_backend_tensor_set(b, ib.data(), 0, ib.size() * sizeof(float));
    }

    void build_derived_weights() {
        const int strides[4] = {10, 6, 4, 2};
        const size_t n_tensors = 80;
        size_t ctx_size = ggml_tensor_overhead() * n_tensors;
        struct ggml_init_params p = {ctx_size, nullptr, true};
        derived_ctx = ggml_init(p);
        if (!derived_ctx) throw std::runtime_error("vae: derived ctx init failed");

        for (int bi = 0; bi < 4; ++bi) {
            const std::string bprefix = "decoder.layers." + std::to_string(bi + 1);
            alloc_snake_pair(bprefix + ".layers.0", &blk_sa[bi], &blk_sb[bi]);

            struct ggml_tensor *w = model.tensor(bprefix + ".layers.1.weight");  // [K, OC, IC]
            ct_k[bi] = static_cast<int>(w->ne[0]);
            ct_oc[bi] = static_cast<int>(w->ne[1]);
            ct_ic[bi] = static_cast<int>(w->ne[2]);
            ct_stride[bi] = strides[bi];
            ct_pad[bi] = static_cast<int>(std::ceil(strides[bi] / 2.0));
            ct_w[bi] = ggml_new_tensor_2d(derived_ctx, GGML_TYPE_F16, ct_ic[bi],
                                          ct_k[bi] * ct_oc[bi]);
            ggml_set_name(ct_w[bi], (bprefix + ".layers.1.weight_perm").c_str());

            for (int r = 0; r < 3; ++r) {
                const std::string rp = bprefix + ".layers." + std::to_string(r + 2);
                alloc_snake_pair(rp + ".layers.0", &res_s1a[bi][r], &res_s1b[bi][r]);
                alloc_snake_pair(rp + ".layers.2", &res_s2a[bi][r], &res_s2b[bi][r]);
            }
        }
        alloc_snake_pair("decoder.layers.5", &snake_out_a, &snake_out_b);

        derived_buf = ggml_backend_alloc_ctx_tensors(derived_ctx, bp.backend);
        if (!derived_buf) throw std::runtime_error("vae: derived buffer alloc failed");
        ggml_backend_buffer_set_usage(derived_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (int bi = 0; bi < 4; ++bi) {
            const std::string bprefix = "decoder.layers." + std::to_string(bi + 1);
            fill_snake(blk_sa[bi], blk_sb[bi], bprefix + ".layers.0");
            for (int r = 0; r < 3; ++r) {
                const std::string rp = bprefix + ".layers." + std::to_string(r + 2);
                fill_snake(res_s1a[bi][r], res_s1b[bi][r], rp + ".layers.0");
                fill_snake(res_s2a[bi][r], res_s2b[bi][r], rp + ".layers.2");
            }

            // Permute CT weight [K, OC, IC] → ggml [IC, K*OC] storage for mul_mat
            // Torch ConvTranspose weight [IC, OC, K]; GGUF flat = k + oc*K + ic*K*OC
            // acestep dst: data[k_oc * IC + ic] with k_oc = k + oc*K (fan order)
            std::vector<float> w_f32;
            tensor_to_f32(model.tensor(bprefix + ".layers.1.weight"), w_f32);
            const int K = ct_k[bi], OC = ct_oc[bi], IC = ct_ic[bi];
            std::vector<float> perm(static_cast<size_t>(IC) * K * OC);
            for (int ic = 0; ic < IC; ++ic) {
                for (int k_oc = 0; k_oc < K * OC; ++k_oc) {
                    const int k = k_oc % K;
                    const int oc = k_oc / K;
                    const float v = w_f32[static_cast<size_t>(k) +
                                          static_cast<size_t>(oc) * K +
                                          static_cast<size_t>(ic) * K * OC];
                    perm[static_cast<size_t>(k_oc) * IC + static_cast<size_t>(ic)] = v;
                }
            }
            std::vector<ggml_fp16_t> w16(perm.size());
            ggml_fp32_to_fp16_row(perm.data(), w16.data(), static_cast<int64_t>(perm.size()));
            ggml_backend_tensor_set(ct_w[bi], w16.data(), 0, w16.size() * sizeof(ggml_fp16_t));
        }
        fill_snake(snake_out_a, snake_out_b, "decoder.layers.5");
    }

    struct ggml_tensor *res_unit(struct ggml_context *ctx, struct ggml_tensor *x, int bi, int r) {
        const int dil = (r == 0) ? 1 : (r == 1) ? 3 : 9;
        const std::string rp =
            "decoder.layers." + std::to_string(bi + 1) + ".layers." + std::to_string(r + 2);
        struct ggml_tensor *skip = x;
        x = snake(ctx, x, res_s1a[bi][r], res_s1b[bi][r]);
        x = conv1d(ctx, model.tensor(rp + ".layers.1.weight"), model.tensor(rp + ".layers.1.bias"), x,
                   1, (dil * (7 - 1)) / 2, dil);
        x = snake(ctx, x, res_s2a[bi][r], res_s2b[bi][r]);
        x = conv1d(ctx, model.tensor(rp + ".layers.3.weight"), model.tensor(rp + ".layers.3.bias"), x,
                   1, 0, 1);
        return ggml_add(ctx, skip, x);
    }

    struct ggml_tensor *build_graph(struct ggml_context *ctx, struct ggml_tensor *latent) {
        // latent: [T, 128]
        struct ggml_tensor *x =
            conv1d(ctx, model.tensor("decoder.layers.0.weight"), model.tensor("decoder.layers.0.bias"),
                   latent, 1, 3, 1);

        for (int bi = 0; bi < 4; ++bi) {
            x = snake(ctx, x, blk_sa[bi], blk_sb[bi]);
            x = conv_t1d(ctx, ct_w[bi],
                         model.tensor("decoder.layers." + std::to_string(bi + 1) + ".layers.1.bias"), x,
                         ct_stride[bi], ct_pad[bi], ct_oc[bi]);
            for (int r = 0; r < 3; ++r) {
                x = res_unit(ctx, x, bi, r);
            }
        }

        x = snake(ctx, x, snake_out_a, snake_out_b);
        x = conv1d(ctx, model.tensor("decoder.layers.6.weight"), nullptr, x, 1, 3, 1);
        return x;  // [T_audio, 1]
    }

    void ensure_graph(int T) {
        if (graph_T == T) return;
        if (graph_ctx) {
            ggml_backend_sched_reset(sched);
            ggml_free(graph_ctx);
            free(graph_buf);
            graph_ctx = nullptr;
            graph_buf = nullptr;
            graph_T = 0;
        }

        size_t ctx_size =
            ggml_tensor_overhead() * kMaxGraphSize + ggml_graph_overhead_custom(kMaxGraphSize, false);
        graph_buf = static_cast<uint8_t *>(malloc(ctx_size));
        struct ggml_init_params p = {ctx_size, graph_buf, true};
        graph_ctx = ggml_init(p);
        if (!graph_ctx) throw std::runtime_error("vae: graph ctx init failed");

        graph_input = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, T, kLatentDim);
        ggml_set_name(graph_input, "vae_input");
        ggml_set_input(graph_input);

        graph_output = build_graph(graph_ctx, graph_input);
        ggml_set_name(graph_output, "vae_output");
        ggml_set_output(graph_output);

        graph = ggml_new_graph_custom(graph_ctx, kMaxGraphSize, false);
        ggml_build_forward_expand(graph, graph_output);

        ggml_backend_sched_reset(sched);
        if (bp.has_gpu) {
            ggml_backend_sched_set_tensor_backend(sched, graph_input, bp.backend);
        }
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            throw std::runtime_error("vae: graph alloc failed");
        }
        graph_T = T;
        std::fprintf(stderr, "[vae] graph nodes=%d T=%d splits=%d on %s\n", ggml_graph_n_nodes(graph),
                     T, ggml_backend_sched_get_n_splits(sched), ggml_backend_name(bp.backend));
    }

    std::vector<float> decode(const std::vector<float> &latent, int T) {
        if (static_cast<int>(latent.size()) < T * kLatentDim) {
            throw std::runtime_error("vae: latent buffer too small for T");
        }
        ensure_graph(T);

        // DiT [T, C] row-major → ggml [T, C] storage (c*T + t)
        scratch_in.resize(static_cast<size_t>(T) * kLatentDim);
        for (int c = 0; c < kLatentDim; ++c) {
            for (int t = 0; t < T; ++t) {
                scratch_in[static_cast<size_t>(c) * T + t] =
                    latent[static_cast<size_t>(t) * kLatentDim + c];
            }
        }
        ggml_backend_tensor_set(graph_input, scratch_in.data(), 0,
                                scratch_in.size() * sizeof(float));

        if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("vae: graph compute failed");
        }

        const int T_audio = static_cast<int>(graph_output->ne[0]);
        const int C_out = static_cast<int>(graph_output->ne[1]);
        if (C_out != 1) {
            throw std::runtime_error("vae: expected mono output channels=1, got " +
                                     std::to_string(C_out));
        }
        if (T_audio != T * kHop) {
            std::fprintf(stderr, "[vae] WARNING: T_audio=%d expected %d\n", T_audio, T * kHop);
        }

        std::vector<float> out(static_cast<size_t>(T_audio));
        ggml_backend_tensor_get(graph_output, out.data(), 0, out.size() * sizeof(float));
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
