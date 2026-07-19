#include "content_adapter.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"

namespace uniflow {

namespace {

constexpr int kContentDim = 1024;
constexpr int kNumHeads = 16;
constexpr int kHeadDim = kContentDim / kNumHeads;  // 64
constexpr float kDurationOffset = 1.0f;
constexpr int kLatentTokenRate = 50;  // 24000 / 480
constexpr float kLnEps = 1e-5f;

std::vector<float> tensor_to_f32(const struct ggml_tensor *t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(static_cast<size_t>(n));
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, static_cast<size_t>(n) * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t *>(t->data), out.data(), n);
    } else {
        throw std::runtime_error("content_adapter: unsupported tensor dtype");
    }
    return out;
}

}  // namespace

struct ContentAdapter::Impl {
    explicit Impl(GgufModel &m, int n_threads) : model(m), n_threads(n_threads) {}
    GgufModel &model;
    int n_threads;
};

ContentAdapter::ContentAdapter(GgufModel &dit_model, int n_threads)
    : impl_(new Impl(dit_model, n_threads)) {}

ContentAdapter::~ContentAdapter() { delete impl_; }

ContentAdapterOutput ContentAdapter::run(const std::vector<float> &t5_hidden, int seq_len,
                                         const std::vector<float> &instruction, int instr_len) {
    GgufModel &model = impl_->model;

    const size_t mem_size = 64ull * 1024 * 1024;
    struct ggml_init_params cparams = {mem_size, nullptr, false};
    struct ggml_context *ctx = ggml_init(cparams);
    if (!ctx) {
        throw std::runtime_error("content_adapter: ggml_init failed");
    }

    auto mha = [&](const char *attn_prefix, struct ggml_tensor *query, int q_len,
                   struct ggml_tensor *kv, int kv_len) -> struct ggml_tensor * {
        std::string p(attn_prefix);
        struct ggml_tensor *in_proj_w = model.tensor(p + ".in_proj_weight");  // [3072, 1024]
        struct ggml_tensor *in_proj_b = model.tensor(p + ".in_proj_bias");

        auto slice_w = [&](int chunk) {
            return ggml_cont(ctx, ggml_view_2d(ctx, in_proj_w, kContentDim, kContentDim,
                                               in_proj_w->nb[1],
                                               static_cast<size_t>(chunk) * kContentDim * in_proj_w->nb[1]));
        };
        auto slice_b = [&](int chunk) {
            return ggml_view_1d(ctx, in_proj_b, kContentDim,
                                static_cast<size_t>(chunk) * kContentDim * in_proj_b->nb[0]);
        };

        // PyTorch MHA in_proj: [q,k,v] stacked; Q from query, K/V from kv (kdim=vdim=prefix)
        struct ggml_tensor *q =
            ggml_add(ctx, ggml_mul_mat(ctx, slice_w(0), query), slice_b(0));
        struct ggml_tensor *k =
            ggml_add(ctx, ggml_mul_mat(ctx, slice_w(1), kv), slice_b(1));
        struct ggml_tensor *v =
            ggml_add(ctx, ggml_mul_mat(ctx, slice_w(2), kv), slice_b(2));

        q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, kHeadDim, kNumHeads, q_len), 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, kHeadDim, kNumHeads, kv_len), 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, kHeadDim, kNumHeads, kv_len), 1, 2, 0, 3));

        struct ggml_tensor *scores = ggml_mul_mat(ctx, k, q);
        struct ggml_tensor *attn =
            ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt(static_cast<float>(kHeadDim)), 0.0f);
        struct ggml_tensor *kqv = ggml_mul_mat(ctx, v, attn);
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
        kqv = ggml_reshape_2d(ctx, kqv, kContentDim, q_len);

        return ggml_add(ctx,
                        ggml_mul_mat(ctx, model.tensor(p + ".out_proj.weight"), kqv),
                        model.tensor(p + ".out_proj.bias"));
    };

    auto layer_norm = [&](struct ggml_tensor *x, const char *prefix) {
        struct ggml_tensor *n = ggml_norm(ctx, x, kLnEps);
        n = ggml_mul(ctx, n, model.tensor(std::string(prefix) + ".weight"));
        return ggml_add(ctx, n, model.tensor(std::string(prefix) + ".bias"));
    };

    // --- inputs ---
    struct ggml_tensor *h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kContentDim, seq_len);
    std::memcpy(h->data, t5_hidden.data(), t5_hidden.size() * sizeof(float));

    struct ggml_tensor *nta = ggml_add(
        ctx, ggml_mul_mat(ctx, model.tensor("content_encoder.text_encoder.proj.weight"), h),
        model.tensor("content_encoder.text_encoder.proj.bias"));

    // TA dummy content: zeros length 1 (matches content_encoder for T2A)
    struct ggml_tensor *ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kContentDim, 1);
    std::memset(ta->data, 0, kContentDim * sizeof(float));

    struct ggml_tensor *instr = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kContentDim, instr_len);
    std::memcpy(instr->data, instruction.data(), instruction.size() * sizeof(float));

    // nta_attn(query=nta, key/value=instruction)
    struct ggml_tensor *nta_out = mha("content_adapter.nta_attn", nta, seq_len, instr, instr_len);
    nta_out = layer_norm(ggml_add(ctx, nta_out, nta), "content_adapter.nta_norm");
    // nta_proj: Conv1d 1x1 → Linear
    struct ggml_tensor *nta_proj_w =
        ggml_reshape_2d(ctx, model.tensor("content_adapter.nta_proj.weight"), kContentDim, kContentDim);
    nta_out = ggml_add(ctx, ggml_mul_mat(ctx, nta_proj_w, nta_out),
                       model.tensor("content_adapter.nta_proj.bias"));

    // ta_attn(query=ta, key/value=instruction)
    struct ggml_tensor *ta_out = mha("content_adapter.ta_attn", ta, 1, instr, instr_len);
    ta_out = layer_norm(ggml_add(ctx, ta_out, ta), "content_adapter.ta_norm");
    struct ggml_tensor *ta_proj_w =
        ggml_reshape_2d(ctx, model.tensor("content_adapter.ta_proj.weight"), kContentDim, kContentDim);
    ta_out = ggml_add(ctx, ggml_mul_mat(ctx, ta_proj_w, ta_out),
                      model.tensor("content_adapter.ta_proj.bias"));

    // cross_attn(query=ta, key/value=nta)
    struct ggml_tensor *cross = mha("content_adapter.cross_attn", ta_out, 1, nta_out, seq_len);
    cross = layer_norm(ggml_add(ctx, cross, ta_out), "content_adapter.cross_norm");

    // global duration MLP on the single TA token
    struct ggml_tensor *gd =
        ggml_add(ctx, ggml_mul_mat(ctx, model.tensor("content_adapter.global_duration_mlp.0.weight"), cross),
                 model.tensor("content_adapter.global_duration_mlp.0.bias"));
    gd = ggml_relu(ctx, gd);
    gd = ggml_add(ctx, ggml_mul_mat(ctx, model.tensor("content_adapter.global_duration_mlp.3.weight"), gd),
                  model.tensor("content_adapter.global_duration_mlp.3.bias"));

    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, nta_out);
    ggml_build_forward_expand(graph, gd);
    ggml_graph_compute_with_ctx(ctx, graph, impl_->n_threads);

    ContentAdapterOutput result;
    result.context.resize(static_cast<size_t>(seq_len) * kContentDim);
    std::memcpy(result.context.data(), nta_out->data, result.context.size() * sizeof(float));
    result.context_len = seq_len;

    const float global_duration_pred = static_cast<const float *>(gd->data)[0];
    const float global_duration = std::exp(global_duration_pred) - kDurationOffset;
    result.global_latent_length =
        std::max(1, static_cast<int>(std::round(global_duration * kLatentTokenRate)));

    // T2A: backbone time_aligned_content = learnable dummy_ta_embed
    std::vector<float> dummy_ta = tensor_to_f32(model.tensor("dummy_ta_embed"));
    result.time_aligned_content.assign(static_cast<size_t>(result.global_latent_length) * kContentDim, 0.0f);
    for (int t = 0; t < result.global_latent_length; ++t) {
        std::memcpy(result.time_aligned_content.data() + static_cast<size_t>(t) * kContentDim,
                    dummy_ta.data(), kContentDim * sizeof(float));
    }

    ggml_free(ctx);
    return result;
}

}  // namespace uniflow
