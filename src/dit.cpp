#include "dit.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.h"
#include "ggml.h"

namespace uniflow {

namespace {

constexpr int kLatentDim = 128;
constexpr int kContentDim = 1024;
constexpr int kFreqEmbedDim = 256;
constexpr float kLnEps = 1e-5f;
// Large (depth 28) needs more nodes than Small; keep headroom.
constexpr size_t kMaxGraphSize = 1u << 17;

std::vector<float> sinusoidal_timestep_embedding(float t, int dim, float max_period = 10000.0f) {
    std::vector<float> out(dim);
    const int half = dim / 2;
    for (int i = 0; i < half; ++i) {
        const float freq = std::exp(-std::log(max_period) * static_cast<float>(i) / static_cast<float>(half));
        const float arg = t * freq;
        out[i] = std::cos(arg);
        out[half + i] = std::sin(arg);
    }
    return out;
}

}  // namespace

struct DiT::Impl {
    explicit Impl(const std::string &gguf_path, int n_threads)
        : bp(global_backend_pair()),
          model(gguf_path, bp.backend),
          sched(nullptr),
          n_threads(n_threads) {
        embed_dim = model.kv_i32("uniflow.dit_embed_dim");
        num_heads = model.kv_i32("uniflow.dit_num_heads");
        n_half_blocks = model.kv_i32("uniflow.dit_n_in_blocks");
        latent_dim = model.kv_i32("uniflow.latent_dim");
        content_dim = model.kv_i32("uniflow.content_dim");
        if (embed_dim <= 0 || num_heads <= 0 || embed_dim % num_heads != 0) {
            throw std::runtime_error("DiT: invalid embed_dim/num_heads in GGUF metadata");
        }
        if (n_half_blocks <= 0) {
            throw std::runtime_error("DiT: invalid dit_n_in_blocks in GGUF metadata");
        }
        if (latent_dim != kLatentDim || content_dim != kContentDim) {
            throw std::runtime_error("DiT: unexpected latent/content dims (expected 128/1024)");
        }
        head_dim = embed_dim / num_heads;
        variant = model.kv_str("uniflow.variant");
        sched = backend_sched_new(bp, kMaxGraphSize);
        std::fprintf(stderr,
                     "[dit] variant=%s embed=%d heads=%d half_blocks=%d via sched on %s (has_gpu=%d)\n",
                     variant.c_str(), embed_dim, num_heads, n_half_blocks,
                     ggml_backend_name(bp.backend), bp.has_gpu ? 1 : 0);
    }

    ~Impl() {
        if (sched) ggml_backend_sched_free(sched);
    }

    BackendPair &bp;
    GgufModelGPU model;
    ggml_backend_sched_t sched;
    int n_threads;
    int embed_dim = 0;
    int num_heads = 0;
    int head_dim = 0;
    int n_half_blocks = 0;
    int latent_dim = 0;
    int content_dim = 0;
    std::string variant;
};

DiT::DiT(const std::string &gguf_path, int n_threads) : impl_(new Impl(gguf_path, n_threads)) {}

DiT::~DiT() { delete impl_; }

int DiT::latent_dim() const { return impl_->latent_dim; }
int DiT::embed_dim() const { return impl_->embed_dim; }

std::vector<float> DiT::forward(const std::vector<float> &x, int T, float timestep,
                                const std::vector<float> &context, int context_len,
                                const std::vector<float> &time_aligned_content) {
    GgufModelGPU &model = impl_->model;
    ggml_backend_sched_t sched = impl_->sched;
    const int embed_dim = impl_->embed_dim;
    const int num_heads = impl_->num_heads;
    const int head_dim = impl_->head_dim;
    const int n_half_blocks = impl_->n_half_blocks;
    const int latent_dim = impl_->latent_dim;
    const int content_dim = impl_->content_dim;

    const size_t mem_size = ggml_tensor_overhead() * kMaxGraphSize + 1024 * 1024;
    struct ggml_init_params cparams = {mem_size, nullptr, true};
    struct ggml_context *ctx = ggml_init(cparams);
    if (!ctx) {
        throw std::runtime_error("DiT::forward: ggml_init failed");
    }

    struct ggml_tensor *x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent_dim, T);
    ggml_set_name(x_in, "x_in");
    ggml_set_input(x_in);

    struct ggml_tensor *context_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, content_dim, context_len);
    ggml_set_name(context_in, "context_in");
    ggml_set_input(context_in);

    struct ggml_tensor *ta_content_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, content_dim, T);
    ggml_set_name(ta_content_in, "ta_content_in");
    ggml_set_input(ta_content_in);

    struct ggml_tensor *positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    auto linear = [&](struct ggml_tensor *w, struct ggml_tensor *b, struct ggml_tensor *in) {
        struct ggml_tensor *out = ggml_mul_mat(ctx, w, in);
        if (b) out = ggml_add(ctx, out, b);
        return out;
    };

    auto layer_norm = [&](struct ggml_tensor *in, const std::string &prefix) {
        struct ggml_tensor *normed = ggml_norm(ctx, in, kLnEps);
        normed = ggml_mul(ctx, normed, model.tensor(prefix + ".weight"));
        normed = ggml_add(ctx, normed, model.tensor(prefix + ".bias"));
        return normed;
    };

    auto film = [&](struct ggml_tensor *normed, struct ggml_tensor *shift, struct ggml_tensor *scale) {
        struct ggml_tensor *out = ggml_add(ctx, ggml_mul(ctx, normed, scale), normed);
        out = ggml_add(ctx, out, shift);
        return out;
    };

    struct ggml_tensor *patch_w =
        ggml_reshape_2d(ctx, model.tensor("backbone.patch_embed.proj.weight"), latent_dim, embed_dim);
    struct ggml_tensor *x_h = linear(patch_w, model.tensor("backbone.patch_embed.proj.bias"), x_in);

    struct ggml_tensor *ctx_h =
        linear(model.tensor("backbone.context_embed.0.weight"), model.tensor("backbone.context_embed.0.bias"),
               context_in);
    ctx_h = ggml_silu(ctx, ctx_h);
    ctx_h = linear(model.tensor("backbone.context_embed.2.weight"), model.tensor("backbone.context_embed.2.bias"),
                   ctx_h);

    struct ggml_tensor *t_freq_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFreqEmbedDim);
    ggml_set_name(t_freq_t, "t_freq");
    ggml_set_input(t_freq_t);

    struct ggml_tensor *time_token =
        linear(model.tensor("backbone.time_embed.mlp.0.weight"), model.tensor("backbone.time_embed.mlp.0.bias"),
               t_freq_t);
    time_token = ggml_silu(ctx, time_token);
    time_token =
        linear(model.tensor("backbone.time_embed.mlp.2.weight"), model.tensor("backbone.time_embed.mlp.2.bias"),
               time_token);
    time_token = ggml_silu(ctx, time_token);

    struct ggml_tensor *time_ada_final =
        linear(model.tensor("backbone.time_ada_final.weight"), model.tensor("backbone.time_ada_final.bias"),
               time_token);
    struct ggml_tensor *shift_final = ggml_view_1d(ctx, time_ada_final, embed_dim, 0);
    struct ggml_tensor *scale_final =
        ggml_view_1d(ctx, time_ada_final, embed_dim, static_cast<size_t>(embed_dim) * time_ada_final->nb[0]);

    auto qk_layernorm = [&](struct ggml_tensor *t, const std::string &prefix) {
        struct ggml_tensor *normed = ggml_norm(ctx, t, kLnEps);
        normed = ggml_mul(ctx, normed, model.tensor(prefix + ".weight"));
        normed = ggml_add(ctx, normed, model.tensor(prefix + ".bias"));
        return normed;
    };

    auto self_attention = [&](struct ggml_tensor *x_norm, const std::string &prefix) {
        struct ggml_tensor *q = ggml_mul_mat(ctx, model.tensor(prefix + ".to_q.weight"), x_norm);
        struct ggml_tensor *k = ggml_mul_mat(ctx, model.tensor(prefix + ".to_k.weight"), x_norm);
        struct ggml_tensor *v = ggml_mul_mat(ctx, model.tensor(prefix + ".to_v.weight"), x_norm);

        q = ggml_reshape_3d(ctx, q, head_dim, num_heads, T);
        k = ggml_reshape_3d(ctx, k, head_dim, num_heads, T);
        v = ggml_reshape_3d(ctx, v, head_dim, num_heads, T);

        q = qk_layernorm(q, prefix + ".norm_q");
        k = qk_layernorm(k, prefix + ".norm_k");

        q = ggml_rope(ctx, q, positions, head_dim, GGML_ROPE_TYPE_NEOX);
        k = ggml_rope(ctx, k, positions, head_dim, GGML_ROPE_TYPE_NEOX);

        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));

        struct ggml_tensor *scores = ggml_mul_mat(ctx, k, q);
        struct ggml_tensor *attn =
            ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);

        struct ggml_tensor *kqv = ggml_mul_mat(ctx, v, attn);
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
        kqv = ggml_reshape_2d(ctx, kqv, embed_dim, T);

        return linear(model.tensor(prefix + ".proj.weight"), model.tensor(prefix + ".proj.bias"), kqv);
    };

    auto cross_attention = [&](struct ggml_tensor *x_norm, struct ggml_tensor *context_normed,
                               const std::string &prefix) {
        struct ggml_tensor *q = ggml_mul_mat(ctx, model.tensor(prefix + ".to_q.weight"), x_norm);
        struct ggml_tensor *k = ggml_mul_mat(ctx, model.tensor(prefix + ".to_k.weight"), context_normed);
        struct ggml_tensor *v = ggml_mul_mat(ctx, model.tensor(prefix + ".to_v.weight"), context_normed);

        q = ggml_reshape_3d(ctx, q, head_dim, num_heads, T);
        k = ggml_reshape_3d(ctx, k, head_dim, num_heads, context_len);
        v = ggml_reshape_3d(ctx, v, head_dim, num_heads, context_len);

        q = qk_layernorm(q, prefix + ".norm_q");
        k = qk_layernorm(k, prefix + ".norm_k");

        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));

        struct ggml_tensor *scores = ggml_mul_mat(ctx, k, q);
        struct ggml_tensor *attn =
            ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);

        struct ggml_tensor *kqv = ggml_mul_mat(ctx, v, attn);
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
        kqv = ggml_reshape_2d(ctx, kqv, embed_dim, T);

        return linear(model.tensor(prefix + ".proj.weight"), model.tensor(prefix + ".proj.bias"), kqv);
    };

    auto geglu_mlp = [&](struct ggml_tensor *x_norm, const std::string &prefix) {
        struct ggml_tensor *proj =
            linear(model.tensor(prefix + ".net.0.proj.weight"), model.tensor(prefix + ".net.0.proj.bias"), x_norm);
        const int64_t inner = proj->ne[0] / 2;
        struct ggml_tensor *hidden = ggml_cont(ctx, ggml_view_2d(ctx, proj, inner, proj->ne[1], proj->nb[1], 0));
        struct ggml_tensor *gate = ggml_cont(
            ctx, ggml_view_2d(ctx, proj, inner, proj->ne[1], proj->nb[1], static_cast<size_t>(inner) * proj->nb[0]));
        struct ggml_tensor *act = ggml_mul(ctx, hidden, ggml_gelu_erf(ctx, gate));
        return linear(model.tensor(prefix + ".net.2.weight"), model.tensor(prefix + ".net.2.bias"), act);
    };

    auto run_block = [&](struct ggml_tensor *x, const std::string &prefix, struct ggml_tensor *skip) {
        if (skip) {
            struct ggml_tensor *cat = ggml_concat(ctx, x, skip, 0);
            cat = layer_norm(cat, prefix + ".skip_norm");
            x = linear(model.tensor(prefix + ".skip_linear.weight"), model.tensor(prefix + ".skip_linear.bias"), cat);
        }

        struct ggml_tensor *time_ada =
            linear(model.tensor(prefix + ".adaln.time_ada.weight"), model.tensor(prefix + ".adaln.time_ada.bias"),
                   time_token);
        auto chunk = [&](int i) {
            return ggml_view_1d(ctx, time_ada, embed_dim, static_cast<size_t>(i) * embed_dim * time_ada->nb[0]);
        };
        struct ggml_tensor *shift_msa = chunk(0);
        struct ggml_tensor *scale_msa = chunk(1);
        struct ggml_tensor *gate_msa = chunk(2);
        struct ggml_tensor *shift_mlp = chunk(3);
        struct ggml_tensor *scale_mlp = chunk(4);
        struct ggml_tensor *gate_mlp = chunk(5);

        struct ggml_tensor *x_norm1 = film(layer_norm(x, prefix + ".norm1"), shift_msa, scale_msa);
        struct ggml_tensor *attn_out = self_attention(x_norm1, prefix + ".attn");
        struct ggml_tensor *tanh_gate_msa = ggml_tanh(ctx, ggml_scale_bias(ctx, gate_msa, -1.0f, 1.0f));
        x = ggml_add(ctx, x, ggml_mul(ctx, attn_out, tanh_gate_msa));

        struct ggml_tensor *ta_normed = layer_norm(ta_content_in, prefix + ".ta_context_norm");
        struct ggml_tensor *ta_proj =
            ggml_mul_mat(ctx, model.tensor(prefix + ".ta_context_projection.weight"), ta_normed);
        x = ggml_add(ctx, x, ta_proj);

        struct ggml_tensor *x_norm2 = layer_norm(x, prefix + ".norm2");
        struct ggml_tensor *context_normed = layer_norm(ctx_h, prefix + ".norm_context");
        struct ggml_tensor *cross_out = cross_attention(x_norm2, context_normed, prefix + ".cross_attn");
        x = ggml_add(ctx, x, cross_out);

        struct ggml_tensor *x_norm3 = film(layer_norm(x, prefix + ".norm3"), shift_mlp, scale_mlp);
        struct ggml_tensor *mlp_out = geglu_mlp(x_norm3, prefix + ".mlp");
        struct ggml_tensor *one_minus_gate_mlp = ggml_scale_bias(ctx, gate_mlp, -1.0f, 1.0f);
        x = ggml_add(ctx, x, ggml_mul(ctx, mlp_out, one_minus_gate_mlp));

        return x;
    };

    std::vector<struct ggml_tensor *> skips;
    skips.reserve(static_cast<size_t>(n_half_blocks));

    for (int i = 0; i < n_half_blocks; ++i) {
        x_h = run_block(x_h, "backbone.in_blocks." + std::to_string(i), nullptr);
        skips.push_back(x_h);
    }

    x_h = run_block(x_h, "backbone.mid_block", nullptr);

    for (int i = 0; i < n_half_blocks; ++i) {
        struct ggml_tensor *skip = skips.back();
        skips.pop_back();
        x_h = run_block(x_h, "backbone.out_blocks." + std::to_string(i), skip);
    }

    struct ggml_tensor *final_normed =
        film(layer_norm(x_h, "backbone.final_block.norm"), shift_final, scale_final);
    struct ggml_tensor *final_lin =
        linear(model.tensor("backbone.final_block.linear.weight"),
               model.tensor("backbone.final_block.linear.bias"), final_normed);

    struct ggml_tensor *conv_in = ggml_cont(ctx, ggml_transpose(ctx, final_lin));
    conv_in = ggml_reshape_3d(ctx, conv_in, T, latent_dim, 1);
    struct ggml_tensor *conv_out =
        ggml_conv_1d(ctx, model.tensor("backbone.final_block.final_layer.weight"), conv_in, 1, 1, 1);
    conv_out = ggml_reshape_2d(ctx, conv_out, T, latent_dim);
    conv_out = ggml_cont(ctx, ggml_transpose(ctx, conv_out));
    conv_out = ggml_add(ctx, conv_out, model.tensor("backbone.final_block.final_layer.bias"));

    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, kMaxGraphSize, false);
    ggml_build_forward_expand(graph, conv_out);
    ggml_set_output(conv_out);

    ggml_backend_sched_reset(sched);
    if (impl_->bp.has_gpu) {
        ggml_backend_sched_set_tensor_backend(sched, x_in, impl_->bp.backend);
        ggml_backend_sched_set_tensor_backend(sched, context_in, impl_->bp.backend);
        ggml_backend_sched_set_tensor_backend(sched, ta_content_in, impl_->bp.backend);
        ggml_backend_sched_set_tensor_backend(sched, positions, impl_->bp.backend);
        ggml_backend_sched_set_tensor_backend(sched, t_freq_t, impl_->bp.backend);
    }
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        ggml_free(ctx);
        throw std::runtime_error("DiT::forward: failed to allocate compute graph");
    }

    ggml_backend_tensor_set(x_in, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(context_in, context.data(), 0, context.size() * sizeof(float));
    ggml_backend_tensor_set(ta_content_in, time_aligned_content.data(), 0,
                            time_aligned_content.size() * sizeof(float));

    std::vector<int32_t> pos_data(T);
    for (int i = 0; i < T; ++i) pos_data[i] = i;
    ggml_backend_tensor_set(positions, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));

    std::vector<float> t_freq = sinusoidal_timestep_embedding(timestep, kFreqEmbedDim);
    ggml_backend_tensor_set(t_freq_t, t_freq.data(), 0, t_freq.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("DiT::forward: graph compute failed");
    }

    std::vector<float> result(static_cast<size_t>(T) * static_cast<size_t>(latent_dim));
    ggml_backend_tensor_get(conv_out, result.data(), 0, result.size() * sizeof(float));

    ggml_free(ctx);
    return result;
}

}  // namespace uniflow
