#include <cstdio>

#include "backend.h"
#include "ggml.h"
#include "ggml-cpu.h"

// Regression: backend init + tiny ggml add graph (Phase 0 conquer).
int main() {
    auto &bp = uniflow::global_backend_pair();
    if (!bp.backend) {
        std::fprintf(stderr, "FAIL: backend is null\n");
        return 1;
    }

    struct ggml_init_params params = {16 * 1024 * 1024, nullptr, false};
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: ggml_init\n");
        return 1;
    }

    auto *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    auto *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ((float *) a->data)[0] = 1.0f;
    ((float *) a->data)[1] = 2.0f;
    ((float *) b->data)[0] = 3.0f;
    ((float *) b->data)[1] = 4.0f;
    auto *sum = ggml_add(ctx, a, b);
    auto *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sum);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    if (((float *) sum->data)[0] != 4.0f || ((float *) sum->data)[1] != 6.0f) {
        std::fprintf(stderr, "FAIL: unexpected sum values\n");
        ggml_free(ctx);
        return 1;
    }
    ggml_free(ctx);
    std::printf("PASS test_smoke\n");
    return 0;
}
