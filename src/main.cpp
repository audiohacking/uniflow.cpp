#include <cstdio>
#include <cstring>
#include <string>

#include "backend.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "scheduler.h"
#include "wav_io.h"

namespace {

void print_usage(const char *argv0) {
    std::fprintf(stderr,
        "uniflow-audio — GGML/GGUF inference for UniFlow-Audio\n"
        "\n"
        "Usage:\n"
        "  %s --smoke-test\n"
        "  %s --help\n"
        "\n"
        "Full T2A/T2M generation (T5 + DiT + StableVAE) is WIP.\n"
        "See DEVELOPMENT.md for status and blockers.\n",
        argv0, argv0);
}

// Phase 0 smoke: tiny ggml graph + backend init + scheduler sanity.
int run_smoke_test() {
    auto &bp = uniflow::global_backend_pair();
    std::fprintf(stderr, "[smoke] backend=%s has_gpu=%d\n",
                 ggml_backend_name(bp.backend), bp.has_gpu ? 1 : 0);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return 1;
    }

    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    for (int i = 0; i < 4; ++i) {
        ((float *) a->data)[i] = static_cast<float>(i);
        ((float *) b->data)[i] = static_cast<float>(10 * i);
    }
    struct ggml_tensor *sum = ggml_add(ctx, a, b);

    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sum);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    const float expected[] = {0.f, 11.f, 22.f, 33.f};
    for (int i = 0; i < 4; ++i) {
        float got = ((float *) sum->data)[i];
        if (got != expected[i]) {
            std::fprintf(stderr, "smoke graph mismatch at %d: got %f want %f\n",
                         i, got, expected[i]);
            ggml_free(ctx);
            return 1;
        }
    }
    std::printf("ggml smoke test: [0.0, 11.0, 22.0, 33.0] OK\n");
    ggml_free(ctx);

    // Scheduler: UniFlow defaults (25 steps, sway=-1) — length must match.
    uniflow::FlowMatchScheduler sched(/*num_steps=*/25, /*sway_coef=*/-1.0f);
    auto sigmas = sched.sigmas();
    // Flow-match Euler uses N+1 sigma knots for N steps.
    if (sigmas.size() != 26) {
        std::fprintf(stderr, "scheduler size mismatch: %zu (want 26)\n", sigmas.size());
        return 1;
    }
    if (sigmas.front() <= sigmas.back()) {
        std::fprintf(stderr, "scheduler expected descending sigmas\n");
        return 1;
    }
    std::printf("scheduler smoke: steps=%d knots=%zu sigma0=%.4f sigmaN=%.4f OK\n",
                sched.num_steps(), sigmas.size(), sigmas.front(), sigmas.back());

    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (std::strcmp(argv[i], "--smoke-test") == 0) {
            return run_smoke_test();
        }
    }
    std::fprintf(stderr,
                 "Full inference CLI not wired yet (need dit.gguf + vae.gguf).\n"
                 "Run: %s --smoke-test\n"
                 "See DEVELOPMENT.md.\n",
                 argv[0]);
    return 1;
}
