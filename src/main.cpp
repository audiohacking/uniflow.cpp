#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "pipeline.h"
#include "scheduler.h"
#include "wav_io.h"

namespace {

void print_usage(const char *argv0) {
    std::fprintf(stderr,
        "uniflow-audio — GGML/GGUF inference for UniFlow-Audio\n"
        "\n"
        "Usage:\n"
        "  %s --smoke-test\n"
        "  %s <t5.gguf> <dit.gguf> <vae.gguf> <instructions.gguf> <spiece.model> \\\n"
        "     --caption \"...\" [--output out.wav] [--steps 25] [--cfg 5.0] [--seed N] \\\n"
        "     [--task t2a|t2m] [--duration SECS] [--threads N]\n",
        argv0, argv0);
}

int run_smoke_test() {
    auto &bp = uniflow::global_backend_pair();
    std::fprintf(stderr, "[smoke] backend=%s has_gpu=%d\n", ggml_backend_name(bp.backend),
                 bp.has_gpu ? 1 : 0);

    struct ggml_init_params params = {16 * 1024 * 1024, nullptr, false};
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) return 1;
    auto *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    for (int i = 0; i < 4; ++i) {
        ((float *) a->data)[i] = static_cast<float>(i);
        ((float *) b->data)[i] = static_cast<float>(10 * i);
    }
    auto *sum = ggml_add(ctx, a, b);
    auto *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sum);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    const float expected[] = {0.f, 11.f, 22.f, 33.f};
    for (int i = 0; i < 4; ++i) {
        if (((float *) sum->data)[i] != expected[i]) {
            ggml_free(ctx);
            return 1;
        }
    }
    ggml_free(ctx);
    std::printf("ggml smoke test OK\n");

    uniflow::FlowMatchScheduler sched(25, -1.0f);
    auto sigmas = sched.sigmas();
    if (sigmas.size() != 26) return 1;
    std::printf("scheduler smoke OK\n");
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

    if (argc < 6) {
        print_usage(argv[0]);
        return 1;
    }

    uniflow::PipelineConfig cfg;
    cfg.t5_gguf_path = argv[1];
    cfg.dit_gguf_path = argv[2];
    cfg.vae_gguf_path = argv[3];
    cfg.instructions_gguf_path = argv[4];
    cfg.spiece_model_path = argv[5];

    std::string caption;
    std::string output = "output.wav";

    for (int i = 6; i < argc; ++i) {
        auto need = [&](const char *flag) -> const char * {
            if (std::strcmp(argv[i], flag) != 0) return nullptr;
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[++i];
        };
        try {
            if (const char *v = need("--caption")) {
                caption = v;
            } else if (const char *v = need("--output")) {
                output = v;
            } else if (const char *v = need("--steps")) {
                cfg.num_steps = std::atoi(v);
            } else if (const char *v = need("--cfg")) {
                cfg.guidance_scale = std::atof(v);
            } else if (const char *v = need("--seed")) {
                cfg.seed = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
            } else if (const char *v = need("--duration")) {
                cfg.duration_seconds = std::atof(v);
            } else if (const char *v = need("--threads")) {
                cfg.n_threads = std::atoi(v);
            } else if (const char *v = need("--task")) {
                if (std::strcmp(v, "t2m") == 0 || std::strcmp(v, "text_to_music") == 0) {
                    cfg.task = "text_to_music";
                } else {
                    cfg.task = "text_to_audio";
                }
            } else if (const char *v = need("--instruction-idx")) {
                cfg.instruction_idx = std::atoi(v);
            } else {
                std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
                return 1;
            }
        } catch (const std::exception &e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }

    if (caption.empty()) {
        std::fprintf(stderr, "--caption is required\n");
        return 1;
    }

    try {
        uniflow::Pipeline pipe(cfg);
        std::vector<float> wav = pipe.generate(caption);
        if (!uniflow::write_wav_f32_mono(output, wav, 24000)) {
            std::fprintf(stderr, "failed to write %s\n", output.c_str());
            return 1;
        }
        std::printf("Wrote %s (%zu samples)\n", output.c_str(), wav.size());
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
