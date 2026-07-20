#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
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
        "  %s <t5.gguf> <dit.gguf> <vae.gguf> <instructions.gguf> <spiece.model> [options]\n"
        "  %s --models-dir DIR [options]\n"
        "  %s --model small|base|large [options]\n"
        "\n"
        "Model paths (one of):\n"
        "  positional: <t5> <dit> <vae> <instructions> <spiece>\n"
        "  --models-dir DIR   Expects DIR/{t5_encoder,vae,instructions}.gguf + DIR/spiece.model\n"
        "                     + DIR/dit-<QUANT>.gguf (or legacy DIR/dit.gguf)\n"
        "  --model NAME       Shortcut → models/uniflow-audio-v1.1-NAME/ (default: base)\n"
        "  --quant QUANT      DiT quant: F16|Q8_0|Q4_0 (default: Q8_0 if present)\n"
        "\n"
        "Prompt options (use ONE of):\n"
        "  --caption TEXT     Text caption for T2A/T2M (plain text; no Dasheng tags)\n"
        "  --batch FILE       One caption per line (# comments / blank lines skipped)\n"
        "\n"
        "Task / instruction (UniFlow):\n"
        "  --task t2a|t2m     text_to_audio (default) or text_to_music\n"
        "  --instruction-idx N   Instruction embedding index 0..9 (default: 0)\n"
        "\n"
        "Generation options:\n"
        "  --steps N          Flow-matching steps (default: 25)\n"
        "  --cfg SCALE        Classifier-free guidance (default: 5.0; Space T2A/T2M)\n"
        "  --sway COEF        Sway sampling coefficient (default: -1.0)\n"
        "  --duration SECS    Fixed length in seconds (default: 0 = model duration pred)\n"
        "  --seed N           RNG seed (default: random); batch uses seed, seed+1, ...\n"
        "  --threads N        CPU threads for T5/adapter (default: 4)\n"
        "\n"
        "Output options:\n"
        "  --output FILE      Output WAV (default: output.wav); single caption only\n"
        "  --output-dir DIR   Directory for batch mode (default: .)\n"
        "\n"
        "Environment:\n"
        "  GGML_BACKEND=Metal|GPU|CPU|MTL0   Force ggml backend (Metal default on Apple)\n"
        "\n"
        "Examples:\n"
        "  %s --model base --quant Q8_0 --caption \"a dog barking\" --duration 5 --seed 42\n"
        "  %s --model small --quant F16 --caption \"lo-fi hip hop beat\" --task t2m\n"
        "  %s --models-dir models --batch prompts.txt --output-dir out/\n"
        "\n"
        "Download GGUF packs (HF):\n"
        "  ./scripts/download_gguf.sh              # base Q8_0 (default)\n"
        "  ./scripts/download_gguf.sh small F16\n"
        "  ./scripts/download_gguf.sh large Q4_0\n",
        argv0, argv0, argv0, argv0, argv0, argv0, argv0);
}

std::vector<std::string> parse_batch_file(const std::string &path) {
    std::vector<std::string> prompts;
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("failed to open batch file: " + path);
    }
    std::string line;
    while (std::getline(f, line)) {
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        if (line[start] == '#') continue;
        size_t end = line.find_last_not_of(" \t\r\n");
        if (end != std::string::npos) {
            prompts.push_back(line.substr(start, end - start + 1));
        }
    }
    return prompts;
}

bool file_readable(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

// Normalize CLI/HF quant tags: f16 → F16, q8_0 → Q8_0, q4_0 → Q4_0.
std::string normalize_quant(const std::string &raw) {
    std::string s;
    s.reserve(raw.size());
    for (char c : raw) {
        if (c >= 'a' && c <= 'z') {
            s.push_back(static_cast<char>(c - 'a' + 'A'));
        } else {
            s.push_back(c);
        }
    }
    if (s == "F16" || s == "FP16") return "F16";
    if (s == "Q8" || s == "Q8_0") return "Q8_0";
    if (s == "Q4" || s == "Q4_0") return "Q4_0";
    throw std::runtime_error("--quant must be F16|Q8_0|Q4_0 (got: " + raw + ")");
}

// Prefer named dit-<QUANT>.gguf (HF GGUF convention); fall back to legacy dit.gguf.
std::string resolve_dit_path(const std::string &models_dir, const std::string &quant_opt) {
    if (!quant_opt.empty()) {
        const std::string tagged = models_dir + "/dit-" + quant_opt + ".gguf";
        if (file_readable(tagged)) return tagged;
        throw std::runtime_error("missing DiT file: " + tagged);
    }
    static const char *kPrefer[] = {"Q8_0", "F16", "Q4_0"};
    for (const char *q : kPrefer) {
        const std::string tagged = models_dir + "/dit-" + q + ".gguf";
        if (file_readable(tagged)) return tagged;
    }
    const std::string legacy = models_dir + "/dit.gguf";
    if (file_readable(legacy)) return legacy;
    throw std::runtime_error(
        "no DiT GGUF in " + models_dir +
        " (expected dit-F16.gguf / dit-Q8_0.gguf / dit-Q4_0.gguf or dit.gguf)");
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

    uniflow::PipelineConfig cfg;
    std::string caption;
    std::string batch_file;
    std::string output_path = "output.wav";
    std::string output_dir = ".";
    std::string models_dir;
    std::string model_variant;  // small|base|large|xlarge → models/uniflow-audio-v1.1-*/
    std::string quant;          // F16|Q8_0|Q4_0 (empty = auto)
    bool seed_set = false;
    bool have_models = false;

    int argi = 1;
    auto apply_model_variant = [&](const std::string &v) {
        if (v != "small" && v != "base" && v != "large" && v != "xlarge") {
            throw std::runtime_error("--model must be small|base|large|xlarge");
        }
        model_variant = v;
        models_dir = "models/uniflow-audio-v1.1-" + v;
    };

    if (argi < argc && std::strcmp(argv[argi], "--models-dir") == 0) {
        if (argi + 1 >= argc) {
            std::fprintf(stderr, "missing value for --models-dir\n");
            return 1;
        }
        models_dir = argv[++argi];
        ++argi;
    } else if (argi < argc && std::strcmp(argv[argi], "--model") == 0) {
        if (argi + 1 >= argc) {
            std::fprintf(stderr, "missing value for --model\n");
            return 1;
        }
        try {
            apply_model_variant(argv[++argi]);
        } catch (const std::exception &e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
        ++argi;
    } else if (argi + 4 < argc && argv[argi][0] != '-') {
        cfg.t5_gguf_path = argv[argi++];
        cfg.dit_gguf_path = argv[argi++];
        cfg.vae_gguf_path = argv[argi++];
        cfg.instructions_gguf_path = argv[argi++];
        cfg.spiece_model_path = argv[argi++];
        have_models = true;
    }

    for (; argi < argc; ++argi) {
        auto need = [&](const char *flag) -> const char * {
            if (std::strcmp(argv[argi], flag) != 0) return nullptr;
            if (argi + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[++argi];
        };
        try {
            if (const char *v = need("--models-dir")) {
                models_dir = v;
            } else if (const char *v = need("--model")) {
                apply_model_variant(v);
            } else if (const char *v = need("--quant")) {
                quant = normalize_quant(v);
            } else if (const char *v = need("--caption")) {
                caption = v;
            } else if (const char *v = need("--batch")) {
                batch_file = v;
            } else if (const char *v = need("--output")) {
                output_path = v;
            } else if (const char *v = need("--output-dir")) {
                output_dir = v;
            } else if (const char *v = need("--steps")) {
                cfg.num_steps = std::atoi(v);
            } else if (const char *v = need("--cfg")) {
                cfg.guidance_scale = static_cast<float>(std::atof(v));
            } else if (const char *v = need("--sway")) {
                cfg.sway_sampling_coef = static_cast<float>(std::atof(v));
            } else if (const char *v = need("--seed")) {
                cfg.seed = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
                seed_set = true;
            } else if (const char *v = need("--duration")) {
                cfg.duration_seconds = static_cast<float>(std::atof(v));
            } else if (const char *v = need("--threads")) {
                cfg.n_threads = std::atoi(v);
            } else if (const char *v = need("--task")) {
                if (std::strcmp(v, "t2m") == 0 || std::strcmp(v, "text_to_music") == 0) {
                    cfg.task = "text_to_music";
                } else if (std::strcmp(v, "t2a") == 0 || std::strcmp(v, "text_to_audio") == 0) {
                    cfg.task = "text_to_audio";
                } else {
                    throw std::runtime_error(
                        "unsupported --task (MVP: t2a|t2m); got: " + std::string(v));
                }
            } else if (const char *v = need("--instruction-idx")) {
                cfg.instruction_idx = std::atoi(v);
                if (cfg.instruction_idx < 0 || cfg.instruction_idx > 9) {
                    throw std::runtime_error("--instruction-idx must be 0..9");
                }
            } else {
                std::fprintf(stderr, "unknown arg: %s\n", argv[argi]);
                print_usage(argv[0]);
                return 1;
            }
        } catch (const std::exception &e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }

    if (!models_dir.empty()) {
        while (!models_dir.empty() && (models_dir.back() == '/' || models_dir.back() == '\\')) {
            models_dir.pop_back();
        }
        cfg.t5_gguf_path = models_dir + "/t5_encoder.gguf";
        cfg.vae_gguf_path = models_dir + "/vae.gguf";
        cfg.instructions_gguf_path = models_dir + "/instructions.gguf";
        cfg.spiece_model_path = models_dir + "/spiece.model";
        try {
            cfg.dit_gguf_path = resolve_dit_path(models_dir, quant);
        } catch (const std::exception &e) {
            std::fprintf(stderr, "Error: %s\n", e.what());
            if (!model_variant.empty()) {
                std::fprintf(stderr, "  Hint: ./scripts/download_gguf.sh %s %s\n",
                             model_variant.c_str(), quant.empty() ? "Q8_0" : quant.c_str());
            }
            return 1;
        }
        have_models = true;
    }

    if (!have_models) {
        // Default: base + prefer dit-Q8_0 (pack ships T5 Q8_0).
        try {
            apply_model_variant("base");
        } catch (...) {
        }
        while (!models_dir.empty() && (models_dir.back() == '/' || models_dir.back() == '\\')) {
            models_dir.pop_back();
        }
        cfg.t5_gguf_path = models_dir + "/t5_encoder.gguf";
        cfg.vae_gguf_path = models_dir + "/vae.gguf";
        cfg.instructions_gguf_path = models_dir + "/instructions.gguf";
        cfg.spiece_model_path = models_dir + "/spiece.model";
        try {
            cfg.dit_gguf_path = resolve_dit_path(models_dir, quant);
            have_models = true;
        } catch (const std::exception &) {
            have_models = false;
        }
    }

    if (!have_models || cfg.t5_gguf_path.empty()) {
        std::fprintf(stderr, "Error: provide --model / --models-dir / positional model paths\n");
        std::fprintf(stderr, "  Hint: ./scripts/download_gguf.sh   # base Q8_0 default\n\n");
        print_usage(argv[0]);
        return 1;
    }

    for (const auto &p :
         {cfg.t5_gguf_path, cfg.dit_gguf_path, cfg.vae_gguf_path, cfg.instructions_gguf_path,
          cfg.spiece_model_path}) {
        if (!file_readable(p)) {
            std::fprintf(stderr, "Error: cannot read model file: %s\n", p.c_str());
            if (!model_variant.empty()) {
                std::fprintf(stderr, "  Hint: ./scripts/download_gguf.sh %s %s\n",
                             model_variant.c_str(), quant.empty() ? "Q8_0" : quant.c_str());
            }
            return 1;
        }
    }

    std::vector<std::string> prompts;
    try {
        if (!batch_file.empty()) {
            prompts = parse_batch_file(batch_file);
            if (prompts.empty()) {
                std::fprintf(stderr, "Error: batch file is empty\n");
                return 1;
            }
        } else if (!caption.empty()) {
            prompts.push_back(caption);
        } else {
            std::fprintf(stderr, "Error: --caption or --batch is required\n\n");
            print_usage(argv[0]);
            return 1;
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    if (!seed_set) {
        cfg.seed = 0;
    }

    uniflow::BackendPair &bp = uniflow::global_backend_pair();

    std::fprintf(stderr, "uniflow-audio\n");
    std::fprintf(stderr, "  backend:      %s%s\n", ggml_backend_name(bp.backend),
                 bp.has_gpu ? " (GPU)" : "");
    std::fprintf(stderr, "  t5:           %s\n", cfg.t5_gguf_path.c_str());
    std::fprintf(stderr, "  dit:          %s\n", cfg.dit_gguf_path.c_str());
    std::fprintf(stderr, "  vae:          %s\n", cfg.vae_gguf_path.c_str());
    std::fprintf(stderr, "  instructions: %s\n", cfg.instructions_gguf_path.c_str());
    std::fprintf(stderr, "  spiece:       %s\n", cfg.spiece_model_path.c_str());
    std::fprintf(stderr, "  task:         %s (instruction_idx=%d)\n", cfg.task.c_str(),
                 cfg.instruction_idx);
    std::fprintf(stderr, "  prompts:      %zu\n", prompts.size());
    std::fprintf(stderr, "  steps:        %d\n", cfg.num_steps);
    std::fprintf(stderr, "  cfg:          %.1f\n", cfg.guidance_scale);
    std::fprintf(stderr, "  sway:         %.1f\n", cfg.sway_sampling_coef);
    if (cfg.duration_seconds > 0.0f) {
        std::fprintf(stderr, "  duration:     %.2fs\n", cfg.duration_seconds);
    } else {
        std::fprintf(stderr, "  duration:     auto (model pred)\n");
    }
    std::fprintf(stderr, "  threads:      %d\n", cfg.n_threads);
    if (seed_set) {
        std::fprintf(stderr, "  seed:         %u\n", cfg.seed);
    } else {
        std::fprintf(stderr, "  seed:         random\n");
    }
    if (prompts.size() == 1) {
        std::fprintf(stderr, "  caption:      %s\n", prompts[0].c_str());
        std::fprintf(stderr, "  output:       %s\n", output_path.c_str());
    } else {
        std::fprintf(stderr, "  output-dir:   %s/\n", output_dir.c_str());
    }

    try {
        uniflow::Pipeline pipe(cfg);
        const unsigned base_seed = cfg.seed;

        for (size_t i = 0; i < prompts.size(); ++i) {
            const std::string &p = prompts[i];
            std::fprintf(stderr, "\n[%zu/%zu] Generating: %s\n", i + 1, prompts.size(),
                         p.length() > 60 ? (p.substr(0, 57) + "...").c_str() : p.c_str());

            if (seed_set) {
                pipe.set_seed(base_seed + static_cast<unsigned>(i));
            } else {
                pipe.set_seed(0);
            }

            std::vector<float> wav = pipe.generate(p);

            std::string out;
            if (prompts.size() == 1) {
                out = output_path;
            } else {
                std::ostringstream oss;
                oss << output_dir << "/output_" << std::setfill('0') << std::setw(4) << i << ".wav";
                out = oss.str();
            }

            if (!uniflow::write_wav_f32_mono(out, wav, 24000)) {
                std::fprintf(stderr, "failed to write %s\n", out.c_str());
                return 1;
            }
            std::fprintf(stderr, "Saved: %s (%.2fs @ 24kHz, %zu samples)\n", out.c_str(),
                         static_cast<float>(wav.size()) / 24000.0f, wav.size());
            if (prompts.size() == 1) {
                std::printf("Wrote %s (%zu samples)\n", out.c_str(), wav.size());
            }
        }

        std::fprintf(stderr, "\nDone. Generated %zu audio file(s).\n", prompts.size());
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
