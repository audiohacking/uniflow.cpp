#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "wav_io.h"

// Regression: round-trip 24 kHz mono WAV write/read (UniFlow sample rate).
int main() {
    const char *path = "output/test_wav_io_roundtrip.wav";
    std::system("mkdir -p output");

    std::vector<float> src(480);  // 20 ms @ 24 kHz
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * float(i) / 24000.0f);
    }

    if (!uniflow::write_wav_f32_mono(path, src, 24000)) {
        std::fprintf(stderr, "FAIL: write_wav\n");
        return 1;
    }
    int sr = 0;
    auto dst = uniflow::read_wav_f32_mono(path, &sr);
    if (sr != 24000 || dst.size() != src.size()) {
        std::fprintf(stderr, "FAIL: read_wav sr=%d n=%zu\n", sr, dst.size());
        return 1;
    }
    float max_err = 0.0f;
    for (size_t i = 0; i < src.size(); ++i) {
        max_err = std::max(max_err, std::fabs(src[i] - dst[i]));
    }
    // 16-bit PCM quantization noise
    if (max_err > 2.0f / 32768.0f + 1e-4f) {
        std::fprintf(stderr, "FAIL: max_err=%g\n", max_err);
        return 1;
    }
    std::printf("PASS test_wav_io (max_err=%g)\n", max_err);
    return 0;
}
