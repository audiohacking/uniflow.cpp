#include <cmath>
#include <cstdio>
#include <vector>

#include "scheduler.h"

// Regression: UniFlow flow-match schedule (25 steps, sway=-1) must be
// descending, length==steps, and match audiogen.cpp / UniFlow semantics
// (first sigma near 1, last near 0).
int main() {
    uniflow::FlowMatchScheduler sched(25, -1.0f);
    auto s = sched.sigmas();
    auto t = sched.timesteps();

    if (s.size() != 26 || t.size() != 25) {
        std::fprintf(stderr, "FAIL: size sigmas=%zu timesteps=%zu\n", s.size(), t.size());
        return 1;
    }
    if (!(s.front() > 0.9f && s.front() <= 1.0f + 1e-5f)) {
        std::fprintf(stderr, "FAIL: sigma0=%f\n", s.front());
        return 1;
    }
    if (!(s.back() >= 0.0f && s.back() < 0.2f)) {
        std::fprintf(stderr, "FAIL: sigmaN=%f\n", s.back());
        return 1;
    }
    for (size_t i = 1; i < s.size(); ++i) {
        if (!(s[i] <= s[i - 1] + 1e-6f)) {
            std::fprintf(stderr, "FAIL: not descending at %zu\n", i);
            return 1;
        }
    }
    // timesteps = sigmas * 1000
    if (std::fabs(t[0] - s[0] * 1000.0f) > 1e-3f) {
        std::fprintf(stderr, "FAIL: timestep scale\n");
        return 1;
    }
    std::printf("PASS test_scheduler (sigma0=%.6f sigmaN=%.6f)\n", s.front(), s.back());
    return 0;
}
