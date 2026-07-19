#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace uniflow {

// Write mono PCM float [-1, 1] to a 16-bit PCM WAV file.
// UniFlow outputs 24 kHz; sample_rate is parameterized for future tasks.
bool write_wav_f32_mono(const std::string &path,
                        const std::vector<float> &samples,
                        int sample_rate);

// Read a mono (or first-channel) 16-bit PCM WAV into float [-1, 1].
// Returns empty vector on failure; sample_rate out-param set when non-null.
std::vector<float> read_wav_f32_mono(const std::string &path, int *sample_rate);

}  // namespace uniflow
