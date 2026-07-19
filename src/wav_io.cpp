#include "wav_io.h"

#include <cstdio>
#include <cstring>
#include <algorithm>

namespace uniflow {

namespace {

#pragma pack(push, 1)
struct WavHeader {
    char riff[4];
    uint32_t chunk_size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];
    uint32_t data_size;
};
#pragma pack(pop)

}  // namespace

bool write_wav_f32_mono(const std::string &path,
                        const std::vector<float> &samples,
                        int sample_rate) {
    if (samples.empty() || sample_rate <= 0) {
        return false;
    }

    const uint16_t bits = 16;
    const uint16_t channels = 1;
    const uint32_t data_size = static_cast<uint32_t>(samples.size() * sizeof(int16_t));

    WavHeader h{};
    std::memcpy(h.riff, "RIFF", 4);
    h.chunk_size = 36 + data_size;
    std::memcpy(h.wave, "WAVE", 4);
    std::memcpy(h.fmt, "fmt ", 4);
    h.fmt_size = 16;
    h.audio_format = 1;  // PCM
    h.num_channels = channels;
    h.sample_rate = static_cast<uint32_t>(sample_rate);
    h.byte_rate = static_cast<uint32_t>(sample_rate * channels * bits / 8);
    h.block_align = static_cast<uint16_t>(channels * bits / 8);
    h.bits_per_sample = bits;
    std::memcpy(h.data, "data", 4);
    h.data_size = data_size;

    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    if (std::fwrite(&h, sizeof(h), 1, f) != 1) {
        std::fclose(f);
        return false;
    }
    for (float s : samples) {
        float clamped = std::max(-1.0f, std::min(1.0f, s));
        int16_t pcm = static_cast<int16_t>(clamped * 32767.0f);
        if (std::fwrite(&pcm, sizeof(pcm), 1, f) != 1) {
            std::fclose(f);
            return false;
        }
    }
    std::fclose(f);
    return true;
}

std::vector<float> read_wav_f32_mono(const std::string &path, int *sample_rate) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return {};
    }

    WavHeader h{};
    if (std::fread(&h, sizeof(h), 1, f) != 1) {
        std::fclose(f);
        return {};
    }
    if (std::memcmp(h.riff, "RIFF", 4) != 0 || std::memcmp(h.wave, "WAVE", 4) != 0) {
        std::fclose(f);
        return {};
    }
    // Minimal reader: expects canonical header layout (fmt then data).
    if (h.audio_format != 1 || h.bits_per_sample != 16) {
        std::fclose(f);
        return {};
    }
    if (sample_rate) {
        *sample_rate = static_cast<int>(h.sample_rate);
    }

    const size_t n_frames = h.data_size / (h.num_channels * sizeof(int16_t));
    std::vector<int16_t> pcm(n_frames * h.num_channels);
    if (std::fread(pcm.data(), sizeof(int16_t), pcm.size(), f) != pcm.size()) {
        std::fclose(f);
        return {};
    }
    std::fclose(f);

    std::vector<float> out(n_frames);
    for (size_t i = 0; i < n_frames; ++i) {
        out[i] = static_cast<float>(pcm[i * h.num_channels]) / 32768.0f;
    }
    return out;
}

}  // namespace uniflow
