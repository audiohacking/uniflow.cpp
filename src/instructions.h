#pragma once

#include <string>
#include <vector>

#include "gguf_util.h"

namespace uniflow {

// Loads instructions.gguf (ARCH uniflow_instructions).
class InstructionBank {
public:
    explicit InstructionBank(const std::string &gguf_path);

    // Returns [seq, 1024] row-major float32 for key like "text_to_audio_0"
    // (tensor name in GGUF: instr.text_to_audio_0).
    std::vector<float> get(const std::string &task_key, int *out_len) const;

private:
    GgufModel model_;
};

}  // namespace uniflow
