#include "instructions.h"

#include <cstring>
#include <stdexcept>

#include "ggml.h"

namespace uniflow {

InstructionBank::InstructionBank(const std::string &gguf_path) : model_(gguf_path) {}

std::vector<float> InstructionBank::get(const std::string &task_key, int *out_len) const {
    const std::string name = "instr." + task_key;
    const struct ggml_tensor *t = model_.tensor(name);
    // GGUF may store as [1024, seq] (ne[0]=1024, ne[1]=seq) matching row-major [seq,1024] write
    const int d0 = static_cast<int>(t->ne[0]);
    const int d1 = static_cast<int>(t->ne[1]);
    int seq = 0;
    int dim = 0;
    if (d0 == 1024) {
        dim = d0;
        seq = d1;
    } else if (d1 == 1024) {
        dim = d1;
        seq = d0;
    } else {
        throw std::runtime_error("instruction tensor unexpected shape for " + name);
    }
    if (out_len) *out_len = seq;

    std::vector<float> out(static_cast<size_t>(seq) * static_cast<size_t>(dim));
    if (t->type == GGML_TYPE_F32) {
        // Stored contiguous; if ne[0]==1024, memory is seq major of 1024-vectors — same as row-major [seq,1024]
        std::memcpy(out.data(), t->data, out.size() * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t *>(t->data), out.data(),
                              static_cast<int64_t>(out.size()));
    } else {
        throw std::runtime_error("instruction dtype unsupported");
    }
    return out;
}

}  // namespace uniflow
