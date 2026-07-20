#!/usr/bin/env python3
"""Convert google/flan-t5-large's encoder stack to a standalone GGUF file for
the uniflow.cpp engine's vendored T5 encoder-only graph (src/t5_encoder.cpp).

Only the encoder is needed (UniFlow content encoding never runs the T5
decoder), so decoder.*, lm_head.* and encoder-decoder-only bits beyond the
token embedding table are skipped.

Tensor naming is our own minimal scheme (not llama.cpp's GGUF vocab layout)
since src/t5_encoder.cpp is a small hand-written graph, not a llama.cpp
architecture-registry consumer:

  token_embd.weight              [vocab, d_model]
  enc.blk.{i}.attn_norm.weight   [d_model]
  enc.blk.{i}.attn_q.weight      [d_model, d_model]
  enc.blk.{i}.attn_k.weight      [d_model, d_model]
  enc.blk.{i}.attn_v.weight      [d_model, d_model]
  enc.blk.{i}.attn_o.weight      [d_model, d_model]
  enc.blk.0.attn_rel_b.weight    [num_buckets, n_head]   (block 0 only; the
                                  relative-position bias table is shared
                                  across all layers in T5)
  enc.blk.{i}.ffn_norm.weight    [d_model]
  enc.blk.{i}.ffn_gate.weight    [d_ff, d_model]   (wi_0, gated-gelu gate)
  enc.blk.{i}.ffn_up.weight      [d_ff, d_model]   (wi_1)
  enc.blk.{i}.ffn_down.weight    [d_model, d_ff]   (wo)
  enc.output_norm.weight         [d_model]
"""
import argparse
import json
import os

import gguf
import numpy as np
from safetensors import safe_open

ARCH = "uniflow_t5enc"

META_INT_FIELDS = {
    "n_layer": "num_layers",
    "n_head": "num_heads",
    "d_model": "d_model",
    "d_kv": "d_kv",
    "d_ff": "d_ff",
    "vocab_size": "vocab_size",
    "relative_attention_num_buckets": "relative_attention_num_buckets",
    "relative_attention_max_distance": "relative_attention_max_distance",
}


def resolve_paths(src):
    if os.path.isdir(src):
        return (
            os.path.join(src, "config.json"),
            os.path.join(src, "model.safetensors"),
        )
    from huggingface_hub import hf_hub_download
    config_path = hf_hub_download(src, "config.json")
    weights_path = hf_hub_download(src, "model.safetensors")
    return config_path, weights_path


def cast(tensor, np_dtype, force_f32=False):
    if force_f32:
        return tensor.astype(np.float32) if tensor.dtype != np.float32 else tensor
    if tensor.dtype == np.float32 and np_dtype != np.float32 and tensor.ndim >= 2:
        return tensor.astype(np_dtype)
    return tensor


def should_quantize(name: str, tensor: np.ndarray) -> bool:
    """Quantize 2D matmul / embedding weights; keep norms and relative bias in float."""
    if tensor.ndim < 2 or tensor.size < 256:
        return False
    if not (tensor.shape[-1] % 32 == 0):
        return False
    lower = name.lower()
    if any(s in lower for s in ("norm", "attn_rel_b", "bias")):
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="local checkpoint dir or HF repo id (google/flan-t5-large)")
    ap.add_argument("-o", "--out", default="t5-encoder.gguf")
    ap.add_argument("--dtype", default="f16", choices=["f32", "f16", "q8_0", "q4_0"])
    args = ap.parse_args()

    config_path, weights_path = resolve_paths(args.src)
    with open(config_path) as f:
        config = json.load(f)

    n_layer = config["num_layers"]

    writer = gguf.GGUFWriter(args.out, ARCH)
    writer.add_name("uniflow-audiogen-t5-encoder")
    writer.add_string("uniflow.t5_dtype", args.dtype)
    for gguf_key, cfg_key in META_INT_FIELDS.items():
        writer.add_int32(f"t5enc.{gguf_key}", int(config[cfg_key]))
    writer.add_float32("t5enc.layer_norm_eps", float(config["layer_norm_epsilon"]))

    use_quant = args.dtype in ("q8_0", "q4_0")
    if use_quant:
        quant_type = (
            gguf.GGMLQuantizationType.Q8_0
            if args.dtype == "q8_0"
            else gguf.GGMLQuantizationType.Q4_0
        )
        np_dtype = np.float32
    else:
        quant_type = None
        np_dtype = np.float16 if args.dtype == "f16" else np.float32

    n_written = 0
    n_quantized = 0
    with safe_open(weights_path, framework="numpy") as f:
        tensors = [("shared.weight", "token_embd.weight", False)]

        for i in range(n_layer):
            prefix = f"encoder.block.{i}"
            tensors.extend([
                (f"{prefix}.layer.0.layer_norm.weight", f"enc.blk.{i}.attn_norm.weight", True),
                (f"{prefix}.layer.0.SelfAttention.q.weight", f"enc.blk.{i}.attn_q.weight", False),
                (f"{prefix}.layer.0.SelfAttention.k.weight", f"enc.blk.{i}.attn_k.weight", False),
                (f"{prefix}.layer.0.SelfAttention.v.weight", f"enc.blk.{i}.attn_v.weight", False),
                (f"{prefix}.layer.0.SelfAttention.o.weight", f"enc.blk.{i}.attn_o.weight", False),
                (f"{prefix}.layer.1.layer_norm.weight", f"enc.blk.{i}.ffn_norm.weight", True),
                (f"{prefix}.layer.1.DenseReluDense.wi_0.weight", f"enc.blk.{i}.ffn_gate.weight", False),
                (f"{prefix}.layer.1.DenseReluDense.wi_1.weight", f"enc.blk.{i}.ffn_up.weight", False),
                (f"{prefix}.layer.1.DenseReluDense.wo.weight", f"enc.blk.{i}.ffn_down.weight", False),
            ])
            if i == 0:
                tensors.append((
                    f"{prefix}.layer.0.SelfAttention.relative_attention_bias.weight",
                    "enc.blk.0.attn_rel_b.weight",
                    True,
                ))

        tensors.append(("encoder.final_layer_norm.weight", "enc.output_norm.weight", True))

        for src_name, dst_name, force_f32 in tensors:
            tensor = f.get_tensor(src_name)
            if use_quant and not force_f32 and should_quantize(dst_name, tensor):
                q = gguf.quants.quantize(tensor.astype(np.float32), quant_type)
                writer.add_tensor(dst_name, q, raw_dtype=quant_type)
                n_quantized += 1
            else:
                writer.add_tensor(dst_name, cast(tensor, np_dtype, force_f32=force_f32))
            n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {n_written} tensors ({n_quantized} quantized) to {args.out}")


if __name__ == "__main__":
    main()
