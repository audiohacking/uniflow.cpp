#!/usr/bin/env python3
"""Convert UniFlow-Audio DiT + content adapter (+ text proj) to GGUF.

Source: models/uniflow-*/model.safetensors + config.yaml
ARCH: uniflow_dit

T2A/T2M subset only (default):
  - backbone.*
  - content_adapter.*
  - content_encoder.text_encoder.proj.*
  - dummy_nta_embed, dummy_ta_embed

Does NOT include phoneme/MIDI/speaker/video/audio encoders (Phase 4+).
Does NOT invent Dasheng metadata — all KV comes from UniFlow config.yaml.

Instructions live in instructions/t5_embeddings.h5 → convert_instructions.py.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import gguf
import numpy as np
import yaml
from safetensors import safe_open

ARCH = "uniflow_dit"

TENSOR_PREFIXES = (
    "backbone.",
    "content_adapter.",
    "content_encoder.text_encoder.proj.",
)
EXTRA_TENSORS = ("dummy_nta_embed", "dummy_ta_embed")


def load_config(model_dir: Path) -> dict:
    with open(model_dir / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_metadata(writer: gguf.GGUFWriter, cfg: dict, variant: str) -> None:
    m = cfg["model"]
    bb = m["backbone"]
    ae = m["autoencoder"]
    ca = m["content_adapter"]
    prefix = "uniflow."

    writer.add_string(prefix + "model_target", str(m["_target_"]))
    writer.add_string(prefix + "backbone_target", str(bb["_target_"]))
    writer.add_string(prefix + "variant", variant)

    writer.add_int32(prefix + "sample_rate", int(cfg.get("sample_rate", ae["sample_rate"])))
    writer.add_int32(prefix + "latent_dim", int(ae["latent_dim"]))
    writer.add_int32(prefix + "downsampling_ratio", int(ae["downsampling_ratio"]))
    writer.add_int32(prefix + "content_dim", int(m["content_dim"]))
    writer.add_float32(prefix + "frame_resolution", float(m["frame_resolution"]))
    writer.add_float32(prefix + "duration_offset", float(m["duration_offset"]))

    writer.add_int32(prefix + "dit_img_size", int(bb["img_size"]))
    writer.add_int32(prefix + "dit_patch_size", int(bb["patch_size"]))
    writer.add_int32(prefix + "dit_in_chans", int(bb["in_chans"]))
    writer.add_int32(prefix + "dit_out_chans", int(bb["out_chans"]))
    writer.add_int32(prefix + "dit_embed_dim", int(bb["embed_dim"]))
    writer.add_int32(prefix + "dit_depth", int(bb["depth"]))
    writer.add_int32(prefix + "dit_num_heads", int(bb["num_heads"]))
    writer.add_float32(prefix + "dit_mlp_ratio", float(bb["mlp_ratio"]))
    writer.add_int32(prefix + "dit_ada_sola_rank", int(bb["ada_sola_rank"]))
    writer.add_int32(prefix + "dit_ada_sola_alpha", int(bb["ada_sola_alpha"]))
    writer.add_int32(prefix + "dit_context_dim", int(bb["context_dim"]))
    writer.add_int32(prefix + "dit_ta_context_dim", int(bb["ta_context_dim"]))
    writer.add_int32(prefix + "adapter_num_heads", int(ca["num_heads"]))
    writer.add_int32(prefix + "adapter_content_dim", int(ca["content_dim"]))
    writer.add_int32(prefix + "adapter_d_out", int(ca["d_out"]))
    writer.add_int32(prefix + "adapter_prefix_dim", int(ca["prefix_dim"]))

    # U-Net skip layout for Small: depth=20 → 10 in + mid + 10 out
    writer.add_int32(prefix + "dit_n_in_blocks", int(bb["depth"] // 2))
    writer.add_int32(prefix + "dit_n_out_blocks", int(bb["depth"] // 2))
    writer.add_int32(prefix + "dit_n_mid_blocks", 1)

    for key, val in (
        ("dit_qk_norm", bb["qk_norm"]),
        ("dit_norm_layer", bb["norm_layer"]),
        ("dit_act_layer", bb["act_layer"]),
        ("dit_time_fusion", bb["time_fusion"]),
        ("dit_ta_context_fusion", bb["ta_context_fusion"]),
        ("dit_context_fusion", bb["context_fusion"]),
        ("dit_context_pe_method", bb["context_pe_method"]),
        ("dit_pe_method", bb["pe_method"]),
        ("dit_rope_mode", bb["rope_mode"]),
        ("dit_input_type", bb["input_type"]),
        ("dit_attn_mode", bb["attn_mode"]),
    ):
        writer.add_string(prefix + key, str(val))

    for key, val in (
        ("dit_context_norm", bb["context_norm"]),
        ("dit_ta_context_norm", bb["ta_context_norm"]),
        ("dit_use_conv", bb["use_conv"]),
        ("dit_skip", bb["skip"]),
        ("dit_skip_norm", bb["skip_norm"]),
        ("dit_qkv_bias", bb["qkv_bias"]),
    ):
        writer.add_bool(prefix + key, bool(val))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "src",
        help="local UniFlow model dir containing config.yaml + model.safetensors",
    )
    ap.add_argument("-o", "--out", default="dit.gguf")
    ap.add_argument("--dtype", default="f16", choices=["f32", "f16", "q8_0", "q4_0"])
    ap.add_argument("--variant", default="small", help="metadata label (small/base/large/xlarge)")
    args = ap.parse_args()

    model_dir = Path(args.src)
    if not model_dir.is_dir():
        raise SystemExit(f"expected model directory, got {args.src}")
    weights_path = model_dir / "model.safetensors"
    if not weights_path.is_file():
        raise SystemExit(f"missing {weights_path}")

    cfg = load_config(model_dir)
    writer = gguf.GGUFWriter(args.out, ARCH)
    writer.add_name("uniflow-dit")
    add_metadata(writer, cfg, args.variant)

    use_quant = args.dtype in ("q8_0", "q4_0")
    if use_quant:
        quant_type = (
            gguf.GGMLQuantizationType.Q8_0
            if args.dtype == "q8_0"
            else gguf.GGMLQuantizationType.Q4_0
        )
        np_dtype = np.float32
    else:
        np_dtype = np.float16 if args.dtype == "f16" else np.float32
        quant_type = None

    n_written = 0
    n_quantized = 0
    n_skipped = 0
    with safe_open(str(weights_path), framework="numpy") as f:
        for name in f.keys():
            if name == "dummy_param":
                n_skipped += 1
                continue
            if not (name.startswith(TENSOR_PREFIXES) or name in EXTRA_TENSORS):
                n_skipped += 1
                continue
            tensor = f.get_tensor(name)

            block_size = 32
            row_compatible = (
                tensor.shape[-1] % block_size == 0 if tensor.ndim >= 1 else False
            )
            should_quantize = (
                use_quant
                and name.startswith("backbone.")
                and tensor.ndim >= 2
                and tensor.size >= 256
                and row_compatible
                and not any(s in name.lower() for s in ("bias", "norm", "ln_", "inv_freq"))
            )

            if should_quantize:
                tensor = gguf.quants.quantize(tensor.astype(np.float32), quant_type)
                writer.add_tensor(name, tensor, raw_dtype=quant_type)
                n_quantized += 1
            else:
                if tensor.dtype == np.float32 and np_dtype != np.float32 and tensor.ndim >= 2:
                    tensor = tensor.astype(np_dtype)
                writer.add_tensor(name, tensor)
            n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(
        f"wrote {n_written} tensors ({n_quantized} quantized) to {args.out}; "
        f"skipped {n_skipped} non-T2A / dummy tensors"
    )


if __name__ == "__main__":
    main()
