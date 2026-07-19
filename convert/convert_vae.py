#!/usr/bin/env python3
"""Convert UniFlow StableVAE (Oobleck) checkpoint to GGUF.

Source: vae/speech_audio_sound_step=1000000.ckpt (PyTorch Lightning)
ARCH: uniflow_vae

Policy:
  - Keep encoder + decoder (T2A needs decode; SE/SR need encode).
  - Drop discriminator / loss modules.
  - Fuse torch.nn.utils.weight_norm pairs (weight_g, weight_v) → weight
    using w = g * v / ||v|| (norm over all dims except 0). Documented in
    DEVELOPMENT.md — C++ must NOT re-apply weight_norm.
  - SnakeBeta alpha/beta stored as-is (log-scale; exp at runtime).

Tensor names in GGUF: encoder.* / decoder.* (autoencoder. prefix stripped).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import gguf
import numpy as np
import torch


ARCH = "uniflow_vae"


def fuse_weight_norm(state: dict) -> dict:
    """Replace (*.weight_g, *.weight_v) with fused *.weight."""
    out = {}
    pairs = defaultdict(dict)
    for name, tensor in state.items():
        if name.endswith(".weight_g"):
            pairs[name[: -len(".weight_g")]]["g"] = tensor
        elif name.endswith(".weight_v"):
            pairs[name[: -len(".weight_v")]]["v"] = tensor
        else:
            out[name] = tensor

    for base, parts in pairs.items():
        if "g" not in parts or "v" not in parts:
            raise RuntimeError(f"incomplete weight_norm pair for {base}: {parts.keys()}")
        g = parts["g"]
        v = parts["v"]
        # PyTorch weight_norm: norm over all dims except dim 0
        dims = list(range(1, v.ndim))
        v_norm = torch.norm(v, p=2, dim=dims, keepdim=True)
        w = g * (v / (v_norm + 1e-12))
        out[base + ".weight"] = w
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="path to speech_audio_sound_step=*.ckpt or model dir")
    ap.add_argument("-o", "--out", default="vae.gguf")
    ap.add_argument("--dtype", default="f16", choices=["f32", "f16"])
    ap.add_argument("--decoder-only", action="store_true", help="omit encoder tensors")
    args = ap.parse_args()

    src = Path(args.src)
    if src.is_dir():
        ckpt_path = src / "vae" / "speech_audio_sound_step=1000000.ckpt"
    else:
        ckpt_path = src
    if not ckpt_path.is_file():
        raise SystemExit(f"missing VAE checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    model_cfg = ckpt.get("model_config", {})

    # Keep only autoencoder.*
    ae_state = {
        k[len("autoencoder.") :]: v
        for k, v in state.items()
        if k.startswith("autoencoder.")
    }
    fused = fuse_weight_norm(ae_state)

    writer = gguf.GGUFWriter(args.out, ARCH)
    writer.add_name("uniflow-vae")
    writer.add_string("uniflow.vae_target", "models.autoencoder.waveform.stable_vae.StableVAE")

    # Metadata from Lightning model_config when present
    model = model_cfg.get("model", {})
    writer.add_int32("uniflow.sample_rate", int(model_cfg.get("sample_rate", 24000)))
    writer.add_int32("uniflow.latent_dim", int(model.get("latent_dim", 128)))
    writer.add_int32("uniflow.downsampling_ratio", int(model.get("downsampling_ratio", 480)))
    writer.add_int32("uniflow.io_channels", int(model.get("io_channels", 1)))
    writer.add_bool("uniflow.vae_weight_norm_fused", True)
    writer.add_bool("uniflow.vae_snake_logscale", True)

    enc_cfg = model.get("encoder", {}).get("config", {})
    dec_cfg = model.get("decoder", {}).get("config", {})
    if enc_cfg:
        writer.add_int32("uniflow.vae_channels", int(enc_cfg.get("channels", 128)))
        strides = enc_cfg.get("strides", [2, 4, 6, 10])
        writer.add_string("uniflow.vae_strides", ",".join(str(s) for s in strides))
        c_mults = enc_cfg.get("c_mults", [1, 2, 4, 8])
        writer.add_string("uniflow.vae_c_mults", ",".join(str(s) for s in c_mults))
        writer.add_bool("uniflow.vae_use_snake", bool(enc_cfg.get("use_snake", True)))
    if dec_cfg:
        writer.add_bool("uniflow.vae_final_tanh", bool(dec_cfg.get("final_tanh", False)))
        writer.add_int32(
            "uniflow.vae_encoder_latent_dim",
            int(enc_cfg.get("latent_dim", 256)),
        )  # encoder emits 2*latent before bottleneck

    np_dtype = np.float16 if args.dtype == "f16" else np.float32
    n_written = 0
    for name, tensor in sorted(fused.items()):
        if args.decoder_only and name.startswith("encoder."):
            continue
        if not (name.startswith("encoder.") or name.startswith("decoder.")):
            continue
        arr = tensor.detach().cpu().numpy()
        if arr.dtype == np.float32 and np_dtype != np.float32 and arr.ndim >= 2:
            arr = arr.astype(np_dtype)
        writer.add_tensor(name, arr)
        n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {n_written} fused VAE tensors to {args.out}")


if __name__ == "__main__":
    main()
