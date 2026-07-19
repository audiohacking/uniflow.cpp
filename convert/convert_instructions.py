#!/usr/bin/env python3
"""Pack UniFlow instruction T5 embeddings (H5) into a GGUF file.

Source: instructions/t5_embeddings.h5
ARCH: uniflow_instructions

Each dataset key `{task}_{idx}` becomes tensor `instr.{task}_{idx}` with shape
[seq, 1024]. Also stores per-key sequence lengths as metadata strings list.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gguf
import h5py
import numpy as np

ARCH = "uniflow_instructions"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="path to t5_embeddings.h5 or model dir")
    ap.add_argument("-o", "--out", default="instructions.gguf")
    args = ap.parse_args()

    src = Path(args.src)
    if src.is_dir():
        h5_path = src / "instructions" / "t5_embeddings.h5"
    else:
        h5_path = src
    if not h5_path.is_file():
        raise SystemExit(f"missing {h5_path}")

    writer = gguf.GGUFWriter(args.out, ARCH)
    writer.add_name("uniflow-instructions")
    writer.add_int32("uniflow.instruction_dim", 1024)

    keys = []
    lengths = []
    with h5py.File(h5_path, "r") as hf:
        for key in sorted(hf.keys()):
            arr = np.asarray(hf[key][()], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 1024:
                raise RuntimeError(f"unexpected shape for {key}: {arr.shape}")
            writer.add_tensor(f"instr.{key}", arr)
            keys.append(key)
            lengths.append(arr.shape[0])

    writer.add_string("uniflow.instruction_keys", ",".join(keys))
    writer.add_string(
        "uniflow.instruction_lengths", ",".join(str(n) for n in lengths)
    )
    writer.add_int32("uniflow.n_instructions", len(keys))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {len(keys)} instruction tensors to {args.out}")


if __name__ == "__main__":
    main()
