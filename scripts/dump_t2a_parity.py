#!/usr/bin/env python3
"""Dump UniFlow T2A intermediate tensors / WAV for C++ parity (offline only).

Requires a working UniFlow Python env + models/uniflow-small + FLAN_T5_PATH.
Produces output/parity/ for comparison — not used at C++ runtime.

Example:
  python3 scripts/dump_t2a_parity.py \\
    --model-dir models/uniflow-small \\
    --caption "a man is speaking while a dog barks" \\
    --seed 42 --steps 25 --cfg 5.0 \\
    --out-dir output/parity
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="models/uniflow-small")
    ap.add_argument("--caption", default="a man is speaking while a dog barks")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--out-dir", default="output/parity")
    ap.add_argument("--instruction-idx", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    ref = root / "reference" / "python"
    sys.path.insert(0, str(ref))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import soundfile as sf

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from modeling_uniflow_audio import UniFlowAudioModel
    from constants import TIME_ALIGNED_TASKS, NON_TIME_ALIGNED_TASKS

    model = UniFlowAudioModel(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    task = "text_to_audio"
    time_aligned = torch.zeros(1, 2).bool()
    if task in TIME_ALIGNED_TASKS:
        time_aligned[0, 0] = True
    if task in NON_TIME_ALIGNED_TASKS:
        time_aligned[0, 1] = True

    with torch.inference_mode():
        waveform = model.sample(
            content=[args.caption],
            task=[task],
            time_aligned=time_aligned,
            instruction_idx=[args.instruction_idx],
            num_steps=args.steps,
            guidance_scale=args.cfg,
            sway_sampling_coef=-1.0,
            disable_progress=False,
        )
    wav = waveform[0, 0].detach().cpu().numpy().astype(np.float32)
    sr = int(model.config["sample_rate"])
    wav_path = out / "python_t2a.wav"
    sf.write(str(wav_path), wav, sr)

    meta = {
        "caption": args.caption,
        "seed": args.seed,
        "steps": args.steps,
        "cfg": args.cfg,
        "instruction_idx": args.instruction_idx,
        "sample_rate": sr,
        "n_samples": int(wav.shape[0]),
        "duration_sec": float(wav.shape[0]) / sr,
        "wav_rms": float(np.sqrt(np.mean(wav**2))),
        "wav_max_abs": float(np.max(np.abs(wav))),
        "wav_path": str(wav_path),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {wav_path} — HUMAN LISTEN GATE G1")


if __name__ == "__main__":
    main()
