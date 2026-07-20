#!/usr/bin/env bash
# Download a UniFlow GGUF pack from audiohacking/uniflow-audio-gguf
# Usage: ./scripts/download_gguf.sh [small|base|large] [F16|Q8_0|Q4_0|all] [out_root]
# Defaults: base + Q8_0 → models/uniflow-audio-v1.1-base/
set -euo pipefail

REPO_ID="${HF_REPO_ID:-audiohacking/uniflow-audio-gguf}"
VARIANT="${1:-base}"
QUANT_RAW="${2:-Q8_0}"
OUT_ROOT="${3:-models}"

normalize_quant() {
  case "$(echo "$1" | tr '[:lower:]' '[:upper:]')" in
    F16|FP16) echo F16 ;;
    Q8|Q8_0) echo Q8_0 ;;
    Q4|Q4_0) echo Q4_0 ;;
    ALL) echo all ;;
    *) return 1 ;;
  esac
}

case "${VARIANT}" in
  -h|--help|help)
    cat <<EOF
Usage: $0 [small|base|large] [F16|Q8_0|Q4_0|all] [out_root]

Model size (folder on HF):
  small   Fastest / smallest DiT; pack uses T5 Q4_0
  base    Default. Mid capacity; pack uses T5 Q8_0
  large   Highest capacity; pack uses T5 Q8_0

DiT quant (filename tag; all sizes publish all three):
  F16     Full half-precision DiT (dit-F16.gguf)
  Q8_0    Default. Near-lossless 8-bit DiT (dit-Q8_0.gguf)
  Q4_0    Smaller 4-bit DiT (dit-Q4_0.gguf)
  all     Every dit-*.gguf in the pack

Each pack ships its own t5_encoder.gguf (small=Q4_0, base/large=Q8_0).
No F16 T5 — that path is unsupported.

Downloads into: <out_root>/uniflow-audio-v1.1-<size>/

Examples:
  $0                         # base Q8_0 (default)
  $0 small F16
  $0 large Q4_0
  $0 base all
EOF
    exit 0
    ;;
  small|base|large) ;;
  *)
    echo "error: unknown size '${VARIANT}' (use small|base|large)" >&2
    echo "usage: $0 [small|base|large] [F16|Q8_0|Q4_0|all] [out_root]" >&2
    exit 1
    ;;
esac

QUANT="$(normalize_quant "${QUANT_RAW}" || true)"
if [[ -z "${QUANT}" ]]; then
  echo "error: unknown quant '${QUANT_RAW}' (use F16|Q8_0|Q4_0|all)" >&2
  exit 1
fi

PACK="uniflow-audio-v1.1-${VARIANT}"
DEST="${OUT_ROOT}/${PACK}"

if ! command -v hf >/dev/null 2>&1; then
  echo "error: install Hugging Face CLI: pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

mkdir -p "${DEST}"
echo "Downloading ${REPO_ID}:${PACK} (dit=${QUANT}) → ${DEST}"

INCLUDE_ARGS=(
  --include "${PACK}/t5_encoder.gguf"
  --include "${PACK}/vae.gguf"
  --include "${PACK}/instructions.gguf"
  --include "${PACK}/spiece.model"
)

if [[ "${QUANT}" == "all" ]]; then
  INCLUDE_ARGS+=(--include "${PACK}/dit-F16.gguf")
  INCLUDE_ARGS+=(--include "${PACK}/dit-Q8_0.gguf")
  INCLUDE_ARGS+=(--include "${PACK}/dit-Q4_0.gguf")
else
  INCLUDE_ARGS+=(--include "${PACK}/dit-${QUANT}.gguf")
fi

hf download "${REPO_ID}" "${INCLUDE_ARGS[@]}" --local-dir "${OUT_ROOT}"

if [[ "${QUANT}" != "all" && ! -f "${DEST}/dit-${QUANT}.gguf" ]]; then
  echo "note: dit-${QUANT}.gguf missing; trying legacy dit.gguf" >&2
  hf download "${REPO_ID}" --include "${PACK}/dit.gguf" --local-dir "${OUT_ROOT}"
fi

echo "Done. Generate with:"
if [[ "${QUANT}" == "all" ]]; then
  echo "  ./build-metal/uniflow-audio --model ${VARIANT} --quant Q8_0 --caption \"...\" --duration 5"
else
  echo "  ./build-metal/uniflow-audio --model ${VARIANT} --quant ${QUANT} --caption \"...\" --duration 5"
fi
