#!/usr/bin/env bash
# Download a UniFlow GGUF pack from audiohacking/uniflow-audio-gguf
# Usage: ./scripts/download_gguf.sh [small|base|large] [F16|Q8_0|Q4_0|all] [out_root]
# Defaults: small + F16 → models/uniflow-audio-v1.1-small/
set -euo pipefail

REPO_ID="${HF_REPO_ID:-audiohacking/uniflow-audio-gguf}"
VARIANT="${1:-small}"
QUANT_RAW="${2:-F16}"
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

  small   Default size. Fastest download and inference.
  base    Mid size.
  large   Highest capacity (F16 / Q8_0 / Q4_0 DiT files on HF).

  F16     Default DiT precision (dit-F16.gguf).
  Q8_0    Near-lossless 8-bit DiT (large).
  Q4_0    Smaller 4-bit DiT (large).
  all     Download every dit-*.gguf in the pack.

Downloads shared T5/VAE/instructions/tokenizer plus the selected DiT into
  <out_root>/uniflow-audio-v1.1-<size>/

Examples:
  $0
  $0 large Q8_0
  $0 large all
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
echo "Downloading ${REPO_ID}:${PACK} (quant=${QUANT}) → ${DEST}"

SHARED=(
  "${PACK}/t5_encoder.gguf"
  "${PACK}/vae.gguf"
  "${PACK}/instructions.gguf"
  "${PACK}/spiece.model"
)

# Always fetch shared files.
for f in "${SHARED[@]}"; do
  hf download "${REPO_ID}" --include "${f}" --local-dir "${OUT_ROOT}"
done

if [[ "${QUANT}" == "all" ]]; then
  hf download "${REPO_ID}" --include "${PACK}/dit-*.gguf" --local-dir "${OUT_ROOT}"
  # legacy fallback if only dit.gguf exists
  hf download "${REPO_ID}" --include "${PACK}/dit.gguf" --local-dir "${OUT_ROOT}" 2>/dev/null || true
else
  if ! hf download "${REPO_ID}" --include "${PACK}/dit-${QUANT}.gguf" --local-dir "${OUT_ROOT}"; then
    echo "note: dit-${QUANT}.gguf missing; trying legacy dit.gguf" >&2
    hf download "${REPO_ID}" --include "${PACK}/dit.gguf" --local-dir "${OUT_ROOT}"
  fi
fi

echo "Done. Generate with:"
if [[ "${QUANT}" == "all" ]]; then
  echo "  ./build-metal/uniflow-audio --model ${VARIANT} --quant Q8_0 --caption \"...\" --duration 5"
else
  echo "  ./build-metal/uniflow-audio --model ${VARIANT} --quant ${QUANT} --caption \"...\" --duration 5"
fi
