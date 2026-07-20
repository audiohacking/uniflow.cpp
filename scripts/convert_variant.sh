#!/usr/bin/env bash
# Convert UniFlow HF checkpoint → staged GGUF pack under dist/hf/
# Usage: ./scripts/convert_variant.sh <small|base|large|xlarge> [F16|Q8_0|Q4_0|all]
# Default quant: F16. "all" writes dit-F16.gguf + dit-Q8_0.gguf + dit-Q4_0.gguf.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VARIANT="${1:?usage: $0 <small|base|large|xlarge> [F16|Q8_0|Q4_0|all]}"
QUANT_RAW="${2:-F16}"
SRC="${ROOT}/models/uniflow-${VARIANT}"
PACK="uniflow-audio-v1.1-${VARIANT}"
DEST="${ROOT}/dist/hf/${PACK}"

case "${VARIANT}" in
  small) HF_ID="wsntxxn/UniFlow-Audio-v1.1-Small" ;;
  base) HF_ID="wsntxxn/UniFlow-Audio-v1.1-Base" ;;
  large) HF_ID="wsntxxn/UniFlow-Audio-v1.1-Large" ;;
  xlarge) HF_ID="wsntxxn/UniFlow-Audio-v1.1-XLarge" ;;
  *) echo "usage: $0 <small|base|large|xlarge> [F16|Q8_0|Q4_0|all]" >&2; exit 1 ;;
esac

normalize_quant() {
  case "$(echo "$1" | tr '[:lower:]' '[:upper:]')" in
    F16|FP16) echo F16 ;;
    Q8|Q8_0) echo Q8_0 ;;
    Q4|Q4_0) echo Q4_0 ;;
    ALL) echo all ;;
    *) return 1 ;;
  esac
}

QUANT="$(normalize_quant "${QUANT_RAW}" || true)"
if [[ -z "${QUANT}" ]]; then
  echo "error: unknown quant '${QUANT_RAW}' (use F16|Q8_0|Q4_0|all)" >&2
  exit 1
fi

if [[ ! -f "${SRC}/model.safetensors" ]]; then
  echo "Missing ${SRC}/model.safetensors" >&2
  echo "  hf download ${HF_ID} --local-dir ${SRC}" >&2
  exit 1
fi

mkdir -p "${DEST}"

dtype_for_tag() {
  case "$1" in
    F16) echo f16 ;;
    Q8_0) echo q8_0 ;;
    Q4_0) echo q4_0 ;;
  esac
}

convert_dit() {
  local tag="$1"
  local dtype
  dtype="$(dtype_for_tag "${tag}")"
  echo "=== DiT (${VARIANT}, ${tag}) → ${DEST}/dit-${tag}.gguf ==="
  python3 "${ROOT}/convert/convert_dit.py" "${SRC}" \
    -o "${DEST}/dit-${tag}.gguf" --dtype "${dtype}" --variant "${VARIANT}"
}

if [[ "${QUANT}" == "all" ]]; then
  for tag in F16 Q8_0 Q4_0; do
    convert_dit "${tag}"
  done
else
  convert_dit "${QUANT}"
fi

if [[ ! -f "${DEST}/vae.gguf" ]]; then
  if [[ -f "${ROOT}/models/vae.gguf" ]]; then
    echo "=== VAE: copy shared models/vae.gguf ==="
    cp "${ROOT}/models/vae.gguf" "${DEST}/vae.gguf"
  else
    echo "=== VAE (F16, fused) ==="
    python3 "${ROOT}/convert/convert_vae.py" "${SRC}" -o "${DEST}/vae.gguf" --dtype f16
  fi
fi

if [[ ! -f "${DEST}/instructions.gguf" ]]; then
  if [[ -f "${ROOT}/models/instructions.gguf" ]]; then
    echo "=== Instructions: copy shared ==="
    cp "${ROOT}/models/instructions.gguf" "${DEST}/instructions.gguf"
  else
    echo "=== Instructions ==="
    python3 "${ROOT}/convert/convert_instructions.py" "${SRC}" -o "${DEST}/instructions.gguf"
  fi
fi

if [[ ! -f "${DEST}/t5_encoder.gguf" ]]; then
  if [[ -f "${ROOT}/models/t5_encoder.gguf" ]]; then
    echo "=== T5: copy shared ==="
    cp "${ROOT}/models/t5_encoder.gguf" "${DEST}/t5_encoder.gguf"
  else
    echo "=== T5 encoder ==="
    python3 "${ROOT}/convert/convert_t5_encoder.py" google/flan-t5-large -o "${DEST}/t5_encoder.gguf" --dtype f32
  fi
fi

if [[ ! -f "${DEST}/spiece.model" ]]; then
  if [[ -f "${ROOT}/models/spiece.model" ]]; then
    cp "${ROOT}/models/spiece.model" "${DEST}/spiece.model"
  else
    echo "NOTE: place spiece.model in ${DEST}/ (from Flan-T5)" >&2
  fi
fi

ls -lh "${DEST}"
echo "Pack ready: ${DEST}"
echo "Upload: ${ROOT}/scripts/upload_gguf_hf.sh ${PACK}"
