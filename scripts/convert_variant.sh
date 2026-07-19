#!/usr/bin/env bash
# Convert one UniFlow HF checkpoint dir → staged GGUF pack under dist/hf/
# Reuses shared T5/VAE/instructions/spiece when present in models/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VARIANT="${1:?usage: $0 <small|base|large|xlarge>}"
SRC="${ROOT}/models/uniflow-${VARIANT}"
PACK="uniflow-audio-v1.1-${VARIANT}"
DEST="${ROOT}/dist/hf/${PACK}"

case "${VARIANT}" in
  small) HF_ID="wsntxxn/UniFlow-Audio-v1.1-Small" ;;
  base) HF_ID="wsntxxn/UniFlow-Audio-v1.1-Base" ;;
  large) HF_ID="wsntxxn/UniFlow-Audio-v1.1-Large" ;;
  xlarge) HF_ID="wsntxxn/UniFlow-Audio-v1.1-XLarge" ;;
  *) echo "usage: $0 <small|base|large|xlarge>" >&2; exit 1 ;;
esac

if [[ ! -f "${SRC}/model.safetensors" ]]; then
  echo "Missing ${SRC}/model.safetensors" >&2
  echo "  hf download ${HF_ID} --local-dir ${SRC}" >&2
  exit 1
fi

mkdir -p "${DEST}"

echo "=== DiT (${VARIANT}, F16) → ${DEST}/dit-F16.gguf ==="
python3 "${ROOT}/convert/convert_dit.py" "${SRC}" -o "${DEST}/dit-F16.gguf" --dtype f16 --variant "${VARIANT}"

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
