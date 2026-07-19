#!/usr/bin/env bash
# Download a UniFlow GGUF pack from audiohacking/uniflow-audio-gguf
set -euo pipefail

REPO_ID="${HF_REPO_ID:-audiohacking/uniflow-audio-gguf}"
VARIANT="${1:-small}"
OUT_ROOT="${2:-models}"

case "${VARIANT}" in
  small|base|large|xlarge) ;;
  *)
    echo "usage: $0 <small|base|large|xlarge> [out_root]" >&2
    exit 1
    ;;
esac

PACK="uniflow-audio-v1.1-${VARIANT}"
DEST="${OUT_ROOT}/${PACK}"

if ! command -v hf >/dev/null 2>&1; then
  echo "error: install Hugging Face CLI (pip install -U 'huggingface_hub[cli]')" >&2
  exit 1
fi

mkdir -p "${DEST}"
echo "Downloading ${REPO_ID}:${PACK} → ${DEST}"
hf download "${REPO_ID}" --include "${PACK}/*" --local-dir "${OUT_ROOT}"

echo "Done. Run with:"
echo "  ./build-metal/uniflow-audio --models-dir ${DEST} --caption \"...\" --duration 5"
