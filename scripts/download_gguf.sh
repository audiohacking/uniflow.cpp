#!/usr/bin/env bash
# Download a UniFlow GGUF pack from audiohacking/uniflow-audio-gguf
# Usage: ./scripts/download_gguf.sh [small|base|large] [out_root]
# Default: small → models/uniflow-audio-v1.1-small/
set -euo pipefail

REPO_ID="${HF_REPO_ID:-audiohacking/uniflow-audio-gguf}"
VARIANT="${1:-small}"
OUT_ROOT="${2:-models}"

case "${VARIANT}" in
  -h|--help|help)
    cat <<EOF
Usage: $0 [small|base|large] [out_root]

  small   Default. Fastest download and inference.
  base    Mid size.
  large   Highest capacity.

Downloads into <out_root>/uniflow-audio-v1.1-<size>/ (default out_root: models).

Examples:
  $0
  $0 base
  $0 large models
EOF
    exit 0
    ;;
  small|base|large) ;;
  *)
    echo "error: unknown size '${VARIANT}' (use small|base|large)" >&2
    echo "usage: $0 [small|base|large] [out_root]" >&2
    exit 1
    ;;
esac

PACK="uniflow-audio-v1.1-${VARIANT}"
DEST="${OUT_ROOT}/${PACK}"

if ! command -v hf >/dev/null 2>&1; then
  echo "error: install Hugging Face CLI: pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

mkdir -p "${DEST}"
echo "Downloading ${REPO_ID}:${PACK} → ${DEST}"
hf download "${REPO_ID}" --include "${PACK}/*" --local-dir "${OUT_ROOT}"

echo "Done. Generate with:"
echo "  ./build-metal/uniflow-audio --model ${VARIANT} --caption \"...\" --duration 5"
