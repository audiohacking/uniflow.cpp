#!/usr/bin/env bash
# Upload local GGUF packs to https://huggingface.co/audiohacking/uniflow-audio-gguf
# Weights stay out of git (dist/ and models/ are gitignored).
set -euo pipefail

REPO_ID="${HF_REPO_ID:-audiohacking/uniflow-audio-gguf}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="${ROOT}/dist/hf"
VARIANT="${1:-uniflow-audio-v1.1-small}"

if ! command -v hf >/dev/null 2>&1; then
  echo "error: install Hugging Face CLI: pip install -U huggingface_hub[cli]" >&2
  exit 1
fi

WHOAMI="$(hf auth whoami 2>&1 || true)"
if echo "${WHOAMI}" | grep -qiE 'not logged|error|unauthorized'; then
  echo "error: not logged in to Hugging Face." >&2
  echo "  Run:  hf auth login" >&2
  echo "  Or:   export HF_TOKEN=hf_...   # write access to ${REPO_ID}" >&2
  exit 1
fi
echo "Logged in as: ${WHOAMI}"

PACK="${STAGING}/${VARIANT}"
if [[ ! -d "${PACK}" ]]; then
  echo "error: missing pack dir ${PACK}" >&2
  exit 1
fi

for f in t5_encoder.gguf dit.gguf vae.gguf instructions.gguf spiece.model; do
  if [[ ! -e "${PACK}/${f}" ]]; then
    echo "error: missing ${PACK}/${f}" >&2
    exit 1
  fi
done

echo "Creating repo if needed: ${REPO_ID}"
hf repo create "${REPO_ID}" --type model --private false 2>/dev/null || true

echo "Uploading model card..."
hf upload "${REPO_ID}" "${STAGING}/README.md" README.md

echo "Uploading ${VARIANT}/ ..."
hf upload "${REPO_ID}" "${PACK}" "${VARIANT}"

echo "Done: https://huggingface.co/${REPO_ID}/tree/main/${VARIANT}"
