#!/usr/bin/env bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

set -euo pipefail

cd /mnt/c/vllm-ascend

TAG="v0.1.0-310p-tests"
ARCHIVE="ascend310p3-aarch64-csrc-tests.tar.gz"

# Автовыбор: нативный linux gh или windows gh.exe
if command -v gh >/dev/null 2>&1; then
  GH_BIN="gh"
elif command -v gh.exe >/dev/null 2>&1; then
  GH_BIN="gh.exe"
elif [ -f "/mnt/c/Program Files/GitHub CLI/gh.exe" ]; then
  GH_BIN="/mnt/c/Program Files/GitHub CLI/gh.exe"
else
  echo "Error: gh / gh.exe not found neither in WSL nor in Windows."
  exit 1
fi

echo "=== Using GitHub CLI: ${GH_BIN} ==="

if [ ! -f "${ARCHIVE}" ]; then
  echo "Error: ${ARCHIVE} not found. Run packaging task first."
  exit 1
fi

echo "=== 1. Checking GitHub authentication ==="
"${GH_BIN}" auth status

echo "=== 2. Uploading ${ARCHIVE} to Release [${TAG}] ==="
"${GH_BIN}" release upload "${TAG}" "${ARCHIVE}" --clobber

echo "=== Success: Artifact uploaded to release ${TAG} ==="