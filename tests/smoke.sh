#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/pde_sim"

if [[ ! -x "${BIN}" ]]; then
  echo "pde_sim not found; build first: cmake -S . -B build && cmake --build build"
  exit 1
fi

echo "Running command: ${BIN} --self-test --backend cpu"
"${BIN}" --self-test --backend cpu

