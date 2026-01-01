#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/pde_sim"
OUT_DIR="$(mktemp -d)"
OUT_FILE="${OUT_DIR}/poisson.vtk"

if [[ ! -x "${BIN}" ]]; then
  echo "pde_sim not found; build first: cmake -S . -B build && cmake --build build"
  exit 1
fi

"${BIN}" \
  --pde "-u_{xx} - u_{yy} = 1" \
  --domain 0,1,0,1 \
  --grid 16,16 \
  --bc "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0" \
  --method sor \
  --omega 1.5 \
  --tol 1e-4 \
  --max-iter 2000 \
  --out "${OUT_FILE}"

if [[ -s "${OUT_FILE}" ]]; then
  echo "Poisson smoke test: PASS (${OUT_FILE})"
else
  echo "Poisson smoke test: FAIL (no output)"
  exit 1
fi

