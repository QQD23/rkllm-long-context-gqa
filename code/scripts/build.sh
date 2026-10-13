#!/usr/bin/env bash
set -euo pipefail
# Native build on RK3588 Linux. SDK files remain outside this repository.
: "SDK_INCLUDE_DIR:?directory containing rkllm.h, rknn_api.h, rknn_matmul_api.h}"
: "RKLLM_RUNTIME_DIR:?directory containing the two pinned shared libraries}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -m)" != aarch64 ]]; then
  echo 'Native AArch64 build required; this script does not configure cross-compilation.' >&2
  exit 2
fi
python3 "$ROOT/scripts/verify_runtime.py" "$RKLLM_RUNTIME_DIR"
for header in rkllm.h rknn_api.h rknn_matmul_api.h; do
  test -f "$SDK_INCLUDE_DIR/$header"
done
mkdir -p "$ROOT/build"
CXX="${CXX:-g++}"
COMMON=(-O3 -std=c++17 -I"$SDK_INCLUDE_DIR" -I"$ROOT/src" -L"$RKLLM_RUNTIME_DIR")
"$CXX" "${COMMON[@]}" -shared -fPIC "$ROOT/src/rkllm_pv_gqa.cpp" -lrknnrt -ldl -o "$ROOT/build/libpv_gqa_v23.so"
"$CXX" "${COMMON[@]}" "$ROOT/src/position_worker.cpp" -lrkllmrt -ldl -pthread -o "$ROOT/build/position_worker"
"$CXX" "${COMMON[@]}" "$ROOT/tests/pv_gqa_probe.cpp" -lrknnrt -pthread -o "$ROOT/build/pv_gqa_probe"
"$CXX" --version > "$ROOT/build/compiler.txt"
sha256sum "$ROOT/build/libpv_gqa_v23.so" "$ROOT/build/position_worker" "$ROOT/build/pv_gqa_probe" > "$ROOT/build/SHA256SUMS"
echo 'Build finished. No inference has been started.'
