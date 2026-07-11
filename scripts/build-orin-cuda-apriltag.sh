#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build/orin-apriltag"
CHECK_ONLY="no"
SOURCE_REVISION="${PV_SOURCE_REVISION:-}"
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir) BUILD_DIR="${2:?missing build directory}"; shift 2 ;;
        --source-revision) SOURCE_REVISION="${2:?missing source revision}"; shift 2 ;;
        --check-only) CHECK_ONLY="yes"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ "$(uname -m)" == "aarch64" ]] || { echo "CUDA build requires AArch64" >&2; exit 1; }
grep -q '^# R39 (release), REVISION: 2.0' /etc/nv_tegra_release || {
    echo "CUDA build requires JetPack 7.2 / L4T R39.2" >&2
    exit 1
}
command -v nvcc >/dev/null || { echo "Install nvidia-cuda-dev" >&2; exit 1; }
command -v cmake >/dev/null || { echo "Install cmake" >&2; exit 1; }
command -v ninja >/dev/null || { echo "Install ninja-build" >&2; exit 1; }
command -v javac >/dev/null || { echo "Install openjdk-17-jdk" >&2; exit 1; }
[[ "$(javac -version 2>&1)" == javac\ 17.* ]] || { echo "JDK 17 is required" >&2; exit 1; }

[[ "${CHECK_ONLY}" == "yes" ]] && exit 0

if [[ -z "${SOURCE_REVISION}" ]]; then
    SOURCE_REVISION="$(git -C "${ROOT}" rev-parse --short=12 HEAD 2>/dev/null || true)"
fi
[[ -n "${SOURCE_REVISION}" ]] || {
    echo "Pass --source-revision when building from a copied source tree" >&2
    exit 1
}

cmake -S "${ROOT}/native/orin-apriltag" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DPV_SOURCE_REVISION="${SOURCE_REVISION}"
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"
ctest --test-dir "${BUILD_DIR}" --output-on-failure
