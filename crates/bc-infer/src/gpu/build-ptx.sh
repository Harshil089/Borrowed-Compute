#!/bin/sh
# Compiles cuda.cu (with kernels.h) to kernels.ptx, which the CUDA backend embeds. Run it
# after changing either file; a test fails until you do. Needs clang with the NVPTX
# target (Homebrew's llvm has it), nothing from NVIDIA.
# PTX ISA 6.3 for sm_52 and newer: any NVIDIA driver from 2018 on (410+) loads it.
set -e
cd "$(dirname "$0")"
CLANG="${CLANG:-$(brew --prefix llvm 2>/dev/null)/bin/clang}"
hash=$(cat cuda.cu kernels.h | shasum -a 256 | cut -c1-64)
"$CLANG" -x cuda --cuda-device-only --cuda-gpu-arch=sm_52 -nocudainc -nocudalib -Xclang -target-feature -Xclang +ptx63 \
    -O3 -ffp-contract=off -fno-fast-math -S cuda.cu -o kernels.ptx.tmp
{ echo "// source sha256 $hash (cuda.cu + kernels.h); regenerate with build-ptx.sh"; cat kernels.ptx.tmp; } > kernels.ptx
rm kernels.ptx.tmp
