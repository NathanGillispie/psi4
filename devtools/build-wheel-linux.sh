#!/usr/bin/env bash
set -euo pipefail

# Build and repair one Linux x86_64 wheel. The build uses the wheel-specific
# scikit-build override, which selects LP64 OpenBLAS and enables PSI4_WHEEL.
#
# Set PSI4_NATIVE_LIB_DIRS when prebuilt native dependencies (for example,
# Libint2, Libxc, or gau2grid) are outside the system loader path:
#
#   PSI4_NATIVE_LIB_DIRS="$CONDA_PREFIX/lib" devtools/build-wheel-linux.sh
#
# Set PSI4_WHEEL_PLAT when running in a manylinux-compatible build image, for
# example PSI4_WHEEL_PLAT=manylinux_2_28_x86_64.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${PSI4_WHEEL_BUILD_DIR:-${repo_root}/objdir_wheel_linux_x86_64}"
wheel_dir="${PSI4_WHEEL_DIR:-${repo_root}/wheelhouse}"
raw_dir="${wheel_dir}/raw"
repaired_dir="${wheel_dir}/repaired"

rm -rf "${build_dir}" "${raw_dir}" "${repaired_dir}"
mkdir -p "${raw_dir}" "${repaired_dir}"

if ! command -v auditwheel >/dev/null 2>&1; then
    printf 'auditwheel is required; install it in the build environment.\n' >&2
    exit 1
fi
if ! command -v patchelf >/dev/null 2>&1; then
    printf 'patchelf is required by auditwheel; install it in the build environment.\n' >&2
    exit 1
fi

if [[ -n "${PSI4_NATIVE_LIB_DIRS:-}" ]]; then
    export LD_LIBRARY_PATH="${PSI4_NATIVE_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    # This is only for the build/repair process. The resulting wheel must not
    # depend on this path at runtime.
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# The repository's math finder gives MATH_ROOT precedence over MKLROOT. Use
# the system OpenBLAS development installation unless the caller supplies a
# different root.
export MATH_ROOT="${PSI4_MATH_ROOT:-/usr}"

cd "${repo_root}"
openblas_lib="${PSI4_OPENBLAS_LIBRARY:-}"
if [[ -z "${openblas_lib}" ]]; then
    openblas_lib="$(ldconfig -p 2>/dev/null | awk '$1 == "libopenblas.so" {print $NF; exit}')"
fi
if [[ ! -f "${openblas_lib}" ]]; then
    printf 'Unable to locate an unversioned OpenBLAS development library.\n' >&2
    exit 1
fi

# Static-link Conda's C++/GCC runtime so auditwheel can certify manylinux_2_28.
python -m pip wheel -v . \
    --no-deps \
    --no-build-isolation \
    --config-settings="build-dir=${build_dir}" \
    --config-settings="cmake.define.BLAS_LIBRARIES=${openblas_lib}" \
    --config-settings="cmake.define.LAPACK_LIBRARIES=${openblas_lib}" \
    --config-settings="cmake.define.ENABLE_GENERIC=ON" \
    --wheel-dir="${raw_dir}"

raw_wheel="$(printf '%s\n' "${raw_dir}"/*.whl)"
auditwheel_args=(repair --wheel-dir "${repaired_dir}")
if [[ -n "${PSI4_WHEEL_PLAT:-}" ]]; then
    auditwheel_args+=(--plat "${PSI4_WHEEL_PLAT}")
fi
auditwheel "${auditwheel_args[@]}" "${raw_wheel}"

printf 'Repaired wheel:\n%s\n' "${repaired_dir}"/*.whl
