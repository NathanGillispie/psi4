#!/usr/bin/env bash
# =============================================================================
# backend.sh - configure/build Psi4 against a chosen BLAS/LAPACK backend.
#
# Motivation: Psi4's BLAS/LAPACK backend is a configure-time choice
# (-DBLAS_LIBRARIES / -DLAPACK_LIBRARIES / ...), but the flags are easy to get
# wrong and were previously hardcoded into the top-level CMakeLists.txt. This
# script resolves a named backend to concrete library/header paths and
# configures a dedicated per-backend build dir, so switching backends is just
# `backend.sh build <name>` - no editing of CMakeLists.txt or the conda env.
#
# Usage:
#   backend.sh list                                # show backends + resolved paths
#   backend.sh configure <backend> [objdir]        # cmake configure only
#   backend.sh build <backend> [objdir]            # configure (if needed) + build
#   backend.sh env <backend>                       # print exports needed to run
#
# Backends: aocl | mkl | openblas
#   aocl     - AMD AOCL (BLIS-mt + FLAME); root /opt/aocl/gcc/MT (override AOCL_ROOT)
#   mkl      - Intel MKL: taken from the active conda env if it has MKL,
#              otherwise the newest extracted package in ~/.conda/pkgs (no
#              downloads). libiomp5 is required at link time by Psi4's
#              FindMathOpenMP; it is searched in the MKL dir, then the active
#              env, then /opt/miniconda3 (base), then other conda envs.
#   openblas - conda OpenBLAS from the active env.
#
# Examples:
#   conda activate p4addons
#   devtools/backend.sh build mkl                   # -> objdir_p4addons_mkl
#   devtools/backend.sh build openblas objdir_ob    # custom build dir
#   devtools/backend.sh env mkl                     # run with these exports
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "backend.sh: $*" >&2; exit 1; }

# ---------------- backend path resolution ----------------
# Each resolve_<backend> sets (empty = not needed):
#   BLAS_LIB, LAPACK_LIB : full paths to link
#   BLAS_INC, LAPACK_INC : include dirs
#   IOMP_DIR             : extra dir for OpenMP_LIBRARY_DIRS (MKL needs libiomp5)
#   MKL_LIBDIR           : runtime libdir (for `env mkl`)

resolve_aocl() {
    local root="${AOCL_ROOT:-/opt/aocl/gcc/MT}"
    [[ -d "$root" ]] || die "AOCL root '$root' not found (set AOCL_ROOT)"
    BLAS_LIB="$root/lib/libblis-mt.so"
    LAPACK_LIB="$root/lib/libflame.so"
    BLAS_INC="$root/include"
    LAPACK_INC="$root/include"
    IOMP_DIR=""
    MKL_LIBDIR=""
    [[ -f "$BLAS_LIB" ]] || die "missing $BLAS_LIB"
    [[ -f "$LAPACK_LIB" ]] || die "missing $LAPACK_LIB"
}

resolve_mkl() {
    local mkl_libdir="" mkl_incdir=""
    if [[ -n "${CONDA_PREFIX:-}" && -f "$CONDA_PREFIX/lib/libmkl_rt.so" ]]; then
        # active conda env already has MKL installed
        mkl_libdir="$CONDA_PREFIX/lib"
        [[ -f "$CONDA_PREFIX/include/mkl.h" ]] && mkl_incdir="$CONDA_PREFIX/include"
    else
        # newest extracted MKL package in the conda cache (no downloads)
        local pkgs="${CONDA_PKGS_DIRS:-$HOME/.conda/pkgs}"
        mkl_libdir="$(ls -d "$pkgs"/mkl-[0-9]*/lib 2>/dev/null | sort -V | tail -1 || true)"
        mkl_incdir="$(ls -d "$pkgs"/mkl-include-[0-9]*/include 2>/dev/null | sort -V | tail -1 || true)"
    fi
    [[ -n "$mkl_libdir" && -f "$mkl_libdir/libmkl_rt.so" ]] \
        || die "no MKL found (looked in active env and ~/.conda/pkgs for libmkl_rt.so)"
    BLAS_LIB="$mkl_libdir/libmkl_rt.so"
    LAPACK_LIB="$mkl_libdir/libmkl_rt.so"
    BLAS_INC="${mkl_incdir:-}"
    LAPACK_INC="${mkl_incdir:-}"
    MKL_LIBDIR="$mkl_libdir"
    # Psi4's FindMathOpenMP hard-requires libiomp5 for MKL under GNU compilers
    IOMP_DIR=""
    if [[ -f "$mkl_libdir/libiomp5.so" ]]; then
        IOMP_DIR="$mkl_libdir"
    else
        for d in "${CONDA_PREFIX:-}" /opt/miniconda3/lib "$HOME"/.conda/envs/*/lib; do
            if [[ -n "$d" && -f "$d/libiomp5.so" ]]; then IOMP_DIR="$d"; break; fi
        done
    fi
    [[ -n "$IOMP_DIR" ]] || die "libiomp5.so not found; MKL needs it at link time (intel-openmp package)"
}

resolve_openblas() {
    [[ -n "${CONDA_PREFIX:-}" ]] || die "activate a conda env first"
    BLAS_LIB="$CONDA_PREFIX/lib/libopenblas.so"
    LAPACK_LIB="$CONDA_PREFIX/lib/libopenblas.so"
    BLAS_INC=""
    LAPACK_INC=""
    IOMP_DIR=""
    MKL_LIBDIR=""
    [[ -f "$BLAS_LIB" ]] || die "missing $BLAS_LIB (conda openblas not in active env)"
}

resolve() {
    case "$1" in
        aocl)     resolve_aocl ;;
        mkl)      resolve_mkl ;;
        openblas) resolve_openblas ;;
        *) die "unknown backend '$1' (aocl | mkl | openblas)" ;;
    esac
}

default_objdir() { echo "objdir_${CONDA_DEFAULT_ENV:-p4dev}_$1"; }

# ---------------- commands ----------------

cmd_list() {
    for b in aocl mkl openblas; do
        local out
        if out="$({ resolve "$b" 2>&1; echo "BLAS=$BLAS_LIB"; echo "LAPACK=$LAPACK_LIB"; [[ -n "$BLAS_INC" ]] && echo "INC=$BLAS_INC"; [[ -n "$IOMP_DIR" ]] && echo "iomp5=$IOMP_DIR"; true; })"; then
            echo "$b:"
            echo "$out" | sed 's/^/    /'
        else
            echo "$b: unavailable (${out##*: })"
        fi
    done
}

cmd_configure() {
    local backend="$1" objdir="${2:-$(default_objdir "$1")}"
    [[ -n "${CONDA_PREFIX:-}" ]] || die "activate the conda env first (CONDA_PREFIX unset)"
    resolve "$backend"
    local cache=""
    [[ -f "$REPO_ROOT/cache_${CONDA_DEFAULT_ENV}.cmake" ]] && cache="-C $REPO_ROOT/cache_${CONDA_DEFAULT_ENV}.cmake"
    local omp_dirs="$CONDA_PREFIX/lib"
    [[ -n "$IOMP_DIR" ]] && omp_dirs="$omp_dirs;$IOMP_DIR"
    local inc_flags=()
    [[ -n "$BLAS_INC" ]] && inc_flags+=(-DBLAS_INCLUDE_DIRS="$BLAS_INC" -DLAPACK_INCLUDE_DIRS="$LAPACK_INC")
    echo "backend.sh: configuring '$objdir' with backend '$backend'"
    echo "  BLAS_LIBRARIES   = $BLAS_LIB"
    echo "  LAPACK_LIBRARIES = $LAPACK_LIB"
    echo "  OpenMP_LIBRARY_DIRS = $omp_dirs"
    cmake -S "$REPO_ROOT" -GNinja $cache -B "$objdir" \
        -DBLAS_LIBRARIES="$BLAS_LIB" \
        -DLAPACK_LIBRARIES="$LAPACK_LIB" \
        -DOpenMP_LIBRARY_DIRS="$omp_dirs" \
        "${inc_flags[@]}"
}

cmd_build() {
    local backend="$1" objdir="${2:-$(default_objdir "$1")}"
    if [[ ! -f "$objdir/CMakeCache.txt" ]]; then
        cmd_configure "$backend" "$objdir"
    else
        resolve "$backend"
        local cached
        cached="$(grep -E '^LAPACK_LIBRARIES:' "$objdir/CMakeCache.txt" | head -1 | cut -d= -f2- || true)"
        if [[ -n "$cached" && "$cached" != "$LAPACK_LIB" ]]; then
            echo "backend.sh: WARNING '$objdir' was configured with LAPACK_LIBRARIES=$cached" >&2
            echo "backend.sh: WARNING but backend '$backend' resolves to $LAPACK_LIB" >&2
            echo "backend.sh: WARNING re-run configure or use a fresh objdir to switch backends" >&2
        else
            echo "backend.sh: '$objdir' already configured; not re-running configure"
        fi
    fi
    cmake --build "$objdir"
}

cmd_env() {
    local backend="$1"
    resolve "$backend"
    [[ -n "$MKL_LIBDIR" ]] && echo "export LD_LIBRARY_PATH=$MKL_LIBDIR:\$LD_LIBRARY_PATH"
}

# ---------------- main ----------------
cmd="${1:-}"
[[ -n "$cmd" ]] || { echo "usage: backend.sh {list|configure|build|env} [backend] [objdir]" >&2; exit 1; }
case "$cmd" in
    list)      cmd_list ;;
    configure) [[ $# -ge 2 ]] || die "configure needs a backend"; cmd_configure "$2" "${3:-}" ;;
    build)     [[ $# -ge 2 ]] || die "build needs a backend"; cmd_build "$2" "${3:-}" ;;
    env)       [[ $# -ge 2 ]] || die "env needs a backend"; cmd_env "$2" ;;
    *) die "unknown command '$cmd' (list | configure | build | env)" ;;
esac
