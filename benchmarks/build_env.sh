#!/usr/bin/env bash
#SBATCH --job-name=cardio_build
#SBATCH --output=cardio_build-%j.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#
# One-time build of the software stack of the paper benchmarks, in
# $CARDIO_PREFIX (default $HOME/cardioEMI-env, about 5 GB), with sources and
# build trees in $CARDIO_BUILD (removable after the build). Run it from the
# repository root, either as a batch job or directly on a node with internet
# access (the sources are downloaded during the build):
#
#   sbatch benchmarks/build_env.sh        # account/partition: sbatch -A ... -p ... or SBATCH_ACCOUNT/SBATCH_PARTITION
#   bash   benchmarks/build_env.sh
#
# Prerequisites (compilers, MPI, CUDA) come from benchmarks/site_env.sh, see
# benchmarks/site_env.example.sh. Two variants are built:
#
#   common : Python (uv-managed), CMake, Ninja, pugixml, spdlog, Boost.Timer, basix 0.9.0
#   gpu    : PETSc $CARDIO_PETSC_VERSION + CUDA + hypre/MUMPS/PT-Scotch, petsc4py,
#            dolfinx 0.9.0, cuDOLFINx 0.9.0, multiphenicsx 0.3.9   (venv-gpu)
#   cpu    : the same with a CPU-only PETSc and without cuDOLFINx  (venv-cpu)
#
# CARDIO_VARIANTS="cpu" builds only the CPU variant (no CUDA needed).
# Every step is skipped when done ($CARDIO_PREFIX/.done-<step>), so a failed
# build can simply be restarted. FORCE_<STEP>=1 (e.g. FORCE_PETSC_GPU=1) redoes one.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f benchmarks/env.sh ]; then
    echo "run this script from the root of the cardioEMI repository" >&2; exit 1
fi
ROOT=$PWD
source benchmarks/env.sh
cardio_site_env

VARIANTS=${CARDIO_VARIANTS:-gpu cpu}
NP=${CARDIO_BUILD_NP:-${SLURM_CPUS_PER_TASK:-8}}
SRC=$CARDIO_BUILD/src
PETSC_SRC=$SRC/petsc-$CARDIO_PETSC_VERSION
mkdir -p "$CARDIO_PREFIX/tools/bin" "$CARDIO_PREFIX/common" "$SRC"
export PATH=$CARDIO_PREFIX/tools/bin:$PATH
export CMAKE_PREFIX_PATH=$CARDIO_PREFIX/common${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}
unset PYTHONPATH

step() {
    local force; force=FORCE_$(printf %s "$1" | tr a-z A-Z)
    if [ -e "$CARDIO_PREFIX/.done-$1" ] && [ "${!force:-0}" != 1 ]; then
        echo ">>> [$1] already done"; return 1
    fi
    echo ">>> [$1] $(date '+%F %T')"; return 0
}
done_step() { touch "$CARDIO_PREFIX/.done-$1"; echo ">>> [$1] done $(date '+%F %T')"; }
clone() { [ -d "$SRC/$1/.git" ] || git clone --depth 1 -b "$2" "$3" "$SRC/$1"; }
cmake_install() {  # cmake_install <src dir> <build dir> <prefix> [cmake args...]
    local s=$1 b=$2 p=$3; shift 3
    # --fresh: drop the cache of a previous failed attempt
    cmake --fresh -G Ninja -S "$s" -B "$b" -DCMAKE_INSTALL_PREFIX="$p" -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_LIBDIR=lib "$@"
    cmake --build "$b" -j "$NP"
    cmake --install "$b"
}
# run_python <args...>: through srun inside a job (some MPI libraries cannot be
# initialised outside a job step), directly otherwise
run_python() {
    if [ -n "${SLURM_JOB_ID:-}" ]; then srun -n 1 "$@"; else "$@"; fi
}

for tool in mpicc mpicxx mpif90 git curl make python3; do
    command -v "$tool" > /dev/null || { echo "ERROR: $tool not found (see benchmarks/site_env.example.sh)" >&2; exit 1; }
done

# CUDA toolkit prefix (include/cuda.h, lib64/libcudart.so), for the gpu variant only
cuda_defs=()
case " $VARIANTS " in *" gpu "*)
    cuda_ok() { [ -n "$1" ] && [ -f "$1/include/cuda.h" ] && ls "$1"/lib*/libcudart.so > /dev/null 2>&1; }
    command -v nvcc > /dev/null || { echo "ERROR: nvcc not found (needed by the gpu variant)" >&2; exit 1; }
    CUDA_DIR=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")
    cuda_ok "$CUDA_DIR" || CUDA_DIR=${CUDA_HOME:-}
    cuda_ok "$CUDA_DIR" || { echo "ERROR: CUDA toolkit prefix not found (set CUDA_HOME)" >&2; exit 1; }
    export CUDA_HOME=$CUDA_DIR CUDAToolkit_ROOT=$CUDA_DIR
    export PATH=$CUDA_DIR/bin:$PATH
    echo "CUDA toolkit: $CUDA_DIR | $(nvcc --version | tail -1)"
    # explicit paths of the CUDA libraries used by cuDOLFINx, for toolkits whose
    # layout FindCUDAToolkit does not resolve (e.g. installed through symlinks)
    cuda_defs=(CMAKE_CUDA_COMPILER="$CUDA_DIR/bin/nvcc" CUDAToolkit_ROOT="$CUDA_DIR")
    [ -f "$CUDA_DIR/extras/CUPTI/lib64/libcupti.so" ] && cuda_defs+=(
        CUDA_cupti_LIBRARY="$CUDA_DIR/extras/CUPTI/lib64/libcupti.so"
        CUDAToolkit_CUPTI_INCLUDE_DIR="$CUDA_DIR/extras/CUPTI/include")
    [ -f "$CUDA_DIR/lib64/libnvrtc.so" ] && cuda_defs+=(CUDA_nvrtc_LIBRARY="$CUDA_DIR/lib64/libnvrtc.so")
    [ -f "$CUDA_DIR/lib64/stubs/libcuda.so" ] && cuda_defs+=(CUDA_cuda_driver_LIBRARY="$CUDA_DIR/lib64/stubs/libcuda.so")
    ;;
esac
echo "variants: $VARIANTS | $(mpicc --version | head -1) | mpicc: $(command -v mpicc) | make -j $NP"

# ---------------------------------------------------------------- common ----
if step sources; then
    clone basix         v0.9.0   https://github.com/FEniCS/basix.git
    clone ufl           2024.2.0 https://github.com/FEniCS/ufl.git
    clone ffcx          v0.9.0   https://github.com/FEniCS/ffcx.git
    clone dolfinx       v0.9.0   https://github.com/FEniCS/dolfinx.git
    clone cuda-dolfinx  v0.9.0   https://github.com/bpachev/cuda-dolfinx.git
    clone multiphenicsx v0.3.9   https://github.com/multiphenics/multiphenicsx.git
    clone pugixml       v1.14    https://github.com/zeux/pugixml.git
    clone spdlog        v1.15.0  https://github.com/gabime/spdlog.git
    [ -d "$PETSC_SRC/.git" ] || git clone --depth 1 -b "v$CARDIO_PETSC_VERSION" https://gitlab.com/petsc/petsc.git "$PETSC_SRC"
    done_step sources
fi

if step tools; then
    cd "$SRC"
    # flex is needed by PT-Scotch (bison is downloaded by PETSc)
    if ! command -v flex > /dev/null; then
        if ! command -v m4 > /dev/null; then
            curl -fL --retry 3 -o m4.tar.gz https://ftp.gnu.org/gnu/m4/m4-1.4.19.tar.gz
            rm -rf m4-1.4.19 && tar xzf m4.tar.gz
            (cd m4-1.4.19 && ./configure --prefix="$CARDIO_PREFIX/tools" && make -j "$NP" && make install)
        fi
        curl -fL --retry 3 -o flex.tar.gz https://github.com/westes/flex/releases/download/v2.6.4/flex-2.6.4.tar.gz
        rm -rf flex-2.6.4 && tar xzf flex.tar.gz
        # -D_GNU_SOURCE: known flex 2.6.4 build issue with recent glibc
        (cd flex-2.6.4 && CFLAGS="-O2 -D_GNU_SOURCE" ./configure --prefix="$CARDIO_PREFIX/tools" \
            && make -j "$NP" && make install)
    fi
    if ! command -v pkg-config > /dev/null; then
        curl -fL --retry 3 -o pkgconf.tar.xz https://distfiles.ariadne.space/pkgconf/pkgconf-2.3.0.tar.xz
        rm -rf pkgconf-2.3.0 && tar xJf pkgconf.tar.xz
        (cd pkgconf-2.3.0 && ./configure --prefix="$CARDIO_PREFIX/tools" && make -j "$NP" && make install)
        ln -sf pkgconf "$CARDIO_PREFIX/tools/bin/pkg-config"
    fi
    # uv (Python manager), CMake and Ninja in a bootstrap venv on the system Python
    python3 -m venv "$CARDIO_PREFIX/bootstrap"
    "$CARDIO_PREFIX/bootstrap/bin/python" -m pip install --upgrade pip uv ninja "cmake>=3.21,<4"
    for t in uv cmake ninja; do ln -sf "$CARDIO_PREFIX/bootstrap/bin/$t" "$CARDIO_PREFIX/tools/bin/$t"; done
    UV_PYTHON_INSTALL_DIR=$CARDIO_PREFIX/python uv python install --python-preference only-managed \
        "$CARDIO_PYTHON_VERSION"
    flex --version; pkg-config --version; uv --version; cmake --version | head -1; ninja --version
    cd "$ROOT"
    done_step tools
fi

if step common_cpp; then
    C=$CARDIO_PREFIX/common
    cmake_install "$SRC/pugixml" "$CARDIO_BUILD/pugixml" "$C" -DBUILD_SHARED_LIBS=ON
    cmake_install "$SRC/spdlog"  "$CARDIO_BUILD/spdlog"  "$C" -DSPDLOG_BUILD_SHARED=ON \
        -DSPDLOG_FMT_EXTERNAL=OFF -DSPDLOG_BUILD_EXAMPLE=OFF
    # dolfinx 0.9 needs Boost.Timer
    if [ -z "${CARDIO_BOOST_DIR:-}" ]; then
        cd "$SRC"
        curl -fL --retry 3 -o boost.tar.gz https://archives.boost.io/release/1.88.0/source/boost_1_88_0.tar.gz
        rm -rf boost_1_88_0 && tar xzf boost.tar.gz && cd boost_1_88_0
        ./bootstrap.sh --prefix="$C" --with-libraries=timer,chrono,system
        ./b2 -j "$NP" link=shared variant=release install
        cd "$ROOT"
    fi
    cmake_install "$SRC/basix/cpp" "$CARDIO_BUILD/basix" "$C"
    done_step common_cpp
fi
[ -n "${CARDIO_BOOST_DIR:-}" ] && export CMAKE_PREFIX_PATH=$CARDIO_BOOST_DIR:$CMAKE_PREFIX_PATH

# ------------------------------------------------------------- variants ----
petsc_options() {  # petsc_options gpu|cpu
    local o=(
        --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --with-mpiexec=srun
        --COPTFLAGS=-O2 --CXXOPTFLAGS=-O2 --FOPTFLAGS=-O2 --with-debugging=no
        --with-shared-libraries --with-64-bit-indices=no --with-scalar-type=real
        --with-precision=double --with-fortran-bindings=no
        --download-hypre --download-metis --download-ptscotch --download-bison
        --download-mumps --download-mumps-avoid-mpi-in-place --download-scalapack
    )
    if [ -n "${CARDIO_BLASLAPACK_DIR:-}" ]; then
        o+=(--with-blaslapack-dir="$CARDIO_BLASLAPACK_DIR")
    else
        o+=(--download-f2cblaslapack)
    fi
    [ -z "${CARDIO_HDF5_DIR:-}" ] && o+=(--download-hdf5)
    if [ "$1" = gpu ]; then
        o+=(--with-cuda --with-cuda-dir="$CUDA_DIR" --with-cuda-arch="$CARDIO_CUDA_ARCH")
    fi
    printf '%s\n' "${o[@]}"
}

for v in $VARIANTS; do
    P=$CARDIO_PREFIX
    HDF5_DIR=${CARDIO_HDF5_DIR:-$P/petsc-$v}   # parallel HDF5 for dolfinx

    if step "venv_$v"; then
        rm -rf "$P/venv-$v"
        UV_PYTHON_INSTALL_DIR=$P/python uv venv --seed --python-preference only-managed \
            --python "$CARDIO_PYTHON_VERSION" "$P/venv-$v"
        ( cardio_activate "$v"
          python -m pip install --upgrade pip
          # nanobind < 2.14: from 2.14 integer arguments need __index__, which the cffi
          # pointers passed by dolfinx 0.9 to its C++ Form constructor do not have
          python -m pip install numpy "cython<3.1" "setuptools<81" wheel "nanobind<2.14" \
              "scikit-build-core[pyproject]" cffi pyyaml
          MPICC=mpicc python -m pip install --no-binary=mpi4py --no-cache-dir mpi4py
          python -m pip install --no-build-isolation --no-deps "$SRC/basix/python"
          python -m pip install --no-deps "$SRC/ufl" "$SRC/ffcx" )
        done_step "venv_$v"
    fi

    if step "petsc_$v"; then
        ( cd "$PETSC_SRC"
          unset PETSC_DIR PETSC_ARCH
          mapfile -t opts < <(petsc_options "$v")
          python3 ./configure PETSC_ARCH="arch-$v" --prefix="$P/petsc-$v" --with-make-np="$NP" "${opts[@]}"
          make PETSC_DIR="$PWD" PETSC_ARCH="arch-$v" all
          make PETSC_DIR="$PWD" PETSC_ARCH="arch-$v" install )
        done_step "petsc_$v"
    fi

    if step "petsc4py_$v"; then
        ( cardio_activate "$v"
          # petsc4py 3.23 build: Cython >= 3.1 rejects its index typing, setuptools >= 81
          # removed the distutils dry_run used by its setup
          python -m pip install --no-build-isolation --no-deps --no-cache-dir "$PETSC_SRC/src/binding/petsc4py" )
        done_step "petsc4py_$v"
    fi

    if step "dolfinx_$v"; then
        ( cardio_activate "$v"
          export HDF5_ROOT=$HDF5_DIR PATH=$HDF5_DIR/bin:$PATH
          cmake_install "$SRC/dolfinx/cpp" "$CARDIO_BUILD/dolfinx-$v" "$P/dolfinx-$v" \
              -DDOLFINX_ENABLE_SLEPC=OFF -DDOLFINX_ENABLE_ADIOS2=OFF -DDOLFINX_ENABLE_PARMETIS=OFF \
              -DDOLFINX_ENABLE_KAHIP=OFF -DDOLFINX_ENABLE_SCOTCH=ON -DSCOTCH_ROOT="$P/petsc-$v" \
              -DHDF5_ROOT="$HDF5_DIR" -DDOLFINX_SKIP_BUILD_TESTS=ON
          python -m pip install -v --no-build-isolation --no-deps --no-cache-dir \
              --config-settings=cmake.build-type=Release "$SRC/dolfinx/python" )
        done_step "dolfinx_$v"
    fi

    if [ "$v" = gpu ] && step cudolfinx; then
        ( cardio_activate gpu
          cmake_install "$SRC/cuda-dolfinx/cpp" "$CARDIO_BUILD/cudolfinx" "$P/dolfinx-gpu" \
              -DCUDOLFINX_SKIP_BUILD_TESTS=YES -DCMAKE_CUDA_ARCHITECTURES="$CARDIO_CUDA_ARCH" \
              "${cuda_defs[@]/#/-D}"
          python -m pip install -v --no-build-isolation --no-deps --no-cache-dir \
              --config-settings=cmake.build-type=Release \
              "${cuda_defs[@]/#/--config-settings=cmake.define.}" \
              "$SRC/cuda-dolfinx/python" )
        done_step cudolfinx
    fi

    if step "multiphenicsx_$v"; then
        ( cardio_activate "$v"
          # multiphenicsx's CMake locates the dolfinx wrappers with `import dolfinx`,
          # which initialises MPI; some MPI libraries refuse (or hang) when this happens
          # outside a job step. Locate the package without importing it.
          python3 - "$SRC/multiphenicsx/multiphenicsx/cpp/CMakeLists.txt" <<'PY'
import sys
from pathlib import Path
f = Path(sys.argv[1]); s = f.read_text()
old = "import os, sys, dolfinx; print(os.path.join(os.path.dirname(dolfinx.__file__), 'wrappers'))"
new = ("import os, importlib.util; "
       "print(os.path.join(importlib.util.find_spec('dolfinx').submodule_search_locations[0], 'wrappers'))")
if old in s:
    f.write_text(s.replace(old, new)); print("multiphenicsx CMakeLists.txt patched")
elif new in s:
    print("multiphenicsx CMakeLists.txt already patched")
else:
    sys.exit("multiphenicsx CMakeLists.txt: expected dolfinx lookup not found")
PY
          python -m pip install -v --no-build-isolation --no-deps --no-cache-dir "$SRC/multiphenicsx"
          python -m pip freeze > "$P/freeze-$v.txt" )
        done_step "multiphenicsx_$v"
    fi
done

# ---------------------------------------------------------------- checks ----
for v in $VARIANTS; do
    if [ "$v" = gpu ] && ! nvidia-smi -L > /dev/null 2>&1; then
        echo ">>> import check (gpu) skipped: no GPU on this node"; continue
    fi
    echo ">>> import check ($v)"
    ( cardio_activate "$v"
      ldd "$PETSC_DIR/lib/libpetsc.so" | grep -E "mpi|cuda|HYPRE" || true
      run_python python - "$v" <<'PY'
import sys
import dolfinx, basix, ffcx, ufl, mpi4py, petsc4py, multiphenicsx.fem.petsc, yaml
from petsc4py import PETSc
print("dolfinx", dolfinx.__version__, "| basix", basix.__version__, "| ffcx", ffcx.__version__,
      "| ufl", ufl.__version__, "| mpi4py", mpi4py.__version__, "| python", sys.version.split()[0])
print("PETSc", PETSc.Sys.getVersion(), petsc4py.get_config()["PETSC_DIR"],
      "| hypre", PETSc.Sys.hasExternalPackage("hypre"), "| cuda", PETSc.Sys.hasExternalPackage("cuda"))
if sys.argv[1] == "gpu":
    import cudolfinx
    print("cudolfinx", cudolfinx.__file__)
print("IMPORT CHECK OK", sys.argv[1])
PY
    )
done
du -sh "$CARDIO_PREFIX"
echo ">>> build finished $(date '+%F %T')"
