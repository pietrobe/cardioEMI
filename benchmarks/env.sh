# shellcheck shell=bash
# Common environment of the paper benchmarks (see benchmarks/README.md).
# Source it from the repository root, in a job or on a login node:
#   source benchmarks/env.sh
# It only defines variables and functions; the caller's environment is changed
# only by cardio_activate, and only inside the shell that calls it.
#
# Every CARDIO_* variable below can be set before sourcing this file.

CARDIO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CARDIO_ROOT

# ---- site environment --------------------------------------------------------
# Optional script that loads the compilers, MPI and CUDA of the cluster
# (e.g. `module load ...`); see benchmarks/site_env.example.sh.
export CARDIO_SITE_ENV=${CARDIO_SITE_ENV:-$CARDIO_ROOT/benchmarks/site_env.sh}

# ---- versions and locations --------------------------------------------------
# dolfinx and cuDOLFINx 0.9 belong to the PETSc 3.22-3.23 generation; PETSc
# 3.23.7 downloads hypre 2.32.0 and supports CUDA up to 12.9.
export CARDIO_PETSC_VERSION=${CARDIO_PETSC_VERSION:-3.23.7}
export CARDIO_PYTHON_VERSION=${CARDIO_PYTHON_VERSION:-3.12}
export CARDIO_CUDA_ARCH=${CARDIO_CUDA_ARCH:-90}             # 90 = Hopper (GH200)

export CARDIO_PREFIX=${CARDIO_PREFIX:-$HOME/cardioEMI-env}  # installed stack (persistent)
export CARDIO_BUILD=${CARDIO_BUILD:-$CARDIO_PREFIX-build}   # sources and build trees (temporary)
export CARDIO_MESH_DIR=${CARDIO_MESH_DIR:-$CARDIO_ROOT/meshes}
export CARDIO_WORK=${CARDIO_WORK:-$CARDIO_ROOT/benchmarks}  # runs/ and results/ go here

# ---- node mapping --------------------------------------------------------------
# The paper counts nodes as follows:
#   GPU node = one GH200 driven by one MPI rank;
#   CPU node = 144 Grace cores, one MPI rank per core.
# CARDIO_GPUS_PER_NODE GPU ranks and CARDIO_CPU_RANKS_PER_NODE CPU ranks are
# placed on each physical node; CARDIO_CPUS_PER_GPU_RANK cores are bound to each
# GPU rank (72 = one Grace CPU per GH200).
export CARDIO_GPUS_PER_NODE=${CARDIO_GPUS_PER_NODE:-1}
export CARDIO_CPU_RANKS_PER_NODE=${CARDIO_CPU_RANKS_PER_NODE:-144}
export CARDIO_CPUS_PER_GPU_RANK=${CARDIO_CPUS_PER_GPU_RANK:-72}
# GPU-aware MPI for the GPU runs: auto, 1 or 0 (0 passes -use_gpu_aware_mpi 0 to PETSc)
export CARDIO_GPU_AWARE_MPI=${CARDIO_GPU_AWARE_MPI:-auto}

cardio_site_env() {
    if [ -f "$CARDIO_SITE_ENV" ]; then
        # shellcheck disable=SC1090
        source "$CARDIO_SITE_ENV"
    fi
}

# ---- activation ----------------------------------------------------------------
# cardio_activate gpu|cpu : PETSc/dolfinx/venv of one variant. The gpu variant
# has a CUDA PETSc and cuDOLFINx; the cpu variant has a CPU-only PETSc, so that
# hypre runs on the host.
cardio_activate() {
    local v=${1:?usage: cardio_activate gpu|cpu}
    cardio_site_env
    unset PYTHONPATH PETSC_ARCH
    # venv activate scripts are not written for `set -u`
    local had_u=0; case $- in *u*) had_u=1; set +u ;; esac
    # shellcheck disable=SC1090
    source "$CARDIO_PREFIX/venv-$v/bin/activate"
    [ "$had_u" = 1 ] && set -u
    export PETSC_DIR=$CARDIO_PREFIX/petsc-$v
    export CARDIO_DOLFINX_DIR=$CARDIO_PREFIX/dolfinx-$v
    local c=$CARDIO_PREFIX/common
    export LD_LIBRARY_PATH=$CARDIO_DOLFINX_DIR/lib:$PETSC_DIR/lib:$c/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export CMAKE_PREFIX_PATH=$CARDIO_DOLFINX_DIR:$PETSC_DIR:$c${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}
    export PKG_CONFIG_PATH=$CARDIO_DOLFINX_DIR/lib/pkgconfig:$PETSC_DIR/lib/pkgconfig:$c/lib/pkgconfig:$c/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}
    export PATH=$CARDIO_PREFIX/tools/bin:$PETSC_DIR/bin:$PATH
    export OMP_NUM_THREADS=1
    # Cray MPICH's compiler wrappers link libhugetlbfs into every MPI extension
    # module; its segment remapping crashes a PIE Python on import. Huge pages are
    # not used here, so both features are switched off (no effect elsewhere).
    export HUGETLB_ELFMAP=no HUGETLB_MORECORE=no
    if [ "$v" = gpu ]; then
        local aware=$CARDIO_GPU_AWARE_MPI
        if [ "$aware" = auto ]; then
            # Cray MPICH: GPU support is a runtime switch, needed when PETSc links its GTL library
            if ldd "$PETSC_DIR/lib/libpetsc.so" 2>/dev/null | grep -q mpi_gtl_cuda; then
                export MPICH_GPU_SUPPORT_ENABLED=1
            fi
            unset PETSC_OPTIONS   # PETSc detects the CUDA support of the MPI library itself
        elif [ "$aware" = 1 ]; then
            export MPICH_GPU_SUPPORT_ENABLED=1
            unset PETSC_OPTIONS
        else
            export PETSC_OPTIONS="-use_gpu_aware_mpi 0"
        fi
    else
        unset PETSC_OPTIONS
    fi
}

# ---- runs ------------------------------------------------------------------------
# enter_rundir <name>: per-job directory of symlinks, so that the JIT caches
# ($PWD/.cache of main.py, $XDG_CACHE_HOME/fenics of cuDOLFINx) are not shared
# between concurrent jobs.
enter_rundir() {
    local d=$CARDIO_WORK/runs/$1 f
    mkdir -p "$d" "$CARDIO_WORK/results"
    for f in main.py utils.py ionic_model.py data; do ln -sfn "$CARDIO_ROOT/$f" "$d/$f"; done
    ln -sfn "$CARDIO_MESH_DIR" "$d/meshes"
    ln -sfn "$CARDIO_WORK/results" "$d/results"
    cd "$d" || return 1
    export XDG_CACHE_HOME=$PWD/.xdg_cache
    echo "working directory: $PWD"
}

# yml_value <yml> <key>
yml_value() { sed -n "s/^$2 *: *\"\{0,1\}\([^\"]*\)\"\{0,1\} *\$/\1/p" "$1" | head -1; }

# run_case <yml> <gpu|cpu> <nranks> <nodes>
# GPU: one rank per GPU, CARDIO_GPUS_PER_NODE per node. CPU: one rank per core,
# CARDIO_CPU_RANKS_PER_NODE per node, filling the sockets in order (block:block).
# Log in <out_name>.log, statistics in <out_name>/stats-*.json. Never aborts the caller.
run_case() {
    local yml=$1 mode=$2 n=$3 nodes=$4 out log rc t0 per
    out=$(yml_value "$yml" out_name); log="$out.log"
    mkdir -p "$out"
    per=$(( (n + nodes - 1) / nodes ))
    echo "=== $(date '+%F %T') | $yml | $mode | $n ranks on $nodes node(s) | log: $log"
    t0=$SECONDS
    (
        cardio_activate "$mode" || exit 1
        if [ "$mode" = gpu ]; then
            srun -N "$nodes" -n "$n" --ntasks-per-node="$per" --gpus-per-task=1 \
                 --cpus-per-task="$CARDIO_CPUS_PER_GPU_RANK" --cpu-bind=cores \
                 python -u main.py "$yml"
        else
            srun -N "$nodes" -n "$n" --ntasks-per-node="$per" --cpus-per-task=1 \
                 --distribution=block:block --cpu-bind=cores \
                 python -u main.py "$yml"
        fi
    ) > "$log" 2>&1
    rc=$?
    echo "    exit code $rc after $((SECONDS - t0)) s | $(grep -m1 'Average KSP iterations' "$log" || echo 'no average reported')"
    return 0
}
