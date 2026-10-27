# shellcheck shell=bash
# Site environment of the benchmarks: copy to benchmarks/site_env.sh (not
# versioned) and adapt it to your cluster. It is sourced by build_env.sh and by
# every run (cardio_activate), so it must be safe to source more than once.
#
# The build needs, on the PATH:
#   - MPI compiler wrappers mpicc, mpicxx, mpif90 (one MPI library for everything);
#   - the CUDA toolkit (nvcc), version 12.x up to 12.9, for the gpu variant;
#   - git, curl, make, a C/C++/Fortran compiler with C++20 support (e.g. gcc >= 12);
#   - python3 (>= 3.9, with venv) to bootstrap the build tools.
# CMake, Ninja and the Python interpreter of the stack are installed by build_env.sh.
#
# Optional, to reuse libraries of the system instead of building them:
#   CARDIO_BLASLAPACK_DIR  BLAS/LAPACK prefix          (default: PETSc downloads f2cblaslapack)
#   CARDIO_HDF5_DIR        parallel HDF5 prefix        (default: PETSc downloads HDF5)
#   CARDIO_BOOST_DIR       Boost prefix with Boost.Timer (default: built from source)

# Example with environment modules:
# module purge
# module load gcc/13 openmpi/5 cuda/12.6
# export CARDIO_BLASLAPACK_DIR=$OPENBLAS_ROOT

# Slurm account and partition of the jobs (read by sbatch):
# export SBATCH_ACCOUNT=my_account
# export SBATCH_PARTITION=gpu
