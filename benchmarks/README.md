# Paper benchmarks: assembly and linear solver scaling (CPU vs GPU)

Scripts to install the software stack and rerun the scaling experiments of the paper on a Slurm cluster:

| Table | Content | Meshes | Nodes |
|---|---|---|---|
| 4 | right-hand side assembly, strong scaling (s per time step) | A1, B1 | 1, 2, 4, 8 |
| 5 | right-hand side assembly, weak scaling (MDoFs/s per node) | A1-A4, B1-B4 | 1, 2, 4, 8 |
| 8 | AMG-CG linear solve, strong scaling (s per time step, iterations) | B1, B2 | 1, 2, 4, 8 |
| 9 | AMG-CG and AMG-GMRES linear solve, weak scaling (MDoFs/s per node, iterations) | B1-B4 | 1, 2, 4, 8 |

Every run executes `main.py` for 100 time steps and records both the assembly and the solve times, so the
assembly tables come from the AMG-CG runs of the same campaign.

> **Note on portability.** The job scripts use Slurm (`sbatch`, `srun`) and the paper's notion of a node:
> a *GPU node* is one NVIDIA GH200 driven by one MPI rank, a *CPU node* is 144 Grace cores with one MPI rank
> per core. The defaults reproduce this layout, and the mapping can be changed in `env.sh`.
> On other hardware, MPI libraries, interconnects or Slurm configurations, the timings can differ
> significantly from the published ones. The iteration counts should be close to the published ones.

## Contents

| File | Purpose |
|---|---|
| `site_env.example.sh` | template of `site_env.sh`: modules (compilers, MPI, CUDA), Slurm account and partition |
| `env.sh` | versions, paths, node mapping; `cardio_activate gpu\|cpu`, `run_case` (srun) |
| `build_env.sh` | one-time build of the stack (batch job or plain script) |
| `make_inputs.py` | generates `inputs/<model>_n<nodes>/<config>.yml` (already generated and versioned) |
| `inputs/` | input files of every run, plus `inputs/smoke/` for a short installation check |
| `run.sbatch` | one run (model, node count, family); submitted by `submit.sh` |
| `submit.sh` | submits the runs of the selected tables, with the right number of nodes |
| `collect_tables.py` | builds Tables 4, 5, 8 and 9 (Markdown, `tables.tex`, `summary.csv`) |

## Software stack

| Component | Version |
|---|---|
| Python | 3.12 (uv-managed) |
| PETSc / petsc4py | 3.23.7, with hypre 2.32.0, MUMPS, PT-Scotch (downloaded by PETSc) |
| DOLFINx / Basix / FFCx / UFL | 0.9.0 / 0.9.0 / 0.9.0 / 2024.2.0 |
| cuDOLFINx | 0.9.0 (GPU variant only) |
| multiphenicsx | 0.3.9 |
| CUDA | 12.x, at most 12.9 (PETSc 3.23); compute capability 9.0 by default |

Two variants are installed side by side in `$CARDIO_PREFIX` (default `$HOME/cardioEMI-env`):
- `gpu`: CUDA-enabled PETSc and cuDOLFINx, used by the GPU runs;
- `cpu`: CPU-only PETSc, so that hypre runs on the host, used by the CPU runs.

With cuDOLFINx 0.9 the GPU path of `main.py` applies the homogeneous Dirichlet point of the GMRES runs by hand
(zeroed rows and columns), since `apply_lifting_block` is not available in that version.

## Step by step

All commands are run from the repository root.

1. **Site environment.** Copy `benchmarks/site_env.example.sh` to `benchmarks/site_env.sh` and load there
   the compilers, MPI (with `mpicc`, `mpicxx`, `mpif90`) and the CUDA toolkit of your cluster. The Slurm account and
   partition can be set there too (`SBATCH_ACCOUNT`, `SBATCH_PARTITION`).
   Any `CARDIO_*` variable of `env.sh` can also be overridden there, e.g.:
   - `CARDIO_CUDA_ARCH`;
   - `CARDIO_GPUS_PER_NODE`, `CARDIO_CPU_RANKS_PER_NODE`, `CARDIO_CPUS_PER_GPU_RANK`;
   - `CARDIO_PREFIX`, `CARDIO_MESH_DIR`.

2. **Build the stack** (2-3 hours; the sources are downloaded, so the node needs internet access):
   ```bash
   sbatch benchmarks/build_env.sh        # or: bash benchmarks/build_env.sh
   ```
   The log must end with `IMPORT CHECK OK gpu` and `IMPORT CHECK OK cpu`. The GPU check is skipped on a node
   without GPUs. A failed build can be restarted: completed steps are skipped.
   Set `FORCE_<STEP>=1` to redo one, e.g. `FORCE_PETSC_GPU=1`.

3. **Meshes.** The inputs read the meshes from `$CARDIO_MESH_DIR` (default `meshes/`):
   - Model A (`A1`-`A4`): generated with `bash emigrid_mesh/generate_model_A.sh` (see the header of the script
     for its requirements; A3 and A4 need a large-memory node);
   - Model B (`millie-5_clean`, `millie-12_clean`, `millie-22_clean`, `millie-47_clean`, i.e. B1-B4, cleaned of
     nearly degenerate tetrahedra): not distributed with the repository, available from the authors on request.
     Each mesh consists of `.xdmf`, `.h5` and `.pickle` files.

4. **Smoke test** (a few minutes, small mesh of `data/`):
   ```bash
   bash benchmarks/submit.sh smoke both
   ```
   Every run in `cardio_run-smoke-*.out` must report `exit code 0` and an average number of iterations. GPU and CPU
   runs of the same configuration should report nearly the same iterations.

5. **Benchmarks.** Check the jobs first, then submit them:
   ```bash
   bash benchmarks/submit.sh all both --dry-run
   bash benchmarks/submit.sh all both
   ```
   - `all` can be replaced by a comma-separated list of tables (`table4,table5,table8,table9`), or by single
     runs (`B3_n4,B4_n8`); `both` can be replaced by `gpu` or `cpu`.
   - A run shared by several tables is submitted once. Runs whose mesh is missing are skipped.
   - Time limits: `CARDIO_TIME_GPU` and `CARDIO_TIME_CPU` (default 4 h).
   - Extra `sbatch` options, such as a node constraint: `CARDIO_SBATCH_ARGS`.

   Each job writes `cardio_run-<run>-<family>-<jobid>.out` in the submission directory. The output of `main.py` goes to
   `benchmarks/results/<run>/<config>.log`, its timings to `benchmarks/results/<run>/<config>/stats-*.json`.

6. **Tables**:
   ```bash
   python3 benchmarks/collect_tables.py
   ```
   Prints the four tables and writes `benchmarks/results/tables.tex` and `benchmarks/results/summary.csv`.

## Solver settings

All runs use a relative tolerance of 1e-7 on the unpreconditioned residual and at most 2000 iterations per time step.
CG runs use homogeneous Neumann conditions with the constant nullspace; GMRES runs use one Dirichlet point.

- **Model A:** each architecture keeps its default coarsening and interpolation (HMIS on CPUs, PMIS on GPUs), with:
  - the l1-scaled SOR/Jacobi smoother;
  - strong threshold 0.25;
  - no aggressive coarsening.
- **Model B:** the same BoomerAMG on CPUs and GPUs:
  - PMIS coarsening and ext+i interpolation;
  - the Chebyshev smoother on every level, including the coarsest;
  - no C/F-ordered relaxation and no aggressive coarsening;
  - strong threshold 0.25 and minimum coarse-grid size 2.

  The hypre statistics are printed in the logs, so the CPU and GPU hierarchies can be compared.

## Expected behaviour and troubleshooting

- **GPU memory.** With cuDOLFINx 0.9 a GPU needs about 16 KB per DoF, so a 96 GB GH200 holds about five million DoFs.
  B2 does not fit on one GPU: the GPU entry of `B2_n1` in Table 8 is expected to fail (OOM). `run.sbatch` prints the peak
  GPU memory of every run.
- **GPU-aware MPI.** `CARDIO_GPU_AWARE_MPI=auto` (default) leaves the choice to PETSc. With Cray MPICH it also enables
  the MPI GPU support when PETSc links it. Set `CARDIO_GPU_AWARE_MPI=0` if the MPI library is not CUDA-aware, or `1` to force it.
- **HDF5 not found or not parallel when building DOLFINx.** By default PETSc downloads and builds HDF5. To use a parallel HDF5
  of the system instead, set `CARDIO_HDF5_DIR`, remove the markers of the steps that depend on PETSc
  (`$CARDIO_PREFIX/.done-{petsc,petsc4py,dolfinx,multiphenicsx}_<v>`, plus `.done-cudolfinx` for `gpu`) and restart the build.
- **Segmentation fault on `import mpi4py`/`petsc4py`** with Cray MPICH: `cardio_activate` disables the libhugetlbfs
  remapping (`HUGETLB_ELFMAP=no`, `HUGETLB_MORECORE=no`).
- **`TimeoutError: JIT compilation timed out`.** A previous job died while compiling forms. Remove
  `benchmarks/runs/*/.cache` and `benchmarks/runs/*/.xdg_cache`.
- **A run hangs before its first output line.** No module on the job path may import matplotlib: building its font cache
  can hang on parallel file systems. `utils.py` imports it lazily.
