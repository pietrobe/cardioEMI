#!/usr/bin/env bash
#SBATCH --job-name=cardio_stream
#SBATCH --output=cardio_stream-%j.out
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#
# STREAM-like memory bandwidth test on one GPU node, used in the paper to quote the
# ratio between the GPU memory bandwidth (HBM) and the CPU memory bandwidth (LPDDR5X
# on a GH200). Self-contained: the two small benchmarks are written below and
# compiled in the job. Kernels are the STREAM ones (Copy, Scale, Add, Triad),
# 8-byte elements, best of ntimes-1 repetitions (the first one is discarded),
# GB/s = 1e9 bytes/s. These are not McCalpin's reference STREAM codes.
#
#   gpu_hbm        GPU kernels on device memory, 1 rank per GPU, all GPUs at once
#   gpu_host_c2c   GPU kernels on pinned host memory of the local CPU (NVLink-C2C on GH200)
#   cpu_lpddr_4s   OpenMP STREAM on host memory, CARDIO_CPUS_PER_GPU_RANK threads per rank,
#                  one rank per GPU, all at once
#   cpu_lpddr_1s   the same with one rank only (no contention from the others)
#
# Submit from the repository root (a few minutes of one node):
#   sbatch --gpus-per-node=<GPUs per node> benchmarks/stream.sbatch
# Compilers: CC (default gcc) with OpenMP and nvcc, from benchmarks/site_env.sh.
# STREAM_N (elements per array, default 2^28 = 2 GiB) and STREAM_NTIMES (default 50).

cd "$SLURM_SUBMIT_DIR" || exit 1
source benchmarks/env.sh               # only defines variables and functions
cardio_site_env
D=$CARDIO_WORK/runs/stream
mkdir -p "$D" && cd "$D" || exit 1
N=${STREAM_N:-268435456}               # elements per array: 2^28 doubles = 2 GiB per array
NTIMES=${STREAM_NTIMES:-50}
NGPU=$(nvidia-smi -L | wc -l)

echo "##### node: $SLURM_JOB_NODELIST   $(date '+%F %T')   GPUs: $NGPU"
nvidia-smi --query-gpu=index,name,memory.total,clocks.max.memory,driver_version --format=csv
nvidia-smi topo -m | head -6
echo "##### CPU / NUMA"
lscpu | grep -E 'Model name|^CPU\(s\)|NUMA node[0-9]+ CPU' || true
numactl -H 2> /dev/null | head -12 || true

cat > stream_cpu.c <<'EOF'
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <unistd.h>
#include <sched.h>
#include <omp.h>

int main(int argc, char **argv) {
    size_t n = argc > 1 ? strtoull(argv[1], NULL, 10) : ((size_t)1 << 28);
    int ntimes = argc > 2 ? atoi(argv[2]) : 50;
    const char *tag = argc > 3 ? argv[3] : "cpu";
    const char *rank = getenv("SLURM_PROCID");
    char host[64];
    gethostname(host, sizeof host);
    if (!rank) rank = "0";
    if (ntimes < 2) ntimes = 2;
    double *a, *b, *c;
    if (posix_memalign((void **)&a, 4096, n * sizeof(double)) ||
        posix_memalign((void **)&b, 4096, n * sizeof(double)) ||
        posix_memalign((void **)&c, 4096, n * sizeof(double))) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    const double s = 3.0;
    /* first touch by the threads that will use the data: pages land on the local CPU socket */
#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < n; j++) { a[j] = 1.0; b[j] = 2.0; c[j] = 0.0; }

    static const char *name[4] = {"Copy", "Scale", "Add", "Triad"};
    const double bytes[4] = {2.0 * 8 * n, 2.0 * 8 * n, 3.0 * 8 * n, 3.0 * 8 * n};
    double tmin[4] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX}, tsum[4] = {0, 0, 0, 0};
    for (int k = 0; k < ntimes; k++) {
        double t[5];
        t[0] = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < n; j++) c[j] = a[j];
        t[1] = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < n; j++) b[j] = s * c[j];
        t[2] = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < n; j++) c[j] = a[j] + b[j];
        t[3] = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < n; j++) a[j] = b[j] + s * c[j];
        t[4] = omp_get_wtime();
        if (k > 0)
            for (int f = 0; f < 4; f++) {
                double dt = t[f + 1] - t[f];
                if (dt < tmin[f]) tmin[f] = dt;
                tsum[f] += dt;
            }
    }
    /* check against the scalar recurrence */
    double aj = 1.0, bj = 2.0, cj = 0.0;
    for (int k = 0; k < ntimes; k++) { cj = aj; bj = s * cj; cj = aj + bj; aj = bj + s * cj; }
    double err = fmax(fabs(a[0] - aj), fmax(fabs(a[n / 2] - aj), fabs(a[n - 1] - aj))) / fabs(aj);
    printf("INFO %s rank %s host %s threads %d first_cpu %d n %zu check_rel_err %.1e %s\n",
           tag, rank, host, omp_get_max_threads(), sched_getcpu(), n, err, err < 1e-12 ? "OK" : "FAILED");
    for (int f = 0; f < 4; f++)
        printf("RESULT %s %s %s %.1f %.1f\n", tag, name[f], rank,
               bytes[f] / tmin[f] / 1e9, bytes[f] / (tsum[f] / (ntimes - 1)) / 1e9);
    return err < 1e-12 ? 0 : 2;
}
EOF

cat > stream_gpu.cu <<'EOF'
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#define LOOP(body) for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; \
                        i += (size_t)gridDim.x * blockDim.x) { body; }
__global__ void k_init(double *a, double *b, double *c, size_t n) { LOOP(a[i] = 1.0; b[i] = 2.0; c[i] = 0.0) }
__global__ void k_copy(const double *a, double *c, size_t n) { LOOP(c[i] = a[i]) }
__global__ void k_scale(double *b, const double *c, double s, size_t n) { LOOP(b[i] = s * c[i]) }
__global__ void k_add(const double *a, const double *b, double *c, size_t n) { LOOP(c[i] = a[i] + b[i]) }
__global__ void k_triad(double *a, const double *b, const double *c, double s, size_t n) { LOOP(a[i] = b[i] + s * c[i]) }

int main(int argc, char **argv) {
    size_t n = argc > 1 ? strtoull(argv[1], NULL, 10) : ((size_t)1 << 28);
    int ntimes = argc > 2 ? atoi(argv[2]) : 50;
    const char *mode = argc > 3 ? argv[3] : "device";      /* device (HBM) or host (pinned LPDDR5X over C2C) */
    const char *tag = argc > 4 ? argv[4] : "gpu";
    const char *rank = getenv("SLURM_PROCID");
    char host[64];
    gethostname(host, sizeof host);
    if (!rank) rank = "0";
    if (ntimes < 2) ntimes = 2;
    const bool onhost = strcmp(mode, "host") == 0;

    CK(cudaSetDevice(0));                                   /* srun --gpus-per-task=1: one visible GPU */
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    int clk_khz = 0, bus = 0, sms = p.multiProcessorCount;
    CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrMemoryClockRate, 0));
    CK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0));

    double *a, *b, *c;
    size_t sz = n * sizeof(double);
    if (onhost) { CK(cudaMallocHost(&a, sz)); CK(cudaMallocHost(&b, sz)); CK(cudaMallocHost(&c, sz)); }
    else        { CK(cudaMalloc(&a, sz));     CK(cudaMalloc(&b, sz));     CK(cudaMalloc(&c, sz)); }
    const int th = 256;
    const int bl = sms * 16;
    const double s = 3.0;
    k_init<<<bl, th>>>(a, b, c, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    cudaEvent_t e[5];
    for (int i = 0; i < 5; i++) CK(cudaEventCreate(&e[i]));
    static const char *name[4] = {"Copy", "Scale", "Add", "Triad"};
    const double bytes[4] = {2.0 * 8 * n, 2.0 * 8 * n, 3.0 * 8 * n, 3.0 * 8 * n};
    double tmin[4] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX}, tsum[4] = {0, 0, 0, 0};
    for (int k = 0; k < ntimes; k++) {
        CK(cudaEventRecord(e[0]));
        k_copy<<<bl, th>>>(a, c, n);        CK(cudaEventRecord(e[1]));
        k_scale<<<bl, th>>>(b, c, s, n);    CK(cudaEventRecord(e[2]));
        k_add<<<bl, th>>>(a, b, c, n);      CK(cudaEventRecord(e[3]));
        k_triad<<<bl, th>>>(a, b, c, s, n); CK(cudaEventRecord(e[4]));
        CK(cudaGetLastError());
        CK(cudaEventSynchronize(e[4]));
        if (k > 0)
            for (int f = 0; f < 4; f++) {
                float ms;
                CK(cudaEventElapsedTime(&ms, e[f], e[f + 1]));
                double dt = ms * 1e-3;
                if (dt < tmin[f]) tmin[f] = dt;
                tsum[f] += dt;
            }
    }
    /* check the first elements against the scalar recurrence */
    double aj = 1.0, bj = 2.0, cj = 0.0;
    for (int k = 0; k < ntimes; k++) { cj = aj; bj = s * cj; cj = aj + bj; aj = bj + s * cj; }
    size_t m = n < ((size_t)1 << 20) ? n : ((size_t)1 << 20);
    std::vector<double> h(m);
    CK(cudaMemcpy(h.data(), a, m * sizeof(double), cudaMemcpyDefault));
    double err = 0;
    for (size_t i = 0; i < m; i++) err = fmax(err, fabs(h[i] - aj) / fabs(aj));
    printf("INFO %s rank %s host %s gpu '%s' mem %.1f GiB sms %d mem_bus %d bit mem_clk %d kHz "
           "computed_peak %.0f GB/s mode %s n %zu check_rel_err %.1e %s\n",
           tag, rank, host, p.name, p.totalGlobalMem / 1073741824.0, sms, bus, clk_khz,
           2.0 * clk_khz * 1e3 * bus / 8 / 1e9, mode, n, err, err < 1e-12 ? "OK" : "FAILED");
    for (int f = 0; f < 4; f++)
        printf("RESULT %s %s %s %.1f %.1f\n", tag, name[f], rank,
               bytes[f] / tmin[f] / 1e9, bytes[f] / (tsum[f] / (ntimes - 1)) / 1e9);
    return err < 1e-12 ? 0 : 2;
}
EOF

echo "##### compile"
case $(uname -m) in aarch64) native=-mcpu=native ;; *) native=-march=native ;; esac
"${CC:-gcc}" -O3 "$native" -fopenmp -o stream_cpu stream_cpu.c -lm || exit 1
nvcc -O3 -arch=sm_"$CARDIO_CUDA_ARCH" -o stream_gpu stream_gpu.cu || exit 1

ALL=$D/stream-$SLURM_JOB_ID.log
: > "$ALL"
GPU_STEP=(srun -N1 -n "$NGPU" --ntasks-per-node="$NGPU" --gpus-per-task=1 --cpus-per-task="$CARDIO_CPUS_PER_GPU_RANK" --cpu-bind=cores)

echo "##### GPU on device memory, $NGPU GPUs at once"
"${GPU_STEP[@]}" ./stream_gpu "$N" "$NTIMES" device gpu_hbm 2>&1 | sort -s -k1,1 | tee -a "$ALL"

echo "##### GPU on pinned host memory of the local CPU, $NGPU GPUs at once"
"${GPU_STEP[@]}" ./stream_gpu "$N" "$NTIMES" host gpu_host_c2c 2>&1 | sort -s -k1,1 | tee -a "$ALL"

export OMP_NUM_THREADS=$CARDIO_CPUS_PER_GPU_RANK OMP_PROC_BIND=close OMP_PLACES=cores
echo "##### CPU STREAM on host memory, $OMP_NUM_THREADS threads per rank, $NGPU ranks at once"
srun -N1 -n "$NGPU" --ntasks-per-node="$NGPU" --cpus-per-task="$CARDIO_CPUS_PER_GPU_RANK" --cpu-bind=cores \
     ./stream_cpu "$N" "$NTIMES" cpu_lpddr_4s 2>&1 | sort -s -k1,1 | tee -a "$ALL"

echo "##### CPU STREAM on host memory, one rank alone"
srun -N1 -n1 --cpus-per-task="$CARDIO_CPUS_PER_GPU_RANK" --cpu-bind=cores \
     ./stream_cpu "$N" "$NTIMES" cpu_lpddr_1s 2>&1 | sort -s -k1,1 | tee -a "$ALL"

echo "##### summary: best GB/s over ranks, per test and kernel"
awk '$1 == "RESULT" { k = $2 " " $3; v = $5 + 0; n[k]++; s[k] += v
                      if (!(k in mn) || v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v }
     END { for (k in n) printf "%-26s ranks %d   min %.0f   mean %.0f   max %.0f\n", k, n[k], mn[k], s[k] / n[k], mx[k] }' "$ALL" | sort
echo "##### checks"
grep -c 'check_rel_err.* OK' "$ALL" | sed 's/^/results with OK check: /'
grep 'FAILED' "$ALL" || echo "no failed checks"

echo "##### Triad ratios (mean of the best Triad GB/s)"
triad() { awk -v t="$1" '$1 == "RESULT" && $2 == t && $3 == "Triad" { s += $5; n++ } END { if (n) printf "%.1f", s / n }' "$ALL"; }
hbm=$(triad gpu_hbm); c2c=$(triad gpu_host_c2c); c4=$(triad cpu_lpddr_4s); c1=$(triad cpu_lpddr_1s)
echo "HBM (GPU)            : $hbm GB/s"
echo "LPDDR5X from GPU/C2C : $c2c GB/s"
echo "LPDDR5X from CPU 4s  : $c4 GB/s"
echo "LPDDR5X from CPU 1s  : $c1 GB/s"
[ -n "$hbm" ] && [ -n "$c1" ] && awk -v h="$hbm" -v c="$c1" 'BEGIN { printf "HBM / LPDDR5X (CPU, one rank) : %.1fx\n", h / c }'
[ -n "$hbm" ] && [ -n "$c2c" ] && awk -v h="$hbm" -v c="$c2c" 'BEGIN { printf "HBM / LPDDR5X (GPU over C2C)  : %.1fx\n", h / c }'
echo "##### done $(date '+%F %T')   full log: $ALL"
