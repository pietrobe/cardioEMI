#!/usr/bin/env bash
#SBATCH --job-name=cardio_run
#SBATCH --output=cardio_run-%x-%j.out
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#
# One benchmark run: one model and node count, one family (gpu or cpu), one or
# more configurations, executed one after the other in the same allocation.
#   CARDIO_RUN     = <model>_n<nodes>, e.g. B2_n4   (inputs in benchmarks/inputs/<run>/)
#   CARDIO_FAMILY  = gpu (one rank per GPU) | cpu (144 ranks per paper node)
#   CARDIO_CONFIGS = configurations to run, e.g. "gpu_cg_amg gpu_gmres_amg"
#                    (default: all the inputs of the run for this family)
# Nodes, GPUs and time limit are set by benchmarks/submit.sh.

cd "$SLURM_SUBMIT_DIR" || exit 1
source benchmarks/env.sh
: "${CARDIO_RUN:?}" "${CARDIO_FAMILY:?}"
n=${CARDIO_RUN##*_n}
[ "$CARDIO_RUN" = smoke ] && n=1
I=$CARDIO_ROOT/benchmarks/inputs/$CARDIO_RUN
configs=${CARDIO_CONFIGS:-$(cd "$I" && ls "$CARDIO_FAMILY"_*.yml | sed 's/\.yml$//' | tr '\n' ' ')}
enter_rundir "$CARDIO_RUN-$CARDIO_FAMILY"

echo "##### $CARDIO_RUN ($CARDIO_FAMILY), paper nodes n=$n, $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST"
echo "configs: $configs"
if [ "$CARDIO_FAMILY" = gpu ]; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
    # GPU memory of the first node, sampled every 10 s, peak printed per run
    MEM=gpumem-$SLURM_JOB_ID.csv
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -l 10 > "$MEM" &
    SMI=$!
    for c in $configs; do
        start=$(wc -l < "$MEM")
        run_case "$I/$c.yml" gpu "$n" "$SLURM_JOB_NUM_NODES"
        tail -n +"$((start + 1))" "$MEM" | awk -F', *' '$2 > m {m = $2; t = $3} END {printf "    peak GPU memory (first node): %d of %d MiB\n", m, t}'
    done
    kill "$SMI" 2> /dev/null
else
    for c in $configs; do
        run_case "$I/$c.yml" cpu $((144 * n)) "$SLURM_JOB_NUM_NODES"
    done
fi
echo "##### done $(date '+%F %T')"
