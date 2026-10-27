#!/usr/bin/env bash
# Submit the paper benchmarks as one Slurm job per (model, nodes, family).
# Run from the repository root on a login node:
#
#   bash benchmarks/submit.sh <tables|runs|smoke> [gpu|cpu|both] [--dry-run]
#
#   <tables>  comma-separated among table4 ... table9, or all
#             (table4/5: assembly; table6/7: Model A solver; table8/9: Model B solver; strong/weak)
#   <runs>    comma-separated single runs, e.g. B3_n4,B4_n8 (all the inputs of each run)
#   smoke     short installation check on the small mesh of data/ (1 GPU, 144 cores)
#
# A run used by several tables is submitted once, with the union of their
# configurations. Runs whose mesh is not in $CARDIO_MESH_DIR are skipped.
# Node mapping (benchmarks/env.sh): n paper nodes = n GPUs, CARDIO_GPUS_PER_NODE
# per node, or 144 n CPU ranks, CARDIO_CPU_RANKS_PER_NODE per node.
# Time limits: CARDIO_TIME_GPU / CARDIO_TIME_CPU (HH:MM:SS, default 04:00:00).
# Further sbatch options (e.g. account, partition, constraint): CARDIO_SBATCH_ARGS,
# or the SBATCH_* variables read by sbatch itself.

set -euo pipefail
cd "$(dirname "$0")/.."
source benchmarks/env.sh
cardio_site_env
what=${1:?usage: bash benchmarks/submit.sh <tables|runs|smoke> [gpu|cpu|both] [--dry-run]}
families=${2:-both}; dry=${3:-}

case $families in gpu|cpu) fams=$families ;; both) fams="gpu cpu" ;; *) echo "unknown family $families"; exit 2 ;; esac

# lines "<run> [<config suffix> ...]"; no suffix = every input of the run
case $what in
    smoke)          lines="smoke" ;;
    [AB][1-4]_n*)   lines=$(tr ',' '\n' <<< "$what") ;;
    *)              lines=$(python3 benchmarks/make_inputs.py --list "$what") ;;
esac

read -r -a sbatch_args <<< "${CARDIO_SBATCH_ARGS:-}"
while read -r run suffixes; do
    [ -n "$run" ] || continue
    if [ ! -d "benchmarks/inputs/$run" ]; then echo "unknown run $run (no benchmarks/inputs/$run)"; exit 2; fi
    if [ "$run" = smoke ]; then
        n=1
        : "${CARDIO_TIME_GPU:=00:30:00}" "${CARDIO_TIME_CPU:=00:30:00}"
    else
        n=${run##*_n}
        mesh=$(yml_value "benchmarks/inputs/$run/$(cd "benchmarks/inputs/$run" && ls | head -1)" mesh_file)
        if [ ! -f "$CARDIO_MESH_DIR/${mesh#meshes/}" ]; then
            echo "skip $run: $CARDIO_MESH_DIR/${mesh#meshes/} not found"; continue
        fi
    fi
    for fam in $fams; do
        configs=""
        for s in $suffixes; do
            # some configurations exist for one family only (e.g. Jacobi on GPU)
            [ -f "benchmarks/inputs/$run/${fam}_$s.yml" ] && configs+="${fam}_$s "
        done
        [ -n "$suffixes" ] && [ -z "$configs" ] && continue
        if [ "$fam" = gpu ]; then
            nodes=$(( (n + CARDIO_GPUS_PER_NODE - 1) / CARDIO_GPUS_PER_NODE ))
            extra=(--gpus-per-node="$CARDIO_GPUS_PER_NODE" --time="${CARDIO_TIME_GPU:-04:00:00}")
        else
            nodes=$(( (144 * n + CARDIO_CPU_RANKS_PER_NODE - 1) / CARDIO_CPU_RANKS_PER_NODE ))
            extra=(--time="${CARDIO_TIME_CPU:-04:00:00}")
        fi
        # shellcheck disable=SC2054  # commas belong to --export
        cmd=(sbatch --nodes="$nodes" --job-name="$run-$fam" "${extra[@]}" ${sbatch_args[@]+"${sbatch_args[@]}"}
             --export=ALL,CARDIO_RUN="$run",CARDIO_FAMILY="$fam",CARDIO_CONFIGS="${configs% }" benchmarks/run.sbatch)
        echo "$run $fam: $nodes node(s), configs: ${configs:-all} -> ${cmd[*]}"
        [ "$dry" = --dry-run ] || "${cmd[@]}"
    done
done <<< "$lines"
