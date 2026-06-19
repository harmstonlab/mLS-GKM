#!/bin/bash
#
# run_benchmarks.sh — submit the speed-benchmark jobs NUM_RPTS times.
#
# Run this from inside examples/SPEEDUP/ENCODE_DATA/ : the .job scripts use
# relative paths (FASTAs/..., Models/..., ../../../bin/...).
#
# For each repeat it submits all four benchmark scripts. The multithreaded
# scripts are expanded into one job per thread count, with --cpus-per-task set
# equal to the thread count (the 2-thread run reserves 2 CPUs, the 128-thread
# run reserves 128). Under --exclusive the whole node is allocated regardless,
# so this only matters for non-exclusive test runs, where many low-thread jobs
# can then pack onto a node and run concurrently.
#
# Configure with the variables below, or override at submission time:
#   ./run_benchmarks.sh -n 5                 # 5 repeats, exclusive (publication runs)
#   ./run_benchmarks.sh -n 1 --no-exclusive  # quick test runs, pack many jobs per node


# Defaults (overridable by env or the CLI flags below)
NUM_RPTS=${NUM_RPTS:-5}          # number of repeats to submit
EXCLUSIVE=${EXCLUSIVE:-true}   # best for benchmarking; set false for quick testing

# Extra sbatch flags passed to every submission. Leave empty by
# default. For the most consistent benchmark timings, pin all jobs to one
# machine and/or a dedicated reservation, e.g.:
#   EXTRA_ARGS="--nodelist=node01 --reservation=my_benchmark_res"
EXTRA_ARGS=${EXTRA_ARGS:-""}

# {er-script thread sweeps 

THREADS_Speed_EVAL_Predict_LS_GKM="1 4 16"
THREADS_Speed_EVAL_Predict_mLS_GKM="1 2 4 8 16 32 64 128"
THREADS_Speed_EVAL_Explain_mLS_GKM="1 2 4 8 16 32 64 128"
THREADS_Speed_EVAL_Explain_LS_GKM="1"

SCRIPTS="Speed_EVAL_Predict_LS-GKM \
         Speed_EVAL_Predict_mLS-GKM \
         Speed_EVAL_Explain_LS-GKM \
         Speed_EVAL_Explain_mLS-GKM"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-n NUM_RPTS] [--exclusive | --no-exclusive] [-h]

  -n, --num-rpts N   Number of repeats to submit (default: ${NUM_RPTS}).
      --no-exclusive Share nodes; per-job CPUs = thread count (for quick testing).
                     Nodes are reserved exclusively unless this is given.
  -h, --help         Show this help.
EOF
}

# ---- Parse CLI flags --------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        -n|--num-rpts)   NUM_RPTS=$2; shift 2 ;;
        --no-exclusive)  EXCLUSIVE=false; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

EXCL_ARG=""
[ "$EXCLUSIVE" = true ] && EXCL_ARG="--exclusive"

# Look up a script's thread sweep via its sanitised variable name.
threads_for() {
    local var="THREADS_$(echo "$1" | tr '-' '_')"
    printf '%s' "${!var-}"
}

# Submit
for rpt in $(seq 1 "$NUM_RPTS"); do
    for script in $SCRIPTS; do
        for t in $(threads_for "$script"); do
            echo "Submitting ${script}  RPT=${rpt}  THREADS=${t}  cpus=${t}  exclusive=${EXCLUSIVE}  extra_args=${EXTRA_ARGS}"
            sbatch \
                --job-name="${script}_RPT_${rpt}_T${t}" \
                --cpus-per-task="$t" \
                $EXCL_ARG \
                $EXTRA_ARGS \
                --export=ALL,RPT="$rpt",THREADS="$t" \
                "${script}.job"
        done
    done
done
