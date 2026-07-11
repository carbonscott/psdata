#!/bin/bash
# tests/bench_index.sbatch -- recorded milano run of the psdata-vs-psana
# random-access benchmark (docs/benchmark-plan.md Section 5).
#
# Timing is valid ONLY on a dedicated milano compute node: bench_index.py's
# require_batch_node() refuses to emit any S / latency / build number off a
# batch node. --exclusive is load-bearing for cold S (no neighbour pollutes the
# Weka client cache; a freshly scheduled node starts cold for the dataset).
#
# One detector per node (two nodes in parallel halves wall-clock). Select the
# detector at submit time, e.g.:
#   sbatch --job-name=bench_jungfrau --export=ALL,BENCH_DET=jungfrau tests/bench_index.sbatch
#   sbatch --job-name=bench_epixquad --export=ALL,BENCH_DET=epixquad tests/bench_index.sbatch
#
#SBATCH --partition=milano
#SBATCH --account=lcls:prjdat21
#SBATCH --qos=normal
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=bench_index
#SBATCH --output=/sdf/data/lcls/ds/prj/prjcwang31/results/_bench_runs/bench_index-%x-%j.out

echo "=== node: $(hostname)  job: $SLURM_JOB_ID  det: ${BENCH_DET:-all}  $(date) ==="
source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
echo "which python: $(which python)"

REPO=/sdf/data/lcls/ds/prj/prjcwang31/results/software
export PYTHONPATH=$REPO/pscalib/src:$REPO/psdata/src:${PYTHONPATH:-}

python $REPO/psdata/tests/bench_index.py
rc=$?
echo "exit: $rc  end: $(date)"
exit $rc        # propagate python's exit code so sacct ExitCode reflects the
                # fail-closed gate (exit != 0 == correctness gate blocked the clock)
