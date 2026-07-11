#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=lcls:prjdat21
#SBATCH --qos=normal
#SBATCH --job-name=env-store-suite
#SBATCH --output=/sdf/data/lcls/ds/prj/prjcwang31/results/fun/psdata/env-store-suite-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16g
#SBATCH --time=02:00:00
#
# The full psana-oracle regression suite (run_tests.sh's default list).  It runs
# EVERY registered test file -- no test and no check is skipped by this job:
#   * the PSDATA_SKIP_SLOW escape hatch is GONE (GATE-02 removed the only reader
#     of it), so the slow gated-forward regression really runs here;
#   * run_tests.sh now counts every skipped CHECK and exits nonzero on any skip
#     that is not justified in tests/skips_allowed.txt (HYG-03).  Expect a tally
#     of "N passed, 0 failed, S skipped", with S covering only the allowlisted
#     torch skips (torch is an optional extra, absent from psconda).
#
# The multi-chunk psana oracle needs its ground truth captured ONCE before this
# job can come back clean (otherwise it emits an unjustified skip and the suite
# fails, by design):
#     source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
#     cd $SW && PYTHONPATH=src python3 tools/gt_capture.py --out-dir .
#
# run_tests.sh PREPENDS src/ to PYTHONPATH; it must never clobber it, or
# psconda's psana becomes un-importable and every oracle test degrades to a SKIP.
# The suite now FAILS on those skips rather than scoring them as passes, but the
# `psana importable?` probe below still fails the job fast, with a clear reason.
set -uo pipefail
echo "host=$(hostname) jobid=${SLURM_JOB_ID:-?} start=$(date -Is)"
SW=/sdf/data/lcls/ds/prj/prjcwang31/results/software/psdata

(
  set +u   # psconda's conda activate scripts reference unbound vars (ADDR2LINE)
  source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
  echo "===== psana importable? (fail fast: no psana => every oracle SKIPs) ====="
  python3 -c 'import psana; print("psana OK:", psana.__file__)' || {
    echo "FATAL: psana not importable -- every oracle test would SKIP"; exit 90; }
  cd "$SW"
  echo "===== psana-oracle regression suite (full default list) ====="
  bash run_tests.sh
)
suite_rc=$?
echo "suite_rc=$suite_rc"
echo "end=$(date -Is)"
if [ "$suite_rc" -eq 0 ]; then echo "OVERALL: PASS"; exit 0; fi
echo "OVERALL: FAIL"
exit 1
