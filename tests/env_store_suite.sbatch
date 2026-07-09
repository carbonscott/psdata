#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=lcls:prjcwang31
#SBATCH --qos=normal
#SBATCH --job-name=env-store-suite
#SBATCH --output=/sdf/data/lcls/ds/prj/prjcwang31/results/fun/psdata/env-store-suite-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16g
#SBATCH --time=01:00:00
#
# The psana-oracle regression suite INCLUDING the new env-store oracle test
# (tests/test_envstore_us013.py), which was registered in run_tests.sh after
# job env-store-verify had already read the 11-file list.  Expect a final
# tally of "12 passed, 0 failed".
#
# run_tests.sh PREPENDS src/ to PYTHONPATH; it must never clobber it, or
# psconda's psana becomes un-importable and the oracle tests print SKIPPED and
# exit 0 -- a false pass.  The `psana importable?` probe below makes that
# failure mode loud instead of silent.
set -uo pipefail
echo "host=$(hostname) jobid=${SLURM_JOB_ID:-?} start=$(date -Is)"
SW=/sdf/data/lcls/ds/prj/prjcwang31/results/software/psdata

(
  set +u   # psconda's conda activate scripts reference unbound vars (ADDR2LINE)
  source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
  echo "===== psana importable? (guards against a false SKIPPED pass) ====="
  python3 -c 'import psana; print("psana OK:", psana.__file__)' || {
    echo "FATAL: psana not importable -- oracle tests would falsely SKIP"; exit 90; }
  cd "$SW"
  echo "===== psana-oracle regression suite (12 files) ====="
  PSDATA_SKIP_SLOW=1 bash run_tests.sh
)
suite_rc=$?
echo "suite_rc=$suite_rc"
echo "end=$(date -Is)"
if [ "$suite_rc" -eq 0 ]; then echo "OVERALL: PASS"; exit 0; fi
echo "OVERALL: FAIL"
exit 1
