#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=lcls:prjdat21
#SBATCH --qos=normal
#SBATCH --job-name=env-store-verify
#SBATCH --output=/sdf/data/lcls/ds/prj/prjcwang31/results/fun/psdata/env-store-verify-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16g
#SBATCH --time=03:00:00
#
# At-scale no-regression verification for psdata's env store
# (branch feature/env-store, commit 15c1335). Two stages on a milano compute
# node:
#   [1] full psana-oracle regression suite (run_tests.sh, psconda) -- proves
#       parse_configure's new TID_PARENT recursion (scan detector now
#       discovered) and the additive RunIndex.env_records field do NOT change
#       which events Run.events() yields.
#   [2] at-scale clamp/persistence proof (verify_cache.py, editable uv venv
#       pointing at <repo>/src) -- default bigdata index == SMD exactly,
#       include_shutdown_tail recovers the untimed superset, and the new
#       env_records field persists/round-trips.
# Stage [1] runs the FULL suite: the slow full-forward-pass test in
# test_stream_us002.py is NOT skipped.  The PSDATA_SKIP_SLOW escape hatch that
# used to bound this job is gone (GATE-02 removed its only reader), and
# run_tests.sh now fails on any skip that is not justified in
# tests/skips_allowed.txt (HYG-03) -- so budget the wall time for a full pass
# rather than skipping a regression to fit the box.
set -uo pipefail
echo "host=$(hostname) jobid=${SLURM_JOB_ID:-?} start=$(date -Is)"
SW=/sdf/data/lcls/ds/prj/prjcwang31/results/software/psdata
FUN=/sdf/data/lcls/ds/prj/prjcwang31/results/fun/psdata

echo "===== [1] psana-oracle regression suite ====="
(
  set +u   # psconda's conda activate scripts reference unbound vars (ADDR2LINE)
  source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
  cd "$SW"
  bash run_tests.sh
)
suite_rc=$?
echo "suite_rc=$suite_rc"

echo "===== [2] at-scale env-store verify (verify_cache.py) ====="
(
  set +u   # .venv/psconda activate scripts may reference unbound vars
  cd "$FUN"
  source .venv/bin/activate
  python -u verify_cache.py
)
verify_rc=$?
echo "verify_rc=$verify_rc"

echo "end=$(date -Is) suite_rc=$suite_rc verify_rc=$verify_rc"
if [ "$suite_rc" -eq 0 ] && [ "$verify_rc" -eq 0 ]; then
  echo "OVERALL: PASS"
  exit 0
fi
echo "OVERALL: FAIL"
exit 1
