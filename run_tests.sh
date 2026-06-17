#!/usr/bin/env bash
# Run the psdata acceptance suite against the production psana on sdfiana025.
#
# The psana cross-check tests use a two-process oracle: psana (from psconda.sh,
# entered via PYTHONPATH) generates ground truth, and the numpy-only psdata
# reader is compared against it. We expose psdata by prepending this project's
# src/ dir to PYTHONPATH; src/ holds ONLY the psdata package, so `import psdata`
# resolves here while `import psana` resolves to the production env -- no
# shadowing. (The standalone src layout replaces the old .pkgroot/.rundir
# symlink workaround the in-lcls2 copy needed.)
#
# Usage (on sdfiana025):
#   source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
#   bash run_tests.sh [test_file ...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # project root
SRC="$REPO/src"                                        # holds only psdata/

PYPARTS="$SRC"

if [[ -z "${PYTHONPATH:-}" ]]; then
  echo "WARNING: PYTHONPATH is empty -- did you source psconda.sh first?" >&2
  export PYTHONPATH="$PYPARTS"
else
  export PYTHONPATH="$PYPARTS:$PYTHONPATH"
fi

TESTS=("$@")
if [[ ${#TESTS[@]} -eq 0 ]]; then
  TESTS=(
    "$REPO/tests/test_format_us001.py"
    "$REPO/tests/test_stream_us002.py"
    "$REPO/tests/test_index_us003.py"
    "$REPO/tests/test_robust_us004.py"
    "$REPO/tests/test_regression_us005.py"
    "$REPO/tests/test_persist_us008.py"
    "$REPO/tests/test_batch_us009.py"
    "$REPO/tests/test_config_us010.py"
    "$REPO/tests/test_torch_us011.py"
    "$REPO/tests/test_bigdata_scan_us012.py"
  )
fi

status=0
for t in "${TESTS[@]}"; do
  echo "### running $t"
  python3 "$t" || status=$?
done
exit $status
