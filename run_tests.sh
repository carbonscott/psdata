#!/usr/bin/env bash
# Run the psdata test suite against the production psana on sdfiana025.
#
# Why this wrapper exists (see notebook pattern, US-001):
#   * psconda.sh imports psana VIA PYTHONPATH, so we must PREPEND, not replace.
#   * The repo root contains both a single-file `psdata.py` (the byte-exact
#     reference) and an unbuilt `psana/` clone -- either will shadow the real
#     thing if the repo root lands on sys.path / cwd.  So we expose the psdata
#     package through an isolated parent dir that contains ONLY psdata, and we
#     run from a clean working directory.
#
# Usage (on sdfiana025):
#   source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
#   bash psdata/run_tests.sh [test_file ...]
set -euo pipefail

# Repo root = parent of this script's directory (psdata/..).
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../psdata
REPO="$(cd "$PKG_DIR/.." && pwd)"                            # repo root

# Isolated package-parent dir holding only the psdata package (via symlink),
# so `import psdata` resolves here and `import psana` resolves to the prod env.
PKGROOT="$REPO/.pkgroot"
RUNDIR="$REPO/.rundir"
mkdir -p "$PKGROOT" "$RUNDIR"
ln -sfn "$PKG_DIR" "$PKGROOT/psdata"

if [[ -z "${PYTHONPATH:-}" ]]; then
  echo "WARNING: PYTHONPATH is empty -- did you source psconda.sh first?" >&2
  export PYTHONPATH="$PKGROOT"
else
  export PYTHONPATH="$PKGROOT:$PYTHONPATH"
fi

TESTS=("$@")
if [[ ${#TESTS[@]} -eq 0 ]]; then
  TESTS=(
    "$PKG_DIR/tests/test_format_us001.py"
    "$PKG_DIR/tests/test_stream_us002.py"
    "$PKG_DIR/tests/test_index_us003.py"
  )
fi

cd "$RUNDIR"
status=0
for t in "${TESTS[@]}"; do
  echo "### running $t"
  python3 "$t" || status=$?
done
exit $status
