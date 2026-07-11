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
# A SKIP IS NOT A PASS (HYG-03).  Each test file is one pass/fail unit (its exit
# code), but a test may also skip an INDIVIDUAL check -- it does so by printing a
# machine-readable record (see tests/_skips.py):
#
#     ##SKIP## <name> :: <reason>
#
# This runner tees every test's output, counts those records, prints an explicit
# skip count in the final tally, and EXITS NONZERO unless every emitted skip name
# is listed in tests/skips_allowed.txt with a justification.  Historically the
# suite reported "12 passed" while three checks -- including two psana oracles --
# had silently not run.
#
# Usage (on sdfiana025):
#   source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
#   bash run_tests.sh [test_file ...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # project root
SRC="$REPO/src"                                        # holds only psdata/
SKIPS_ALLOWED="$REPO/tests/skips_allowed.txt"          # name :: justification
SKIP_MARKER="##SKIP##"                                 # must match tests/_skips.py

PYPARTS="$SRC"

if [[ -z "${PYTHONPATH:-}" ]]; then
  echo "WARNING: PYTHONPATH is empty -- did you source psconda.sh first?" >&2
  export PYTHONPATH="$PYPARTS"
else
  export PYTHONPATH="$PYPARTS:$PYTHONPATH"
fi

# The default suite.  EVERY tests/test_*.py on disk must appear here -- an
# unregistered test file is another way for a check to silently not run;
# tests/test_runner_hygiene_hyg03.py asserts this list is complete.
TESTS=("$@")
if [[ ${#TESTS[@]} -eq 0 ]]; then
  TESTS=(
    "$REPO/tests/test_format_us001.py"
    "$REPO/tests/test_stream_us002.py"
    "$REPO/tests/test_gate02_gated_forward.py"
    "$REPO/tests/test_index_us003.py"
    "$REPO/tests/test_robust_us004.py"
    "$REPO/tests/test_regression_us005.py"
    "$REPO/tests/test_persist_us008.py"
    "$REPO/tests/test_batch_us009.py"
    "$REPO/tests/test_config_us010.py"
    "$REPO/tests/test_torch_us011.py"
    "$REPO/tests/test_bigdata_scan_us012.py"
    "$REPO/tests/test_uniqueid_us011.py"
    "$REPO/tests/test_envstore_us013.py"
    "$REPO/tests/test_runner_hygiene_hyg03.py"
  )
fi

# ---------------------------------------------------------------------------
# skip allowlist: 'name :: justification' records (blank / '#' lines ignored).
# Echoes the justification and returns 0 if $1 is allowed, else returns 1.
# ---------------------------------------------------------------------------
allowed_justification() {
  local want="$1" line name just
  [[ -f "$SKIPS_ALLOWED" ]] || return 1
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"           # ltrim
    case "$line" in ''|'#'*) continue ;; esac
    [[ "$line" == *"::"* ]] || continue
    name="${line%%::*}"
    just="${line#*::}"
    # strip surrounding whitespace
    name="${name#"${name%%[![:space:]]*}"}"; name="${name%"${name##*[![:space:]]}"}"
    just="${just#"${just%%[![:space:]]*}"}"; just="${just%"${just##*[![:space:]]}"}"
    if [[ "$name" == "$want" ]]; then
      printf '%s' "$just"
      return 0
    fi
  done < "$SKIPS_ALLOWED"
  return 1
}

WORK="$(mktemp -d "${TMPDIR:-/tmp}/psdata-run-tests.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

status=0
passed=0
failed=0
skip_names=()
skip_reasons=()

for t in "${TESTS[@]}"; do
  echo "### running $t"
  out="$WORK/out.$$"
  # Stream the test's output to the terminal AND capture it, so the skip
  # records can be counted.  `pipefail` would give us tee's status, so take
  # PIPESTATUS[0] -- the PYTHON exit code -- explicitly.
  set +e
  python3 "$t" 2>&1 | tee "$out"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -eq 0 ]]; then
    passed=$((passed + 1))
  else
    status=$rc
    failed=$((failed + 1))
  fi
  # collect this file's skip records: '##SKIP## <name> :: <reason>'
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    rec="${line#*"$SKIP_MARKER"}"                       # drop everything up to+incl marker
    rec="${rec#"${rec%%[![:space:]]*}"}"                # ltrim
    name="${rec%%::*}"
    reason="${rec#*::}"
    name="${name#"${name%%[![:space:]]*}"}"; name="${name%"${name##*[![:space:]]}"}"
    reason="${reason#"${reason%%[![:space:]]*}"}"; reason="${reason%"${reason##*[![:space:]]}"}"
    [[ -n "$name" ]] || continue
    skip_names+=("$name")
    skip_reasons+=("$reason")
  done < <(grep -F "$SKIP_MARKER" "$out" || true)
  rm -f "$out"
done

n_skipped=${#skip_names[@]}

# Final tally over the test FILES run (each script is one pass/fail unit) plus
# the INDIVIDUAL checks that skipped inside them.  A skip is not a pass.
echo
echo "${passed} passed, ${failed} failed, ${n_skipped} skipped"

unjustified=0
if [[ $n_skipped -gt 0 ]]; then
  echo "--- skipped checks -------------------------------------------------"
  for i in "${!skip_names[@]}"; do
    name="${skip_names[$i]}"
    reason="${skip_reasons[$i]}"
    if just="$(allowed_justification "$name")"; then
      echo "SKIP ${name}"
      echo "     reason:        ${reason}"
      echo "     justification: ${just}"
    else
      unjustified=$((unjustified + 1))
      echo "SKIP ${name}"
      echo "     reason:        ${reason}"
      echo "     justification: NONE -- not in tests/skips_allowed.txt"
      echo "UNJUSTIFIED SKIP: ${name} -- a skip is not a pass (HYG-03)" >&2
    fi
  done
  echo "--------------------------------------------------------------------"
fi

if [[ $unjustified -gt 0 ]]; then
  echo "FAILED: ${unjustified} unjustified skip(s) -- a check that did not run" \
       "is not a passing check.  Fix the cause, or add the skip to" \
       "tests/skips_allowed.txt with a real justification." >&2
  [[ $status -ne 0 ]] || status=1
fi

exit $status
