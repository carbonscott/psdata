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
# PSDATA_SKIPS_ALLOWED overrides the allowlist path (used by the hygiene test to
# drive malformed-allowlist cases); a real caller never sets it.
SKIPS_ALLOWED="${PSDATA_SKIPS_ALLOWED:-$REPO/tests/skips_allowed.txt}"  # name :: justification
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
# PSDATA_NO_DEFAULT_TESTS=1 suppresses the default list even with no args -- so an
# empty run is expressible (the CLI otherwise can't distinguish "no args" from
# "run nothing").  It exists so the 0-test guard below is reachable and testable;
# a real caller just passes files or nothing.
if [[ ${#TESTS[@]} -eq 0 && "${PSDATA_NO_DEFAULT_TESTS:-}" != "1" ]]; then
  TESTS=(
    "$REPO/tests/test_format_us001.py"
    "$REPO/tests/test_det10_unknown_version.py"
    "$REPO/tests/test_hostile_bytes_fail06.py"
    "$REPO/tests/test_stream_us002.py"
    "$REPO/tests/test_str01_multichunk_forward.py"
    "$REPO/tests/test_str02_stream_dropout.py"
    "$REPO/tests/test_gate02_gated_forward.py"
    "$REPO/tests/test_gate_failclosed_fail02.py"
    "$REPO/tests/test_index_us003.py"
    "$REPO/tests/test_str03_ts_collision.py"
    "$REPO/tests/test_index_format_idx02.py"
    "$REPO/tests/test_idx03_portable_index.py"
    "$REPO/tests/test_robust_us004.py"
    "$REPO/tests/test_fail04_truncated_index.py"
    "$REPO/tests/test_str04_chunk_gap.py"
    "$REPO/tests/test_regression_us005.py"
    "$REPO/tests/test_persist_us008.py"
    "$REPO/tests/test_idx04_invalidation.py"
    "$REPO/tests/test_batch_us009.py"
    "$REPO/tests/test_perf01_coalesced_reads.py"
    "$REPO/tests/test_mem01_bounded_read.py"
    "$REPO/tests/test_config_us010.py"
    "$REPO/tests/test_cal02_multistep_config.py"
    "$REPO/tests/test_torch_us011.py"
    "$REPO/tests/test_bigdata_scan_us012.py"
    "$REPO/tests/test_uniqueid_us011.py"
    "$REPO/tests/test_envstore_us013.py"
    "$REPO/tests/test_gate07_bench_committed.py"
    "$REPO/tests/test_gate06_detector_stream_covered.py"
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
      # An entry with an EMPTY justification is malformed -- do NOT treat it as
      # allowed (that would let a bare 'name ::' rubber-stamp a skip).  Warn and
      # keep looking; if nothing else matches, the skip stays unjustified.
      if [[ -z "$just" ]]; then
        echo "WARNING: malformed allowlist entry for '${name}' in ${SKIPS_ALLOWED}:" \
             "empty justification -- ignoring (a skip is not a pass)" >&2
        continue
      fi
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
bare_skip_files=()          # test files that emitted an UNMARKED skip line
bare_skip_lines=()          # the offending line, verbatim

# Lines that look like the OLD skip idiom -- 'print("[skip] ..."); return',
# 'print("SKIP ...")', 'print("SKIPPED: ...")' and friends -- but carry NO
# ##SKIP## marker.  A future author who writes one of these exits 0 with no
# marker, so the counter above never sees it and it scores as a PASS: exactly the
# pre-fix bug (this repo's old test_config_us010 / test_uniqueid_us011 used the
# 'SKIP ...' form verbatim).  Two passes, then the ##SKIP## marker lines are
# excluded (they are the SANCTIONED form):
#   * case-INSENSITIVE for bracketed/-ing/-ed spellings: [skip], [ skip ],
#     skipping, skipped (so a real record whose reason says "skipped" is only
#     safe because it rides on a ##SKIP## line, which we drop);
#   * case-SENSITIVE for a standalone uppercase SKIP token (\bSKIP\b) -- catches
#     "SKIP ", "SKIP:", "##SKIP##" (dropped by the exclusion).  The word boundary
#     means it does NOT match inside PSDATA_SKIP_SLOW (underscores are word
#     chars) and does NOT flag lowercase "skip" in ordinary prose.
BARE_SKIP_RE_CI='\[[[:space:]]*skip[[:space:]]*\]|skipping|skipped'
BARE_SKIP_RE_CS='\bSKIP\b'

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
  # collect this file's skip records: '##SKIP## <name> :: <reason>'.
  # tests/_skips.py ALWAYS prints the marker at the START of the line, so anchor
  # the match there (leading whitespace tolerated) -- a line that merely mentions
  # ##SKIP## in prose (e.g. test_torch's "see the ##SKIP## records above")
  # is NOT a record and must not be collected.
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    rec="${line#*"$SKIP_MARKER"}"                       # drop everything up to+incl marker
    rec="${rec#"${rec%%[![:space:]]*}"}"                # ltrim
    # A real record has a '::' separator; without it the line is malformed
    # (e.g. '##SKIP## garbage') and must not become a phantom skip.
    if [[ "$rec" != *"::"* ]]; then
      echo "WARNING: malformed skip record (no '::' separator), ignoring: ${line}" >&2
      continue
    fi
    name="${rec%%::*}"
    reason="${rec#*::}"
    name="${name#"${name%%[![:space:]]*}"}"; name="${name%"${name##*[![:space:]]}"}"
    reason="${reason#"${reason%%[![:space:]]*}"}"; reason="${reason%"${reason##*[![:space:]]}"}"
    if [[ -z "$name" ]]; then
      echo "WARNING: malformed skip record (empty name), ignoring: ${line}" >&2
      continue
    fi
    skip_names+=("$name")
    skip_reasons+=("$reason")
  done < <(grep -E "^[[:space:]]*${SKIP_MARKER}[[:space:]]" "$out" || true)
  # UNMARKED skips: skip-looking lines that are NOT ##SKIP## records.  Two grep
  # passes (case-insensitive spellings + case-sensitive uppercase SKIP token),
  # de-duplicated, with the ##SKIP## marker lines excluded last.
  while IFS= read -r bline; do
    [[ -n "$bline" ]] || continue
    bare_skip_files+=("$t")
    bare_skip_lines+=("$bline")
  done < <( { grep -iE "$BARE_SKIP_RE_CI" "$out"
              grep -E  "$BARE_SKIP_RE_CS" "$out"; } 2>/dev/null \
            | grep -vF "$SKIP_MARKER" | awk '!seen[$0]++' || true)
  rm -f "$out"
done

ran=$((passed + failed))

# A vacuous green -- zero test files executed -- must not read as success.
if [[ $ran -eq 0 ]]; then
  echo
  echo "RESULT: FAIL -- ran 0 tests" >&2
  exit 1
fi

n_skipped=${#skip_names[@]}
n_bare=${#bare_skip_lines[@]}

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

# Unmarked (old-idiom) skips: a printed skip that carries no ##SKIP## marker.
if [[ $n_bare -gt 0 ]]; then
  echo "--- unmarked skips (HYG-03) ----------------------------------------"
  for i in "${!bare_skip_lines[@]}"; do
    echo "UNMARKED SKIP in ${bare_skip_files[$i]}:"
    echo "     ${bare_skip_lines[$i]}"
  done
  echo "--------------------------------------------------------------------"
  echo "FAILED: ${n_bare} unmarked skip(s) -- a printed skip with no ##SKIP##" \
       "marker exits 0 and scores as a PASS (the exact HYG-03 bug).  Route it" \
       "through tests/_skips.py: skip(name, reason), and justify it in" \
       "tests/skips_allowed.txt if it is legitimate." >&2
  [[ $status -ne 0 ]] || status=1
fi

exit $status
