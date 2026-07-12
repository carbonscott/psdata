#!/bin/bash
# tests/bench_workers.sbatch -- recorded milano run of the psdata MULTI-WORKER
# read-scaling benchmark (tests/bench_workers.py).
#
# The library END is a PARAMETER of this wrapper: the same, unmodified
# bench_workers.py is run against BOTH psdata ends, selected purely by a
# PYTHONPATH PREPEND, so the only variable between the two jobs is the library.
#
#   BENCH_END=before  -> psdata 1d018423b20509cb9296df1a5d166cc3153910a6
#                        (read_events: one os.pread per (event, stream) pair)
#   BENCH_END=after   -> psdata 15caa3a1695aabc5d148a2efda4ec8aff9fbd582
#                        (read_events: PERF-01 coalesced merge)
#
# HARNESS != LIBRARY.  The harness is run from $BENCH_HARNESS (this checkout, i.e.
# $SLURM_SUBMIT_DIR), NEVER from $BENCH_PSDATA: bench_workers.py does not exist at
# the BEFORE SHA, so running $BENCH_PSDATA/tests/bench_workers.py could not run the
# BEFORE arm at all -- and copying the harness into that checkout would make it
# DIRTY, which the preflight below rightly kills the job for.  $BENCH_PSDATA
# contributes exactly one thing: src/ on PYTHONPATH.  That is what makes "the same
# unmodified harness runs against both ends" literally true.
#
# Submit (one node each; run them SERIALLY, never on a shared node):
#   sbatch --job-name=bw_after  --export=ALL,BENCH_END=after  tests/bench_workers.sbatch
#   sbatch --job-name=bw_before --export=ALL,BENCH_END=before tests/bench_workers.sbatch
#
# Overrides:
#   BENCH_PSDATA      absolute path to the psdata checkout for this end -- the
#                     LIBRARY under test, used ONLY as a PYTHONPATH entry
#                     (default: $BENCH_ROOT/psdata-$BENCH_END)
#   BENCH_PSDATA_SHA  the SHA that checkout MUST be at (default: the pin above)
#   BENCH_HARNESS     absolute path to the checkout the HARNESS is run from
#                     (default: $SLURM_SUBMIT_DIR -- where you sbatch'd from)
#   BENCH_ROOT        where the two checkouts live
#   plus every BENCH_* knob bench_workers.py documents (WORKERS/REPS/EVENTS/...)
#
# --exclusive + --mem=0 are LOAD-BEARING, not decoration: a 16-worker aggregate
# GB/s measured while a neighbour job shares the node's cores, page cache and Weka
# client is not a measurement.  bench_workers.py self-guards and REFUSES to emit a
# number off an exclusive batch node -- and, unlike bench_index.py, it offers NO
# BENCH_ALLOW_NONBATCH override (the campaign forbids login-node numbers; setting
# that variable is itself a hard refusal).
#
#SBATCH --partition=milano
#SBATCH --account=lcls:prjdat21
#SBATCH --qos=normal
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --job-name=bench_workers
#SBATCH --output=/sdf/data/lcls/ds/prj/prjcwang31/results/_bench_runs/bench_workers-%x-%j.out

set -u

echo "=== node: $(hostname)  job: $SLURM_JOB_ID  end: ${BENCH_END:-<unset>}  $(date) ==="

# ---------------------------------------------------------------- 0. the end
if [ -z "${BENCH_END:-}" ]; then
  echo "FATAL: BENCH_END is unset. Submit with --export=ALL,BENCH_END=before|after" >&2
  exit 2
fi

BENCH_ROOT=${BENCH_ROOT:-/sdf/data/lcls/ds/prj/prjcwang31/results/software}
SHA_BEFORE=1d018423b20509cb9296df1a5d166cc3153910a6
SHA_AFTER=15caa3a1695aabc5d148a2efda4ec8aff9fbd582

case "$BENCH_END" in
  before) DEFAULT_SHA=$SHA_BEFORE ;;
  after)  DEFAULT_SHA=$SHA_AFTER  ;;
  *) echo "FATAL: BENCH_END must be 'before' or 'after', got '$BENCH_END'" >&2; exit 2 ;;
esac

BENCH_PSDATA=${BENCH_PSDATA:-$BENCH_ROOT/psdata-$BENCH_END}
BENCH_PSDATA_SHA=${BENCH_PSDATA_SHA:-$DEFAULT_SHA}
# The HARNESS lives in the checkout you submitted from -- decoupled from the
# LIBRARY checkout on purpose (see the header): the BEFORE SHA predates
# bench_workers.py, and copying the harness in would make that checkout DIRTY and
# trip the preflight.  $BENCH_PSDATA is PYTHONPATH-only.
BENCH_HARNESS=${BENCH_HARNESS:-${SLURM_SUBMIT_DIR:-$PWD}}
export BENCH_END BENCH_PSDATA BENCH_PSDATA_SHA BENCH_HARNESS

if [ ! -d "$BENCH_PSDATA/src/psdata" ]; then
  echo "FATAL: no psdata at $BENCH_PSDATA/src/psdata (set BENCH_PSDATA/BENCH_ROOT)" >&2
  exit 2
fi

if [ ! -f "$BENCH_HARNESS/tests/bench_workers.py" ]; then
  echo "FATAL: no harness at $BENCH_HARNESS/tests/bench_workers.py (set BENCH_HARNESS" >&2
  echo "       -- it is NOT expected inside \$BENCH_PSDATA: the BEFORE SHA has no" >&2
  echo "       tests/bench_workers.py, which is the whole point of the split)" >&2
  exit 2
fi

echo "library (PYTHONPATH only): $BENCH_PSDATA"
echo "harness (the script run):  $BENCH_HARNESS/tests/bench_workers.py"

# The campaign forbids the non-batch override outright.
if [ -n "${BENCH_ALLOW_NONBATCH:-}" ]; then
  echo "FATAL: BENCH_ALLOW_NONBATCH is set. This campaign FORBIDS it." >&2
  exit 2
fi

# ------------------------------------------------- 1. psana env, then PREPEND
# psconda.sh SETS PYTHONPATH.  Sourcing it AFTER exporting ours would clobber the
# library selection; assigning ours as PYTHONPATH=<repo>/src (without keeping the
# tail) would hide psana and make every psana probe SKIP -- a silent false pass.
# So: source psconda FIRST, then PREPEND, and probe `import psana` explicitly.
source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
echo "which python: $(which python)"

export PYTHONPATH=$BENCH_PSDATA/src:${PYTHONPATH:-}
echo "PYTHONPATH:   $PYTHONPATH"

# ------------------------------------------------------------ 2. HARD PREFLIGHT
# Fail the job BEFORE any number is produced if the wrong psdata, a psdata at the
# wrong SHA, or no psana is importable.  A benchmark that silently graded the
# other end -- or graded a psdata that shadowed psana out of the environment --
# would be worse than no benchmark.
python - <<'PREFLIGHT'
import os, subprocess, sys

want_root = os.path.realpath(os.environ["BENCH_PSDATA"])
want_sha = os.environ.get("BENCH_PSDATA_SHA", "").strip()
end = os.environ.get("BENCH_END")
fail = []

try:
    import psdata
except Exception as e:
    print(f"PREFLIGHT FATAL: cannot import psdata: {e!r}", file=sys.stderr)
    sys.exit(2)

got = os.path.realpath(psdata.__file__)
want = os.path.realpath(os.path.join(want_root, "src", "psdata", "__init__.py"))
if got != want:
    fail.append(f"WRONG psdata imported: {got}\n                  expected: {want}\n"
                f"                  (PYTHONPATH prepend did not win -- another "
                f"psdata is installed/earlier on sys.path)")

sha = ""
try:
    sha = subprocess.run(["git", "-C", want_root, "rev-parse", "HEAD"],
                         capture_output=True, text=True, timeout=60).stdout.strip()
except Exception as e:
    fail.append(f"cannot read the git SHA of {want_root}: {e!r}")
if want_sha and sha and not sha.startswith(want_sha[:12]):
    fail.append(f"WRONG psdata SHA for BENCH_END={end}: checkout is {sha}, "
                f"expected {want_sha}")
dirty = subprocess.run(["git", "-C", want_root, "status", "--porcelain"],
                       capture_output=True, text=True).stdout.strip()
if dirty:
    fail.append(f"psdata checkout {want_root} is DIRTY -- the measured library is "
                f"not the pinned SHA:\n{dirty[:500]}")

try:
    import psana
    psana_file = psana.__file__
except Exception as e:
    psana_file = None
    fail.append(f"psana is NOT importable ({e!r}) -- PYTHONPATH must PREPEND the "
                f"library, never replace psconda's own PYTHONPATH. A run without "
                f"psana is not the campaign environment.")

print(f"PREFLIGHT end={end}")
print(f"PREFLIGHT psdata.__file__={got}")
print(f"PREFLIGHT psdata sha={sha} (expected {want_sha})")
print(f"PREFLIGHT psana.__file__={psana_file}")
if fail:
    print("PREFLIGHT FAILED:", file=sys.stderr)
    for f in fail:
        print(f"  - {f}", file=sys.stderr)
    sys.exit(2)
print("PREFLIGHT OK")
PREFLIGHT
rc=$?
if [ $rc -ne 0 ]; then
  echo "FATAL: preflight failed (rc=$rc) -- refusing to emit a number." >&2
  exit $rc
fi

# ------------------------------------------------------------------- 3. RUN
# The HARNESS comes from $BENCH_HARNESS (unchanged between the two arms); the
# LIBRARY comes from $BENCH_PSDATA/src via the PYTHONPATH prepend above.  ONE
# variable between the jobs: the library.
python "$BENCH_HARNESS/tests/bench_workers.py"
rc=$?
echo "exit: $rc  end: $(date)"
exit $rc        # propagate python's exit code so sacct ExitCode reflects a refusal
                # (2 = guard/bound refused; 1 = memory bound violated / impure)
