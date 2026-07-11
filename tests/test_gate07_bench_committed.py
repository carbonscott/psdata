#!/usr/bin/env python3
"""GATE-07 regression: the project's strongest correctness gate must live IN
the repository -- not only in a working tree.

The strongest correctness evidence this project has (COR-10) is produced by
``tests/bench_index.py``: raw byte-exact on ALL timing positions (~11.3k on
mfx100848724/r51, ~12.6k on ued1010667/r177), NaN-aware calib on a
3000-position evenly-spread subsample, exact L1 event-count AND timestamp-set
equality, all gating the clock fail-closed on a milano compute node.  For a
long stretch that script existed ONLY as an UNCOMMITTED file in a working tree:
``find`` returned nothing, so the numbers were not reproducible or auditable
from anything shipped.  A psana developer who cloned the repo could not run the
strongest gate; every "green" they could reproduce was a prefix of one run.

This meta-test pins that gate INTO the tree.  It is deliberately cheap --
stdlib only (numpy used opportunistically if present), NO psana, NO SLAC data --
so it runs anywhere the suite runs, on every invocation, and can never be
switched off.  It asserts, of the tree it is itself running in, that:

  1. ``tests/bench_index.py`` EXISTS and is a readable file (located relative to
     this file, so it genuinely inspects the checked-out tree, cwd-robust);
  2. it is SYNTACTICALLY VALID (``ast.parse`` -- also compiles under py_compile);
  3. its load-bearing gate machinery is STILL PRESENT, so a future edit that
     guts the gate down to a stub is caught:
       * a NaN-aware calib equality (``equal_nan=True`` / ``np.isnan``), NOT a
         bare ``np.array_equal`` -- verified structurally against ``eq_calib``;
       * the ~3000-position evenly-spread calib subsample (``GATE_CALIB_CAP``);
       * the fail-closed batch-node guard (``require_batch_node`` + ``sys.exit``);
       * the exact event-count AND timestamp-set equality checks;
       * the random-access read primitives the gate exercises.

It is the DISCRIMINATOR for the fix: on the parent commit ``bench_index.py`` is
untracked, so a clean checkout does NOT contain it -- assertion (1) fails.  On
the fix commit the script is tracked, the checkout has it, and every assertion
below passes.

Deliberately imports NOTHING from the tree under test: it only reads
``bench_index.py`` as text and parses it, so it carries no part of the artifact
it certifies.
"""

import ast
import os
import sys
import warnings

# numpy is OPTIONAL here: the discriminating checks are text/AST-based (pure
# stdlib), so this meta-test certifies the committed gate even in a bare
# interpreter.  When numpy IS present we additionally demonstrate *why* the gate
# needs a NaN-aware compare.  Guarding the import keeps "fix passes" true no
# matter which interpreter the meta-test is run under.
try:
    import numpy as np
except Exception:                              # pragma: no cover - env-dependent
    np = None

# --- locate the gate script relative to THIS file (cwd-robust) --------------
# It must live in the same tests/ directory as this meta-test; deriving the
# path from __file__ (not cwd) means we inspect the tree we actually run in --
# which is exactly what makes this a discriminator between the parent (script
# absent) and the fix (script committed).
_HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_PY = os.path.join(_HERE, "bench_index.py")
BENCH_SBATCH = os.path.join(_HERE, "bench_index.sbatch")


def _read(path):
    """Read a file as text, or return None if it is not present/readable.

    Returning None (rather than raising) lets the existence assertion below own
    the failure message on the parent commit, where bench_index.py is absent."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _source():
    """The bench_index.py source, or None if it is not in this tree."""
    return _read(BENCH_PY)


# ---------------------------------------------------------------------------
# 1. the gate script is IN the tree and readable  (the discriminator)
# ---------------------------------------------------------------------------
def test_bench_index_committed_and_readable():
    """tests/bench_index.py must exist as a readable file in this tree.  This is
    the assertion that FAILS on the parent commit, where the script is untracked
    and therefore absent from a clean checkout."""
    assert os.path.exists(BENCH_PY), (
        "GATE-07: tests/bench_index.py is NOT in the repository (%s does not "
        "exist).  The project's strongest correctness gate (COR-10) must be "
        "committed so its numbers are reproducible and auditable from a clean "
        "clone -- an uncommitted working-tree-only gate is not evidence anyone "
        "else can run." % BENCH_PY)
    assert os.path.isfile(BENCH_PY), (
        "GATE-07: %s exists but is not a regular file" % BENCH_PY)
    assert os.access(BENCH_PY, os.R_OK), (
        "GATE-07: %s exists but is not readable" % BENCH_PY)
    src = _source()
    assert src and src.strip(), "GATE-07: tests/bench_index.py is empty/unreadable"
    print("[gate-07] tests/bench_index.py is committed and readable (%d bytes)"
          % len(src))


# ---------------------------------------------------------------------------
# 2. the gate script is syntactically valid
# ---------------------------------------------------------------------------
def _parse():
    """Parse bench_index.py, returning (src, tree).  Asserts it is present and
    parses.  A committed gate that does not even parse is not auditable."""
    src = _source()
    assert src is not None, (
        "GATE-07: tests/bench_index.py is not present in this tree (cannot "
        "parse what is not committed)")
    try:
        # bench_index.py carries a cosmetic invalid-escape ('/\\') in its
        # docstring that trips a SyntaxWarning under 3.12; it is not a parse
        # error and is out of scope for this meta-test, so contain the noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(src, filename=BENCH_PY)
    except SyntaxError as e:                      # pragma: no cover - failure path
        raise AssertionError(
            "GATE-07: tests/bench_index.py does not parse: %s" % e)
    return src, tree


def test_bench_index_parses():
    _parse()
    print("[gate-07] tests/bench_index.py parses cleanly (ast.parse)")


# ---------------------------------------------------------------------------
# 3. the load-bearing gate machinery is still present
# ---------------------------------------------------------------------------
def _function_source(src, tree, name):
    """Return the source segment of the top-level function `name`, or None."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src, node)
            if seg is not None:
                return seg
    return None


def test_calib_comparison_is_nan_aware():
    """The calib equality must be NaN-aware -- jungfrau calib masks bad pixels to
    NaN, so a bare np.array_equal would spuriously report inequality (NaN != NaN)
    and either mask real regressions or fail every run.  Verify structurally
    against the eq_calib function, not just anywhere in the file."""
    src, tree = _parse()

    # Self-documenting reminder of WHY the gate needs equal_nan: the bare form
    # never matches on NaN, the NaN-aware form does.  Only when numpy is present.
    if np is not None:
        a = np.array([1.0, np.nan, 2.0])
        assert not np.array_equal(a, a), "bare array_equal should differ on NaN"
        assert np.array_equal(a, a, equal_nan=True), "equal_nan must treat NaN==NaN"

    eq_calib = _function_source(src, tree, "eq_calib")
    assert eq_calib is not None, (
        "GATE-07: eq_calib() is gone from tests/bench_index.py -- the NaN-aware "
        "calib comparison is the heart of the COR-10 gate")
    nan_aware = ("equal_nan" in eq_calib) or ("isnan" in eq_calib)
    assert nan_aware, (
        "GATE-07: eq_calib() no longer does a NaN-aware comparison (no "
        "equal_nan / np.isnan) -- a bare np.array_equal would neuter the calib "
        "gate on any NaN-masked detector:\n%s" % eq_calib)
    print("[gate-07] calib comparison is NaN-aware (equal_nan / isnan in eq_calib)")


def test_batch_node_guard_present():
    """Timing is only valid on a milano compute node; the gate must refuse to
    emit numbers off one (fail-closed).  Both the guard and its hard exit must
    survive."""
    src, _ = _parse()
    assert "require_batch_node" in src, (
        "GATE-07: the fail-closed batch-node guard (require_batch_node) is gone")
    assert "SLURM_JOB_ID" in src, (
        "GATE-07: the batch-node guard no longer checks $SLURM_JOB_ID")
    assert "sys.exit" in src, (
        "GATE-07: require_batch_node no longer hard-exits -- the guard must "
        "fail closed, not just warn")
    print("[gate-07] fail-closed batch-node guard present "
          "(require_batch_node + SLURM_JOB_ID + sys.exit)")


def test_calib_subsample_present():
    """Calib runs on a ~3000-position evenly-spread subsample of the timing
    positions -- the cap and the spread selection must both be present."""
    src, _ = _parse()
    assert "GATE_CALIB_CAP" in src, (
        "GATE-07: the calib subsample cap (GATE_CALIB_CAP) is gone")
    assert "3000" in src, (
        "GATE-07: the default 3000-position calib subsample size is gone")
    assert "calib_targets" in src, (
        "GATE-07: the evenly-spread calib subsample selection (calib_targets) "
        "is gone -- calib would no longer run on the spread subsample")
    print("[gate-07] ~3000-position evenly-spread calib subsample present "
          "(GATE_CALIB_CAP / calib_targets)")


def test_count_and_timestamp_equality_present():
    """Exact L1 event-count equality AND timestamp-SET equality (the ragged
    DAQ-shutdown-tail catch) must both survive."""
    src, _ = _parse()
    # event-count equality: psana's forward count vs the index's n_events
    assert "psana_count" in src and "n_events" in src, (
        "GATE-07: the event-count equality check (psana_count vs n_events) is "
        "gone")
    # timestamp-SET equality
    assert "ts_psana" in src and "timestamps" in src, (
        "GATE-07: the timestamp-set equality check (ts_psana vs "
        "ridx.timestamps) is gone")
    assert "set(" in src, (
        "GATE-07: the timestamp-SET comparison no longer uses set() -- a list "
        "compare would miss ordering-independent tail divergence")
    print("[gate-07] event-count + timestamp-set equality checks present")


def test_random_access_primitives_present():
    """The gate exercises the random-access read primitives that are the whole
    reason the benchmark exists (read_event_at / read_events / read_event)."""
    src, _ = _parse()
    for prim in ("read_event_at", "read_events", "read_event"):
        assert prim in src, (
            "GATE-07: random-access primitive %r is not exercised by the gate"
            % prim)
    print("[gate-07] random-access read primitives exercised "
          "(read_event_at / read_events / read_event)")


def test_bench_sbatch_committed():
    """The recorded milano batch script must ship alongside the gate, so the run
    that produced the numbers is itself reproducible/auditable."""
    assert os.path.exists(BENCH_SBATCH) and os.path.isfile(BENCH_SBATCH), (
        "GATE-07: tests/bench_index.sbatch is NOT in the repository (%s) -- the "
        "recorded run of the strongest gate must ship with it" % BENCH_SBATCH)
    sb = _read(BENCH_SBATCH)
    assert sb and "bench_index.py" in sb, (
        "GATE-07: bench_index.sbatch does not invoke bench_index.py")
    assert "#SBATCH" in sb, (
        "GATE-07: bench_index.sbatch has no #SBATCH directives")
    print("[gate-07] tests/bench_index.sbatch is committed and invokes the gate")


def main():
    print("=" * 72)
    print("GATE-07: the strongest correctness gate must be IN the repository")
    print("=" * 72)
    # existence FIRST -- this is the assertion that discriminates the parent
    # (script absent -> fails here) from the fix (script committed -> passes).
    test_bench_index_committed_and_readable()
    test_bench_index_parses()
    test_calib_comparison_is_nan_aware()
    test_batch_node_guard_present()
    test_calib_subsample_present()
    test_count_and_timestamp_equality_present()
    test_random_access_primitives_present()
    test_bench_sbatch_committed()
    print("\n[ok] the COR-10 gate (tests/bench_index.py) is committed, parses, "
          "and retains its load-bearing machinery")
    print("\nGATE-07 REGRESSION PASSED")


if __name__ == "__main__":
    main()
