#!/usr/bin/env python3
"""GATE-06 regression: the "SMD is optional" gate must scan a DETECTOR stream.

``tests/test_bigdata_scan_us012.py`` validates that psdata can build its
random-access index from the BIGDATA directly, with no ``.smd.xtc2`` sidecar
(``build_from_bigdata`` -- the "SMD is optional" claim).  But on the parent it
did so ONLY over the run's three SMALL non-detector streams::

    SMALL_STREAMS = (0, 1, 4)

The jungfrau DETECTOR streams (5 & 7) were never walked -- a full bigdata-header
walk of a ~600 GB stream "didn't finish in 480 s", so they were excluded.  Two
consequences:

  * the SMD-free build was NEVER exercised on real detector data; and
  * r51's ragged DAQ end-of-run tail lives precisely on streams 5 & 7 -- exactly
    the omitted ones -- so the gate could assert *exact* ts equality and never
    meet the superset / ragged-tail case (17982 physical L1Accepts vs 17872
    SMD-indexed) that "SMD is optional" advertises.

This meta-test AST-inspects ``test_bigdata_scan_us012.py`` and pins the fix
BEHAVIOURALLY (the file exists on the parent too -- what changed is its
covered-stream set):

  1. its module-level covered-stream constants now include a jungfrau DETECTOR
     stream (5 or 7), not just the small non-detector streams (0, 1, 4); and
  2. a detector/tail test function asserts a STRICT superset (the ragged tail),
     not mere ts equality.

It imports NO psana, needs NO SLAC data, and reads only the sibling test's
source -- so the orchestrator can run it against a bare parent worktree.

Discriminator:
  * PARENT (origin/main @ 0f9e8031ec6b9f10fe3a30712f444d23766d2dab):
    ``test_bigdata_scan_us012.py`` defines only ``SMALL_STREAMS = (0, 1, 4)`` --
    covered set {0, 1, 4}, no detector stream -> assertion (1) FAILS.
  * FIX (fix/gate-06): a ``DETECTOR_STREAMS = (5, 7)`` constant is scanned by
    ``test_bigdata_detector_stream_reaches_ragged_tail`` -> covered set gains
    5 & 7 and the strict-superset assertion is present -> PASSES.
"""
import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# cwd-robust: resolve the sibling under test relative to THIS file, not cwd.
TARGET = os.path.join(_HERE, "test_bigdata_scan_us012.py")

# The jungfrau DETECTOR streams of mfx100848724/r51 -- where the ragged
# shutdown tail (17982 vs 17872) lives.  At least one of these must now be
# covered by the "SMD is optional" gate.
DETECTOR_STREAMS_UNIVERSE = frozenset({5, 7})
# The small non-detector streams the parent covered -- must be KEPT (the fix
# ADDS detector coverage, it does not replace the small-stream coverage).
REQUIRED_SMALL = frozenset({0, 1, 4})


def _load_target():
    """Return ``(source, ast_module)`` for the gate under test, cwd-robustly."""
    assert os.path.exists(TARGET), (
        f"cannot find the gate under test at {TARGET!r} -- this meta-test must "
        f"sit beside tests/test_bigdata_scan_us012.py")
    with open(TARGET, "r") as fh:
        src = fh.read()
    return src, ast.parse(src, filename=TARGET)


def _int_literals(node):
    """The plain ints in a tuple/list literal AST node (bool excluded)."""
    out = set()
    if isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            if (isinstance(elt, ast.Constant) and isinstance(elt.value, int)
                    and not isinstance(elt.value, bool)):
                out.add(elt.value)
    return out


def _stream_constants(tree):
    """``{NAME: {ints}}`` for every MODULE-LEVEL assignment whose target name
    mentions STREAM and whose value is a tuple/list of ints (e.g. SMALL_STREAMS,
    DETECTOR_STREAMS).  Restricted to ``tree.body`` so function parameters /
    defaults (the synthetic ``det_streams=(5, 7)`` fixtures) are never counted --
    only the gate's declared covered-stream constants."""
    consts = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        ints = _int_literals(node.value)
        if not ints:
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and "STREAM" in tgt.id.upper():
                consts[tgt.id] = ints
    return consts


def covered_streams(tree):
    """The union of every module-level ``*STREAM*`` constant -- the set of
    streams the gate declares it walks."""
    consts = _stream_constants(tree)
    covered = set()
    for ints in consts.values():
        covered |= ints
    return covered, consts


def _detector_test_fn(tree):
    """The module-level test function that covers the detector / tail case, by
    name (contains 'detector' or 'tail'), or ``None``."""
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef) and node.name.startswith("test")
                and ("detector" in node.name.lower()
                     or "tail" in node.name.lower())):
            return node
    return None


def _asserts_strict_superset(fn):
    """True iff ``fn`` references ``DETECTOR_STREAMS`` AND contains a strict
    ``>`` comparison -- a superset / ragged-tail relationship, not mere ts
    equality.  This is what makes the detector-stream assertion MEANINGFUL."""
    if fn is None:
        return False
    refs_detector = any(isinstance(n, ast.Name) and n.id == "DETECTOR_STREAMS"
                        for n in ast.walk(fn))
    has_strict_gt = any(
        isinstance(n, ast.Compare) and any(isinstance(op, ast.Gt) for op in n.ops)
        for n in ast.walk(fn))
    return refs_detector and has_strict_gt


def test_detector_stream_now_covered():
    """The gate's covered-stream set must include a jungfrau DETECTOR stream
    (5 or 7), keep the small-stream coverage, and assert a strict superset."""
    _src, tree = _load_target()
    covered, consts = covered_streams(tree)

    # The small-stream coverage must be KEPT (the fix ADDS to it).
    assert REQUIRED_SMALL <= covered, (
        f"the gate must keep its small-stream coverage {sorted(REQUIRED_SMALL)}; "
        f"the covered-stream set parsed from {os.path.basename(TARGET)} is "
        f"{sorted(covered)} (constants: { {k: sorted(v) for k, v in consts.items()} })")

    # THE FIX: a jungfrau DETECTOR stream (5 or 7) is now covered.
    det_covered = covered & DETECTOR_STREAMS_UNIVERSE
    assert det_covered, (
        "GATE-06 NOT fixed: the covered-stream set of "
        f"{os.path.basename(TARGET)} is {sorted(covered)} -- it includes NO "
        f"jungfrau DETECTOR stream {sorted(DETECTOR_STREAMS_UNIVERSE)}.  The "
        "'SMD is optional' gate still only walks the small non-detector streams "
        "(0, 1, 4), so the SMD-free bigdata build is never exercised on real "
        "detector data and the ragged-tail superset case (17982 vs 17872, on "
        "streams 5 & 7) is never met.  Add DETECTOR_STREAMS and a detector-stream "
        "tail scan.")

    # The detector coverage must be MEANINGFUL: a test function that reaches the
    # tail and asserts a STRICT superset, not just exact ts equality.
    fn = _detector_test_fn(tree)
    assert fn is not None, (
        f"a detector stream {sorted(det_covered)} appears in the covered set but "
        "no detector/tail test function scans it -- the stream is declared, not "
        "actually walked to the ragged tail")
    assert _asserts_strict_superset(fn), (
        f"the detector test {fn.name!r} must assert a STRICT superset -- a '>' "
        "comparison of the SMD-free event set against the SMD-indexed set, "
        "referencing DETECTOR_STREAMS -- so it reaches the ragged tail rather "
        "than asserting mere ts equality (the head-prefix / equality-only "
        "disease GATE-06 fixes)")

    print(f"[gate-06] {os.path.basename(TARGET)} covers detector stream(s) "
          f"{sorted(det_covered)} (covered set {sorted(covered)}); "
          f"{fn.name!r} asserts a strict superset -- the ragged tail is reached")
    return sorted(det_covered)


def main():
    print("=" * 72)
    print("GATE-06 regression: the 'SMD is optional' gate must scan a DETECTOR "
          "stream")
    print("=" * 72)
    covered = test_detector_stream_now_covered()
    print(f"[ok] detector stream(s) {covered} covered by the bigdata-scan gate")
    print("\nGATE-06 REGRESSION PASSED")


if __name__ == "__main__":
    main()
