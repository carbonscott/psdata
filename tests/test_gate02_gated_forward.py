#!/usr/bin/env python3
"""GATE-02 regression: the gated-forward check must not be able to skip itself.

US-002's ``test_forward_gated_to_index_event_set`` is the gate that locks the
SMD-gating invariant (FAIL-01): on a run with a ragged DAQ-shutdown tail the
ungated k-way merge surfaces trailing L1Accepts the SMD writer never indexed --
17982 in bigdata vs 17872 indexed on mfx100848724/r51 -- and ``Run.events()``
must filter the merge to the indexed timestamps.

That gate used to open with::

    if os.environ.get("PSDATA_SKIP_SLOW"):
        print("SKIP gated-forward regression (PSDATA_SKIP_SLOW set)")
        return

and the canonical runner invokes the suite as ``PSDATA_SKIP_SLOW=1 bash
run_tests.sh`` (tests/env_store_suite.sbatch, tests/env_store_verify.sbatch).
So in every "green" run this project ever recorded, the gate printed SKIP,
returned None, the file still exited 0, and the runner counted it a PASS.  The
invariant was never actually checked: a gate that the runner always switches off
is not a gate.

This test reproduces that exact condition -- it sets PSDATA_SKIP_SLOW=1 itself,
so it pins the behaviour no matter what the ambient environment does -- and
demands a real event count back from the gate.  A ``None`` return means the gate
skipped itself, and that is the defect.

Deliberately lives in its OWN file, importing the gate from the tree it runs in,
so that it carries no part of the fix: dropped into a tree whose
``test_stream_us002.py`` still has the escape hatch, it fails.

SLOW: it drives one full forward pass over a 10-stream Jungfrau run.  That is
the point.  Needs the production psana env (psconda.sh) on host sdfiana025 and
the reference dataset.
"""

import os
import sys

# --- locate the package and the sibling test module -------------------------
# Mirror test_stream_us002.py: src/ (holds psdata/) is the parent dir's src.
# Also put this tests/ dir on sys.path so the import below resolves regardless
# of the cwd the runner invokes us from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../<repo>/src
for _p in (_PKG_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The gate under test, taken from the tree we are running in.
from test_stream_us002 import test_forward_gated_to_index_event_set


def test_gated_forward_cannot_be_skipped():
    """The gate must run -- and report how far it walked -- even under the
    canonical runner's PSDATA_SKIP_SLOW=1."""
    prev = os.environ.get("PSDATA_SKIP_SLOW")
    os.environ["PSDATA_SKIP_SLOW"] = "1"    # the condition the bug fired under
    try:
        n = test_forward_gated_to_index_event_set()
    finally:
        if prev is None:
            os.environ.pop("PSDATA_SKIP_SLOW", None)
        else:
            os.environ["PSDATA_SKIP_SLOW"] = prev

    # bool is an int subclass; exclude it so a `return True` stub cannot pass.
    assert isinstance(n, int) and not isinstance(n, bool), (
        f"GATE-02: test_forward_gated_to_index_event_set() returned {n!r} under "
        "PSDATA_SKIP_SLOW=1 instead of an event count -- a None return means the "
        "gate skipped itself.  That is the defect: the canonical runner always "
        "sets PSDATA_SKIP_SLOW=1, so the SMD-gating regression (FAIL-01) never "
        "executed, yet the suite still reported green")
    # A gate that silently walks zero events is not a gate.
    assert n > 0, (
        f"GATE-02: the gated-forward check reported {n} events under "
        "PSDATA_SKIP_SLOW=1 -- it must walk the whole run, not short-circuit")
    print(f"[gate-02] gated forward ran under PSDATA_SKIP_SLOW=1 and walked "
          f"{n} events; no environment variable can disable the gate")
    return n


def main():
    print("=" * 72)
    print("GATE-02 regression: the gated-forward check cannot skip itself")
    print("=" * 72)
    test_gated_forward_cannot_be_skipped()
    print("[ok] PSDATA_SKIP_SLOW=1 no longer disables the SMD-gating regression")
    print("\nGATE-02 REGRESSION PASSED")


if __name__ == "__main__":
    main()
