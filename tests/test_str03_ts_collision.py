#!/usr/bin/env python3
"""STR-03 regression: the ts-keyed merge must not lose or misclassify events on
a timestamp collision.

The bug
-------
``RunIndex._merge_streams`` builds the unified ``timestamps`` / ``entries``
index by keying every per-stream ``(ts, chunk_path, off, size)`` record into a
dict by its 64-bit timestamp::

    merged.setdefault(ts, {})[stream] = (chunk_path, off, size)   # parent

Two hazards hide in that one assignment:

1. **Duplicate L1Accept ts within one stream.**  If a stream contributes two
   L1Accepts with the SAME ts (a DAQ/clock anomaly), the second assignment
   silently overwrites the first in ``merged[ts][stream]`` -- one event vanishes
   from the index with no error, no count, no word.  ``read_event(ts)`` /
   ``_position_of`` can then only ever return one of them.

2. **A transition sharing a ts with an L1Accept.**  A transition dgram is not a
   real event.  If it is keyed into the same slot as an L1Accept it can flip the
   event's classification -- the entry ends up pointing at transition bytes (a
   real event dropped) or a transition is counted as an event.

Note that a ts shared ACROSS streams is NOT a collision: that is the normal
cross-stream merge (the same event assembled from several streams) and must be
preserved exactly.  Only a repeat WITHIN one stream, or a transition landing on
an event's slot, is the fault.

The fix (STR-03)
----------------
* A second event landing on an already-filled ``(ts, stream)`` slot RAISES a
  clear ``ValueError`` naming the colliding ts and stream, instead of silently
  overwriting.  The entry shape holds only one dgram per ``(ts, stream)`` and the
  ts-unique bisect index (:meth:`RunIndex._position_of`) cannot address two
  events at one ts, so "preserve both" is not representable without changing the
  persisted format -- raising loudly is the safe minimal policy, and it never
  fires on healthy data (each real L1Accept has a unique ts within its stream).
* A transition (``is_event=False``) that shares a ts with an L1Accept never
  becomes -- nor displaces -- an event: the L1Accept stays indexed, the
  transition is additive-only.

The discriminator
-----------------
On the PARENT commit:

* case (a) does NOT raise -- it silently keeps one event and drops the other, so
  the ``assertRaises(ValueError)`` fails;
* case (b) with a transition record either crashes the parent's 4-tuple unpack
  or (if fed as a plain event) silently overwrites, so the "event survives,
  transition excluded" assertions fail.

On the FIXED commit both cases behave as asserted, and the no-collision control
is byte-for-byte identical to the parent (verified separately by a differential
run and pinned here as an explicit expected value).

This test is SELF-CONTAINED: stdlib + numpy only, NO psana, NO SLAC data.  It
drives ``RunIndex._merge_streams`` directly on synthetic per-stream records
(the exact shape the SMD / bigdata scans feed it), built on a bare
``RunIndex.__new__`` instance so no run config, no files, and no timing-stream
clamp are needed (``include_shutdown_tail=True`` keeps every event).  Runs from
any cwd::

    python3 tests/test_str03_ts_collision.py
"""

import os
import sys

import numpy as np  # noqa: F401  (asserts the numpy-only env is present)

# --- locate the package (parent of this tests dir) --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")  # .../<repo>/src (holds psdata/)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import psdata.index as IX  # noqa: E402


def _fresh_index():
    """A bare :class:`RunIndex` with just the two lists ``_merge_streams``
    writes -- no run config, no files.  ``_merge_streams`` with
    ``include_shutdown_tail=True`` never consults the run config or the
    timing-stream clamp, so this is all it needs to run."""
    idx = IX.RunIndex.__new__(IX.RunIndex)
    idx.timestamps = []
    idx.entries = []
    return idx


def _merge(per_stream):
    idx = _fresh_index()
    # include_shutdown_tail=True -> no timing clamp, so an event is indexed iff
    # it is kept by the merge; is_event(ts) <=> (ts in idx.timestamps).
    idx._merge_streams(per_stream, include_shutdown_tail=True)
    return idx


# --------------------------------------------------------------------------
# control: the normal, no-collision case is unchanged (byte-exact)
# --------------------------------------------------------------------------
def test_no_collision_control_unchanged():
    """Distinct-ts events, plus a ts shared ACROSS streams (the normal
    cross-stream merge), produce exactly the event set and ordering the parent
    produced -- pinned to explicit expected values so any drift reddens.

    A ts shared across streams is NOT a collision; it is one event assembled
    from several streams and must merge into a single entry carrying both.
    """
    per_stream = {
        0: [(200, "c000", 50, 60), (100, "c000", 0, 50)],   # unshuffled ts order
        1: [(100, "c000", 0, 40)],                           # ts=100 also in s0
        2: [(300, "c001", 0, 70)],
    }
    idx = _merge(per_stream)

    # ascending, one position per DISTINCT ts, cross-stream ts=100 combined
    assert idx.timestamps == [100, 200, 300], idx.timestamps
    assert idx.entries == [
        {0: ("c000", 0, 50), 1: ("c000", 0, 40)},   # ts=100: both streams merged
        {0: ("c000", 50, 60)},                        # ts=200: stream 0 only
        {2: ("c001", 0, 70)},                         # ts=300: stream 2 only
    ], idx.entries
    print("[ok] no-collision control: distinct-ts events kept, cross-stream "
          "same-ts merged into one entry -- event set/order byte-unchanged")


# --------------------------------------------------------------------------
# (a) two L1Accepts sharing a ts WITHIN one stream -> neither silently dropped
# --------------------------------------------------------------------------
def test_duplicate_l1accept_ts_not_silently_dropped():
    """Two L1Accepts with the SAME ts in the SAME stream is a DAQ/clock anomaly.

    Chosen policy: RAISE a clear ``ValueError`` naming the colliding ts, so an
    event is never lost without a word (the ts-unique index cannot address two
    events at one ts, so keeping both is not representable).

    PARENT: the second assignment silently overwrites the first in
    ``merged[ts][stream]`` -- exactly ONE event is indexed and no error is
    raised, so this ``assertRaises`` fails.
    FIX: raises, naming the ts.
    """
    per_stream = {0: [(100, "c000", 0, 50), (100, "c000", 50, 60)]}

    raised = None
    try:
        _merge(per_stream)
    except ValueError as e:
        raised = e
    assert raised is not None, (
        "STR-03: two L1Accepts sharing a ts in one stream did NOT raise -- one "
        "event was silently dropped by the dict overwrite (parent behaviour)")
    msg = str(raised)
    assert "100" in msg, ("the diagnostic must name the colliding timestamp; "
                          "got: " + msg)
    assert "duplicate" in msg.lower(), ("the diagnostic must call out the "
                                        "duplicate ts; got: " + msg)

    # and prove the parent bug it guards: without the guard, only one of the two
    # distinct dgrams would survive -- both are genuinely different records.
    assert (100, "c000", 0, 50) != (100, "c000", 50, 60)
    print("[ok] (a) duplicate L1Accept ts in one stream RAISES a named "
          "ValueError -- no event silently dropped")


# --------------------------------------------------------------------------
# (b) a transition sharing a ts with an L1Accept must not flip is_event
# --------------------------------------------------------------------------
def test_transition_sharing_ts_does_not_flip_is_event():
    """A transition dgram (``is_event=False``) that shares a ts with an
    L1Accept must NOT change the event's classification: the L1Accept stays
    indexed (``is_event`` True) and the transition is excluded from the entry
    and never counted as an event.

    PARENT: has no notion of a transition record in the merge -- it either
    crashes on the 5-tuple unpack or, fed as a plain 4-tuple event, silently
    overwrites the L1Accept's slot; either way the assertions below fail.
    FIX: transition is additive-only; the event survives untouched.
    """
    # transition rides on the SAME ts=100 as the L1Accept, in another stream
    per_stream = {
        0: [(100, "c000", 0, 50)],              # the real L1Accept (is_event)
        1: [(100, "c000", 999, 8, False)],      # a transition sharing ts=100
    }
    idx = _merge(per_stream)

    assert idx.timestamps == [100], (
        "the L1Accept must stay indexed (is_event True) despite a colliding "
        "transition ts; got " + repr(idx.timestamps))
    assert idx.entries == [{0: ("c000", 0, 50)}], (
        "the transition must not enter the event's entry nor displace it; got "
        + repr(idx.entries))
    assert 1 not in idx.entries[0], "transition stream leaked into the event entry"

    # a transition on the SAME stream+ts as the event must likewise not clobber
    # the event (the event wins, no raise, transition dropped)
    per_stream2 = {
        0: [(100, "c000", 0, 50), (100, "c000", 999, 8, False)],
    }
    idx2 = _merge(per_stream2)
    assert idx2.timestamps == [100] and idx2.entries == [{0: ("c000", 0, 50)}], (
        "a transition sharing the event's own (ts, stream) slot must not flip "
        "or drop the event; got " + repr((idx2.timestamps, idx2.entries)))

    # a LONE transition ts creates no phantom event
    idx3 = _merge({0: [(300, "c000", 0, 8, False)]})
    assert idx3.timestamps == [] and idx3.entries == [], (
        "a transition with no matching L1Accept must not create an event; got "
        + repr((idx3.timestamps, idx3.entries)))

    print("[ok] (b) a transition sharing a ts with an L1Accept leaves is_event "
          "True, is excluded from the entry, and creates no phantom event")


def main():
    print("=" * 72)
    print("STR-03 regression: timestamp-collision safety in the ts-keyed merge")
    print("=" * 72)
    test_no_collision_control_unchanged()
    test_duplicate_l1accept_ts_not_silently_dropped()
    test_transition_sharing_ts_does_not_flip_is_event()
    print("\nALL STR-03 TS-COLLISION CHECKS PASSED")


if __name__ == "__main__":
    main()
