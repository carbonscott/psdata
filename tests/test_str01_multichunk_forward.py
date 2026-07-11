#!/usr/bin/env python3
"""STR-01 regression: multi-chunk FORWARD streaming must follow the chunk roll.

Run on sdfiana025 with the production psana env (for the SLAC data mount):

    source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
    bash psdata/run_tests.sh <abs path to this file>

The bug (STR-01)
----------------
A long run's bigdata is split into chunk files ``...-s0NN-c000.xtc2``,
``-c001``, ...; the DAQ rolls a stream from ``c000`` to ``c001`` at 250-500 GB
and per-chunk dgram offsets restart at 0.  The random-access INDEX path already
follows this roll (``index._scan_bigdata_stream`` over ``_enumerate_bd_chunks``;
``read_event_at`` reads the right chunk file).  But the FORWARD streaming path's
``_StreamCursor`` used to open only ``c000`` and, when it ran off the end, go
silently dead -- so for events past the roll, a detector carried on a rolled
stream lost its segments and ``evt.stack(det)`` returned ``None``, while
``read_event_at(k)`` returned a full frame from ``c001`` for the SAME timestamp.
Same run, two answers -- the reader contradicting itself on the tail of the run.

What this test does (the discriminator)
---------------------------------------
On a real multi-chunk run whose jungfrau streams roll (``mfx101343025 / r35``:
streams 7/8/9/10 roll ``c000 -> c001`` around k=36600 and k=42720 of 48000), it:

  1. Builds the random-access index (the CORRECT reference) and finds ``roll_k``
     -- the first event position whose entry references a non-``c000`` chunk.
  2. Collects a handful of post-roll checkpoint events whose jungfrau frame is
     FULLY present in the index, keyed by TIMESTAMP (so the comparison is robust
     to any EOB-event position skew between the forward and index event sets).
  3. Forward-streams with ``psdata.events`` (the path under test) and, for each
     checkpoint timestamp, asserts ``evt.stack('jungfrau')`` is a full-shaped,
     non-None frame that is byte-identical to the index path's frame for that
     timestamp.

On the PARENT (unfixed) the forward cursor is dead past the roll, so
``stack('jungfrau')`` is ``None`` at the first post-roll checkpoint -> the
``got is not None`` assertion fails and the test exits nonzero.  On the FIX the
cursor follows the roll, so every post-roll frame matches the index -> passes.

A pre-roll control checkpoint is also matched (forward == index before the roll)
so a failure past the roll is unambiguously the roll bug, not a harness fault.

This test needs the real rolled run (the bug is positional and late: reaching it
means a forward pass to ~k=36600); it CANNOT run without SLAC data, so it runs on
a compute node, not locally.  It needs NO psana -- it compares psdata's forward
path against psdata's own index path.
"""

import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")   # .../<repo>/src (holds psdata/)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import psdata                       # noqa: E402
from psdata import index as psindex  # noqa: E402

from _skips import skip             # noqa: E402  (machine-readable skips, HYG-03)

# ---- multi-chunk reference run (streams 7/8/9/10 roll c000 -> c001) --------
MC_EXP = "mfx101343025"
MC_RUN = 35
# The instrument dir has been seen spelled both ways; test_robust/test_batch read
# this same run and take whichever exists so a path-case mismatch cannot silently
# turn the check into a no-op (HYG-03: an un-run check is not a pass).
MC_DIR_CANDIDATES = (
    "/sdf/data/lcls/ds/mfx/mfx101343025/xtc",
    "/sdf/data/lcls/ds/MFX/mfx101343025/xtc",
)
JUNGFRAU = "jungfrau"          # the detector carried on the rolled streams
POST_ROLL_TO_CHECK = 5         # a handful of post-roll events (keeps the gate cheap)
# Safety bound so a pathological run never streams forever; the loop normally
# stops right after the last checkpoint (well below the run's 48000 events).
MAX_STREAM_EVENTS = 200_000


def _mc_dir():
    """The multi-chunk run's xtc dir if reachable on this node, else ``None``
    (the caller emits a named SKIP -- a data-absent node cannot run the oracle
    and SHOULD redden the suite, but a named skip is clearer than an
    AssertionError; mirrors ``test_batch_us009::test_multichunk_batch``)."""
    return next((c for c in MC_DIR_CANDIDATES if os.path.isdir(c)), None)


def _mc_stream_files(d):
    """The rolled run's per-stream c000 bigdata files, keyed by real stream index
    (so jungfrau segments map to the right streams)."""
    paths = sorted(glob.glob(os.path.join(
        d, f"{MC_EXP}-r{MC_RUN:04d}-s*-c000.xtc2")))
    paths = psdata.filter_c000(paths)
    assert paths, f"no c000 stream files for {MC_EXP} r{MC_RUN} under {d}"
    return {psdata.stream_index_of(p): p for p in paths}


def _first_complete_jungfrau_at_or_after(ridx, k0, n):
    """Scan indexed positions from ``k0`` upward; return ``(k, ts, stack)`` for
    the first event whose jungfrau frame is fully present in the INDEX path, or
    ``None`` if none before the end.  Restricting checkpoints to
    index-complete events makes a forward-path ``None`` at the same timestamp
    unambiguously the STR-01 data loss (not a genuine missing segment)."""
    k = k0
    while k < n:
        ts = ridx.timestamps[k]
        st = ridx.read_event(ts).stack(JUNGFRAU)
        if st is not None:
            return k, ts, st
        k += 1
    return None


def test_forward_streaming_follows_chunk_roll():
    d = _mc_dir()
    if d is None:
        # Data-absent node: the oracle genuinely cannot run here.  Emit a named
        # SKIP (NOT allowlisted -- it SHOULD redden the suite) rather than a hard
        # AssertionError.  On a data-PRESENT node d is not None and this is inert,
        # so the milano gate is unaffected.
        return skip(
            "str01_multichunk_run_missing",
            f"the multi-chunk run {MC_EXP}/r{MC_RUN} is not reachable on this "
            f"node (looked under {list(MC_DIR_CANDIDATES)}); the forward-roll "
            f"check needs real multi-chunk data (streams that roll c000->c001)")
    files = _mc_stream_files(d)
    rc = psdata.discover(files)

    # --- the CORRECT reference: the random-access index already follows the roll
    ridx = psindex.build_index(files, run_config=rc)
    n = ridx.n_events
    assert n > 0, "empty index"
    assert ridx.multichunk_streams, (
        f"{MC_EXP}/r{MC_RUN} shows no rolled stream (multichunk_streams empty); "
        f"this run cannot exercise STR-01 -- expected streams 7/8/9/10 to roll "
        f"c000->c001. Wrong run or a data-layout change.")

    # earliest event position served from a non-c000 chunk == the roll boundary
    # (the same discovery test_batch_us009.test_multichunk_batch uses).
    roll_k = None
    for k in range(n):
        if any("c000" not in path
               for (path, _o, _s) in ridx.entries[k].values()):
            roll_k = k
            break
    assert roll_k is not None, (
        "no chunk-path switch found in the index entries despite a multi-chunk "
        "stream -- index inconsistency")
    roll_ts = ridx.timestamps[roll_k]

    # --- pre-roll control: forward must already agree with the index here ---
    pre = _first_complete_jungfrau_at_or_after(ridx, max(0, roll_k // 2), roll_k)
    assert pre is not None, "no complete-jungfrau indexed event found before the roll"
    pre_k, pre_ts, pre_ref = pre

    # --- post-roll checkpoints: complete-in-index jungfrau events past the roll,
    #     keyed by timestamp (robust to EOB-event position skew) ---
    refs = {}            # ts -> reference jungfrau stack from the index path
    scan_k = roll_k
    while scan_k < n and len(refs) < POST_ROLL_TO_CHECK:
        hit = _first_complete_jungfrau_at_or_after(ridx, scan_k, n)
        if hit is None:
            break
        hk, hts, hst = hit
        refs[hts] = hst
        scan_k = hk + 1
    assert refs, (
        f"no complete-jungfrau indexed events found at/after the roll (k>="
        f"{roll_k}); cannot discriminate the STR-01 regression on this run")
    check_ts = set(refs)
    last_check_ts = max(check_ts)

    # --- the path under test: forward-stream in ascending ts and follow (or,
    #     on the parent, fail to follow) the roll ---
    matched = {}         # ts -> forward jungfrau stack (post-roll checkpoints)
    pre_ok = False
    seen = 0
    for evt in psdata.events(files, run_config=rc):
        seen += 1
        ts = evt.timestamp

        if ts == pre_ts and not pre_ok:
            got = evt.stack(JUNGFRAU)
            assert got is not None, (
                f"pre-roll control: forward stack('{JUNGFRAU}') is None at "
                f"ts={pre_ts} (k={pre_k}, BEFORE the roll) though the index has "
                f"a {pre_ref.shape} frame -- harness/reader fault, not STR-01")
            assert got.shape == pre_ref.shape and np.array_equal(got, pre_ref), (
                f"pre-roll control: forward != index at ts={pre_ts} (k={pre_k}) "
                f"before any roll -- harness/reader fault, not STR-01")
            pre_ok = True

        if ts in check_ts and ts not in matched:
            got = evt.stack(JUNGFRAU)
            ref = refs[ts]
            # THE STR-01 ASSERTION.  Past the roll the forward path must return a
            # full frame that byte-matches the index path for the SAME timestamp.
            # PARENT: the rolled cursor is dead here -> jungfrau missing its
            # rolled segments -> stack() is None -> this fails (exit nonzero).
            # FIX: the cursor followed the roll -> full frame == index -> passes.
            assert got is not None, (
                f"STR-01: forward stack('{JUNGFRAU}') is None at ts={ts} (an "
                f"event PAST the chunk roll at k={roll_k}), but the index path "
                f"returns a full {ref.shape} frame for the same timestamp -- the "
                f"forward path silently lost the rolled chunk's data")
            assert got.shape == ref.shape and got.dtype == ref.dtype, (
                f"STR-01: forward stack shape/dtype {got.shape}/{got.dtype} != "
                f"index {ref.shape}/{ref.dtype} at ts={ts}")
            assert np.array_equal(got, ref), (
                f"STR-01: forward stack != index stack at ts={ts} past the roll "
                f"-- the forward and random-access paths disagree on the rolled "
                f"chunk's bytes")
            matched[ts] = got

        # Stop shortly after the last checkpoint -- do NOT run the whole run.
        if ts >= last_check_ts and len(matched) >= len(check_ts) and pre_ok:
            break
        if seen > MAX_STREAM_EVENTS:
            break

    assert pre_ok, (
        f"pre-roll control event ts={pre_ts} (k={pre_k}) was never yielded by "
        f"the forward path -- unexpected; cannot trust the post-roll comparison")
    missing = check_ts - set(matched)
    assert not missing, (
        f"STR-01: forward streaming never yielded {len(missing)} post-roll "
        f"checkpoint event(s) that the index has ({sorted(missing)}) -- the "
        f"forward path stopped at the roll instead of following it")

    rolled = sorted(ridx.multichunk_streams)
    ridx.close()
    print(f"[ok] STR-01: forward streaming followed the chunk roll -- pre-roll "
          f"control matched at k={pre_k}; {len(matched)} post-roll jungfrau "
          f"frames (ts>={roll_ts}, first roll at k={roll_k}) byte-identical to "
          f"the random-access index path (rolled streams={rolled})")


def main():
    print("=" * 72)
    print("STR-01 regression: multi-chunk forward streaming follows the roll")
    print("=" * 72)
    test_forward_streaming_follows_chunk_roll()
    print()
    print("STR-01 REGRESSION CHECK PASSED")


if __name__ == "__main__":
    main()
