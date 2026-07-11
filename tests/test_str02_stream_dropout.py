#!/usr/bin/env python3
"""STR-02 regression: a stream that DROPS OUT mid-run must be a DISTINCT,
queryable condition -- not an indistinguishable ``None``.

The bug (STR-02)
----------------
``psdata.stream.events`` assembles per-event detector data by a k-way merge over
the run's several bigdata streams.  When a stream stops producing dgrams partway
through a run (a killed DAQ / transport dropout -- it hits EOF while the OTHER
streams carry the run on), the merge simply skips it::

    for cur in cursors:
        ts = cur.head_ts()
        if ts is None:
            continue            # <-- dropped stream silently skipped

so every detector carried on that stream is absent from ``seg_index`` for every
later event, and ``evt.stack(det)`` / ``evt.raw(det)`` return ``None``.  That
``None`` is byte-for-byte the SAME answer a detector gives when it is genuinely,
legitimately not present this event (the missing-segment rule).  A scientist
processing the run therefore cannot tell a *data-quality fault* ("this detector's
stream dropped out at event N") from a *legitimate absence* ("this detector was
simply not present this event").  psana names this condition
(``DroppedContribution`` -- WRITE-03); psdata, on the parent, surfaces it as an
unnamed hole.

The fix (STR-02)
----------------
``events`` records the streams that had dropped out by the time each event was
assembled on the :class:`~psdata.stream.Event` (``Event.dropped_streams``), and
``Event.detector_status(det)`` classifies each detector's ``None`` as:

  * ``"dropped"`` -- a stream carrying it dropped out (data-quality fault), vs
  * ``"absent"``  -- its streams are all still live; it merely did not contribute
    a complete frame this event (legitimate).

The returned frame VALUES are unchanged: ``dropped_streams`` is a pure
side-channel, so ``stack`` / ``raw`` never change.  On the canonical (gated)
event set of a healthy run no stream drops, so the signal stays empty and no
detector reports ``"dropped"``.  (The test is structural, so on the ungated path
a benign DAQ-shutdown tail -- some streams writing extra trailing L1Accepts past
where others stopped -- would flag the earlier-stopping stream on those tail
events; the gated ``Run.events()`` default filters them out.  This synthetic
control below is a clean run, where the signal is genuinely silent.)

The discriminator (fail on parent, pass on fix)
-----------------------------------------------
On a synthetic two-stream run where stream 1 stops after two events while stream
0 carries on:

  * Both a dropped-stream detector and a genuinely-absent detector return
    ``None`` from ``stack()`` -- on BOTH parent and fix (the fix does not change
    the value).
  * The parent exposes NO way to tell them apart: it has no ``dropped_streams``
    attribute and no ``detector_status`` method.  This probe reads them through
    ``getattr(..., None)``, so on the parent the discriminating status is
    ``None`` and the ``== "dropped"`` assertion FAILS -> the test exits non-zero.
  * On the fix the dropped detector reports ``"dropped"`` and the absent one
    reports ``"absent"`` -- distinguishable -> the test PASSES.

A no-dropout control (stream 1 also runs to the end) asserts the signal never
fires: ``dropped_streams`` is empty for every event, no detector is ``"dropped"``,
and the yielded event set/services are exactly as before (byte-unchanged).

Self-contained: standard library + numpy only.  NO psana, NO SLAC data, NO real
xtc2 file -- the per-stream dgram files are built by hand, in bytes, to the exact
24-byte dgram header ``parse_dgram_header`` reads (empty-payload L1Accepts), and
the RunConfig (detector -> stream mapping) is fabricated directly, as the sibling
synthetic tests do (``test_bigdata_scan_us012``'s ``_fake_rc_with_timing``,
``test_det10_unknown_version``'s hand-built Names bytes).  It drives the real
``psdata.stream.events`` k-way merge -- the code under test -- end to end.
cwd-robust; ``main()`` + ``__main__``.

Parent SHA (silent-None behaviour this probe reddens): b12e5ca6355676e2ad5fad970a0c5031d11dc8ff
"""

import os
import struct
import sys
import tempfile

import numpy as np  # noqa: F401  (psdata's only declared dependency; asserts env)

# --- locate the package under test (parent-of-tests/src), cwd-robust ---------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")  # .../<repo>/src (holds psdata/)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import psdata.format as fmt      # noqa: E402
import psdata.stream as pstream  # noqa: E402


# ===========================================================================
# Hand-build a minimal L1Accept dgram, byte-for-byte to the 24-byte header
# parse_dgram_header reads (src/psdata/format.py):
#   Transition = uint32 nsec, uint32 sec, uint32 env   (service = (env>>24)&0xf)
#   Xtc        = uint32 src, uint16 damage, uint16 typeid, uint32 extent
# extent == XTC_HDR (12) means an empty payload -> total on-disk size 24 bytes.
# An empty payload carries no ShapesData, so every detector is `None` from
# stack()/raw() -- exactly the "indistinguishable None" this probe is about.
# ===========================================================================
def make_l1_dgram(ts):
    nsec = ts & 0xFFFFFFFF
    sec = (ts >> 32) & 0xFFFFFFFF
    env = fmt.SERVICE_L1ACCEPT << 24         # service field lives in bits 24..27
    transition = struct.pack("<III", nsec, sec, env)
    xtc = struct.pack("<IHHI", 0, 0, 0, fmt.XTC_HDR)  # src, damage, typeid, extent
    return transition + xtc


def _write_stream(path, ts_list):
    with open(path, "wb") as fh:
        for ts in ts_list:
            fh.write(make_l1_dgram(ts))


def _det(name, det_type, stream, seg=0, alg="raw"):
    """A one-segment DetectorInfo on ``stream`` (fabricated directly, as
    test_bigdata_scan_us012._fake_rc_with_timing does)."""
    d = fmt.DetectorInfo(name, det_type, f"{name}-id")
    d.algs[alg] = {}                          # key presence is all status needs
    d.segments[alg] = {seg}
    d.seg_to_stream[(alg, seg)] = stream
    return d


def build_run(tmpdir, s0_ts, s1_ts):
    """A synthetic two-stream run in ``tmpdir``.

    Stream 0 carries detA and detC (both on the SAME live stream); stream 1
    carries detB.  Returns ``(files, run_config)`` ready for
    ``psdata.stream.events``.  Detectors declare no data payload, so every
    detector is ``None`` this event -- the point is *why* it is ``None``.
    """
    p0 = os.path.join(tmpdir, "synth-r0001-s000.xtc2")
    p1 = os.path.join(tmpdir, "synth-r0001-s001.xtc2")
    _write_stream(p0, s0_ts)
    _write_stream(p1, s1_ts)

    rc = fmt.RunConfig()
    rc.stream_files = {0: p0, 1: p1}
    # cfg_end == 0 -> cursors read dgrams from offset 0 (no Configure prefix).
    rc.stream_configs = {0: (None, 0), 1: (None, 0)}
    rc.raw_tables = {0: {}, 1: {}}            # events() indexes against these
    rc.detectors = {
        "detA": _det("detA", "epixA", stream=0),
        "detB": _det("detB", "epixB", stream=1),  # the stream that drops out
        "detC": _det("detC", "epixC", stream=0),
    }
    return rc.stream_files, rc


# --- read the fix's signal through getattr so the PARENT (which has neither the
#     attribute nor the method) yields a clean, discriminating assertion failure
#     rather than an AttributeError -----------------------------------------
def _dropped_streams(evt):
    return getattr(evt, "dropped_streams", None)


def _status(evt, det, alg="raw"):
    fn = getattr(evt, "detector_status", None)
    return None if fn is None else fn(det, alg)


def _stream_events(files, rc):
    return list(pstream.events(files, run_config=rc))


# ===========================================================================
# 1. THE DISCRIMINATOR: a mid-run drop-out is distinguishable from absence
# ===========================================================================
def test_dropout_is_distinguishable():
    with tempfile.TemporaryDirectory() as td:
        # stream 0: events at ts 10,20,30,40; stream 1 STOPS after ts 20.
        files, rc = build_run(td, s0_ts=[10, 20, 30, 40], s1_ts=[10, 20])
        evts = _stream_events(files, rc)

    # the k-way merge yields one event per distinct ts, ascending.
    assert [e.timestamp for e in evts] == [10, 20, 30, 40], \
        [e.timestamp for e in evts]
    assert all(e.service == fmt.SERVICE_L1ACCEPT for e in evts)

    by_ts = {e.timestamp: e for e in evts}

    # --- THE DISCRIMINATOR, first: post-dropout events (stream 1 gone, stream
    #     0 carries the run).  This is what must fail on the parent. ---------
    for ts in (30, 40):
        e = by_ts[ts]

        # PRECONDITION (true on parent AND fix): at the VALUE level a
        # dropped-stream detector and a genuinely-absent detector are the same
        # indistinguishable None.  The fix must NOT change this value.
        assert e.stack("detB") is None, f"ts={ts}: detB value must stay None"
        assert e.stack("detC") is None, f"ts={ts}: detC value must stay None"

        # THE STR-02 ASSERTIONS.  The reader must expose a side-channel that
        # tells the two Nones apart.  On the PARENT there is none:
        # dropped_streams is absent (getattr -> None) and detector_status does
        # not exist (-> None), so these fail and the probe exits non-zero.
        sB = _status(e, "detB")
        sC = _status(e, "detC")
        assert sB == "dropped", (
            f"STR-02: ts={ts}: detB is None because its stream (1) DROPPED OUT "
            f"mid-run, but the reader reports {sB!r} -- indistinguishable from a "
            f"genuine absence. A drop-out must be a distinct, queryable condition.")
        assert sC == "absent", (
            f"STR-02: ts={ts}: detC is a genuine absence (its stream 0 is live) "
            f"and must report 'absent', not {sC!r}.")
        assert sB != sC, (
            f"STR-02: ts={ts}: a stream drop-out ({sB!r}) must be DISTINGUISHABLE "
            f"from a legitimate absence ({sC!r}); on the parent both are an "
            f"indistinguishable None.")

        # the queryable stream-level and detector-level views agree.
        assert _dropped_streams(e) == frozenset({1}), (
            f"ts={ts}: dropped_streams must name stream 1; got "
            f"{_dropped_streams(e)!r}")
        assert e.dropped_detectors() == ["detB"], (
            f"ts={ts}: dropped_detectors must be exactly [detB]; got "
            f"{e.dropped_detectors()!r}")
        # detA is also on the (live) stream 0 with no data -> a plain absence.
        assert _status(e, "detA") == "absent", (
            f"ts={ts}: detA (live stream 0, no data) is a genuine absence; got "
            f"{_status(e, 'detA')!r}")

    # --- CONTROL: pre-dropout events (stream 1 still live).  Tolerant of the
    #     parent (None == "no signal") so it never discriminates -- only the
    #     post-dropout block above does.  On the fix the signal is present and
    #     must be silent here (empty / not "dropped"). ----------------------
    for ts in (10, 20):
        e = by_ts[ts]
        assert _dropped_streams(e) in (None, frozenset()), (
            f"ts={ts}: no stream has dropped yet, so the drop-out signal must be "
            f"silent (empty); got {_dropped_streams(e)!r}")
        assert _status(e, "detB") in (None, "absent"), (
            f"ts={ts}: detB's stream is live -> a plain (legitimate) absence, "
            f"never 'dropped'; got {_status(e, 'detB')!r}")

    print("[ok] STR-02: a mid-run stream drop-out is a DISTINCT, queryable "
          "condition (detector_status 'dropped' vs 'absent'; dropped_streams={1}) "
          "-- not an indistinguishable None")


# ===========================================================================
# 2. NO-DROPOUT CONTROL: the signal never fires; event set/values unchanged
# ===========================================================================
def test_no_dropout_unchanged():
    with tempfile.TemporaryDirectory() as td:
        # both streams run to the end -- a healthy run, no drop-out.
        files, rc = build_run(td, s0_ts=[10, 20, 30, 40], s1_ts=[10, 20, 30, 40])
        evts = _stream_events(files, rc)

    # event set + services are exactly what the parent merge produces (the fix
    # must not add, drop, or reorder events).
    assert [e.timestamp for e in evts] == [10, 20, 30, 40], \
        [e.timestamp for e in evts]
    assert all(e.service == fmt.SERVICE_L1ACCEPT for e in evts)

    for e in evts:
        # returned VALUES are unchanged (None for these payload-less detectors,
        # on both parent and fix -- the fix touches no frame value).
        assert e.stack("detA") is None
        assert e.stack("detB") is None
        assert e.stack("detC") is None
        # and the STR-02 signal is SILENT on a healthy run: no spurious firing.
        # Tolerant of the parent (None == no such signal) so this control never
        # discriminates; on the fix the signal is present and must be empty.
        assert _dropped_streams(e) in (None, frozenset()), (
            f"ts={e.timestamp}: healthy run must drop no stream; got "
            f"{_dropped_streams(e)!r}")
        assert _status(e, "detB") in (None, "absent"), (
            f"ts={e.timestamp}: detB is a plain absence on a healthy run, never "
            f"'dropped'; got {_status(e, 'detB')!r}")
        dd = getattr(e, "dropped_detectors", lambda *a, **k: [])()
        assert dd == [], (
            f"ts={e.timestamp}: no detector may be flagged dropped on a healthy "
            f"run; got {dd!r}")

    print("[ok] STR-02 control: a no-dropout run drops no stream, flags no "
          "detector, and yields the same event set -- byte-unchanged")


# ===========================================================================
# 3. import purity: the streaming path pulls in no framework
# ===========================================================================
def test_import_purity():
    pstream.assert_no_framework_imports()
    for m in ("psana", "mpi4py", "h5py"):
        assert m not in sys.modules, f"{m} leaked into sys.modules"
    print("[ok] STR-02: streaming path imports no psana / mpi4py / h5py")


def main():
    print("=" * 72)
    print("STR-02 regression: a mid-run stream drop-out is distinguishable from "
          "a genuine detector absence")
    print("=" * 72)
    test_dropout_is_distinguishable()
    test_no_dropout_unchanged()
    test_import_purity()
    print()
    print("ALL STR-02 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
