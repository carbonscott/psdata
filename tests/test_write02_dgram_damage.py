#!/usr/bin/env python3
"""WRITE-02 regression: psdata's damage API must read DGRAM-TOP damage.

The bug (WRITE-02)
------------------
Every xtc2 dgram has a ``damage`` word in its TOP-LEVEL Xtc header (the
Transition + Xtc header ``parse_dgram_header`` reads).  The DAQ flags a
dropped/damaged detector *contribution* there -- psana reports it as
``{stream: DroppedContribution}`` -- while it leaves the nested ShapesData-level
Xtc damage 0.  On real data this is stark: across 3 runs ShapesData damage is 0
on ~450,000 ShapesData Xtc, yet 17,872/17,872 events on r51 and 38,400/38,400 on
r107 carry DGRAM-TOP damage.

``psdata``'s damage accessor read ONLY the ShapesData level, so
``evt.is_damaged()`` reported **0 damaged events across all 130,072 events of all
3 runs** -- the API was blind to the only damage the DAQ actually writes.

The fix (WRITE-02)
------------------
``events`` / the random-access reader now thread each contributing stream's
DGRAM-TOP ``Xtc.damage`` word onto the assembled :class:`~psdata.stream.Event`
(``Event._stream_damage``), and the Event surfaces it:

  * ``evt.is_damaged()``           -- whole-event: True if ANY contributing
                                      stream's dgram-top damage is nonzero.
  * ``evt.damage_by_stream()``     -- ``{stream: (damage_id, userbits)}`` for
                                      every contributing stream (the damaged
                                      subset is psana's ``{16: DroppedContribution}``).
  * ``evt.is_detector_damaged(d)`` -- True iff an OWNING stream of ``d`` (a
                                      dropped contribution on stream 16 damages
                                      every detector on stream 16) is damaged.

Damage is a pure side-channel: the raw frame VALUES are byte-unchanged.  The
legacy per-detector ``evt.damage(det)`` / ``evt.is_damaged(det)`` (ShapesData
level) is preserved unchanged -- and, being ShapesData-only, stays blind, which
is exactly why the new dgram-top accessors are needed.

Self-contained
--------------
stdlib + numpy only -- NO psana, NO SLAC data.  It hand-builds synthetic xtc2
dgrams (byte-for-byte to the header layout in ``src/psdata/format.py``): a stream
whose DGRAM-TOP damage word is nonzero (a DroppedContribution bit) while its
ShapesData-level damage is 0, then drives BOTH read paths the fix threads the
damage through -- the forward ``psdata.stream.events`` k-way merge AND the
random-access ``RunIndex.read_event_at`` (single) / ``read_events`` (batch) over
a hand-built index on the same on-disk file -- and checks the assertions below.

Fail on parent / pass on fix
----------------------------
Parent SHA: 1a0e9f926accc80c617526eeb9e63147ad6d93b4 (origin/main).

On the PARENT, ``Event`` reads only ShapesData-level damage (always 0) and has
no dgram-top API: ``is_damaged`` REQUIRES a ``det_name`` (so no-arg
``evt.is_damaged()`` raises ``TypeError``), and ``damage_by_stream`` /
``is_detector_damaged`` do not exist.  This probe reads them through ``getattr``
(and catches the no-arg ``TypeError``) so the parent yields ``None`` for the
discriminating verdict -- the ``is True`` assertions then FAIL and the test exits
non-zero.  On the FIX the accessors return the real dgram-top verdict, so every
assertion holds and the test exits 0.

Contains no part of the fix; cwd-robust; ``main()`` + ``__main__``.
"""

import os
import struct
import sys
import tempfile

import numpy as np

# --- locate the package under test (parent-of-tests/src), cwd-robust ---------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")  # .../<repo>/src (holds psdata/)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import psdata.format as fmt      # noqa: E402
import psdata.stream as pstream  # noqa: E402
import psdata.index as pidx      # noqa: E402

# DroppedContribution is value 1 in the xtc2 Damage enum -- the bit the DAQ sets
# on the top-level dgram Xtc when a detector's contribution is dropped (psana:
# ``{stream: DroppedContribution}``).  The EXACT numeric is immaterial to the
# fix: we inject it into the synthetic bytes and read it back, and ANY nonzero
# dgram-top damage id must make ``is_damaged()`` True.
DROPPED_CONTRIBUTION = 1

# The stream carrying the dropped contribution -- 16 to mirror psana's observed
# ``{16: DroppedContribution}`` on r51.
DROPPED_STREAM = 16
HEALTHY_STREAM = 0


# ===========================================================================
# Hand-build xtc2 dgram bytes, byte-for-byte to src/psdata/format.py:
#   Transition = uint32 nsec, uint32 sec, uint32 env   (service=(env>>24)&0xf)
#   Xtc        = uint32 src, uint16 damage, uint16 typeid, uint32 extent
#     (extent INCLUDES the 12B Xtc header; src packs (nodeId,namesId))
# The DGRAM-TOP damage word is the damage field of the dgram's OWN (top) Xtc
# header -- byte offset off+16 -- which parse_dgram_header returns as h["damage"].
# ===========================================================================
def _pack_xtc(src, damage, tid, payload):
    """One Xtc: 12B header (src, uint16 damage, uint16 typeid, uint32 extent)
    then ``payload``.  ``extent`` includes the header."""
    extent = fmt.XTC_HDR + len(payload)
    return struct.pack("<IHHI", src, damage, tid, extent) + payload


def _make_shapesdata(src, frame_u16):
    """A ShapesData Xtc (its OWN Xtc.damage == 0 -- the level the DAQ leaves 0)
    holding a Shapes child (the frame's dims) and a Data child (the frame bytes)
    for one rank-2 uint16 field named 'raw'.  ``src`` packs (nodeId, namesId) so
    _index_dgram keys it to the fabricated Names table."""
    h, w = frame_u16.shape
    shapes_payload = struct.pack("<5I", h, w, 0, 0, 0)          # MAXRANK dims
    shapes = _pack_xtc(0, 0, fmt.TID_SHAPES, shapes_payload)
    data = _pack_xtc(0, 0, fmt.TID_DATA, frame_u16.astype("<u2").tobytes())
    return _pack_xtc(src, 0, fmt.TID_SHAPESDATA, shapes + data)  # ShapesData damage = 0


def _make_l1(ts, dgram_damage, payload):
    """An L1Accept dgram whose TOP Xtc carries ``dgram_damage`` and whose payload
    (the top Xtc's children) is ``payload`` (a ShapesData, or b'' for a dropped
    contribution)."""
    nsec = ts & 0xFFFFFFFF
    sec = (ts >> 32) & 0xFFFFFFFF
    env = fmt.SERVICE_L1ACCEPT << 24
    transition = struct.pack("<III", nsec, sec, env)
    top = _pack_xtc(0, dgram_damage, fmt.TID_PARENT, payload)   # dgram-top damage here
    return transition + top


def _write(path, dgrams):
    with open(path, "wb") as fh:
        for d in dgrams:
            fh.write(d)


def _det(name, det_type, stream, seg=0, alg="raw"):
    """A one-segment DetectorInfo on ``stream`` (fabricated directly, as
    test_str02_stream_dropout._det / test_bigdata_scan_us012 do)."""
    d = fmt.DetectorInfo(name, det_type, f"{name}-id")
    d.algs[alg] = {}
    d.segments[alg] = {seg}
    d.seg_to_stream[(alg, seg)] = stream
    return d


def _healthy_table():
    """The fabricated Names table for detHealthy seg 0: one rank-2 uint16 field
    'raw'.  Only the keys _index_dgram / extract_field consult are populated."""
    return dict(det_type="epixH", det_name="detHealthy", det_id="detHealthy-id",
                alg_name="raw", alg_version=0, segment=0, num_arrays=1,
                names=[dict(name="raw", type=1, rank=2)])   # type 1 == uint16


def build_run(tmpdir, ts_list, healthy_damage, dropped_damage, frame):
    """A synthetic two-stream run.

    HEALTHY_STREAM (0) carries ``detHealthy`` with a real ShapesData frame;
    DROPPED_STREAM (16) carries ``detDropped`` whose contribution is DROPPED
    (empty payload -> no ShapesData).  Each stream's top-level dgram damage is
    ``healthy_damage`` / ``dropped_damage`` -- either a single int (same for
    every event) or a per-event list aligned with ``ts_list`` (so a mix of
    damaged and clean events can share one on-disk file, for the random-access
    index reads).  Returns ``(files, rc)`` for ``psdata.stream.events`` and for
    a hand-built :class:`psdata.index.RunIndex` (:func:`_build_index`).
    """
    def _per_event(d):
        return list(d) if isinstance(d, (list, tuple)) else [d] * len(ts_list)
    hd = _per_event(healthy_damage)
    dd = _per_event(dropped_damage)
    p0 = os.path.join(tmpdir, "synth-r0001-s000.xtc2")
    p16 = os.path.join(tmpdir, "synth-r0001-s016.xtc2")
    _write(p0, [_make_l1(ts, h, _make_shapesdata(0, frame))
                for ts, h in zip(ts_list, hd)])
    _write(p16, [_make_l1(ts, d, b"") for ts, d in zip(ts_list, dd)])

    rc = fmt.RunConfig()
    rc.stream_files = {HEALTHY_STREAM: p0, DROPPED_STREAM: p16}
    # cfg_end == 0 -> cursors read dgrams from offset 0 (no Configure prefix).
    rc.stream_configs = {HEALTHY_STREAM: (None, 0), DROPPED_STREAM: (None, 0)}
    rc.raw_tables = {HEALTHY_STREAM: {(0, 0): _healthy_table()},
                     DROPPED_STREAM: {}}
    rc.detectors = {
        "detHealthy": _det("detHealthy", "epixH", stream=HEALTHY_STREAM),
        "detDropped": _det("detDropped", "epixD", stream=DROPPED_STREAM),
    }
    return rc.stream_files, rc


# --- read the fix's dgram-top signal through getattr so the PARENT (which has
#     neither the no-arg is_damaged nor the new methods) yields a clean,
#     discriminating None rather than an AttributeError/TypeError -------------
def _whole_event_damaged(evt):
    fn = getattr(evt, "is_damaged", None)
    if fn is None:
        return None
    try:
        return fn()                      # fix: whole-event bool; parent: TypeError
    except TypeError:
        return None                      # parent: is_damaged REQUIRES a det_name


def _damage_by_stream(evt):
    fn = getattr(evt, "damage_by_stream", None)
    return None if fn is None else fn()


def _is_detector_damaged(evt, det):
    fn = getattr(evt, "is_detector_damaged", None)
    return None if fn is None else fn(det)


def _build_index(files, rc):
    """A :class:`psdata.index.RunIndex` over the synthetic on-disk streams,
    populated by hand (no SMD sidecar needed): walk each stream file's dgrams,
    keying every L1Accept's (path, offset, size) by timestamp into
    ``entries[k]`` with ``timestamps[k]`` ascending -- exactly the shape the
    real SMD/bigdata scan produces.  ``read_event_at`` / ``read_events`` then
    pread these offsets and assemble Events through the SAME
    ``_assemble_stream_dgram`` the fix threads dgram-top damage through.  Works
    identically on parent and fix (both build/serve the index the same way);
    only the damage the assembled Event exposes differs."""
    ridx = pidx.RunIndex(rc)
    per_stream = {}
    for stream, path in rc.stream_files.items():
        with open(path, "rb") as fh:
            data = fh.read()
        recs, off = [], 0
        while off + fmt.DGRAM_HDR <= len(data):
            h = fmt.parse_dgram_header(data, off)
            size = fmt.XTC_HDR + h["extent"]
            recs.append((h["ts"], off, size))
            off += size
        per_stream[stream] = recs
    all_ts = sorted({ts for recs in per_stream.values() for (ts, _, _) in recs})
    for ts in all_ts:
        entry = {}
        for stream, recs in per_stream.items():
            for (t, o, sz) in recs:
                if t == ts:
                    entry[stream] = (rc.stream_files[stream], o, sz)
        ridx.timestamps.append(ts)
        ridx.entries.append(entry)
    return ridx


FRAME = np.arange(6, dtype=np.uint16).reshape(2, 3)   # known, byte-checkable


# ===========================================================================
# 1. THE DISCRIMINATOR: a dropped contribution's DGRAM-TOP damage is surfaced
#    and attributed to the owning stream / its detectors.
# ===========================================================================
def test_dropped_contribution_surfaced():
    with tempfile.TemporaryDirectory() as td:
        files, rc = build_run(td, [100, 200], healthy_damage=0,
                              dropped_damage=DROPPED_CONTRIBUTION, frame=FRAME)
        evts = list(pstream.events(files, run_config=rc))

    assert [e.timestamp for e in evts] == [100, 200], [e.timestamp for e in evts]

    for e in evts:
        ts = e.timestamp

        # --- PRECONDITION (true on parent AND fix): the frame VALUES are
        #     unchanged.  The healthy frame reads byte-exact; the dropped
        #     contribution is an (indistinguishable-at-the-value-level) None. ---
        got = e.stack("detHealthy")
        assert got is not None and got.shape == (1, 2, 3), got
        assert np.array_equal(got[0], FRAME), (got, FRAME)
        assert e.stack("detDropped") is None, "dropped contribution -> None"

        # legacy ShapesData-level damage is 0 (this is WHY the parent is blind):
        # it exists on parent AND fix and reports undamaged for both detectors.
        assert e.damage("detHealthy") == {0: (0, 0)}, e.damage("detHealthy")
        assert e.is_damaged("detHealthy") is False   # ShapesData path -> blind

        # --- THE WRITE-02 ASSERTIONS (fail on parent -> None; pass on fix) ----
        wd = _whole_event_damaged(e)
        assert wd is True, (
            f"WRITE-02: ts={ts}: an event with a DGRAM-TOP DroppedContribution "
            f"on stream {DROPPED_STREAM} must report is_damaged()==True; got "
            f"{wd!r} (the parent reads only ShapesData damage, which is 0, so it "
            f"is blind to the only damage the DAQ writes).")

        dbs = _damage_by_stream(e)
        assert dbs is not None, (
            f"WRITE-02: ts={ts}: no per-stream damage view -- the parent cannot "
            f"say WHICH stream's contribution was dropped.")
        assert dbs.get(DROPPED_STREAM) == (DROPPED_CONTRIBUTION, 0), (
            f"ts={ts}: stream {DROPPED_STREAM} must carry the DroppedContribution "
            f"dgram-top damage; damage_by_stream()={dbs!r}")
        assert dbs.get(HEALTHY_STREAM) == (0, 0), (
            f"ts={ts}: the healthy stream must be undamaged; got {dbs!r}")
        # the damaged subset is exactly psana's {16: DroppedContribution}.
        damaged_only = {s: v for s, v in dbs.items() if v[0] != 0}
        assert damaged_only == {DROPPED_STREAM: (DROPPED_CONTRIBUTION, 0)}, \
            damaged_only

        # per-DETECTOR attribution: the dropped stream's detector is damaged,
        # the healthy stream's detector is not (damage per owning stream).
        assert _is_detector_damaged(e, "detDropped") is True, (
            f"ts={ts}: detDropped is on the dropped stream {DROPPED_STREAM}; its "
            f"contribution was dropped -> must be flagged damaged.")
        assert _is_detector_damaged(e, "detHealthy") is False, (
            f"ts={ts}: detHealthy is on the undamaged stream {HEALTHY_STREAM}; "
            f"must NOT be flagged damaged.")

    print("[ok] WRITE-02: DGRAM-TOP DroppedContribution surfaced -- "
          "is_damaged()==True, damage_by_stream()=={16: DroppedContribution}, "
          "attributed to detDropped (owning stream 16), not detHealthy; frame "
          "VALUES byte-unchanged")


# ===========================================================================
# 2. THE CRUX + BYTE-EXACTNESS: ShapesData damage 0 while DGRAM-TOP is set on a
#    detector that STILL delivers its frame -- the frame is byte-unchanged, only
#    the new dgram-top API sees the damage.
# ===========================================================================
def test_shapesdata_zero_but_dgramtop_set():
    with tempfile.TemporaryDirectory() as td:
        # detHealthy's OWN stream carries a DGRAM-TOP damage word, but still
        # delivers its full ShapesData frame (ShapesData damage stays 0).
        files_d, rc_d = build_run(td, [100], healthy_damage=DROPPED_CONTRIBUTION,
                                  dropped_damage=0, frame=FRAME)
        e_dmg = list(pstream.events(files_d, run_config=rc_d))[0]
        # a byte-for-byte identical run EXCEPT no dgram-top damage anywhere.
        files_c, rc_c = build_run(td, [100], healthy_damage=0,
                                  dropped_damage=0, frame=FRAME)
        e_clean = list(pstream.events(files_c, run_config=rc_c))[0]

    # ShapesData level is 0 on BOTH (parent + fix see this) -> the legacy
    # per-detector accessor is blind to the dgram-top damage.
    assert e_dmg.damage("detHealthy") == {0: (0, 0)}
    assert e_dmg.is_damaged("detHealthy") is False       # legacy/ShapesData: blind

    # BYTE-EXACTNESS: setting the dgram-top damage word did NOT change the frame.
    frame_dmg = e_dmg.raw("detHealthy")[0]
    frame_clean = e_clean.raw("detHealthy")[0]
    assert np.array_equal(frame_dmg, frame_clean), (frame_dmg, frame_clean)
    assert np.array_equal(frame_dmg, FRAME), (frame_dmg, FRAME)

    # ...yet the fix's dgram-top API DOES see it (fails on parent -> None).
    assert _is_detector_damaged(e_dmg, "detHealthy") is True, (
        "WRITE-02: detHealthy's stream carries a DGRAM-TOP damage word (ShapesData "
        "0) -> is_detector_damaged must be True; the parent, reading only "
        "ShapesData, is blind.")
    assert _whole_event_damaged(e_dmg) is True

    print("[ok] WRITE-02: ShapesData damage 0 while DGRAM-TOP set -- the frame is "
          "byte-identical to the clean read, the legacy ShapesData accessor is "
          "blind, and only the dgram-top API reports the damage")


# ===========================================================================
# 3. CONTROL: all-zero dgram-top damage -> is_damaged() is False (no spurious
#    firing).  Tolerant of the parent (no no-arg API -> None) so this control
#    never discriminates; on the fix the verdict must be a real False.
# ===========================================================================
def test_clean_run_is_undamaged():
    with tempfile.TemporaryDirectory() as td:
        files, rc = build_run(td, [100, 200], healthy_damage=0,
                              dropped_damage=0, frame=FRAME)
        evts = list(pstream.events(files, run_config=rc))

    assert [e.timestamp for e in evts] == [100, 200]
    for e in evts:
        # frame VALUES unchanged, detDropped still absent (structurally).
        assert np.array_equal(e.stack("detHealthy")[0], FRAME)
        assert e.stack("detDropped") is None

        api_present = _damage_by_stream(e) is not None      # True only on the fix
        wd = _whole_event_damaged(e)
        if api_present:
            assert wd is False, (
                f"WRITE-02: ts={e.timestamp}: an all-zero-dgram-top event must "
                f"report is_damaged()==False; got {wd!r}")
            assert _is_detector_damaged(e, "detDropped") is False
            assert _is_detector_damaged(e, "detHealthy") is False
            assert all(v == (0, 0) for v in _damage_by_stream(e).values())
        else:
            assert wd is None            # parent: no whole-event API (tolerated)

    print("[ok] WRITE-02 control: an all-zero DGRAM-TOP run reports is_damaged() "
          "False, flags no stream/detector, and yields byte-unchanged frames")


# ===========================================================================
# 4. RANDOM ACCESS: read_event_at (single) and read_events (batch) surface the
#    SAME dgram-top damage the forward path does -- WRITE-02 threads it through
#    both random-access read paths (RunIndex._assemble_stream_dgram), so the
#    regression must exercise them, not just pstream.events().
# ===========================================================================
def test_random_access_dgram_damage():
    # One on-disk run with a MIX of clean and damaged events so a stop-short /
    # cross-event bug is catchable: only k=1 (ts=200) carries dgram-top damage
    # (on BOTH streams -- so detHealthy's own frame rides a damaged stream yet
    # must read back byte-exact); k=0 and k=2 are clean.
    ts_list = [100, 200, 300]
    hdmg = [0, DROPPED_CONTRIBUTION, 0]       # stream 0 (detHealthy) per event
    ddmg = [0, DROPPED_CONTRIBUTION, 0]       # stream 16 (detDropped) per event
    with tempfile.TemporaryDirectory() as td:
        files, rc = build_run(td, ts_list, healthy_damage=hdmg,
                              dropped_damage=ddmg, frame=FRAME)
        ridx = _build_index(files, rc)
        assert ridx.timestamps == ts_list, ridx.timestamps

        # --- read_event_at (single random access) -------------------------
        e_dmg = ridx.read_event_at(1)         # the damaged event
        e_clean = ridx.read_event_at(0)       # a clean event

        # frame VALUES unchanged (parent AND fix): damaged frame == clean FRAME.
        assert np.array_equal(e_dmg.stack("detHealthy")[0], FRAME)
        assert np.array_equal(e_dmg.raw("detHealthy")[0], FRAME)
        assert e_dmg.stack("detDropped") is None          # dropped contribution
        # legacy ShapesData level stays 0 on the random-access read too.
        assert e_dmg.damage("detHealthy") == {0: (0, 0)}

        # THE WRITE-02 ASSERTIONS on the random-access read (fail on parent).
        assert _whole_event_damaged(e_dmg) is True, (
            "WRITE-02: read_event_at must surface DGRAM-TOP damage; got "
            f"{_whole_event_damaged(e_dmg)!r} (parent's random-access Event is "
            "blind -- it reads only ShapesData damage).")
        assert _damage_by_stream(e_dmg) == {0: (DROPPED_CONTRIBUTION, 0),
                                            16: (DROPPED_CONTRIBUTION, 0)}, \
            _damage_by_stream(e_dmg)
        assert _is_detector_damaged(e_dmg, "detDropped") is True
        assert _is_detector_damaged(e_dmg, "detHealthy") is True

        # a clean event reads back UNDAMAGED (on the fix; parent -> None).
        assert np.array_equal(e_clean.stack("detHealthy")[0], FRAME)
        if _damage_by_stream(e_clean) is not None:        # API present == fix
            assert _whole_event_damaged(e_clean) is False
            assert _is_detector_damaged(e_clean, "detDropped") is False
            assert all(v == (0, 0)
                       for v in _damage_by_stream(e_clean).values())
        else:
            assert _whole_event_damaged(e_clean) is None  # parent (tolerated)

        # --- read_events (batch) ------------------------------------------
        # Cover the damaged k=1 in a multi-event batch with k=1 NOT first/only,
        # so a batch assembly that stopped short or crossed events is caught.
        batch = ridx.read_events([0, 1, 2])
        assert [e.timestamp for e in batch] == ts_list, \
            [e.timestamp for e in batch]

        b_clean0, b_dmg, b_clean2 = batch
        # the damaged event in the batch surfaces the damage identically...
        assert _whole_event_damaged(b_dmg) is True, (
            "WRITE-02: read_events (batch) must surface DGRAM-TOP damage on the "
            f"damaged event; got {_whole_event_damaged(b_dmg)!r}.")
        assert _damage_by_stream(b_dmg) == {0: (DROPPED_CONTRIBUTION, 0),
                                            16: (DROPPED_CONTRIBUTION, 0)}, \
            _damage_by_stream(b_dmg)
        assert _is_detector_damaged(b_dmg, "detDropped") is True
        # ...its frame is still byte-exact (damage is a side-channel)...
        assert np.array_equal(b_dmg.stack("detHealthy")[0], FRAME)
        assert b_dmg.stack("detDropped") is None
        # ...and the batch's clean neighbours are not spuriously flagged (fix).
        if _damage_by_stream(b_clean0) is not None:       # API present == fix
            assert _whole_event_damaged(b_clean0) is False
            assert _whole_event_damaged(b_clean2) is False

    print("[ok] WRITE-02: read_event_at (single) AND read_events (batch) surface "
          "the same DGRAM-TOP damage the forward path does -- damaged event "
          "flagged, clean events not, frames byte-exact")


# ===========================================================================
# 5. import purity: the streaming path pulls in no framework.
# ===========================================================================
def test_import_purity():
    pstream.assert_no_framework_imports()
    for m in ("psana", "mpi4py", "h5py"):
        assert m not in sys.modules, f"{m} leaked into sys.modules"
    print("[ok] WRITE-02: streaming path imports no psana / mpi4py / h5py")


def main():
    print("=" * 72)
    print("WRITE-02 regression: psdata's damage API must read DGRAM-TOP damage")
    print("=" * 72)
    test_dropped_contribution_surfaced()
    test_shapesdata_zero_but_dgramtop_set()
    test_clean_run_is_undamaged()
    test_random_access_dgram_damage()
    test_import_purity()
    print()
    print("ALL WRITE-02 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
