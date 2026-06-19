#!/usr/bin/env python3
"""US-012 -- build the random-access index from BIGDATA, no SMD artifact.

``psdata`` already builds its ``timestamp -> {stream: (offset, size)}`` index by
scanning the small ``.smd.xtc2`` sidecars (US-003).  Those sidecars are produced
only by the DAQ DRP or by xtcdata's ``smdwriter`` -- the very toolchain psdata
exists to avoid depending on.  This story removes the *hard* dependency on that
artifact: :meth:`RunIndex.build_from_bigdata` rebuilds the identical index by
walking the bigdata dgram headers directly (the ``smdwriter`` algorithm in pure
Python), and :func:`build_index` defaults to ``source="auto"`` -- SMD when
present (fast cache), bigdata scan otherwise.

What this suite proves (oracle = psdata's OWN SMD-scan path):

  1. the bigdata-scan index is byte-exact against the SMD-scan index on every
     event the SMD indexes (same timestamps, same per-stream offset/size);
  2. a randomly-read event is byte-identical whichever index served it;
  3. ``build_index`` routing -- ``source="bigdata"`` forces the scan,
     ``source="auto"`` falls back to it when a sidecar is missing, and uses SMD
     when all sidecars are present;
  4. the bigdata path stays framework-pure (no psana / xtcdata / mpi4py / h5py).

For speed it uses only the run's three SMALL streams (s000/s001/s004): the
bigdata-header walk of a small file is sub-second, while the giant detector
streams (tens-to-hundreds of GB) are validated at scale outside the suite.
"""
import os
import sys
import tempfile

import numpy as np

# --- locate the package (parent of this tests dir) --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")  # .../<repo>/src (holds psdata/)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import psdata
import psdata.format  # noqa: F401  (DGRAM_HDR / parse_dgram_header for the oracle)
import psdata.index as _ix

# Reference dataset -- single-chunk run; SMALL streams only, for a fast scan.
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
EXP = "mfx100848724"
RUN = 51
SMALL_STREAMS = (0, 1, 4)
FILES = [f"{DIR}/{EXP}-r{RUN:04d}-s{s:03d}-c000.xtc2" for s in SMALL_STREAMS]


def _entry_key(entry):
    """Normalise one event's index entry to {stream: (basename, off, size)} so
    SMD- and bigdata-built entries compare regardless of absolute path object."""
    return {s: (os.path.basename(p), o, sz) for s, (p, o, sz) in entry.items()}


def test_bigdata_index_byte_exact_against_smd():
    """The default bigdata-built index is byte-identical to the SMD-built index:
    same timestamps, same per-stream (offset, size).  ``build_from_bigdata``
    clamps to the canonical event set (events carrying the timing/master
    stream), which equals the SMD/``smdwriter`` set exactly.  On these small
    streams -- which include the timing stream s004 and have no ragged DAQ tail
    gap -- ``include_shutdown_tail=True`` is a no-op, so it too equals SMD; the
    17872-vs-17982 superset gap only appears on the full 10-stream scan (the
    long-running detector streams), validated at scale by verify_cache.py."""
    idx_smd = psdata.build_index(FILES, source="smd")
    idx_bd = psdata.build_index(FILES, source="bigdata")
    assert idx_smd.scan_source == "smd" and idx_bd.scan_source == "bigdata"
    assert idx_bd.include_shutdown_tail is False, "default must clamp the tail"

    bd_pos = {ts: k for k, ts in enumerate(idx_bd.timestamps)}
    assert len(idx_bd.timestamps) >= len(idx_smd.timestamps), \
        "bigdata index must contain at least every SMD-indexed event"
    for ts, smd_entry in zip(idx_smd.timestamps, idx_smd.entries):
        assert ts in bd_pos, f"bigdata index missing SMD event ts={ts}"
        assert _entry_key(smd_entry) == _entry_key(idx_bd.entries[bd_pos[ts]]), \
            f"offset/size mismatch at ts={ts}"
    # default clamp == SMD exactly (these small streams carry s004 and have no
    # DAQ tail gap, so the canonical set is the whole set).
    assert idx_smd.timestamps == idx_bd.timestamps, \
        "default bigdata index must equal the SMD index on run 51's small streams"
    # the include_shutdown_tail flag plumbs through and, absent a tail here,
    # yields the same set -- proving it is a no-op when nothing is clamped.
    idx_bd_full = psdata.build_index(FILES, source="bigdata",
                                     include_shutdown_tail=True)
    assert idx_bd_full.include_shutdown_tail is True
    assert idx_bd_full.timestamps == idx_smd.timestamps, \
        "no DAQ tail on the small streams -> include_shutdown_tail is a no-op"


# --------------------------------------------------------------------------
# Synthetic clamp-logic tests (no data, no scan): exercise the canonical /
# shutdown-tail merge directly via a minimal fabricated RunConfig.  The
# 17872-vs-17982 behaviour is real-data-validated in verify_cache.py; here we
# pin the merge PREDICATE itself -- fast and deterministic.
# --------------------------------------------------------------------------
import psdata.format as _fmt


def _fake_rc_with_timing(timing_stream=4, det_streams=(5, 7)):
    """A minimal RunConfig: a timing detector (det_type 'ts', alg 'raw') on
    ``timing_stream``, plus a detector spread over ``det_streams``.  Enough for
    ``_timing_streams`` / ``_merge_streams`` (no real Names tables needed)."""
    rc = _fmt.RunConfig()
    rc.stream_files = {s: f"/x/fake-s{s:03d}-c000.xtc2"
                       for s in {timing_stream, *det_streams}}
    ts = _fmt.DetectorInfo("timing", "ts", "ts0")
    ts.algs["raw"] = {}                       # key presence is all that matters
    ts.segments["raw"] = {0}
    ts.seg_to_stream[("raw", 0)] = timing_stream
    rc.detectors["timing"] = ts
    det = _fmt.DetectorInfo("det", "jungfrau", "jf0")
    det.algs["raw"] = {}
    det.segments["raw"] = set(range(len(det_streams)))
    for i, s in enumerate(det_streams):
        det.seg_to_stream[("raw", i)] = s
    rc.detectors["det"] = det
    return rc


def test_merge_streams_clamps_shutdown_tail():
    """Default merge drops events lacking the timing/master stream; the flag
    keeps them (the 17872-vs-17982 behaviour, in miniature)."""
    rc = _fake_rc_with_timing(timing_stream=4, det_streams=(5, 7))
    cp = "/x/fake-c000.xtc2"
    # timing stream 4 stops at ts=200; detector streams 5,7 run on to ts=300
    # (the shutdown tail -- present on disk, no timing/pulseId).
    per_stream = {
        4: [(100, cp, 0, 10), (200, cp, 10, 10)],
        5: [(100, cp, 0, 10), (200, cp, 10, 10), (300, cp, 20, 10)],
        7: [(100, cp, 0, 10), (200, cp, 10, 10), (300, cp, 20, 10)],
    }
    # default: clamp -> the timing-less ts=300 event is dropped.
    idx = _ix.RunIndex(rc)
    assert idx.include_shutdown_tail is False
    assert _ix.RunIndex(rc)._timing_streams() == frozenset({4})
    idx._merge_streams(per_stream)
    assert idx.timestamps == [100, 200], "clamp must drop the shutdown tail"

    # include_shutdown_tail=True: keep the full union, tail and all.
    idx2 = _ix.RunIndex(rc)
    idx2._merge_streams(per_stream, include_shutdown_tail=True)
    assert idx2.timestamps == [100, 200, 300], "tail must be kept when asked"
    tail_entry = idx2.entries[idx2.timestamps.index(300)]
    assert set(tail_entry) == {5, 7} and 4 not in tail_entry, \
        "the kept tail event indeed lacks the timing/master stream"


def test_merge_streams_no_timing_detector_keeps_all():
    """Graceful degradation: a run that declares NO timing detector cannot be
    clamped, so every assembled event is kept (no spurious dropping)."""
    rc = _fmt.RunConfig()
    rc.stream_files = {5: "/x/a-c000.xtc2", 7: "/x/b-c000.xtc2"}
    det = _fmt.DetectorInfo("det", "jungfrau", "jf0")
    det.algs["raw"] = {}
    det.segments["raw"] = {0, 1}
    det.seg_to_stream[("raw", 0)] = 5
    det.seg_to_stream[("raw", 1)] = 7
    rc.detectors["det"] = det
    cp = "/x/c-c000.xtc2"
    per_stream = {5: [(1, cp, 0, 5), (2, cp, 5, 5)], 7: [(1, cp, 0, 5)]}
    idx = _ix.RunIndex(rc)
    assert idx._timing_streams() == frozenset(), "no 'ts' detector -> no clamp"
    idx._merge_streams(per_stream)
    assert idx.timestamps == [1, 2], "no timing detector -> keep every event"


def test_include_shutdown_tail_round_trips():
    """The clamp flag survives the save/ship round-trips (to_dict/from_dict --
    and pickle save/load, which share _persist_state), and a pre-clamp blob
    that lacks the field loads as canonical (False), the back-compat default.
    Pure-synthetic: no scan, no real data."""
    rc = _fake_rc_with_timing()
    idx = _ix.RunIndex(rc)
    idx.include_shutdown_tail = True
    back = _ix.RunIndex.from_dict(idx.to_dict())
    assert back.include_shutdown_tail is True, \
        "include_shutdown_tail must survive to_dict/from_dict"
    # an older blob predating the clamp has no such field -> must load as False.
    state = idx.to_dict()
    del state["include_shutdown_tail"]
    assert _ix.RunIndex.from_dict(state).include_shutdown_tail is False, \
        "a persisted blob lacking the field must restore as canonical (False)"


def test_read_event_on_clamped_tail_raises_hinted_keyerror():
    """read_event(ts) for a shutdown-tail timestamp dropped by the default
    clamp raises KeyError, and the message names include_shutdown_tail so the
    caller knows how to recover it (build_from_bigdata(..., include_shutdown_
    tail=True)).  Pure-synthetic: _position_of raises before any pread."""
    rc = _fake_rc_with_timing(timing_stream=4, det_streams=(5, 7))
    cp = "/x/fake-c000.xtc2"
    per_stream = {
        4: [(100, cp, 0, 10), (200, cp, 10, 10)],            # timing stops @200
        5: [(100, cp, 0, 10), (200, cp, 10, 10), (300, cp, 20, 10)],
        7: [(100, cp, 0, 10), (200, cp, 10, 10), (300, cp, 20, 10)],
    }
    idx = _ix.RunIndex(rc)
    idx._merge_streams(per_stream)               # default clamp drops ts=300
    assert 300 not in idx.timestamps, "precondition: the tail ts is clamped out"
    try:
        idx.read_event(300)
    except KeyError as e:
        assert "include_shutdown_tail" in str(e), \
            "the KeyError must hint at include_shutdown_tail=True for recovery"
    else:
        raise AssertionError(
            "read_event on a clamped-out tail ts must raise KeyError")


_BOOKKEEPING = {"smdinfo", "chunkinfo", "runinfo", "epicsinfo"}


def _segs_equal(a, b):
    """``raw()`` returns a per-segment list of arrays; compare them pairwise."""
    if a is None or b is None:
        return a is None and b is None
    if len(a) != len(b):
        return False
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def _readable_triples(run_config):
    """All ``(det, alg, field)`` triples declared in the run's Names tables,
    minus the bookkeeping detectors -- so the read comparison is detector- and
    field-agnostic (the small-stream scalars do not have a ``raw``/``raw``
    field, only the area detectors do)."""
    triples = set()
    for tables in run_config.raw_tables.values():
        for t in tables.values():
            if t["det_name"] in _BOOKKEEPING:
                continue
            for nm in t["names"]:
                triples.add((t["det_name"], t["alg_name"], nm["name"]))
    return sorted(triples)


def test_random_read_identical_either_index():
    """A randomly-read event is byte-identical whichever index served it,
    across every readable field of every (non-bookkeeping) detector."""
    idx_smd = psdata.build_index(FILES, source="smd")
    idx_bd = psdata.build_index(FILES, source="bigdata")
    n = idx_smd.n_events
    triples = _readable_triples(idx_smd.run_config)
    for k in (0, n // 2, n - 1):
        e_smd = idx_smd.read_event_at(k)
        e_bd = idx_bd.read_event_at(k)
        assert e_smd.timestamp == e_bd.timestamp
        compared = 0
        for det, alg, field in triples:
            a = e_smd.raw(det, field=field, alg=alg)
            b = e_bd.raw(det, field=field, alg=alg)
            if a is None and b is None:
                continue
            assert a is not None and b is not None, \
                f"{det}.{alg}.{field} presence differs at k={k}"
            assert _segs_equal(a, b), \
                f"{det}.{alg}.{field} bytes differ at k={k}"
            compared += 1
        assert compared > 0, f"no field compared at k={k}"


def test_build_index_source_routing():
    """``source`` selects the index origin; ``auto`` falls back when a sidecar
    is missing and uses SMD when all are present."""
    # all sidecars present -> auto and smd both use the SMD cache
    assert psdata.build_index(FILES).scan_source == "smd"            # default auto
    assert psdata.build_index(FILES, source="auto").scan_source == "smd"
    assert psdata.build_index(FILES, source="smd").scan_source == "smd"
    # explicit bigdata always scans bigdata
    assert psdata.build_index(FILES, source="bigdata").scan_source == "bigdata"
    # a missing sidecar makes auto fall back to the bigdata scan (no error)
    bogus = {s: f"/nonexistent/smalldata/{EXP}-r{RUN:04d}-s{s:03d}-c000.smd.xtc2"
             for s in SMALL_STREAMS}
    idx = psdata.build_index(FILES, smd_files=bogus, source="auto")
    assert idx.scan_source == "bigdata"
    # ... whereas forcing source="smd" with a missing sidecar must error
    try:
        psdata.build_index(FILES, smd_files=bogus, source="smd")
    except (FileNotFoundError, OSError):
        pass
    else:
        raise AssertionError("source='smd' should fail when a sidecar is missing")
    # an invalid source is rejected
    try:
        psdata.build_index(FILES, source="nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid source must raise ValueError")


def test_build_index_auto_partial_sidecar():
    """``source="auto"`` must fall back to the bigdata scan when even ONE
    sidecar is missing -- not just when all are absent.  The routing test
    above covers the all-present (SMD) and all-bogus (fallback) extremes; this
    pins the in-between case: the real sidecars resolved by
    :func:`psdata.index.smd_files_for`, with exactly one entry swapped for a
    nonexistent path."""
    smd = _ix.smd_files_for(FILES)
    assert smd, "smd_files_for must resolve the sidecar mapping"
    # sanity: with the genuine (all-present) sidecars, auto picks the SMD cache.
    assert psdata.build_index(FILES, smd_files=smd, source="auto").scan_source \
        == "smd"
    # break exactly ONE sidecar -> a single miss must trigger bigdata fallback.
    partial = dict(smd)
    a_stream = sorted(partial)[0]
    partial[a_stream] = f"/nonexistent/smalldata/missing-s{a_stream:03d}.smd.xtc2"
    n_present = sum(1 for p in partial.values() if os.path.exists(p))
    assert n_present == len(partial) - 1, \
        "exactly one sidecar should be missing for this case"
    idx = psdata.build_index(FILES, smd_files=partial, source="auto")
    assert idx.scan_source == "bigdata", \
        "a single missing sidecar must trigger the bigdata fallback"


def _walk_l1accepts(buf):
    """Walk every dgram of a single bigdata chunk's bytes from offset 0 (the
    same cursor algorithm as :func:`psdata.index._scan_bigdata_stream`), and
    return two parallel lists: ``boundaries`` -- the start offset of every
    dgram (a valid split point) -- and ``l1`` -- the ``(ts, offset, size)`` of
    every L1Accept.  Pure local re-implementation, used as the oracle."""
    boundaries = []
    l1 = []
    cursor = 0
    n = len(buf)
    while cursor + psdata.format.DGRAM_HDR <= n:
        h = psdata.format.parse_dgram_header(buf, cursor)
        total = psdata.format.XTC_HDR + h["extent"]
        if cursor + total > n:
            break
        boundaries.append(cursor)
        if h["service"] == psdata.format.SERVICE_L1ACCEPT:
            l1.append((h["ts"], cursor, total))
        cursor += total
    return boundaries, l1


def test_bigdata_scan_multichunk_synthetic():
    """The bigdata chunk-roll + per-chunk offset RESET, exercised on a
    SYNTHETIC two-chunk fixture cut from a real small bigdata file.

    Run 51 is single-chunk and real multichunk runs are far too large to
    header-walk in a unit test, so we manufacture the two-chunk case: take a
    small single-chunk bigdata file, split its bytes at a dgram boundary near
    the middle (with L1Accepts on BOTH sides), and write the halves as
    ``...-c000.xtc2`` / ``...-c001.xtc2`` siblings.  We then assert that
    :func:`psdata.index._enumerate_bd_chunks` finds the c001 sibling and that
    :func:`psdata.index._scan_bigdata_stream` restarts offsets at 0 inside
    c001 -- verified record-for-record against an independent walk of the
    original bytes (``original_offset == c001_offset + split``)."""
    src = f"{DIR}/{EXP}-r{RUN:04d}-s001-c000.xtc2"        # ~1.5 MB, single chunk
    with open(src, "rb") as fh:
        data = fh.read()

    # Oracle: independent walk of the WHOLE original file.
    boundaries, orig_l1 = _walk_l1accepts(data)
    assert len(orig_l1) >= 2, "need >=2 L1Accepts to split between"

    # Choose a split at a dgram boundary near the middle with L1Accepts on both
    # sides.  Boundaries are sorted; scan outward from the midpoint.
    mid = len(data) // 2
    split = None
    order = sorted(range(len(boundaries)),
                   key=lambda i: abs(boundaries[i] - mid))
    for i in order:
        b = boundaries[i]
        if b == 0:
            continue                                  # c000 must be non-empty
        before = any(off < b for _ts, off, _sz in orig_l1)
        after = any(off >= b for _ts, off, _sz in orig_l1)
        if before and after:
            split = b
            break
    assert split is not None, \
        "no dgram boundary has L1Accepts on both sides -- pick another file"

    with tempfile.TemporaryDirectory() as td:
        stem = "synth-r0001-s000"
        c000 = os.path.join(td, f"{stem}-c000.xtc2")
        c001 = os.path.join(td, f"{stem}-c001.xtc2")
        with open(c000, "wb") as fh:
            fh.write(data[:split])
        with open(c001, "wb") as fh:
            fh.write(data[split:])

        # the -c000/-c001 filename convention must discover the sibling.
        assert _ix._enumerate_bd_chunks(c000) == [c000, c001], \
            "enumerate must find the c001 sibling by filename convention"

        recs, _nbytes, chunk_paths = _ix._scan_bigdata_stream(c000)
        assert chunk_paths == [c000, c001], \
            "scan must walk both chunks in order"

        # split records by their chunk file.
        recs0 = [r for r in recs if r[1] == c000]
        recs1 = [r for r in recs if r[1] == c001]
        assert recs0 and recs1, "both chunks must contribute L1Accepts"

        # the c001 cursor RESTARTS at 0: its first record's offset is the
        # offset of the first L1Accept WITHIN the c001 bytes (not a continuation
        # of c000's cursor, which would be >= split).
        first_c001_off = recs1[0][2]
        assert first_c001_off < split, \
            "c001 offsets must restart from 0, not continue past the split"

        # oracle partition of the original L1Accepts at the same split.
        orig0 = [(ts, off, sz) for ts, off, sz in orig_l1 if off < split]
        orig1 = [(ts, off, sz) for ts, off, sz in orig_l1 if off >= split]
        assert len(recs0) == len(orig0) and len(recs1) == len(orig1)

        # c000 part: byte-exact (ts, offset, size) vs the original walk.
        for (ts, cp, off, sz), (ots, ooff, osz) in zip(recs0, orig0):
            assert cp == c000
            assert (ts, off, sz) == (ots, ooff, osz), \
                "c000 records must match the original file exactly"

        # c001 part: same ts/size, and original_offset == c001_offset + split,
        # proving the per-chunk cursor reset reproduces the right offsets.
        for (ts, cp, off, sz), (ots, ooff, osz) in zip(recs1, orig1):
            assert cp == c001
            assert ts == ots and sz == osz, \
                "c001 records must carry the original ts/size"
            assert ooff == off + split, \
                "per-chunk reset: original_offset == c001_offset + split"

    # the whole synthetic path stays framework-pure.
    _ix.assert_no_framework_imports()


def test_bigdata_path_is_framework_pure():
    """Building from bigdata must not import any framework."""
    psdata.build_index(FILES, source="bigdata")
    _ix.assert_no_framework_imports()


if __name__ == "__main__":
    test_bigdata_index_byte_exact_against_smd()
    _n = len(psdata.build_index(FILES, source="smd").timestamps)
    print(f"OK  bigdata index byte-exact vs SMD ({_n} events, small streams)")
    test_random_read_identical_either_index()
    print("OK  random reads byte-identical from either index")
    test_build_index_source_routing()
    print("OK  build_index source routing (smd / bigdata / auto-fallback)")
    test_build_index_auto_partial_sidecar()
    print("OK  build_index auto falls back on a single missing sidecar")
    test_bigdata_scan_multichunk_synthetic()
    print("OK  bigdata multichunk synthetic: chunk-roll + per-chunk offset reset")
    test_bigdata_path_is_framework_pure()
    print("OK  bigdata scan path is framework-pure")
    test_merge_streams_clamps_shutdown_tail()
    print("OK  merge clamps the shutdown tail by default; flag keeps it")
    test_merge_streams_no_timing_detector_keeps_all()
    print("OK  no timing detector -> no clamp (graceful)")
    test_include_shutdown_tail_round_trips()
    print("OK  include_shutdown_tail survives save/ship + back-compat default")
    test_read_event_on_clamped_tail_raises_hinted_keyerror()
    print("OK  read_event on a clamped tail ts raises a hinted KeyError")
    print("ALL PASS")
