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
    """Every event the SMD path indexes is present in the bigdata-built index
    with byte-identical (offset, size); the bigdata index is a superset (it may
    also recover trailing dgrams the offline smdwriter never wrote to the SMD)."""
    idx_smd = psdata.build_index(FILES, source="smd")
    idx_bd = psdata.build_index(FILES, source="bigdata")
    assert idx_smd.scan_source == "smd" and idx_bd.scan_source == "bigdata"

    bd_pos = {ts: k for k, ts in enumerate(idx_bd.timestamps)}
    assert len(idx_bd.timestamps) >= len(idx_smd.timestamps), \
        "bigdata index must contain at least every SMD-indexed event"
    for ts, smd_entry in zip(idx_smd.timestamps, idx_smd.entries):
        assert ts in bd_pos, f"bigdata index missing SMD event ts={ts}"
        assert _entry_key(smd_entry) == _entry_key(idx_bd.entries[bd_pos[ts]]), \
            f"offset/size mismatch at ts={ts}"
    # on THIS run the small streams have no smdwriter tail gap -> exact match.
    assert idx_smd.timestamps == idx_bd.timestamps, \
        "expected exact equality on the small streams of run 51"


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
    print("ALL PASS")
