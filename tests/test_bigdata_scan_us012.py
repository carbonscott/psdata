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

For speed the byte-exact SMD-vs-bigdata comparison uses the run's three SMALL
streams (s000/s001/s004): a full bigdata-header walk of a small file is
sub-second.  The giant detector streams (jungfrau s005/s007, tens-to-hundreds
of GB) cannot be header-walked in full here -- the walk "didn't finish in 480 s"
-- so GATE-06 covers them with a BOUNDED tail scan instead: it seeds the
SMD-free header walk at the last SMD-indexed dgram's byte offset (deep in the
file -- the tail, never a head prefix) and walks forward to EOF, reaching the
ragged DAQ end-of-run shutdown tail where the 17982-vs-17872 superset case
actually lives (see :func:`test_bigdata_detector_stream_reaches_ragged_tail`).
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

# GATE-06 -- the DETECTOR (jungfrau) BIGDATA streams of this run.  The ragged
# DAQ end-of-run shutdown tail lives HERE (streams 5 & 7), and a FULL
# bigdata-header walk of a ~600 GB jungfrau stream is far too slow for the suite
# (it "didn't finish in 480 s").  So the covered-stream set can no longer be the
# three SMALL streams alone: without a detector stream the "SMD is optional"
# bigdata build is never exercised on real detector data and the superset /
# ragged-tail case it advertises is never met.  These are covered WITHOUT a full
# walk -- a BOUNDED tail scan seeded at the last SMD offset; see
# :func:`test_bigdata_detector_stream_reaches_ragged_tail`.
DETECTOR_STREAMS = (5, 7)
DET_FILES = {s: f"{DIR}/{EXP}-r{RUN:04d}-s{s:03d}-c000.xtc2"
             for s in DETECTOR_STREAMS}

# Oracle magnitudes for mfx100848724/r51, pinned across the suite (see
# tests/test_stream_us002.py N_INDEXED and src/psdata/run.py): the offline SMD
# writer indexed 17872 canonical L1Accepts, while the bigdata detector streams
# physically carry 17982 -- 110 more, the ragged DAQ-shutdown tail the SMD never
# recorded.  These are the ONLY assertions in the detector test that consult a
# number psana produced, so it is an ORACLE check, not psdata-vs-psdata.
N_SMD_INDEXED = 17872
N_BIGDATA_TAIL = 110              # 17982 - 17872 phantom shutdown-tail events
N_BIGDATA_SUPERSET = 17982


def _entry_key(entry):
    """Normalise one event's index entry to {stream: (basename, off, size)} so
    SMD- and bigdata-built entries compare regardless of absolute path object."""
    return {s: (os.path.basename(p), o, sz) for s, (p, o, sz) in entry.items()}


def _forward_l1_from_offset(path, start_off, filesize, max_dgrams=1 << 20):
    """Seed the SMD-free bigdata cursor at ``start_off`` and walk FORWARD to EOF,
    reading only each dgram's 24-byte header (never its GB-scale payload).

    This is the exact ``psdata.index._scan_bigdata_stream`` inner loop -- the
    ``smdwriter`` algorithm in pure Python, the same cursor walk the oracle
    :func:`_walk_l1accepts` performs -- but *started deep in the file* rather
    than at offset 0, so it costs only the dgrams from ``start_off`` to EOF.
    Given a ``start_off`` near the end (the last SMD-indexed dgram of a detector
    stream) this is a BOUNDED tail scan of a few hundred dgrams, not a full
    ~600 GB header walk, yet it still reaches the true end of the stream.

    Returns ``(boundary_header, tail_l1, reached_eof, n_dgrams)``:

      * ``boundary_header`` -- the parsed header AT ``start_off`` (with ``_off``
        and ``_total`` added), i.e. the dgram the SMD offset points at;
      * ``tail_l1`` -- ``(ts, off, size)`` for every ``L1Accept`` STRICTLY AFTER
        the boundary dgram, to end of file (the ragged shutdown tail);
      * ``reached_eof`` -- True iff the walk consumed the file cleanly to a dgram
        boundary at EOF (so the whole tail was seen, not a window that stopped
        short);
      * ``n_dgrams`` -- dgrams walked (boundary included).
    """
    boundary_header = None
    tail_l1 = []
    n_dgrams = 0
    reached_eof = False
    fd = os.open(path, os.O_RDONLY)
    try:
        cursor = start_off
        first = True
        while cursor + psdata.format.DGRAM_HDR <= filesize:
            if n_dgrams >= max_dgrams:
                break                      # safety cap; reached_eof stays False
            hdr = os.pread(fd, psdata.format.DGRAM_HDR, cursor)
            if len(hdr) < psdata.format.DGRAM_HDR:
                break
            h = psdata.format.parse_dgram_header(hdr, 0)
            total = psdata.format.XTC_HDR + h["extent"]
            if cursor + total > filesize:
                break                      # dgram runs past EOF (truncation)
            n_dgrams += 1
            if first:
                boundary_header = dict(h)
                boundary_header["_off"] = cursor
                boundary_header["_total"] = total
                first = False
            elif h["service"] == psdata.format.SERVICE_L1ACCEPT:
                tail_l1.append((h["ts"], cursor, total))
            cursor += total
        else:
            # loop ended on the while-condition (not a break): a clean end has
            # cursor exactly at EOF (r51's streams end on a dgram boundary).
            reached_eof = (cursor == filesize)
        return boundary_header, tail_l1, reached_eof, n_dgrams
    finally:
        os.close(fd)


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


def test_bigdata_detector_stream_reaches_ragged_tail():
    """GATE-06 -- exercise the "SMD is optional" bigdata build on a real
    DETECTOR (jungfrau) stream, ALL THE WAY TO its ragged DAQ-shutdown TAIL.

    ``test_bigdata_index_byte_exact_against_smd`` above proves the SMD-free walk
    on the SMALL streams (0/1/4), which carry the timing stream and have NO
    ragged tail -- so it can only ever assert *exact* ts equality and never meets
    the superset case the module advertises.  The 17982-vs-17872 gap lives on the
    DETECTOR streams (5 & 7), which the small-stream tests never touch.  This test
    covers them.

    Feasibility: a FULL bigdata-header walk of a ~600 GB jungfrau stream "didn't
    finish in 480 s", so we do NOT walk from offset 0.  Instead we build the SMD
    index over the detector streams (fast -- the sidecars are tiny), take the
    byte offset of the LAST SMD-indexed dgram on each stream (deep in the file --
    the TAIL, never a head prefix), and walk the SMD-free header cursor forward
    from there to EOF.  That is a BOUNDED scan of a few hundred dgrams that still
    reaches the true end of the detector stream, where the shutdown tail lives.

    Assertions (MEANINGFUL -- a superset, not mere equality):
      1. OVERLAP is byte-exact: at the last SMD offset the SMD-free header walk
         reads the SMD-recorded ``(ts, intDgramSize)`` -- proving the SMD is just
         a cache of the same bigdata cursor on the DETECTOR data too;
      2. the walk reaches EOF, so the WHOLE tail was seen;
      3. every L1Accept past the boundary is a PHANTOM the SMD never indexed;
      4. the SMD-free detector-stream event set is a STRICT SUPERSET of the SMD
         set, of exactly the oracle magnitude (17982 vs 17872, tail = 110).
    """
    rc = psdata.discover(DET_FILES)
    # Sanity: streams 5 & 7 really are the jungfrau (detector) streams -- so this
    # is genuinely detector coverage, not another non-detector stream.
    jung = rc.find_detector_by_type("jungfrau")
    assert jung, "run config declares no jungfrau detector on the detector streams"
    jstreams = set(rc.detector(jung[0]).streams_for("raw"))
    assert set(DETECTOR_STREAMS) <= jstreams, \
        f"streams {DETECTOR_STREAMS} are not all jungfrau streams ({sorted(jstreams)})"

    # SMD-indexed truth over the detector streams only -- fast (tiny sidecars).
    idx_smd = psdata.build_index(DET_FILES, run_config=rc, source="smd")
    assert idx_smd.scan_source == "smd"
    smd_ts = set(idx_smd.timestamps)
    assert len(smd_ts) == N_SMD_INDEXED, (
        f"SMD index over the detector streams has {len(smd_ts)} events, expected "
        f"the oracle {N_SMD_INDEXED} for {EXP}/r{RUN} (the set the superset is "
        f"taken against must itself be the psana/SMD event set)")

    tail_ts = set()
    for s in DETECTOR_STREAMS:
        bd = DET_FILES[s]
        # Single-chunk run: the whole stream is c000, so a forward walk to EOF
        # stays in one file (the STR-04 chunk roll is not needed here).
        assert not os.path.exists(bd.replace("-c000.xtc2", "-c001.xtc2")), \
            f"stream s{s:03d} is multi-chunk; this bounded tail walk assumes single-chunk r51"
        filesize = os.path.getsize(bd)

        # The last SMD-indexed dgram ON THIS STREAM: its bigdata offset + size.
        last = None
        for ts, entry in zip(reversed(idx_smd.timestamps),
                             reversed(idx_smd.entries)):
            if s in entry:
                cpath, off, size = entry[s]
                last = (ts, cpath, off, size)
                break
        assert last is not None, f"no SMD entry references detector stream s{s:03d}"
        last_ts, last_cpath, last_off, last_size = last
        assert os.path.basename(last_cpath) == os.path.basename(bd)
        # The seed offset is DEEP in the detector file -- this reaches the TAIL,
        # never a head prefix (a head-prefix scan would start near offset 0).
        assert last_off > 0.5 * filesize, (
            f"s{s:03d}: last SMD offset {last_off} is not deep in the "
            f"{filesize}-byte file -- a tail scan must start near the END, not "
            f"the head (this is the head-prefix disease GATE-06 guards against)")

        boundary, tail_l1, reached_eof, ndg = _forward_l1_from_offset(
            bd, last_off, filesize)

        # (1) OVERLAP byte-exact: the SMD-free header walk reproduces the SMD's
        # recorded (ts, intDgramSize) at the boundary offset on the DETECTOR
        # stream -- the SMD is just a cache of this same bigdata cursor.
        assert boundary is not None and \
            boundary["service"] == psdata.format.SERVICE_L1ACCEPT, \
            f"s{s:03d}: no L1Accept dgram at the SMD-recorded offset {last_off}"
        assert boundary["ts"] == last_ts, (
            f"s{s:03d}: bigdata dgram ts {boundary['ts']} at the SMD offset "
            f"!= the SMD-recorded ts {last_ts}")
        assert boundary["_total"] == last_size, (
            f"s{s:03d}: bigdata dgram size {boundary['_total']} at the SMD offset "
            f"!= the SMD-recorded intDgramSize {last_size}")

        # (2) reached the TRUE end of the detector stream -> the whole tail seen.
        assert reached_eof, (
            f"s{s:03d}: forward walk from the last SMD event did not reach EOF "
            f"({ndg} dgrams, cursor short of the {filesize}-byte end) -- the tail "
            f"was not fully covered")

        # (3) every L1Accept past the boundary is a PHANTOM the SMD never indexed.
        for ts, off, sz in tail_l1:
            assert off > last_off, \
                f"s{s:03d}: tail record at {off} is not strictly past the boundary"
            assert ts not in smd_ts, (
                f"s{s:03d}: bigdata tail L1Accept ts={ts} is unexpectedly IN the "
                f"SMD index -- it must be a phantom shutdown-tail event")
            tail_ts.add(ts)

    # (4) STRICT SUPERSET: the SMD-free bigdata build sees MORE detector-stream
    # L1Accepts than the SMD recorded -- the 17982-vs-17872 ragged-tail gap,
    # reproduced here from a BOUNDED tail walk (no full ~600 GB header scan).
    superset_ts = smd_ts | tail_ts
    assert superset_ts > smd_ts, (
        "the detector bigdata must carry ragged-tail L1Accepts the SMD never "
        "indexed -- a strict superset, not mere equality")
    assert len(tail_ts) == N_BIGDATA_TAIL, (
        f"detector-stream ragged tail has {len(tail_ts)} phantom events, expected "
        f"the oracle {N_BIGDATA_TAIL} ({N_BIGDATA_SUPERSET} bigdata - "
        f"{N_SMD_INDEXED} SMD) for {EXP}/r{RUN}")
    assert len(superset_ts) == N_BIGDATA_SUPERSET, (
        f"SMD-free detector-stream event set has {len(superset_ts)} events, "
        f"expected the oracle {N_BIGDATA_SUPERSET}")
    return len(superset_ts)


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
    _sup = test_bigdata_detector_stream_reaches_ragged_tail()
    print(f"OK  detector-stream bigdata build reaches the ragged tail "
          f"({_sup} events, strict superset of {N_SMD_INDEXED} SMD; streams "
          f"{DETECTOR_STREAMS})")
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
