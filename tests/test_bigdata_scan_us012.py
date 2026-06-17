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

import numpy as np

# --- locate the package (parent of this tests dir) --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")  # .../<repo>/src (holds psdata/)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import psdata
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
    return len(idx_smd.timestamps)


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


def test_bigdata_path_is_framework_pure():
    """Building from bigdata must not import any framework."""
    psdata.build_index(FILES, source="bigdata")
    _ix.assert_no_framework_imports()


if __name__ == "__main__":
    n = test_bigdata_index_byte_exact_against_smd()
    print(f"OK  bigdata index byte-exact vs SMD ({n} events, small streams)")
    test_random_read_identical_either_index()
    print("OK  random reads byte-identical from either index")
    test_build_index_source_routing()
    print("OK  build_index source routing (smd / bigdata / auto-fallback)")
    test_bigdata_path_is_framework_pure()
    print("OK  bigdata scan path is framework-pure")
    print("ALL PASS")
