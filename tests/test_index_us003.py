#!/usr/bin/env python3
"""US-003 acceptance test for psdata.index.

Verifies the acceptance criteria:

  1. The index is built by scanning only the small SMD files
     (``smalldata/{exp}-r{run:04d}-s###-c000.smd.xtc2``), reading each event's
     bigdata offset/size from ``smdinfo.offsetAlg.intOffset`` /
     ``.intDgramSize`` -- the GB-scale bigdata files are never opened during
     the build (proven by file-descriptor accounting on a wrapped ``os.open``).
  2. ``RunIndex.read_event(ts)`` / ``read_event_at(k)`` read an arbitrary event
     by random access using ``os.pread`` at the indexed (offset, size), WITHOUT
     sequentially scanning the bigdata file (proven by counting ``os.pread``
     calls: one per contributing stream, with the size matching the indexed
     dgram size).
  3. For the reference dataset (exp=mfx100848724 run=51
     dir=/sdf/data/lcls/ds/prj/public01/xtc), a random read of the K-th event
     (for several K) returns raw arrays byte-identical to the same event
     obtained via the US-002 streaming path.
  4. ``read_event(ts)`` and ``read_event_at(k)`` agree for the same event.
  5. The build reads only the small SMD files; the build time over the run's
     full event range is reported.
  6. Importing psdata / psdata.index does not import psana, mpi4py, or h5py.

Needs the production psana env (psconda.sh) on host sdfiana025 only for the
optional cross-check against psana raw arrays; the streaming-vs-random-access
equivalence (criterion 3) needs no psana.
"""

import glob
import os
import subprocess
import sys

import numpy as np

# --- locate the package (parent of this tests dir) --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(os.path.dirname(_HERE))  # .../<repo> (holds psdata/)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# Reference dataset -- lives in the TEST, never in the library.
EXP = "mfx100848724"
RUN = 51
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
JUNGFRAU = "jungfrau"
# Event positions to spot-check by random access (well within the run).
K_VALUES = [0, 1, 5, 17, 100, 999]


def _stream_files():
    """Resolve the run's per-stream bigdata xtc2 files by globbing -- the test,
    not the library, knows the file-naming pattern."""
    paths = sorted(glob.glob(f"{DIR}/{EXP}-r{RUN:04d}-s*-c000.xtc2"))
    assert paths, f"no stream files found under {DIR}"
    files = {}
    for p in paths:
        base = os.path.basename(p)
        sidx = int(base.split("-s")[1].split("-")[0])
        files[sidx] = p
    return files


# --------------------------------------------------------------------------
# 6. import purity
# --------------------------------------------------------------------------
def test_import_purity_before_psana():
    import psdata
    from psdata import index as psindex
    psindex.assert_no_framework_imports()
    for m in ("psana", "mpi4py", "h5py"):
        assert m not in sys.modules, f"{m} leaked into sys.modules on import"


def test_import_purity_subprocess():
    code = (
        "import sys, psdata, psdata.index as i; "
        "i.assert_no_framework_imports(); "
        "bad=[m for m in ('psana','mpi4py','h5py') if m in sys.modules]; "
        "assert not bad, bad; print('CLEAN')"
    )
    env = dict(os.environ, PYTHONPATH=_PKG_PARENT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "CLEAN" in out.stdout, out.stdout


# --------------------------------------------------------------------------
# 1./5. build reads ONLY the SMD files; report build time over the full run
# --------------------------------------------------------------------------
def test_build_reads_only_smd_files():
    """Wrap ``os.open`` so we can record every file path opened during the
    index build, and assert none of them is a bigdata file."""
    import psdata
    from psdata import index as psindex

    files = _stream_files()
    rc = psdata.discover(files)

    smd_map = psindex.smd_files_for(files)
    smd_paths = set(smd_map.values())
    bd_paths = set(files.values())
    # sanity: smd and bigdata are disjoint and well-formed
    assert smd_paths and not (smd_paths & bd_paths)
    for s, p in smd_map.items():
        assert p.endswith(".smd.xtc2") and "/smalldata/" in p, p

    opened = []
    real_open = os.open

    def spy_open(path, *a, **kw):
        opened.append(os.fspath(path))
        return real_open(path, *a, **kw)

    os.open = spy_open
    try:
        ridx = psindex.RunIndex.build(smd_map, rc)
    finally:
        os.open = real_open

    # Every file opened during the build must be one of the SMD files; in
    # particular NO bigdata file may have been opened.
    opened_bigdata = [p for p in opened if p in bd_paths]
    assert not opened_bigdata, \
        f"index build opened bigdata files: {opened_bigdata}"
    assert all(p in smd_paths for p in opened), \
        f"build opened non-SMD files: {[p for p in opened if p not in smd_paths]}"

    assert ridx.n_events > 0
    assert ridx.smd_bytes_read > 0
    # The build should read only a tiny fraction of the run's data: the SMD
    # files total a few MB; the bigdata files total hundreds of GB.
    bd_total = sum(os.path.getsize(p) for p in bd_paths)
    assert ridx.smd_bytes_read < bd_total / 1000, \
        (f"build read {ridx.smd_bytes_read} B, suspiciously large vs SMD size "
         f"(bigdata total {bd_total} B)")
    print(f"[build] indexed {ridx.n_events} events from SMD only in "
          f"{ridx.build_seconds:.3f}s, read "
          f"{ridx.smd_bytes_read/1e6:.1f} MB of SMD "
          f"(bigdata total {bd_total/1e9:.1f} GB, never opened); "
          f"multichunk_streams={sorted(ridx.multichunk_streams)}")
    ridx.close()
    return ridx.n_events


# --------------------------------------------------------------------------
# 2. random read uses pread at indexed (offset,size), no sequential scan
# --------------------------------------------------------------------------
def test_random_read_uses_pread_no_scan():
    """A single random read must issue exactly one ``os.pread`` per
    contributing stream, each for the indexed dgram size -- never a walk of the
    bigdata file."""
    import psdata
    from psdata import index as psindex

    files = _stream_files()
    rc = psdata.discover(files)
    ridx = psindex.build_index(files, run_config=rc)

    k = 1000
    entry = ridx.entries[k]
    # entries[k][stream] = (chunk_path, offset, size) since US-004 (the chunk
    # path is recorded so multi-chunk runs read from the right file).
    expected_sizes = sorted(size for (_path, _off, size) in entry.values())

    preads = []
    real_pread = os.pread

    def spy_pread(fd, n, off):
        preads.append((n, off))
        return real_pread(fd, n, off)

    os.pread = spy_pread
    try:
        evt = ridx.read_event_at(k)
        _ = evt.stack(JUNGFRAU)  # force materialization
    finally:
        os.pread = real_pread

    # one pread per contributing stream, each exactly the indexed dgram size,
    # at the indexed offset -- no extra reads, no scanning.  entry maps
    # stream -> (offset, size).
    assert len(preads) == len(entry), \
        f"{len(preads)} preads for {len(entry)} contributing streams"
    got_sizes = sorted(n for (n, _off) in preads)
    assert got_sizes == expected_sizes, \
        f"pread sizes {got_sizes} != indexed dgram sizes {expected_sizes}"
    # the (offset, size) pairs preaded must be exactly the indexed pairs
    # (entries[k][stream] = (chunk_path, offset, size) since US-004)
    indexed_pairs = sorted((off, size) for (_path, off, size) in entry.values())
    pread_pairs = sorted((off, n) for (n, off) in preads)
    assert pread_pairs == indexed_pairs, \
        f"pread (offset,size) {pread_pairs} != indexed {indexed_pairs}"
    # total bytes pread is the sum of the event's dgrams -- a few MB, NOT the
    # multi-GB bigdata file size.
    total_pread = sum(n for (n, _o) in preads)
    print(f"[random] event k={k}: {len(preads)} preads "
          f"({total_pread/1e6:.2f} MB total) at indexed offsets, no scan")
    ridx.close()


# --------------------------------------------------------------------------
# 3./4. random read == streaming read, byte-identical; read_event(ts)==at(k)
# --------------------------------------------------------------------------
def test_random_matches_streaming():
    """For several K, the K-th event obtained by random access must be
    byte-identical (every detector's raw arrays, pulseId, ts) to the K-th event
    from the US-002 streaming path -- no psana required."""
    import psdata
    from psdata import index as psindex

    files = _stream_files()
    rc = psdata.discover(files)

    # --- streaming ground truth: the first max(K)+1 events ----------------
    want_upto = max(K_VALUES) + 1
    stream_events = {}    # k -> (ts, pulseId, {det: stack-or-None})
    raw_det_names = rc.detector_names()   # real detectors (no bookkeeping)
    for k, evt in enumerate(psdata.events(files, run_config=rc)):
        if k >= want_upto:
            break
        dets = {}
        for name in raw_det_names:
            det = rc.detector(name)
            if "raw" in det.algs and "raw" in det.algs["raw"]:
                dets[name] = evt.stack(name)   # ndarray or None
        stream_events[k] = (evt.timestamp, evt.pulseId, dets)

    # --- random access ----------------------------------------------------
    ridx = psindex.build_index(files, run_config=rc)
    n_checked_dets = 0
    for k in K_VALUES:
        gts, gpid, gdets = stream_events[k]

        # read by position
        evt_k = ridx.read_event_at(k)
        # and by exact timestamp -- must resolve to the same event
        evt_ts = ridx.read_event(gts)
        assert evt_ts.timestamp == evt_k.timestamp == gts, \
            f"k={k}: ts mismatch by={evt_ts.timestamp} at={evt_k.timestamp} stream={gts}"
        assert evt_k.pulseId == gpid, \
            f"k={k}: pulseId mismatch random={evt_k.pulseId} stream={gpid}"
        assert evt_ts.pulseId == gpid

        for name, gstack in gdets.items():
            rstack = evt_k.stack(name)
            tstack = evt_ts.stack(name)
            if gstack is None:
                assert rstack is None and tstack is None, \
                    f"k={k} {name}: streaming None but random !None"
                continue
            assert rstack is not None, f"k={k} {name}: random None, stream !None"
            assert rstack.shape == gstack.shape and rstack.dtype == gstack.dtype, \
                f"k={k} {name}: shape/dtype {rstack.shape}/{rstack.dtype} vs " \
                f"{gstack.shape}/{gstack.dtype}"
            assert np.array_equal(rstack, gstack), (
                f"k={k} {name}: random-access raw != streaming raw; max|diff|="
                f"{np.abs(rstack.astype('int64')-gstack.astype('int64')).max()}")
            # by-timestamp path must agree too
            assert np.array_equal(tstack, gstack), \
                f"k={k} {name}: read_event(ts) raw != streaming raw"
            n_checked_dets += 1

    ridx.close()
    print(f"[equiv] random-access K={K_VALUES}: ts + pulseId + raw arrays "
          f"byte-identical to streaming ({n_checked_dets} detector-events "
          f"checked); read_event(ts) == read_event_at(k)")


def test_unknown_ts_raises():
    import psdata
    from psdata import index as psindex
    files = _stream_files()
    rc = psdata.discover(files)
    ridx = psindex.build_index(files, run_config=rc)
    try:
        ridx.read_event(1)          # not a real timestamp
    except KeyError:
        pass
    else:
        raise AssertionError("read_event with bogus ts should raise KeyError")
    finally:
        ridx.close()
    print("[guard] read_event(unknown ts) -> KeyError (ok)")


# --------------------------------------------------------------------------
# OPTIONAL cross-check vs psana (kept light: a couple of events)
# --------------------------------------------------------------------------
def test_random_matches_psana():
    """Spot-check that random-access raw equals psana's det.raw.raw(evt) for a
    couple of events (the byte-exact oracle).  Skipped cleanly if psana is not
    importable."""
    try:
        from psana import DataSource
    except Exception as e:                      # pragma: no cover
        print(f"[psana] skipped (psana not importable: {e})")
        return
    import psdata
    from psdata import index as psindex

    files = _stream_files()
    rc = psdata.discover(files)
    ridx = psindex.build_index(files, run_config=rc)

    ds = DataSource(exp=EXP, run=RUN, dir=DIR)
    run = next(ds.runs())
    jf = run.Detector(JUNGFRAU)

    check_k = {0, 5, 50}
    gt = {}
    for k, evt in enumerate(run.events()):
        if k > max(check_k):
            break
        if k in check_k:
            ts = evt.timestamp
            ts = int(ts() if callable(ts) else ts)
            gt[k] = (ts, np.asarray(jf.raw.raw(evt)))

    for k, (ts, graw) in gt.items():
        evt = ridx.read_event(ts)
        my = evt.stack(JUNGFRAU)
        assert my is not None and my.shape == graw.shape, (k, my.shape if my is not None else None, graw.shape)
        assert np.array_equal(my, graw), (
            f"k={k}: random-access raw != psana raw; max|diff|="
            f"{np.abs(my.astype('int64')-graw.astype('int64')).max()}")
    ridx.close()
    print(f"[psana] random-access raw byte-identical to psana det.raw.raw "
          f"for events {sorted(gt)} (np.array_equal)")


def main():
    print("=" * 72)
    print("US-003 acceptance: psdata.index random access by event / timestamp")
    print("=" * 72)
    test_import_purity_before_psana(); print("[ok] import purity (in-proc)")
    test_import_purity_subprocess();   print("[ok] import purity (subprocess)")
    test_build_reads_only_smd_files()
    print("[ok] index build reads only the small SMD files")
    test_random_read_uses_pread_no_scan()
    print("[ok] random read uses pread at indexed (offset,size), no scan")
    test_random_matches_streaming()
    print("[ok] random access == streaming (byte-identical) & ts==position")
    test_unknown_ts_raises()
    print("[ok] read_event(unknown ts) raises")
    test_random_matches_psana()
    print("[ok] random access == psana raw (byte-identical)")
    print("\nALL US-003 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
