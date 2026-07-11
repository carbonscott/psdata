#!/usr/bin/env python3
"""US-009 -- batch random-read (``RunIndex.read_events`` / ``read_stack``).

psdata already serves a single event by random access (``read_event_at(k)``,
one ``os.pread`` per contributing stream).  This story adds a **batch** read:

  * ``RunIndex.read_events(ks) -> list[Event]`` -- read many positions in ONE
    call, coalescing the ``pread``s (grouped per bigdata chunk file, ascending
    offset within each), returned in ``ks`` order; and
  * ``RunIndex.read_stack(ks, det, field='raw', alg='raw') -> np.ndarray`` of
    shape ``(len(ks), n_seg, *seg_shape)`` filled into ONE preallocated buffer.

This suite is **self-contained** -- the oracle is psdata's own single-event
path:

  * ``read_events(ks)`` must be byte-identical to
    ``[read_event_at(k) for k in ks]`` (ts, pulseId, every detector's raw
    arrays), for ``ks`` in non-contiguous / shuffled / repeated order;
  * ``read_stack(ks, 'jungfrau')`` must equal
    ``np.stack([read_event_at(k).stack('jungfrau') for k in ks])``;
  * a coalesced batch must issue STRICTLY FEWER ``os.pread``s than the serial
    ``sum over ks of contributing streams`` (PERF-01: adjacent per-stream dgrams
    are merged into one syscall over the covering span), while every indexed
    dgram is still covered by exactly one issued span -- no extra scan -- and each
    chunk file is opened at most once.

No psana is required.

Run (on sdfiana025, from the repo root):
    PYTHONPATH=src .venv/bin/python tests/test_batch_us009.py
or via the suite:
    bash run_tests.sh tests/test_batch_us009.py
"""

import glob
import os
import sys

import numpy as np

# --- locate the package (parent of this tests dir) --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")  # .../<repo>/src (holds psdata/)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _HERE not in sys.path:                           # for _skips (sibling module)
    sys.path.insert(0, _HERE)

from _skips import skip   # noqa: E402  (machine-readable skip records, HYG-03)

# Reference dataset -- lives in the TEST, never in the library.  Single-chunk
# primary run (mfx100848724 r51): jungfrau raw (32,512,1024) uint16.
EXP = "mfx100848724"
RUN = 51
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
JUNGFRAU = "jungfrau"

# Multi-chunk run (streams roll c000 -> c001): exercises read_events reading
# from more than one chunk file in a single coalesced batch.
MC_EXP = "mfx101343025"
MC_RUN = 35
# The instrument dir has been seen spelled both ways; test_robust_us004 reads
# this same run from the lowercase path.  Take whichever exists (first by
# default) so a path-case mismatch cannot silently turn this check into a skip
# -- an un-run check is not a passing check (HYG-03).
MC_DIR_CANDIDATES = (
    "/sdf/data/lcls/ds/mfx/mfx101343025/xtc",
    "/sdf/data/lcls/ds/MFX/mfx101343025/xtc",
)
MC_DIR = next((d for d in MC_DIR_CANDIDATES if os.path.isdir(d)),
              MC_DIR_CANDIDATES[0])

# A deliberately non-contiguous / shuffled / repeated set of positions, plus 0
# and a far-out position, to prove order independence and de-duplication.
KS = [17, 0, 100, 5, 999, 1, 17, 0, 250, 42]


def _stream_files(directory, exp, run):
    """Resolve a run's per-stream bigdata c000 xtc2 files by globbing -- the
    test (not the library) knows the file-naming pattern.  The index follows the
    ``chunkinfo`` roll to later chunks itself."""
    paths = sorted(glob.glob(f"{directory}/{exp}-r{run:04d}-s*-c000.xtc2"))
    assert paths, f"no stream files found under {directory} for {exp} r{run}"
    files = {}
    for p in paths:
        base = os.path.basename(p)
        sidx = int(base.split("-s")[1].split("-")[0])
        files[sidx] = p
    return files


def _event_signature(evt, det_names, rc):
    """A fully-materialized, comparable snapshot of an Event: ts, pulseId, and
    every real detector's stacked raw array (or None).  Used to assert two
    Events are byte-identical regardless of how they were read."""
    dets = {}
    for name in det_names:
        det = rc.detector(name)
        if "raw" in det.algs and "raw" in det.algs["raw"]:
            dets[name] = evt.stack(name)   # ndarray or None
    return evt.timestamp, evt.pulseId, dets


def _assert_same_event(a, b, where):
    ats, apid, adets = a
    bts, bpid, bdets = b
    assert ats == bts, f"{where}: ts {ats} != {bts}"
    assert apid == bpid, f"{where}: pulseId {apid} != {bpid}"
    assert set(adets) == set(bdets), f"{where}: detector set differs"
    for name in adets:
        x, y = adets[name], bdets[name]
        if x is None or y is None:
            assert x is None and y is None, \
                f"{where} {name}: one None, one not ({x is None} vs {y is None})"
            continue
        assert x.shape == y.shape and x.dtype == y.dtype, \
            f"{where} {name}: shape/dtype {x.shape}/{x.dtype} vs {y.shape}/{y.dtype}"
        assert np.array_equal(x, y), (
            f"{where} {name}: arrays differ; max|diff|="
            f"{np.abs(x.astype('int64') - y.astype('int64')).max()}")


# --------------------------------------------------------------------------
# import purity (in-proc + subprocess) -- read_events/read_stack add no deps
# --------------------------------------------------------------------------
def test_import_purity():
    import subprocess
    import psdata
    from psdata import index as psindex
    psindex.assert_no_framework_imports()
    for m in ("psana", "mpi4py", "h5py"):
        assert m not in sys.modules, f"{m} leaked into sys.modules on import"
    code = (
        "import sys, psdata, psdata.index as i; "
        "i.assert_no_framework_imports(); "
        "bad=[m for m in ('psana','mpi4py','h5py') if m in sys.modules]; "
        "assert not bad, bad; print('CLEAN')"
    )
    env = dict(os.environ, PYTHONPATH=_SRC)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "CLEAN" in out.stdout, out.stdout
    print("[ok] import purity (read_events/read_stack add no framework deps)")


# --------------------------------------------------------------------------
# read_events(ks) is byte-identical to [read_event_at(k) for k in ks]
# --------------------------------------------------------------------------
def test_read_events_matches_serial():
    """Coalesced batch read == serial per-event read, for shuffled / repeated
    positions, returned in ``ks`` order."""
    import psdata
    from psdata import index as psindex

    files = _stream_files(DIR, EXP, RUN)
    rc = psdata.discover(files)
    ridx = psindex.build_index(files, run_config=rc)
    det_names = rc.detector_names()

    # serial oracle (each read independently, fresh single-event path)
    serial = [_event_signature(ridx.read_event_at(k), det_names, rc) for k in KS]

    # batch
    batch_events = ridx.read_events(KS)
    assert len(batch_events) == len(KS), \
        f"read_events returned {len(batch_events)} events for {len(KS)} ks"
    batch = [_event_signature(e, det_names, rc) for e in batch_events]

    for i, k in enumerate(KS):
        _assert_same_event(serial[i], batch[i], where=f"ks[{i}]=k{k}")

    # order independence: the same set in a different order yields the same
    # per-position events (so the output truly tracks ks order, not read order).
    perm = list(reversed(KS))
    perm_events = ridx.read_events(perm)
    perm_sig = [_event_signature(e, det_names, rc) for e in perm_events]
    for i, k in enumerate(perm):
        # find the matching serial signature (by position k)
        j = KS.index(k)
        _assert_same_event(serial[j], perm_sig[i], where=f"perm k{k}")

    ridx.close()
    print(f"[ok] read_events({KS}) byte-identical to serial read_event_at "
          f"(ts+pulseId+raw for {len(det_names)} detectors), order-independent")


# --------------------------------------------------------------------------
# read_stack(ks, det) == np.stack of per-event stack(det)
# --------------------------------------------------------------------------
def test_read_stack_matches_per_event():
    import psdata
    from psdata import index as psindex

    files = _stream_files(DIR, EXP, RUN)
    rc = psdata.discover(files)
    ridx = psindex.build_index(files, run_config=rc)

    # oracle: stack each event's jungfrau frame individually
    per_event = [ridx.read_event_at(k).stack(JUNGFRAU) for k in KS]
    assert all(s is not None for s in per_event), \
        "reference jungfrau frames unexpectedly missing a segment"
    oracle = np.stack(per_event, axis=0)

    out = ridx.read_stack(KS, JUNGFRAU)
    assert out.shape == oracle.shape, \
        f"read_stack shape {out.shape} != np.stack shape {oracle.shape}"
    assert out.shape[0] == len(KS), \
        f"read_stack first axis {out.shape[0]} != len(ks) {len(KS)}"
    assert out.dtype == oracle.dtype, \
        f"read_stack dtype {out.dtype} != {oracle.dtype}"
    assert np.array_equal(out, oracle), (
        "read_stack != np.stack of per-event stacks; max|diff|="
        f"{np.abs(out.astype('int64') - oracle.astype('int64')).max()}")

    # ONE preallocated buffer: each row equals the matching read_event_at stack
    for i, k in enumerate(KS):
        assert np.array_equal(out[i], ridx.read_event_at(k).stack(JUNGFRAU)), \
            f"read_stack row {i} (k={k}) != read_event_at(k).stack"

    ridx.close()
    print(f"[ok] read_stack({KS}, {JUNGFRAU!r}) shape {out.shape} "
          f"== np.stack of per-event Event.stack (byte-identical)")


# --------------------------------------------------------------------------
# pread-count guard: total preads == sum of contributing streams, no scan,
# and each chunk file opened at most once
# --------------------------------------------------------------------------
def test_batch_coalesces_pread_no_scan():
    """The whole batch must issue STRICTLY FEWER ``os.pread``s than the serial
    ``sum over ks of contributing streams`` (PERF-01: adjacent per-stream dgrams
    coalesce into one syscall), while every indexed dgram is still covered by
    exactly one issued span (no stray reads, no scan), and each distinct chunk
    file is opened at most once (lazy ``_bd_fd`` reuse)."""
    import psdata
    from psdata import index as psindex

    files = _stream_files(DIR, EXP, RUN)
    rc = psdata.discover(files)
    ridx = psindex.build_index(files, run_config=rc)

    # distinct positions actually read (repeats are de-duplicated by the batch)
    distinct = []
    seen = set()
    for k in KS:
        if k not in seen:
            seen.add(k)
            distinct.append(k)

    # expected preads = sum of contributing streams over the DISTINCT positions;
    # expected per-read (offset,size) pairs = exactly the indexed pairs.
    expected_pairs = []
    expected_chunk_paths = set()
    for k in distinct:
        for stream, (path, off, size) in ridx.entries[k].items():
            expected_pairs.append((off, size))
            expected_chunk_paths.add(path)
    expected_n = len(expected_pairs)

    preads = []                      # (fd, off, n) in call order
    opens = []
    real_pread, real_open = os.pread, os.open

    def spy_pread(fd, n, off):
        preads.append((fd, off, n))
        return real_pread(fd, n, off)

    def spy_open(path, *a, **kw):
        opens.append(os.fspath(path))
        return real_open(path, *a, **kw)

    os.pread, os.open = spy_pread, spy_open
    try:
        events = ridx.read_events(KS)
        for e in events:
            _ = e.stack(JUNGFRAU)  # materialize (no new preads -- bytes cached)
    finally:
        os.pread, os.open = real_pread, real_open

    # Attribute each pread / expected dgram to the chunk file (fd) it belongs to;
    # offsets restart per chunk file so they are only comparable within one fd.
    path_to_fd = dict(ridx._bd_fds)          # {chunk_path: fd} opened this batch
    fd_to_path = {fd: p for p, fd in path_to_fd.items()}

    # PERF-01: the batch now COALESCES adjacent per-stream reads (contiguous /
    # near-adjacent dgrams merged into ONE pread over the covering span, sliced
    # back), so it issues STRICTLY FEWER syscalls than the serial K*S path.  Two
    # of the requested positions -- events 0 and 1 -- are adjacent, and their
    # per-stream dgrams are contiguous on disk in every stream that carries both,
    # so at least those merge; the whole batch is therefore < the serial sum.
    #
    # [UPDATED for PERF-01.  The pre-fix assertions here were
    #    assert len(preads) == expected_n                 # 80 for 8 events
    #    assert sorted(preads) == sorted(expected_pairs)
    # which ENCODED the no-coalescing behaviour PERF-01 fixes (one pread per
    # (event, stream), byte ranges never merged).  They are replaced by the
    # strictly-fewer count PLUS the real coalescing contract below (every indexed
    # dgram covered by exactly one merged pread span, spans disjoint & ascending
    # per chunk, no stray reads).  End-to-end byte-exactness of the coalesced
    # bytes is proven separately by test_read_events_matches_serial.]
    assert len(preads) < expected_n, (
        f"{len(preads)} preads for {len(distinct)} distinct events -- expected "
        f"STRICTLY FEWER than the serial K*S={expected_n} (sum of contributing "
        f"streams): the batch must COALESCE adjacent per-stream reads (PERF-01), "
        f"not issue one pread per (event, stream).")

    # Group the issued preads and the expected dgram ranges by chunk file (fd).
    expected_by_fd = {}
    for k in distinct:
        for stream, (path, off, size) in ridx.entries[k].items():
            expected_by_fd.setdefault(path_to_fd[path], []).append((off, size))
    issued_by_fd = {}
    for fd, off, n in preads:
        issued_by_fd.setdefault(fd, []).append((off, off + n))

    for fd, spans in issued_by_fd.items():
        base = os.path.basename(fd_to_path[fd])
        ordered = spans[:]                                  # call order per fd
        assert ordered == sorted(ordered), \
            f"{base}: preads not issued in ascending offset order: {ordered}"
        # merged spans must be disjoint (a dgram is never read twice)
        for (s0, e0), (s1, e1) in zip(ordered, ordered[1:]):
            assert e0 <= s1, f"{base}: issued pread spans overlap: {ordered}"
        # every indexed dgram range is covered by EXACTLY ONE issued span, and
        # every issued span covers at least one requested dgram (no stray reads).
        exp = expected_by_fd.get(fd, [])
        for off, size in exp:
            covering = [(s, e) for (s, e) in ordered if s <= off and off + size <= e]
            assert len(covering) == 1, (
                f"{base}: indexed dgram [{off},{off + size}) covered by "
                f"{len(covering)} pread spans (want exactly 1): {covering}")
        for s, e in ordered:
            assert any(s <= o and o + sz <= e for (o, sz) in exp), \
                f"{base}: issued pread span [{s},{e}) covers no requested dgram"

    # each distinct chunk file opened at most once during the batch.
    opened_chunks = [p for p in opens if p in expected_chunk_paths]
    assert len(opened_chunks) == len(set(opened_chunks)), \
        f"a chunk file was opened more than once: {opened_chunks}"
    assert set(opened_chunks) <= expected_chunk_paths, \
        f"opened unexpected files: {set(opened_chunks) - expected_chunk_paths}"

    ridx.close()
    print(f"[ok] batch of {len(KS)} ks ({len(distinct)} distinct) COALESCED the "
          f"serial K*S={expected_n} reads into {len(preads)} preads "
          f"(< K*S; PERF-01), every indexed dgram covered by exactly one span, "
          f"{len(set(opened_chunks))} chunk file(s) opened once, ascending offset")


# --------------------------------------------------------------------------
# missing-segment policy: read_stack raises on an incomplete event;
# read_events still returns it (None surfaces lazily)
# --------------------------------------------------------------------------
def test_read_stack_missing_segment_policy():
    """If a requested event is missing a segment for the detector,
    ``read_stack`` raises ``ValueError`` (a dense numeric buffer cannot hold a
    None row), while ``read_events`` returns the Event and ``stack`` is None --
    consistent with the existing missing-segment rule, applied eagerly only
    where a dense array forces it.

    Built synthetically by dropping one stream from one event's index entry so
    the detector's received segment set no longer matches its declared set."""
    import copy
    import psdata
    from psdata import index as psindex

    files = _stream_files(DIR, EXP, RUN)
    rc = psdata.discover(files)
    ridx = psindex.build_index(files, run_config=rc)

    # pick an event and find a jungfrau-contributing stream to drop
    jf = rc.detector(JUNGFRAU)
    jf_streams = jf.streams_for("raw")
    assert jf_streams, "jungfrau declares no streams?"
    drop_stream = sorted(jf_streams)[0]

    k_bad = 7
    # remove one contributing stream so the segment set is incomplete
    orig_entry = ridx.entries[k_bad]
    assert drop_stream in orig_entry, \
        f"event {k_bad} does not carry stream {drop_stream}"
    ridx.entries[k_bad] = {s: v for s, v in orig_entry.items()
                           if s != drop_stream}

    # read_events returns the (incomplete) Event; stack(det) is None
    evt = ridx.read_events([k_bad])[0]
    assert evt.stack(JUNGFRAU) is None, \
        "expected jungfrau stack to be None for the incomplete event"

    # read_stack raises ValueError naming the offending position
    raised = False
    try:
        ridx.read_stack([0, k_bad, 1], JUNGFRAU)
    except ValueError as e:
        raised = True
        assert str(k_bad) in str(e), \
            f"ValueError should name position {k_bad}: {e}"
    assert raised, "read_stack should raise ValueError on an incomplete event"

    # an empty ks is a clean ValueError too
    try:
        ridx.read_stack([], JUNGFRAU)
    except ValueError:
        pass
    else:
        raise AssertionError("read_stack([]) should raise ValueError")

    ridx.entries[k_bad] = orig_entry   # restore (defensive)
    ridx.close()
    print("[ok] read_stack raises on missing-segment event; read_events returns "
          "it with stack=None (policy consistent + documented)")


# --------------------------------------------------------------------------
# multi-chunk: a single coalesced batch reading across chunk files (c000+c001)
# --------------------------------------------------------------------------
def test_multichunk_batch():
    """On the multi-chunk run, a batch that spans the chunk roll reads from more
    than one chunk file in one call and stays byte-identical to the serial
    path."""
    paths = sorted(glob.glob(f"{MC_DIR}/{MC_EXP}-r{MC_RUN:04d}-s*-c000.xtc2"))
    if not paths:
        return skip(
            "multichunk_batch_run_missing",
            f"the multi-chunk run {MC_EXP}/r{MC_RUN} was not found under any of "
            f"{list(MC_DIR_CANDIDATES)}; a coalesced batch spanning the chunk "
            f"roll (c000 -> c001) is therefore unverified -- the single-chunk "
            f"batch checks above cannot exercise a roll")
    import psdata
    from psdata import index as psindex

    files = _stream_files(MC_DIR, MC_EXP, MC_RUN)
    rc = psdata.discover(files)
    ridx = psindex.build_index(files, run_config=rc)
    det_names = rc.detector_names()
    n = ridx.n_events
    assert n > 0

    # choose positions spanning before and after a chunk roll, if any stream
    # rolled; otherwise just spread across the run.
    if ridx.multichunk_streams:
        # find the first position whose entry references a non-c000 chunk
        roll_k = None
        for k in range(n):
            if any("c000" not in path
                   for (path, _o, _s) in ridx.entries[k].values()):
                roll_k = k
                break
        post = [roll_k, roll_k + 1, n - 1] if roll_k is not None else [n - 1]
    else:
        post = [n - 1]
    ks = [0, 1, n // 2] + post
    ks = [k for k in ks if 0 <= k < n]
    # shuffle so the batch must reorder across chunk files
    ks = list(reversed(ks))

    serial = [_event_signature(ridx.read_event_at(k), det_names, rc) for k in ks]
    batch = [_event_signature(e, det_names, rc) for e in ridx.read_events(ks)]
    for i, k in enumerate(ks):
        _assert_same_event(serial[i], batch[i], where=f"mc ks[{i}]=k{k}")

    n_chunks = len({path for k in ks for (path, _o, _s) in ridx.entries[k].values()})
    ridx.close()
    print(f"[ok] multi-chunk batch ks={ks} across {n_chunks} chunk file(s) "
          f"byte-identical to serial (multichunk_streams="
          f"{sorted(ridx.multichunk_streams)})")


def main():
    print("=" * 72)
    print("US-009 acceptance: batch random-read (read_events / read_stack)")
    print("=" * 72)
    test_import_purity()
    test_read_events_matches_serial()
    test_read_stack_matches_per_event()
    test_batch_coalesces_pread_no_scan()
    test_read_stack_missing_segment_policy()
    test_multichunk_batch()
    print("\nALL US-009 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
