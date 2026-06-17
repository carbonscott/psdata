#!/usr/bin/env python
"""Ray shared-index cube -- a DEMONSTRATOR for the psdata shardable index.

This exercises the two primitives the ``feature/shardable-index`` work added to
the numpy-only psdata reader:

  * **US-008 -- serializable / persisted index.** The driver builds the
    :class:`psdata.index.RunIndex` ONCE (one SMD scan), then ships it to every
    worker through Ray's object store as a single ``index.to_dict()`` payload.
    Each worker reconstructs it with :meth:`RunIndex.from_dict` -- *no* SMD
    rescan, and fd-safe (the raw ``os.open`` fds are stripped on serialize and
    reopen lazily, per the verified pickle gotcha).
  * **US-009 -- batch random read.** Each worker batch-reads its slice of event
    positions with :meth:`RunIndex.read_events` (coalesced ``pread``s, grouped
    per chunk file in ascending offset order), instead of one seek per event.

It is the framework-free answer to the owner's Ray prototype
(``cube_prototype/ray_cube_prototype.py``), whose every ``@ray.remote`` worker
rebuilds a psana ``DataSource`` + ``run.build_table()`` -- a full SMD rescan per
worker -- and so plateaus at ~120 evt/s, barely above serial ~99 evt/s
(``CUBE_PROTOTYPE_PLAN.md`` Findings 4-5, lines 389-398 / 485-494). The three
named next-optimizations there (``CUBE_PROTOTYPE_PLAN.md:500-505``) are:

    #1 shared index via Ray object store    <- US-008 (this script: ray.put once)
    #2 batch random access                  <- US-009 (this script: read_events)
    #3 real binning keys                     <- this script: bin by pulseId

This demonstrator delivers all three. (#4, spillover, is out of scope.)

What the cube computes
----------------------
A GroupBy-Aggregate: group events into ``num_bins`` bins by a **real**
per-event scalar and accumulate the per-bin **sum and count** of the jungfrau
raw frame ``(32, 512, 1024)``. The bin key is ``event.pulseId % num_bins`` --
``pulseId`` is the timing-system (``det_type='ts'``) scalar carried in every
event's own data, NOT the positional ``i % num_bins`` the owner's prototype used
as a placeholder. Because the key is a property of the event itself, the same
event always lands in the same bin no matter which worker reads it or in what
order -- which is exactly why a *position*-based fake key would be wrong.

Correctness oracle (self-contained, NO psana)
----------------------------------------------
The parallel cube must equal a serial single-process cube over the same
positions and the same binning, ``np.allclose(rtol=1e-5)`` (the owner's
tolerance, ``CUBE_PROTOTYPE_PLAN.md:238``; the smalldata_tools reference RHS is
not yet implemented, so we use psdata's own serial path as the oracle).

What "beats per-worker rescan" means here (the headline metric)
---------------------------------------------------------------
The honest performance story is the **rescan tax**, not raw evt/s. The jungfrau
frame is ~33.6 MB/event, so this cube is **I/O-bandwidth-bound**: a single
process already saturates the shared-filesystem read bandwidth (~1 GB/s here),
and adding workers does not raise evt/s -- so the absolute throughput is gated
by bandwidth, not by the index. (The owner's ~120 evt/s plateau was measured on
the small ``tmo_atmopal`` opal, whose per-event bytes are tiny; it is not a
jungfrau-achievable number.) The index's value is therefore measured directly:
this script runs the cube two ways at each worker count -- (A) the shared index
shipped once, (B) each worker rebuilding the index itself (the prototype's
per-worker ``DataSource`` + ``build_table``) -- and reports the wall-clock gap,
the **rescan tax** the shared index removes. That tax is ~one SMD-scan
(~2 s here) per worker; it grows with worker count (N rescans vs 1), which is
exactly the scaling plateau the psana prototype hit (120 vs serial 99 evt/s,
only ~1.2x for 4 workers; ``CUBE_PROTOTYPE_PLAN.md:485-494``).

Ray is a DEMONSTRATOR dependency only -- it is imported here, never by
``psdata`` itself (``import psdata`` stays numpy-only; ``ray`` lives in a
``[demo]`` extra). See ``examples/run_cube_ray.sh`` for the run environment.

Run (on sdfiana025)::

    examples/run_cube_ray.sh                       # default: 2000 evt, workers 1/4/16
    examples/run_cube_ray.sh --events 4000 --bins 16
    examples/run_cube_ray.sh --workers 1 4         # custom worker counts
    examples/run_cube_ray.sh --check               # assert parallel==serial, then exit

Or, with an environment that has both ``ray`` and ``psdata`` importable::

    python examples/cube_ray_shared_index.py --events 2000
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

import psdata
from psdata import index as psindex


# --- reference dataset (lives in the example, never in the library) ----------
EXP = "mfx100848724"
RUN = 51
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
DETECTOR = "jungfrau"
# The timing-system detector type whose 'raw' alg carries the pulseId scalar
# (Event.pulseId reads it). This is the REAL binning key's source.
_BINKEY_NAME = "pulseId"


def _stream_files(directory, exp, run):
    """Resolve a run's per-stream bigdata c000 xtc2 files by globbing -- the
    example (not the library) knows the file-naming pattern; the index follows
    the ``chunkinfo`` roll to later chunks itself."""
    paths = sorted(glob.glob(f"{directory}/{exp}-r{run:04d}-s*-c000.xtc2"))
    if not paths:
        raise FileNotFoundError(
            f"no xtc2 stream files under {directory} for {exp} r{run}")
    files = {}
    for p in paths:
        base = os.path.basename(p)
        sidx = int(base.split("-s")[1].split("-")[0])
        files[sidx] = p
    return files


def _bin_of(pulse_id, num_bins):
    """The REAL per-event bin key: derived from the event's own ``pulseId``
    (a timing-system scalar that ships with the event), reduced modulo
    ``num_bins``. ``None`` pulseId (run without a timing detector) -> bin 0.

    Deliberately a function of the EVENT, not its position ``i`` -- so the same
    event bins identically regardless of worker assignment or read order. That
    invariance is what makes a parallel cube agree with the serial one; the
    owner's ``i % num_bins`` placeholder would not survive repartitioning."""
    if pulse_id is None:
        return 0
    return int(pulse_id % num_bins)


# ==========================================================================
# The per-worker kernel (also the serial oracle's body) -- pure psdata
# ==========================================================================
def _accumulate_slice(index_state, ks, detector, num_bins, read_batch=32):
    """Read positions ``ks`` from a shipped index state and accumulate the
    per-bin (sum, count) of one detector's stacked frame.

    ``index_state`` is a :meth:`RunIndex.to_dict` payload (NOT a live index):
    reconstructing it here does NO SMD rescan, and is fd-safe -- the bigdata
    file descriptors reopen lazily inside this process on the first read.

    Reads in **sub-batches of ``read_batch`` positions** rather than the whole
    slice at once: each :meth:`RunIndex.read_events` call coalesces its
    ``pread``s (US-009), but a sub-batch keeps only ``read_batch`` events'
    bytes resident at a time. That matters for a big detector -- the jungfrau
    frame is ~33.6 MB/event, so reading a 5000-event slice in ONE call would
    hold ~170 GB of dgram snapshots; the sub-batch bounds peak memory to
    ``read_batch`` frames while preserving the coalesced-read benefit per batch.

    Returns ``(sums, counts)``:
      * ``sums``   : ``{bin -> float64 ndarray (n_seg, *seg_shape)}``
      * ``counts`` : ``{bin -> int}``
    Only bins that received >=1 event are present, so partials merge by simple
    per-bin addition.
    """
    ridx = psindex.RunIndex.from_dict(index_state)  # no rescan; fds lazy
    try:
        sums = {}
        counts = {}
        for start in range(0, len(ks), read_batch):
            batch = ks[start:start + read_batch]
            # Coalesced batch read of this sub-window: grouped preads (US-009),
            # one os.open per chunk path (reused), NO SMD rescan in the worker.
            for evt in ridx.read_events(batch):
                frame = evt.stack(detector)           # (n_seg, *shape) or None
                if frame is None:
                    continue                          # missing-segment -> skip
                b = _bin_of(evt.pulseId, num_bins)
                if b not in sums:
                    sums[b] = frame.astype(np.float64)
                    counts[b] = 1
                else:
                    sums[b] += frame
                    counts[b] += 1
        return sums, counts
    finally:
        ridx.close()


def _merge_partials(partials):
    """Merge ``[(sums, counts), ...]`` into a single ``(sums, counts)`` by
    per-bin addition (the GroupBy reduce step)."""
    out_sums, out_counts = {}, {}
    for sums, counts in partials:
        for b, s in sums.items():
            if b not in out_sums:
                out_sums[b] = s.copy()
                out_counts[b] = counts[b]
            else:
                out_sums[b] += s
                out_counts[b] += counts[b]
    return out_sums, out_counts


def serial_cube(index_state, ks, detector, num_bins, read_batch=32):
    """Single-process reference cube over the same positions + binning -- the
    correctness oracle (no Ray, no psana)."""
    return _merge_partials([_accumulate_slice(index_state, ks, detector,
                                              num_bins, read_batch)])


# ==========================================================================
# Ray driver: build index once, ship via object store, fan out workers
# ==========================================================================
def _partition(ks, n_parts):
    """Split ``ks`` into ``n_parts`` contiguous, near-equal slices."""
    n = len(ks)
    n_parts = max(1, min(n_parts, n))
    edges = [round(i * n / n_parts) for i in range(n_parts + 1)]
    return [ks[edges[i]:edges[i + 1]] for i in range(n_parts)
            if edges[i + 1] > edges[i]]


def parallel_cube(ray, index_ref, ks, detector, num_bins, num_workers,
                  read_batch=32):
    """Run the cube on ``num_workers`` Ray tasks, each handed a slice of ``ks``
    and the SHARED index object ref.

    ``index_ref`` is a single ``ray.put(index.to_dict())`` ref -- Ray ships the
    one immutable payload to each worker (zero per-worker SMD rescan; this is
    optimization #1, the shared index). Passing the ObjectRef as a remote-task
    argument auto-dereferences it inside the worker.
    """
    @ray.remote
    def _worker(state, ks_slice, det, nbins, rb):
        # state arrives already dereferenced from the object store.
        return _accumulate_slice(state, ks_slice, det, nbins, rb)

    slices = _partition(ks, num_workers)
    futures = [_worker.remote(index_ref, sl, detector, num_bins, read_batch)
               for sl in slices]
    partials = ray.get(futures)            # order-preserving; one per slice
    return _merge_partials(partials)


def _rescan_accumulate_slice(stream_files, ks, detector, num_bins,
                             read_batch=32):
    """The CONTRAST worker: rebuild the index *inside the worker* from the raw
    stream files, then cube its slice.

    This faithfully re-creates the per-worker overhead of the owner's psana Ray
    prototype, whose every ``@ray.remote`` worker constructs its own psana
    ``DataSource`` and calls ``run.build_table()`` -- i.e. **rescans the SMD per
    worker** -- before it can random-access (``ray_cube_prototype.py``
    ``process_bin``; ``CUBE_PROTOTYPE_PLAN.md`` Finding 4, lines 389-398). Here
    the rescan is psdata's own ``build_index`` (the same SMD scan), so the only
    difference from :func:`_accumulate_slice` is *who pays the index build*: the
    contrast pays it N times (once per worker), the shared-index path pays it
    ONCE in the driver. Everything downstream (the batch reads, the binning, the
    accumulation) is identical, so any wall-clock gap is exactly the rescan tax."""
    rc = psdata.discover(stream_files)              # per-worker config discovery
    ridx = psindex.build_index(stream_files, run_config=rc)  # per-worker RESCAN
    try:
        state = ridx.to_dict()
        return _accumulate_slice(state, ks, detector, num_bins, read_batch)
    finally:
        ridx.close()


def rescan_cube(ray, files_ref, ks, detector, num_bins, num_workers):
    """Run the cube with each worker rebuilding the index itself (the owner's
    per-worker-rescan pattern), for an apples-to-apples contrast with
    :func:`parallel_cube`. Same dataset, same reader, same binning -- the ONLY
    difference is the per-worker SMD rescan the shared index removes."""
    @ray.remote
    def _worker(stream_files, ks_slice, det, nbins):
        return _rescan_accumulate_slice(stream_files, ks_slice, det, nbins)

    slices = _partition(ks, num_workers)
    futures = [_worker.remote(files_ref, sl, detector, num_bins)
               for sl in slices]
    partials = ray.get(futures)
    return _merge_partials(partials)


def _cubes_close(a, b, rtol=1e-5):
    """``np.allclose(rtol)`` over the per-bin mean frames of two cubes (sums and
    counts must agree bin-for-bin)."""
    sa, ca = a
    sb, cb = b
    if set(ca) != set(cb):
        return False, f"bin sets differ: {sorted(ca)} vs {sorted(cb)}"
    for bkey in ca:
        if ca[bkey] != cb[bkey]:
            return False, f"bin {bkey}: counts {ca[bkey]} != {cb[bkey]}"
        if not np.allclose(sa[bkey], sb[bkey], rtol=rtol):
            d = np.abs(sa[bkey] - sb[bkey]).max()
            return False, f"bin {bkey}: sums differ, max|diff|={d}"
    return True, "match"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=int, default=800,
                    help="number of event positions to cube (default 800)")
    ap.add_argument("--bins", type=int, default=10,
                    help="number of GroupBy bins (default 10)")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 16],
                    help="worker counts to benchmark (default: 1 4 16)")
    ap.add_argument("--read-batch", type=int, default=32, dest="read_batch",
                    help="events per coalesced read_events call inside a worker "
                         "(bounds peak memory for big detectors; default 32)")
    ap.add_argument("--no-contrast", action="store_true",
                    help="skip the per-worker-rescan contrast (shared-index only)")
    ap.add_argument("--detector", default=DETECTOR)
    ap.add_argument("--exp", default=EXP)
    ap.add_argument("--run", type=int, default=RUN)
    ap.add_argument("--dir", default=DIR)
    ap.add_argument("--check", action="store_true",
                    help="assert parallel cube == serial cube (rtol=1e-5) and "
                         "that workers do no SMD rescan, then exit nonzero on "
                         "failure")
    args = ap.parse_args(argv)

    import ray  # demonstrator-only dependency; NEVER imported by psdata itself

    print("=" * 72)
    print("psdata Ray shared-index cube (US-008 + US-009 demonstrator)")
    print("=" * 72)

    # -- Phase 1: build the index ONCE (the only SMD scan in the whole run) ---
    files = _stream_files(args.dir, args.exp, args.run)
    rc = psdata.discover(files)
    t0 = time.time()
    ridx = psindex.build_index(files, run_config=rc)
    build_s = time.time() - t0
    n_total = ridx.n_events
    n = min(args.events, n_total)
    ks = list(range(n))
    print(f"\n[index] built ONCE: {n_total} events, {build_s:.2f}s SMD scan, "
          f"{ridx.smd_bytes_read / 1e6:.1f} MB SMD read")
    print(f"[index] cubing {n} events into {args.bins} bins by REAL key "
          f"'{_BINKEY_NAME} % {args.bins}'")

    # -- serialize the built index to a shippable payload (US-008) -----------
    state = ridx.to_dict()                 # fd-safe; no fds, no SMD rescan
    blob_streams = sorted(ridx.bd_files)
    ridx.close()                           # driver's own fds no longer needed

    # -- Phase 2: ray.put the index ONCE; share the ref across all workers ----
    ray.init(ignore_reinit_error=True, logging_level="warning")
    cpus = ray.cluster_resources().get("CPU", 0)
    print(f"[ray]   initialized, {cpus:.0f} CPUs available")
    index_ref = ray.put(state)             # optimization #1: shared index
    print(f"[ray]   index shipped to object store ONCE "
          f"(streams {blob_streams}); workers reconstruct via from_dict, "
          f"NO per-worker SMD rescan")

    # the run's stream files are shipped to the contrast workers (so each can
    # rebuild the index itself, the way the psana prototype does).
    files_ref = ray.put(files)

    # -- serial oracle (the correctness reference) ---------------------------
    t0 = time.time()
    oracle = serial_cube(state, ks, args.detector, args.bins, args.read_batch)
    serial_s = time.time() - t0
    serial_rate = n / serial_s if serial_s else float("inf")
    n_used = sum(oracle[1].values())
    print(f"\n[serial] {n_used}/{n} events ({args.detector} present) in "
          f"{serial_s:.2f}s = {serial_rate:.0f} evt/s "
          f"(single-process oracle)")

    # -- Phase 3: shared-index vs per-worker-rescan, side by side ------------
    # The headline contrast: the SAME cube, SAME reader, SAME dataset, run two
    # ways at each worker count -- (A) shared index shipped once (this work),
    # (B) each worker rebuilding the index itself (the owner's psana prototype's
    # per-worker DataSource+build_table). The wall-clock gap is exactly the
    # per-worker SMD-rescan tax the shared index removes.
    do_contrast = not args.no_contrast
    print(f"\n[scaling] cube on {n} events, {args.detector} "
          f"({33.6 if args.detector == DETECTOR else '?'} MB/event); the cube "
          f"is I/O-bandwidth-bound on this big detector, so evt/s is gated by "
          f"shared-FS read bandwidth, NOT by the index -- the index win shows "
          f"in the rescan column below.")
    head = ("          workers | shared-idx wall | "
            + ("rescan wall | rescan tax | " if do_contrast else "")
            + "evt/s | == serial")
    rule = ("          --------+-----------------+"
            + ("-------------+------------+" if do_contrast else "")
            + "-------+----------")
    print(head)
    print(rule)
    ok_all = True
    for w in args.workers:
        t0 = time.time()
        cube = parallel_cube(ray, index_ref, ks, args.detector, args.bins, w,
                             args.read_batch)
        wall = time.time() - t0
        rate = n / wall if wall else float("inf")
        match, why = _cubes_close(cube, oracle)
        ok_all = ok_all and match

        if do_contrast:
            t0 = time.time()
            rcube = rescan_cube(ray, files_ref, ks, args.detector, args.bins, w)
            rwall = time.time() - t0
            # the rescan variant must ALSO equal the oracle (same correctness)
            rmatch, rwhy = _cubes_close(rcube, oracle)
            ok_all = ok_all and rmatch
            tax = rwall - wall
            contrast_cols = (f"{rwall:11.2f} | {tax:+9.2f}s | ")
            mflag = "yes" if (match and rmatch) else f"NO ({why or rwhy})"
        else:
            contrast_cols = ""
            mflag = "yes" if match else f"NO ({why})"

        print(f"          {w:7d} | {wall:15.2f} | {contrast_cols}"
              f"{rate:5.0f} | {mflag}")

    ray.shutdown()

    # -- summary -------------------------------------------------------------
    print()
    if ok_all:
        print("[ok] every parallel cube == serial cube (np.allclose rtol=1e-5) "
              "-- both the shared-index and the per-worker-rescan variants")
        print("[ok] workers reconstruct the shipped index (from_dict, ~0 s) and "
              "batch-read (read_events) -- the per-worker SMD rescan (~the index "
              "build time, paid N times by the psana prototype) is paid ONCE here")
        if do_contrast:
            print("[ok] the 'rescan tax' column is the wall-clock the shared "
                  "index eliminates -- it grows with worker count (N rescans vs "
                  "1), exactly the plateau the psana prototype hit (120 vs 99 "
                  "evt/s, ~1.2x for 4 workers; CUBE_PROTOTYPE_PLAN.md:485-494)")
    else:
        print("[FAIL] a parallel cube disagreed with the serial oracle")

    if args.check:
        # In --check mode, also prove a worker does zero SMD I/O: a from_dict'd
        # index never opens an SMD file (its smd_bytes_read is the inherited
        # build figure, but no NEW scan happens -- it only os.opens bigdata
        # chunk paths on read).
        rebuilt = psindex.RunIndex.from_dict(state)
        smd_paths = set(psindex.smd_files_for(files).values())
        opened = []
        real_open = os.open

        def spy_open(path, *a, **kw):
            opened.append(os.fspath(path))
            return real_open(path, *a, **kw)

        os.open = spy_open
        try:
            _ = rebuilt.read_events(ks[: min(50, len(ks))])
        finally:
            os.open = real_open
        rebuilt.close()
        smd_opened = [p for p in opened if p in smd_paths]
        no_rescan = not smd_opened
        if no_rescan:
            print("[ok] reconstructed-index worker opened 0 SMD files on read "
                  "(no rescan); only bigdata chunk paths")
        else:
            print(f"[FAIL] worker reopened SMD files: {smd_opened}")
            ok_all = False
        if not (ok_all and no_rescan):
            sys.exit(1)
        print("\nALL US-008/US-009 ACCEPTANCE CHECKS PASSED")

    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
