#!/usr/bin/env python3
"""Capture the psana ground truth for the MULTI-CHUNK oracle (US-004 / HYG-03).

``tests/test_robust_us004.py::test_chunk_roll_vs_psana`` is the only check that
proves psdata reads events **byte-identically to psana across a chunk-file roll**
(``c000 -> c001``, followed via ``chunkinfo`` on Enable).  It consumes a ground
truth this script produces:

    <out-dir>/gt_mc35_manifest.json     -- {"events": [[i, ts, pulseId, shape, dtype], ...]}
    <out-dir>/gt_mc35_<i>.npy           -- psana's det.raw.raw(evt) for event i

Without them the test SKIPs -- and that skip is NOT in ``tests/skips_allowed.txt``,
so ``run_tests.sh`` fails the suite (HYG-03: an oracle that never runs is not a
passing oracle).  Historically the ground truth was absent, the skip was scored
as a pass, and the reader's silent-data-loss bug sat on exactly this path.

THIS SCRIPT IS THE ORACLE.  It is one of the two files in the repo allowed to
import psana (``tests/_env_oracle.py`` is the other); psdata itself never does.

How it picks the events
-----------------------
1. Builds psdata's SMD-only ``RunIndex`` for the run (numpy only, no psana) and
   asks it where each multi-chunk stream ROLLS -- the first event position whose
   entry for that stream names a chunk file other than the stream's first
   (for mfx101343025 r35 this is the s007 boundary the test's docstring names).
2. Targets a window of event positions on BOTH sides of every roll
   (``--window`` before and after, deduplicated).  Those positions' timestamps
   are the capture set -- so every captured event is guaranteed to be in the
   index the test looks it up in (``ridx.read_event(ts)``), and the set spans
   both chunk files.
3. Iterates ``psana``'s ``run.events()`` IN ORDER (psana's env/config state is
   populated online -- never random-access it) and captures each event whose
   timestamp is in the target set: ``det.raw.raw(evt)`` and the timing
   detector's ``pulseId(evt)``.

The index is only used to CHOOSE which events to capture; the recorded values
are 100% psana's, and the test looks them up by TIMESTAMP.  So this is not a
circular oracle: if psdata's index had the wrong offsets, the arrays would still
be psana's and the comparison would fail, which is the point.

Invocation (on sdfiana025, from the psdata repo root)
----------------------------------------------------
    source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
    cd <psdata repo root>
    PYTHONPATH=src python3 tools/gt_capture.py --out-dir .

``--out-dir .`` (the repo root) is what the test expects by default: it looks for
``$PWD/gt_mc35_manifest.json``, and ``run_tests.sh`` is run from the repo root.
To keep the capture elsewhere, write it there and point the test at it::

    PYTHONPATH=src python3 tools/gt_capture.py --out-dir /sdf/data/lcls/ds/prj/public01/scratch/gt
    PSDATA_GT_MANIFEST=/sdf/.../gt/gt_mc35_manifest.json bash run_tests.sh

Cost: one full psana forward pass up to the roll, and ~32 MiB per captured
jungfrau event on disk (32 segs x 512 x 1024 uint16).  ``--window 3`` (the
default) captures 6 events per rolled stream; raise it for more margin, lower it
if disk is tight.  ``--dry-run`` prints the plan (roll positions, target events,
estimated bytes) WITHOUT importing psana or writing anything.

Exit status: 0 = every targeted event captured; 3 = the run was scanned but psana
never yielded some targeted timestamp (the manifest still holds what WAS
captured -- and a target timestamp that psdata's index has but psana's event
stream never produces is itself a finding worth chasing, so it is not silently
swallowed); nonzero otherwise.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# --- reference run: the multi-chunk run test_robust_us004.py cross-checks ----
MC_EXP = "mfx101343025"
MC_RUN = 35
# The instrument dir has been seen spelled both ways; take whichever exists.
MC_DIR_CANDIDATES = (
    "/sdf/data/lcls/ds/mfx/mfx101343025/xtc",
    "/sdf/data/lcls/ds/MFX/mfx101343025/xtc",
)
MC_DET = "jungfrau"
# File naming the test hard-codes: gt_mc35_manifest.json / gt_mc35_<i>.npy
GT_PREFIX = "gt_mc35"


def _default_dir():
    for d in MC_DIR_CANDIDATES:
        if os.path.isdir(d):
            return d
    return MC_DIR_CANDIDATES[0]


# ---------------------------------------------------------------------------
# psdata side (numpy only): build the index, find the roll, pick the events
# ---------------------------------------------------------------------------
def _stream_files(directory, exp, run):
    """The run's per-stream c000 bigdata files keyed by REAL stream index --
    identical to test_robust_us004._mc_stream_files (so the index this script
    reasons about is the index the test builds)."""
    import glob
    import psdata
    paths = sorted(glob.glob(os.path.join(
        directory, "%s-r%04d-s*-c000.xtc2" % (exp, run))))
    if not paths:
        raise SystemExit(
            "FATAL: no bigdata stream files for %s r%d under %s"
            % (exp, run, directory))
    paths = psdata.filter_c000(paths)
    return {psdata.stream_index_of(p): p for p in paths}


def _build_index(directory, exp, run):
    import psdata
    from psdata import index as psindex
    files = _stream_files(directory, exp, run)
    rc = psdata.discover(files)
    smd = psindex.smd_files_for(files)
    ridx = psindex.RunIndex.build(smd, rc)
    return rc, ridx


def _roll_positions(ridx):
    """{stream: k} -- the first event position at which each multi-chunk stream
    reads from a chunk file other than its first (i.e. where it rolled)."""
    rolls = {}
    for stream in sorted(ridx.multichunk_streams):
        first_chunk = ridx.chunk_files[stream][0]
        for k, entry in enumerate(ridx.entries):
            if stream not in entry:
                continue
            if entry[stream][0] != first_chunk:
                rolls[stream] = k
                break
    return rolls


def _target_positions(ridx, rolls, window):
    """Event positions on BOTH sides of every roll: [k-window, k+window)."""
    n = ridx.n_events
    ks = set()
    for k in rolls.values():
        for j in range(k - window, k + window):
            if 0 <= j < n:
                ks.add(j)
    return sorted(ks)


def _timing_det_name(rc, override):
    """The detector psana exposes pulseId on (det_type 'ts'); psdata and psana
    take the name from the same Names table, so this resolves for both."""
    if override:
        return override
    for name in rc.detector_names():
        if getattr(rc.detector(name), "det_type", None) == "ts":
            return name
    return "timing"


# ---------------------------------------------------------------------------
# psana side: the ORACLE.  Forward pass, capture the targeted timestamps.
# ---------------------------------------------------------------------------
def _psana_capture(args, targets_ts, det_name, timing_name):
    """Iterate psana in order; return {ts: (raw_or_None, pulseId)}."""
    from psana import DataSource

    ds = DataSource(exp=args.exp, run=args.run, dir=args.dir)
    prun = next(ds.runs())
    det = prun.Detector(det_name)
    try:
        timing = prun.Detector(timing_name)
    except Exception as e:
        raise SystemExit(
            "FATAL: psana has no detector %r for pulseId (%s).\n"
            "       detnames psana reports: %r\n"
            "       Pass --timing-det <name>."
            % (timing_name, e, sorted(getattr(prun, "detnames", []) or [])))

    want = set(targets_ts)
    got = {}
    # A safety bound: the last targeted event lives at a known index, so a scan
    # far past it means psana's event stream and psdata's index have diverged.
    limit = max(args.max_scan, 0) or None
    t0 = time.monotonic()
    for k, evt in enumerate(prun.events()):
        if limit is not None and k > limit:
            print("  [warn] stopped scanning at psana event %d (--max-scan)" % k)
            break
        ts = evt.timestamp
        ts = int(ts() if callable(ts) else ts)
        if ts in want:
            raw = det.raw.raw(evt)
            raw = None if raw is None else np.asarray(raw)
            pid = timing.raw.pulseId(evt)
            pid = int(pid() if callable(pid) else pid)
            got[ts] = (raw, pid)
            want.discard(ts)
            print("  [capture] psana event %d ts=%d pulseId=%d raw=%s"
                  % (k, ts, pid,
                     "None (missing segment)" if raw is None
                     else "%s %s" % (raw.shape, raw.dtype)))
            if not want:
                print("  [capture] all %d targets captured after %d psana events "
                      "(%.1fs)" % (len(got), k + 1, time.monotonic() - t0))
                break
        elif k % 500 == 0 and k:
            print("  ... psana event %d (%d/%d captured, %.1fs)"
                  % (k, len(got), len(targets_ts), time.monotonic() - t0))
    return got, sorted(want)


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Capture the multi-chunk psana ground truth "
                    "(gt_mc35_*.npy + gt_mc35_manifest.json) for "
                    "tests/test_robust_us004.py::test_chunk_roll_vs_psana.")
    p.add_argument("--out-dir", default=os.getcwd(),
                   help="where to write the manifest + .npy arrays "
                        "(default: cwd -- which is where the test looks)")
    p.add_argument("--exp", default=MC_EXP)
    p.add_argument("--run", type=int, default=MC_RUN)
    p.add_argument("--dir", default=None,
                   help="xtc dir (default: the first of %s that exists)"
                        % (MC_DIR_CANDIDATES,))
    p.add_argument("--det", default=MC_DET,
                   help="detector to capture raw for (default: %s)" % MC_DET)
    p.add_argument("--timing-det", default=None,
                   help="detector carrying pulseId (default: the det_type='ts' "
                        "one, else 'timing')")
    p.add_argument("--window", type=int, default=3,
                   help="events to capture on EACH side of every chunk roll "
                        "(default: 3 -> 6 events per rolled stream)")
    p.add_argument("--max-scan", type=int, default=0,
                   help="give up after this many psana events (0 = no bound)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan (rolls, target events, bytes) and exit; "
                        "does not import psana and writes nothing")
    args = p.parse_args(argv)

    if args.dir is None:
        args.dir = _default_dir()
    if args.window < 1:
        p.error("--window must be >= 1 (need events on both sides of the roll)")

    print("=" * 72)
    print("gt_capture: multi-chunk psana ground truth")
    print("  exp=%s run=%d dir=%s det=%s" % (args.exp, args.run, args.dir,
                                             args.det))
    print("=" * 72)

    # ---- psdata (numpy only): where does the run roll? --------------------
    rc, ridx = _build_index(args.dir, args.exp, args.run)
    if args.det not in rc.detector_names():
        raise SystemExit("FATAL: detector %r not in this run; detectors: %r"
                         % (args.det, sorted(rc.detector_names())))
    timing_name = _timing_det_name(rc, args.timing_det)
    print("[index] %d events, streams %s, multichunk streams %s"
          % (ridx.n_events, sorted(ridx.chunk_files),
             sorted(ridx.multichunk_streams)))
    for s, chunks in sorted(ridx.chunk_files.items()):
        if len(chunks) > 1:
            print("[index]   stream %d rolls through %s"
                  % (s, [os.path.basename(c) for c in chunks]))

    rolls = _roll_positions(ridx)
    if not rolls:
        ridx.close()
        raise SystemExit(
            "FATAL: no stream in %s r%d rolls to a second chunk -- this run "
            "cannot produce a multi-chunk ground truth.  (test_robust_us004 "
            "expects mfx101343025 r35, whose s007 stream rolls.)"
            % (args.exp, args.run))
    for s, k in sorted(rolls.items()):
        print("[roll]  stream %d rolls at event k=%d" % (s, k))

    ks = _target_positions(ridx, rolls, args.window)
    targets = [(k, int(ridx.timestamps[k])) for k in ks]
    print("[plan]  capturing %d events around the roll(s): %s"
          % (len(ks), ks))
    print("[plan]  timing detector for pulseId: %r" % (timing_name,))

    if args.dry_run:
        # estimate the on-disk cost from the indexed dgram sizes (no psana)
        est = sum(sum(sz for (_p, _o, sz) in ridx.entries[k].values())
                  for k in ks)
        print("[plan]  DRY RUN -- nothing captured.  ~%.1f MiB of raw dgrams "
              "back these events (the .npy arrays are of that order)."
              % (est / (1 << 20)))
        ridx.close()
        return 0

    # ---- psana: the oracle ------------------------------------------------
    ts_to_k = {ts: k for k, ts in targets}
    captured, missing = _psana_capture(args, list(ts_to_k), args.det,
                                       timing_name)
    ridx.close()

    # ---- write the ground truth ------------------------------------------
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    events = []
    n_bytes = 0
    for ts in sorted(captured, key=lambda t: ts_to_k[t]):
        k = ts_to_k[ts]
        raw, pid = captured[ts]
        path = os.path.join(out_dir, "%s_%d.npy" % (GT_PREFIX, k))
        if raw is None:
            # psana saw no complete detector for this event: record it as an
            # object-dtype None, which is what the test's
            # `gt.dtype == object -> expect psdata None` branch reads.
            arr = np.array(None, dtype=object)
            shape, dtype = None, None
        else:
            arr = raw
            shape, dtype = list(raw.shape), str(raw.dtype)
        np.save(path, arr, allow_pickle=True)
        n_bytes += os.path.getsize(path)
        events.append([k, int(ts), int(pid), shape, dtype])
        print("[write] %s  (i=%d ts=%d pulseId=%d shape=%s dtype=%s)"
              % (os.path.basename(path), k, ts, pid, shape, dtype))

    manifest = {
        "exp": args.exp,
        "run": args.run,
        "dir": args.dir,
        "det": args.det,
        "timing_det": timing_name,
        "window": args.window,
        "roll_positions": {str(s): k for s, k in sorted(rolls.items())},
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generator": "tools/gt_capture.py",
        # what test_robust_us004.py consumes: [i, ts, pulseId, shape, dtype],
        # with the array for i at <manifest dir>/gt_mc35_<i>.npy
        "events": events,
    }
    man_path = os.path.join(out_dir, "%s_manifest.json" % GT_PREFIX)
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("[write] %s  (%d events, %.1f MiB of arrays)"
          % (man_path, len(events), n_bytes / (1 << 20)))

    if missing:
        print("\n[WARN] psana never yielded %d targeted timestamp(s) that "
              "psdata's SMD index HAS: %s" % (len(missing), missing))
        print("[WARN] the manifest holds the %d events that WERE captured; the "
              "test will still run against those.  A timestamp in the index "
              "that psana's event stream does not produce is a real "
              "discrepancy -- chase it, do not paper over it." % len(events))
        return 3
    if not events:
        print("\nFAILED: captured nothing.")
        return 1
    print("\nOK: ground truth for %d events around the chunk roll(s) written to "
          "%s" % (len(events), out_dir))
    print("Run the oracle with:  bash run_tests.sh tests/test_robust_us004.py")
    print("(from %s, or set PSDATA_GT_MANIFEST=%s)" % (out_dir, man_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
