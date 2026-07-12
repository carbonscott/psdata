#!/usr/bin/env python3
"""bench_workers.py -- psdata MULTI-WORKER READ-SCALING benchmark (PAR-03 / PERF-01).

Sibling of tests/bench_index.py.  Where bench_index answers "how many frames does
psana read that psdata skips" (the A headline, single process), this answers the
ONE question bench_index cannot: **does psdata's read throughput SCALE with
parallel workers, or does it collapse?**

  * AGGREGATE evt/s and AGGREGATE GB/s at WORKERS in {1, 4, 8, 16}, >= 3 reps per
    point, every point printed WITH its spread (stdev, min, max).  No point is
    ever printed without a spread.
  * The workers partition a contiguous event window DISJOINTLY and each reads its
    own slice through the SHARED, PERSISTED random-access index (built ONCE in the
    parent, ``RunIndex.save``d, then ``RunIndex.load``ed per worker -- zero SMD
    rescan per worker; the same shared-index pattern as
    examples/cube_ray_shared_index.py, but with plain multiprocessing instead of
    Ray so this measures psdata I/O, NOT Ray task overhead).
  * The SMOKING GUN (the reason this harness exists): per worker we read
    ``/proc/self/io`` around the timed region and report ``rchar`` (bytes read via
    read/pread syscalls) and ``syscr`` (the READ-SYSCALL COUNT), plus ``read_bytes``
    (block layer; usually 0 on a network FS like Weka -- reported, never relied on).

The hypothesis under test
-------------------------
PERF-01 (PR #23) replaced psdata's one-``pread``-per-(event, stream) batch read
with a REAL coalescing merge (40 preads -> 5 for 8 events x 5 streams).  PAR-03
blamed that unbatched pattern for a scaling collapse: 16 workers delivered
0.30 GB/s against a measured 5.54 GB/s node ceiling.  So this harness MUST be able
to see the syscall pattern, and it must run UNCHANGED against BOTH library ends:

    BEFORE = 1d018423b20509cb9296df1a5d166cc3153910a6   (read_events: 1 pread per
                                                          (event, stream) pair)
    AFTER  = 15caa3a1695aabc5d148a2efda4ec8aff9fbd582   (read_events: coalesced)

selected purely by PYTHONPATH prepend (see tests/bench_workers.sbatch).  It
therefore uses ONLY psdata API that exists at BOTH SHAs -- notably
``read_events(ks)`` called POSITIONALLY (the ``max_bytes=`` keyword is AFTER-only)
and never ``iter_events`` / ``iter_stack`` / ``read_stack(max_bytes=)`` /
``DEFAULT_READ_BATCH`` / ``DEFAULT_READ_MEM_LIMIT`` (all AFTER-only).  See the
API-COMPATIBILITY block below.

The predicted signature, at 8 workers x a 32-event sub-batch x 5 streams:
    BEFORE  syscr ~ 32 x 5 = 160 preads per sub-batch, each ~6.7 MB
    AFTER   syscr ~ a handful (one per contiguous run per stream, split only by
            the 128 MiB span cap), each ~100+ MB
so ``syscr_per_evt`` and ``MB_per_read`` are the discriminator -- and if the AFTER
end lifts aggregate GB/s toward the ceiling while BEFORE stays pinned, PAR-03's
diagnosis is confirmed on the same harness, same node, same bytes.

The node ceiling is MEASURED here too (not quoted): the same worker counts, the
same bigdata files, but a dumb 8 MiB-block sequential ``os.pread`` loop over
disjoint byte ranges.  Each ceiling slot draws an EQUAL SHARE from EVERY chunk
file (see ceiling_segments), so the ceiling exercises the same interleaved
5-file set psdata does -- a ceiling that read only stream 0 would be a
1-file sequential number graded against psdata's 5-file interleaved one.  That is
the bandwidth psdata is being graded against.

Correctness / discipline
------------------------
  * EXCLUSIVE BATCH NODE ONLY.  ``require_batch_node()`` is copied from
    tests/bench_index.py, with the ``BENCH_ALLOW_NONBATCH`` escape hatch REMOVED
    (the campaign forbids login-node numbers and forbids the override): setting
    that variable is itself a hard refusal, so nobody can smuggle a shared-node
    number into the campaign.  Exclusivity is additionally attested from
    ``scontrol`` / the Slurm CPU+memory allocation.
  * COLD -- and PROVEN cold, not asserted.  Every (worker-count, rep) cell reads a
    DISJOINT contiguous window of events -- no cell ever re-reads a byte another
    cell read -- and every chunk file is ``posix_fadvise(DONTNEED)``-evicted before
    each cell (the same cold discipline as bench_index).  A hard check refuses a
    configuration whose cells would overlap.  Because DONTNEED is only advice and,
    on Weka, reaches only the kernel page cache, the FIRST eviction is preceded by
    a deliberate prime of a KNOWN number of bytes (read from the TAIL of each chunk
    file, a region neither phase ever times) and the resulting drop in
    /proc/meminfo ``Cached`` is printed as a greppable ``COLDPROOF`` line.  A drop
    smaller than half the primed bytes is a loud WARNING (not a refusal): the
    numbers still stand, they are simply WARM numbers.
  * BALANCED CELL ORDER.  The worker order is ROTATED per rep (Latin-square style),
    so no worker count is pinned to a fixed position in the run and a
    position-in-run effect (warm-up, drift) cannot land entirely on the worker-count
    axis -- i.e. straight into speedup/efficiency.  The windows stay disjoint.
  * MEMORY BOUND (modelled on MEM-01 / bench_index's SUBBATCH guardrail).  Reads
    walk each worker's slice in <= SUBBATCH (32) event sub-batches, read-and-
    discard.  The bound is stated, PREDICTED before the clock starts (refuse if it
    exceeds the node budget), and then VERIFIED against each worker's measured peak
    RSS (``VmHWM``): the FAIL gate is the node budget, and the model is printed
    beside it so a model that under-states reality is visible.  See memory_bound().

Env overrides (bench_index style; CONFIG is otherwise hardcoded):
    BENCH_WORKERS   -- comma list overriding {1,4,8,16}
    BENCH_REPS      -- reps per point (default 3; < 3 is refused -- no spread)
    BENCH_EVENTS    -- events per cell (default 1024, i.e. per worker-count/rep)
    BENCH_SUBBATCH  -- events per read_events call (default 32; the memory bound)
    BENCH_DET       -- detector spec name (default "jungfrau")
    BENCH_WORKDIR   -- scratch dir for the persisted index
                       (default /lscratch/<user>/bench_workers_<jobid>)
    BENCH_SKIP_CEILING -- 1 to skip the raw-pread node-ceiling phase
    BENCH_MEM_FRACTION -- fraction of node MemTotal the harness may use (default .25)

Run -- see tests/bench_workers.sbatch (which selects the library end):
    sbatch --job-name=bw_after  --export=ALL,BENCH_END=after  tests/bench_workers.sbatch
    sbatch --job-name=bw_before --export=ALL,BENCH_END=before tests/bench_workers.sbatch
"""
import os
import queue as _queue
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import multiprocessing as mp

# NOTE: psdata is imported lazily inside main()/the workers so that the module
# import (which `spawn` repeats in every child) stays cheap and so that the
# PYTHONPATH-selected library is the one that gets imported, exactly as
# bench_index.py does (plain `import psdata`, never a sys.path hack).

# ==========================================================================
# API-COMPATIBILITY CONTRACT -- every psdata symbol this file touches, and the
# proof it exists at BOTH ends (verified with `git show <sha>:src/psdata/...`):
#
#   psdata.open(exp=, run=, dir=)             run.py:426 (BEFORE) / 730 (AFTER)
#   Run.build_index(rebuild=, source=)        run.py:247 (BEFORE) / 371 (AFTER)
#                                             -- identical body at both ends
#   Run.close()                               run.py:402 (BEFORE) / 706 (AFTER)
#   psdata.RunIndex                           __init__.py exports at both ends
#   RunIndex.save(path)                       index.py:758 (BEFORE) / 1540 (AFTER)
#   RunIndex.load(path)                       index.py:773 (BEFORE) / 1588 (AFTER)
#                                             -- called with ONE positional arg;
#                                                AFTER's dir=/verify_files= are
#                                                defaulted, BEFORE has neither.
#   RunIndex.read_events(ks)                  index.py:477 (BEFORE) / 1074 (AFTER)
#                                             -- called POSITIONALLY.  AFTER's
#                                                max_bytes= is keyword-only and
#                                                MUST NOT be passed (MEM-01, AFTER
#                                                only).  We keep every call under
#                                                AFTER's 2 GiB default guard by
#                                                construction (see memory_bound).
#   RunIndex.close()                          index.py:626 (BEFORE) / 1380 (AFTER)
#   RunIndex.n_events                         index.py:384 (BEFORE) / 843 (AFTER)
#   RunIndex.entries / .timestamps            documented attrs, both ends
#   RunIndex.chunk_files / .bd_files          documented attrs, both ends
#   RunIndex.build_seconds / .smd_bytes_read  documented attrs, both ends
#   RunIndex.scan_bytes_read / .scan_source   __init__ sets both at both ends
#   Event.stack(det)                          stream.py:287 (BEFORE) / (AFTER)
#   psdata.__file__                           (provenance)
#
# DELIBERATELY NOT USED (AFTER-only -- would break the BEFORE run):
#   RunIndex.iter_events / iter_stack, read_events(max_bytes=),
#   read_stack(max_bytes=), DEFAULT_READ_BATCH, DEFAULT_READ_MEM_LIMIT,
#   RunIndex._coalesce_reads, psdata.GateBuildError, Run.seg_configs_at,
#   Run.n_config_steps, RunIndex.load(verify_files=)
# The 32-event sub-batch (bench_index's SUBBATCH, the cube example's read_batch)
# is the ONE batching primitive common to both ends.
# ==========================================================================

# ==========================================================================
# CONFIG -- hardcoded module-level block
# ==========================================================================
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"

# The flagship big-frame detector (identical spec to bench_index.py's DETECTORS
# entry: exp/run/dir/det/N/ev_mb all copied from there, so the two benchmarks
# grade the same bytes).
DETECTORS = {
    "jungfrau": dict(name="jungfrau", exp="mfx100848724", run=51, det="jungfrau",
                     N=17872, ev_mb=33.5),
}
DET_NAME = os.environ.get("BENCH_DET", "jungfrau")

_w_env = os.environ.get("BENCH_WORKERS")
WORKERS = ([int(x) for x in _w_env.split(",") if x.strip()] if _w_env
           else [1, 4, 8, 16])
REPS = int(os.environ.get("BENCH_REPS", "3"))
EVENTS = int(os.environ.get("BENCH_EVENTS", "1024"))   # events per cell
SUBBATCH = int(os.environ.get("BENCH_SUBBATCH", "32"))  # events per read_events
SKIP_CEILING = os.environ.get("BENCH_SKIP_CEILING") == "1"
MEM_FRACTION = float(os.environ.get("BENCH_MEM_FRACTION", "0.25"))

# Memory-bound model constants (see memory_bound()).
COALESCE_SPAN_BYTES = 128 * 1024 ** 2    # AFTER's _COALESCE_MAX_SPAN, as a NUMBER
#   (hardcoded on purpose: the private symbol does not exist at BEFORE, so the
#    bound must not import it -- it is the worst-case transient merged buffer.)
WORKER_BASE_BYTES = 512 * 1024 ** 2      # interpreter + numpy + the loaded index
READ_EVENTS_HARD_CAP = 1536 * 1024 ** 2  # 1.5 GiB: our OWN cap on one read_events
#   call, kept strictly under AFTER's 2 GiB DEFAULT_READ_MEM_LIMIT so the call
#   NEVER raises MemoryError at AFTER while silently proceeding at BEFORE -- the
#   two ends must execute the identical code path.

# Node-ceiling phase (raw 8 MiB preads over the same bigdata files).
CEIL_BLOCK = 8 * 1024 ** 2               # one pread
CEIL_SLOT = 2 * 1024 ** 3                # bytes each ceiling worker reads per rep

# Cold-discipline proof (see prime_page_cache / fadvise_dontneed): bytes read on
# purpose, from the TAIL of the chunk files, so the first DONTNEED has a KNOWN
# quantity to evict.  The tail is read by NEITHER timed phase (phase 1 walks the
# first `need` events, phase 2 the head of each file), so priming it cannot warm a
# byte either phase times.
COLDPROOF_BYTES = 2 * 1024 ** 3

BARRIER_TIMEOUT = 1800.0                 # s; 16 spawned workers + index load


# ==========================================================================
# Self-guard -- an EXCLUSIVE milano compute node, or nothing.
# Copied verbatim from tests/bench_index.py's on_batch_node(), then hardened:
# the BENCH_ALLOW_NONBATCH escape hatch bench_index offers is REMOVED here (the
# campaign forbids login-node numbers and forbids the override), and setting it
# is itself a refusal.
# ==========================================================================
def on_batch_node():
    """(ok, host, why). A batch node has $SLURM_JOB_ID set and is NOT in the
    shared iana/psana interactive pool nor a login node (milano nodes are
    sdfmilanNNN)."""
    host = socket.gethostname()
    jid = os.environ.get("SLURM_JOB_ID")
    low = host.lower()
    if not jid:
        return False, host, "no $SLURM_JOB_ID (not under Slurm)"
    if low.startswith("sdfiana") or low.startswith("psana") or "login" in low:
        return False, host, f"interactive/login pool host {host!r}"
    return True, host, "ok"


def exclusivity():
    """(ok, why). Attest that this job owns the WHOLE node.  A 16-worker scaling
    number measured while a neighbour job shares the node's cores, page cache and
    Weka client is not a number at all.

    TWO independent pieces of evidence; EITHER one attests exclusivity:

      (1) Slurm's own view (`scontrol show job -o`) reporting
          OverSubscribe=EXCLUSIVE / Shared=0.  POSITIVE PROOF -- but scontrol's
          OverSubscribe field is only ADVISORY here: on THIS cluster (S3DF milano)
          a genuinely --exclusive job reports OverSubscribe=NO (verified: job
          31394378 was submitted --exclusive --mem=0, got AllocCPUS=120 and mem=480G
          -- the whole node -- while scontrol said OverSubscribe=NO).  So anything
          other than EXCLUSIVE/Shared=0 is NOT evidence of sharing: fall through.
      (2) RESOURCE ATTESTATION: the job holds effectively every CPU on the node AND
          took all its memory (--mem=0 => SLURM_MEM_PER_NODE=0), which is exactly
          what --exclusive --mem=0 grants.  (os.cpu_count() may report 128 SMT
          threads where SLURM_CPUS_ON_NODE says 120 cores, hence the >= ncpu//2.)

    Only when BOTH fail is the job refused -- and then loudly.
    """
    jid = os.environ.get("SLURM_JOB_ID", "")
    scon = "(scontrol unavailable)"
    try:
        p = subprocess.run(["scontrol", "show", "job", "-o", jid],
                           capture_output=True, text=True, timeout=30)
        if p.returncode == 0:
            out = p.stdout
            if "OverSubscribe=EXCLUSIVE" in out or "Shared=0" in out:
                return True, "scontrol: OverSubscribe=EXCLUSIVE"
            tok = [t for t in out.split()
                   if t.startswith(("OverSubscribe=", "Shared="))]
            scon = (" ".join(tok) if tok
                    else "(scontrol reports no OverSubscribe/Shared field)")
    except Exception:                      # noqa: BLE001  (scontrol not on PATH)
        pass
    ncpu = os.cpu_count() or 0
    scpu = int(os.environ.get("SLURM_CPUS_ON_NODE", "0") or 0)
    smem = os.environ.get("SLURM_MEM_PER_NODE")
    # --exclusive hands the task every CPU on the node.  Accept >= half of
    # os.cpu_count() so an SMT-vs-core accounting difference cannot false-fail.
    if ncpu and scpu >= max(1, ncpu // 2) and (smem in (None, "", "0")):
        return True, (f"SLURM_CPUS_ON_NODE={scpu} of {ncpu} cpus, "
                      f"SLURM_MEM_PER_NODE={smem!r} (whole node); scontrol says "
                      f"{scon} -- ADVISORY only on this cluster, the CPU+memory "
                      f"allocation is the attestation")
    return False, (f"cannot attest exclusivity: scontrol says {scon} AND the "
                   f"allocation is not the whole node: SLURM_CPUS_ON_NODE={scpu!r} "
                   f"of {ncpu} cpus, SLURM_MEM_PER_NODE={smem!r}")


def require_exclusive_batch_node():
    """Refuse to emit ANY number off an exclusive milano compute node.  No
    override exists -- deliberately."""
    if os.environ.get("BENCH_ALLOW_NONBATCH") is not None:
        print("REFUSE: BENCH_ALLOW_NONBATCH is set. This campaign FORBIDS it -- a "
              "multi-worker scaling number off an exclusive batch node is noise, "
              "not a measurement. Unset it and submit tests/bench_workers.sbatch.")
        sys.exit(2)
    ok, host, why = on_batch_node()
    if not ok:
        print(f"REFUSE: multi-worker scaling is invalid off a milano compute node "
              f"({why}); host={host}. Submit tests/bench_workers.sbatch "
              f"(--partition=milano --exclusive --mem=0 --account lcls:prjdat21).")
        sys.exit(2)
    xok, xwhy = exclusivity()
    if not xok:
        print(f"REFUSE: host={host} is a batch node but the job is NOT exclusive "
              f"({xwhy}). A shared node's cores/page-cache/Weka client are shared "
              f"with a neighbour, so the 16-worker point would be meaningless. "
              f"Submit with --exclusive --mem=0.")
        sys.exit(2)
    return host, xwhy


# ==========================================================================
# /proc accounting -- the smoking-gun instruments
# ==========================================================================
_IO_KEYS = ("rchar", "syscr", "read_bytes")


def proc_io():
    """``{rchar, syscr, read_bytes}`` for THIS process.

      * rchar      -- bytes returned by read/pread syscalls.  Always meaningful,
                      including on a network filesystem.  This is the GB/s
                      numerator (bench_index uses the same counter).
      * syscr      -- the number of read syscalls.  THE metric PERF-01 changes:
                      BEFORE issues one pread per (event, stream); AFTER merges.
      * read_bytes -- bytes fetched at the block layer.  Typically 0 on Weka/NFS
                      (no block device in the path) -- reported for completeness,
                      never used as a denominator.

    THE INSTRUMENT IS NOT FREE: reading /proc/self/io is itself read(2), so the
    closing proc_io() call lands INSIDE the io1-io0 window it measures.  Measured:
    two back-to-back calls with zero work between them yield syscr_delta=2,
    rchar_delta=98.  That is noise at BEFORE (~320 preads/worker) but +10% at
    AFTER/W=16 (~20 preads/worker) -- enough to make AFTER's syscr_per_evt appear
    to RISE with worker count, i.e. to fake a coalescing regression out of the
    instrument.  So the overhead is CALIBRATED once at startup (below) and
    SUBTRACTED from every reported delta.
    """
    out = {k: 0 for k in _IO_KEYS}
    try:
        with open("/proc/self/io") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in out:
                    out[k] = int(v)
    except OSError:
        pass
    return out


def _calibrate_io_probe():
    """Cost of ONE proc_io() call, in (read syscalls, bytes) -- see proc_io().
    Runs at import, so a spawned worker (which re-imports this module) calibrates
    its own probe in its own process, exactly like the one it will subtract."""
    a = proc_io()
    b = proc_io()
    return (max(0, b["syscr"] - a["syscr"]), max(0, b["rchar"] - a["rchar"]))


IO_PROBE_SYSCR, IO_PROBE_RCHAR = _calibrate_io_probe()


def io_delta(io0, io1):
    """``io1 - io0`` with the cost of the CLOSING proc_io() call removed (clamped
    at 0).  ``read_bytes`` is untouched: /proc is not a block device."""
    return dict(
        rchar=max(0, io1["rchar"] - io0["rchar"] - IO_PROBE_RCHAR),
        syscr=max(0, io1["syscr"] - io0["syscr"] - IO_PROBE_SYSCR),
        read_bytes=max(0, io1["read_bytes"] - io0["read_bytes"]),
    )


def peak_rss_bytes():
    """VmHWM -- this process's peak resident set size, in bytes.  Turns the
    memory BOUND from a claim into a measurement."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def meminfo_bytes(field):
    """One /proc/meminfo field ("MemTotal", "Cached", "MemAvailable", ...) in
    bytes, or 0.  ``Cached`` is what makes the cold discipline auditable: it is the
    page cache DONTNEED is supposed to drop."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def mem_total_bytes():
    return meminfo_bytes("MemTotal")


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


# ==========================================================================
# Cold discipline -- copied from tests/bench_index.py
# ==========================================================================
def all_chunk_paths(ridx):
    """Every distinct bigdata chunk file the index reads (rolled chunks too)."""
    paths = []
    cf = getattr(ridx, "chunk_files", None)
    if cf:
        for chunks in cf.values():
            paths.extend(chunks)
    else:
        paths.extend(getattr(ridx, "bd_files", {}).values())
    return sorted(set(p for p in paths if p))


_COLDPROOF_PENDING = True


def fadvise_dontneed(paths, expect_drop=0):
    """Best-effort page-cache eviction (POSIX_FADV_DONTNEED) for each path, so a
    re-read starts cold.  'only advice the kernel may decline' -- combined with
    --exclusive and DISJOINT windows (no cell re-reads another cell's bytes) this
    is the cold discipline.

    AND IT IS PROVEN, NOT ASSUMED.  Swallowing every OSError and printing nothing
    is how a benchmark ships warm numbers labelled cold: on Weka, DONTNEED reaches
    only the KERNEL page cache, and if it silently no-ops, nothing in the output
    would say so.  This is not hypothetical -- in a paired job with both arms
    back-to-back on one node, the arm that ran SECOND was ~14% faster at K=1000 on
    all three nodes, in BOTH orders: the cache was demonstrably not cold across
    arms.  So: any OSError is reported, and the FIRST call samples /proc/meminfo
    ``Cached`` / ``MemAvailable`` on both sides of the eviction and prints a
    greppable COLDPROOF line.  ``expect_drop`` is the number of bytes the caller
    has just deliberately pulled into the cache (see prime_page_cache); a drop of
    less than half of it is a loud WARNING -- never an exit, because the numbers
    still stand, they are simply WARM numbers and must be read as such.
    """
    global _COLDPROOF_PENDING
    proof = _COLDPROOF_PENDING
    if proof:
        _COLDPROOF_PENDING = False
        c0, a0 = meminfo_bytes("Cached"), meminfo_bytes("MemAvailable")
    errs, last = 0, ""
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError as e:
            errs += 1
            last = f"{p}: {e}"
    if errs:
        print(f"WARNING: posix_fadvise(DONTNEED) FAILED on {errs} of {len(paths)} "
              f"chunk files (last: {last}) -- those files were NOT evicted, so the "
              f"numbers that follow are WARM, not cold.", flush=True)
    if proof:
        c1, a1 = meminfo_bytes("Cached"), meminfo_bytes("MemAvailable")
        dropped = c0 - c1
        want = 0.5 * expect_drop
        ok = (errs == 0) and (dropped >= want)
        print(f"COLDPROOF cached_before={c0 / 1e6:.0f}MB "
              f"cached_after={c1 / 1e6:.0f}MB dropped_MB={dropped / 1e6:.0f} "
              f"primed_MB={expect_drop / 1e6:.0f} "
              f"expect_drop_MB={want / 1e6:.0f} "
              f"memavail_before_MB={a0 / 1e6:.0f} "
              f"memavail_after_MB={a1 / 1e6:.0f} fadvise_errors={errs} "
              f"verdict={'OK' if ok else 'SUSPECT'}", flush=True)
        if not ok:
            print(f"WARNING: COLDPROOF is SUSPECT -- DONTNEED dropped only "
                  f"{dropped / 1e6:.0f} MB of page cache after "
                  f"{expect_drop / 1e6:.0f} MB were just read from these very "
                  f"files. On Weka, DONTNEED reaches only the kernel page cache; "
                  f"if it no-ops, EVERY number below is a WARM number. The numbers "
                  f"still stand -- they are just not cold, and a BEFORE/AFTER "
                  f"comparison must then be run on SEPARATE nodes (or separate "
                  f"jobs), never back-to-back on one node.", flush=True)


def prime_page_cache(chunk_sizes, nbytes=COLDPROOF_BYTES):
    """Deliberately pull ``nbytes`` (split evenly) into the page cache, read from
    the TAIL of each chunk file -- the one region NEITHER timed phase touches, so
    this cannot warm a byte that is later measured.  Returns the bytes actually
    read: that is the KNOWN quantity the next fadvise_dontneed() must be able to
    drop, and without it 'the cache is cold' is a claim with no evidence."""
    per = nbytes // max(1, len(chunk_sizes))
    got = 0
    for path, sz in chunk_sizes:
        want = min(per, sz)
        start = sz - want                  # the TAIL
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            done = 0
            while done < want:
                b = os.pread(fd, min(CEIL_BLOCK, want - done), start + done)
                if not b:
                    break
                done += len(b)
            got += done
        finally:
            os.close(fd)
    return got


# ==========================================================================
# Provenance
# ==========================================================================
def git_sha(start_path):
    """(sha, dirty) of the git repo containing ``start_path``, or (None, None).

    ``.git`` is tested with os.path.exists, NOT isdir: in a git WORKTREE (and a
    submodule) ``.git`` is a FILE, and isdir would walk straight past it and return
    (None, None) -- which would name the persisted index artifact
    ``...-nosha-<jid>.idx`` for BOTH library ends, so two arms sharing one job would
    COLLIDE on the same path (and one end would load the other's index file)."""
    d = os.path.dirname(os.path.abspath(start_path))
    for _ in range(8):
        if os.path.exists(os.path.join(d, ".git")):
            try:
                sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                                     capture_output=True, text=True,
                                     timeout=30).stdout.strip()
                dirty = bool(subprocess.run(
                    ["git", "-C", d, "status", "--porcelain"],
                    capture_output=True, text=True, timeout=30).stdout.strip())
                return (sha or None), dirty
            except Exception:              # noqa: BLE001
                return None, None
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return None, None


_PSANA_PROBE = r'''
import json, os, subprocess, sys
out = {"python": sys.executable,
       "conda_prefix": os.environ.get("CONDA_PREFIX"),
       "conda_env": os.environ.get("CONDA_DEFAULT_ENV")}
try:
    import psana
    out["file"] = psana.__file__
    out["version"] = getattr(psana, "__version__", None)
    d = os.path.dirname(os.path.abspath(psana.__file__))
    sha = None
    for _ in range(8):
        if os.path.exists(os.path.join(d, ".git")):   # a worktree's .git is a FILE
            sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
            break
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    out["sha"] = sha or None
    out["importable"] = True
except Exception as e:
    out["importable"] = False
    out["error"] = repr(e)
print("PSANA_JSON=" + json.dumps(out))
'''


def psana_provenance():
    """Probe psana in a CLEAN SUBPROCESS.

    Two reasons this is not an in-process ``import psana``:
      (1) purity -- this harness measures psdata, and psdata is psana-free; the
          benchmark process must stay psana-free too (asserted at the end).
      (2) fork/MPI safety -- psana can initialise MPI at import, and initialising
          MPI in a process that later forks/spawns 16 children is a known hazard.
    """
    try:
        p = subprocess.run([sys.executable, "-c", _PSANA_PROBE],
                           capture_output=True, text=True, timeout=600)
        for line in (p.stdout or "").splitlines():
            if line.startswith("PSANA_JSON="):
                import json
                return json.loads(line[len("PSANA_JSON="):])
    except Exception as e:                 # noqa: BLE001
        return {"importable": False, "error": repr(e)}
    return {"importable": False, "error": "probe produced no PSANA_JSON line"}


def print_provenance(host, xwhy, psdata_mod, spec):
    lib_sha, lib_dirty = git_sha(psdata_mod.__file__)
    ps = psana_provenance()
    mt = mem_total_bytes()
    print("#### Provenance", flush=True)
    print(f"  hostname          : {host}")
    print(f"  node type         : partition={os.environ.get('SLURM_JOB_PARTITION')!r} "
          f"cpu={cpu_model()!r}")
    print(f"  cores             : os.cpu_count()={os.cpu_count()} "
          f"SLURM_CPUS_ON_NODE={os.environ.get('SLURM_CPUS_ON_NODE')}")
    print(f"  exclusivity       : {xwhy}")
    print(f"  io probe overhead : syscr={IO_PROBE_SYSCR} rchar={IO_PROBE_RCHAR} B "
          f"per proc_io() call (calibrated at startup; SUBTRACTED from every "
          f"worker's reported syscr/rchar delta -- the closing proc_io() otherwise "
          f"lands inside its own measurement window)")
    print(f"  MemTotal          : {mt / 1024 ** 3:.1f} GiB")
    print(f"  slurm job         : {os.environ.get('SLURM_JOB_ID')} "
          f"account={os.environ.get('SLURM_JOB_ACCOUNT')!r}")
    print(f"  psdata.__file__   : {psdata_mod.__file__}")
    print(f"  psdata git SHA    : {lib_sha} (dirty={lib_dirty})")
    print(f"  psdata END        : {os.environ.get('BENCH_END', '(unset)')}")
    print(f"  psana importable  : {ps.get('importable')}")
    print(f"  psana.__file__    : {ps.get('file')}")
    print(f"  psana release SHA : {ps.get('sha')}  (version={ps.get('version')!r}, "
          f"env={ps.get('conda_env')!r}, prefix={ps.get('conda_prefix')!r})")
    print(f"  python            : {sys.executable}")
    print(f"  PYTHONPATH        : {os.environ.get('PYTHONPATH')}")
    print(f"  dataset           : exp={spec['exp']} run={spec['run']} "
          f"det={spec['det']} dir={DIR} N={spec['N']} ~{spec['ev_mb']} MB/ev")
    # Greppable one-liners.
    print(f"PROV psdata_file={psdata_mod.__file__} psdata_sha={lib_sha} "
          f"psdata_dirty={lib_dirty} psana_sha={ps.get('sha')} "
          f"psana_version={ps.get('version')} host={host} "
          f"partition={os.environ.get('SLURM_JOB_PARTITION')} "
          f"cores={os.cpu_count()} mem_GiB={mt / 1024 ** 3:.1f} "
          f"io_probe_syscr={IO_PROBE_SYSCR} io_probe_rchar={IO_PROBE_RCHAR} "
          f"harness={os.path.abspath(__file__)}", flush=True)
    if not ps.get("importable"):
        print(f"REFUSE: psana is NOT importable in this environment "
              f"({ps.get('error')}). The campaign requires the real psana2 "
              f"environment (psconda) with the library selected by a PYTHONPATH "
              f"PREPEND -- a clobbered PYTHONPATH that hides psana is the known "
              f"false-pass trap.")
        sys.exit(2)
    return lib_sha


# ==========================================================================
# Memory bound (MEM-01 shaped)
# ==========================================================================
def bytes_per_event(ridx, ks):
    """Mean indexed dgram bytes per event over ``ks`` -- read straight off the
    index (``entries[k] == {stream: (chunk_path, offset, size)}``, a documented
    attribute at BOTH ends).  NO I/O.  This is the frame size the memory bound and
    the read-amplification ratio are computed from."""
    tot = 0
    for k in ks:
        for rec in ridx.entries[int(k)].values():
            tot += rec[2]
    return tot / max(1, len(ks))


def memory_bound(b_e, max_workers):
    """THE BOUND THIS HARNESS ENFORCES.

    Per worker, the read path holds at most ONE ``read_events`` sub-batch:

        raw dgram bytes of one sub-batch        SUBBATCH x b_e
      + one decoded stack (transient)         +          b_e
      + one coalesced merged buffer (AFTER)   + 128 MiB   (the span cap; BEFORE
                                                           has no such buffer, so
                                                           this term only ever
                                                           over-states BEFORE)
      + interpreter + numpy + loaded index    + 512 MiB
      ------------------------------------------------------------------
      = per_worker

    and the aggregate is ``max_workers x per_worker``.  With the jungfrau frame
    (b_e ~ 33.5 MB) and SUBBATCH=32:

        32 x 33.5 MB = 1.072 GB
             + 33.5 MB = 1.106 GB
            + 0.134 GB = 1.240 GB
            + 0.537 GB = 1.777 GB per worker
        x 16 workers  = 28.4 GB aggregate

    Two hard checks, BOTH before the clock starts:
      (a) SUBBATCH x b_e must stay under READ_EVENTS_HARD_CAP (1.5 GiB) -- strictly
          below AFTER's 2 GiB ``DEFAULT_READ_MEM_LIMIT``, so a single
          ``read_events`` call NEVER trips AFTER's MEM-01 guard while sailing
          through at BEFORE.  The two ends must run the identical code path.
      (b) the aggregate must stay under MEM_FRACTION (25%) of the node's MemTotal.
    Afterwards the bound is VERIFIED against the BUDGET -- the sum of the workers'
    measured peak RSS (VmHWM) must not exceed MEM_FRACTION x MemTotal, or the run
    fails closed (exit 1).  The BUDGET, not the model, is the gate: the model is a
    prediction, and a prediction is not something a measurement can violate.  The
    verdict line therefore prints BOTH (measured vs model AND measured vs budget),
    so a per-worker model that under-states reality is visible even when the run
    passes.
    """
    per_worker = SUBBATCH * b_e + b_e + COALESCE_SPAN_BYTES + WORKER_BASE_BYTES
    agg = per_worker * max_workers
    mt = mem_total_bytes()
    budget = MEM_FRACTION * mt if mt else float("inf")
    batch = SUBBATCH * b_e
    print("\n#### Memory bound (MEM-01 shaped)", flush=True)
    print(f"  SUBBATCH={SUBBATCH} events/read_events call; bytes/event="
          f"{b_e / 1e6:.1f} MB (measured off the index)")
    print(f"  one read_events call materializes ~{batch / 1024 ** 3:.2f} GiB "
          f"(hard cap {READ_EVENTS_HARD_CAP / 1024 ** 3:.2f} GiB, kept strictly "
          f"under AFTER's 2.00 GiB DEFAULT_READ_MEM_LIMIT)")
    print(f"  per-worker model  = {SUBBATCH}xb_e + b_e + 128MiB(coalesce span) + "
          f"512MiB(base) = {per_worker / 1e9:.2f} GB")
    print(f"  aggregate @ {max_workers:2d}w  = {agg / 1e9:.2f} GB")
    print(f"  node MemTotal     = {mt / 1e9:.1f} GB; budget = "
          f"{MEM_FRACTION:.0%} = {budget / 1e9:.1f} GB")
    print(f"MEMBOUND subbatch={SUBBATCH} bytes_per_event={b_e:.0f} "
          f"per_worker_GB={per_worker / 1e9:.2f} workers={max_workers} "
          f"predicted_aggregate_GB={agg / 1e9:.2f} "
          f"node_mem_GB={mt / 1e9:.1f} budget_GB={budget / 1e9:.1f}", flush=True)
    if batch > READ_EVENTS_HARD_CAP:
        print(f"REFUSE: SUBBATCH={SUBBATCH} x {b_e / 1e6:.1f} MB/event = "
              f"{batch / 1024 ** 3:.2f} GiB per read_events call, over the "
              f"{READ_EVENTS_HARD_CAP / 1024 ** 3:.2f} GiB cap. AFTER would raise "
              f"MemoryError (MEM-01) where BEFORE would silently allocate -- the "
              f"ends would not run the same code. Lower BENCH_SUBBATCH.")
        sys.exit(2)
    if agg > budget:
        print(f"REFUSE: {max_workers} workers x {per_worker / 1e9:.2f} GB = "
              f"{agg / 1e9:.2f} GB exceeds the {MEM_FRACTION:.0%} node budget "
              f"({budget / 1e9:.1f} GB of {mt / 1e9:.1f} GB). Lower BENCH_SUBBATCH "
              f"or BENCH_WORKERS.")
        sys.exit(2)
    print("  -> BOUND OK (16 workers cannot blow the node's RAM)\n", flush=True)
    return per_worker, agg, budget


# ==========================================================================
# WORKERS (spawned; each loads the PERSISTED shared index -- no SMD rescan)
# ==========================================================================
def _worker_read(wid, idx_path, det, ks, subbatch, barrier, q):
    """One psdata reader.  Loads the shared persisted index, waits on the barrier
    so every worker's clock starts together, then reads its OWN disjoint slice in
    <= subbatch sub-batches (read-and-discard: only scalars survive a sub-batch).

    Only ``RunIndex.load(path)`` / ``read_events(ks)`` / ``Event.stack(det)`` /
    ``close()`` are used -- all present at BOTH library ends.
    """
    try:
        import psdata
        ridx = psdata.RunIndex.load(idx_path)      # persisted index, zero rescan
        barrier.wait()                             # <-- all workers start together
        io0 = proc_io()
        t0 = time.monotonic()
        n = 0
        acc = 0.0
        for i in range(0, len(ks), subbatch):
            sub = ks[i:i + subbatch]
            for evt in ridx.read_events(sub):      # POSITIONAL: no max_bytes=
                st = evt.stack(det)
                if st is not None:
                    # The bytes are already off the disk and already decoded: the
                    # read is done by read_events and Event.raw COPIES out of the
                    # dgram buffer.  `.shape` touches no pixel -- this is just a
                    # cheap liveness use of the stack so it cannot be optimised
                    # away or left unbuilt.
                    acc += float(st.shape[0])
                    n += 1
            # `sub` and its events drop out of scope here (read-and-discard)
        t1 = time.monotonic()
        io1 = proc_io()
        ridx.close()
        d = io_delta(io0, io1)                     # probe overhead subtracted
        q.put(dict(wid=wid, ok=True, t0=t0, t1=t1, secs=t1 - t0,
                   n_delivered=n, n_requested=len(ks),
                   rchar=d["rchar"], syscr=d["syscr"],
                   read_bytes=d["read_bytes"],
                   peak_rss=peak_rss_bytes(), acc=acc,
                   psdata_file=psdata.__file__))
    except BaseException:                          # noqa: BLE001
        try:
            barrier.abort()                        # unblock the parent, fail loud
        except Exception:                          # noqa: BLE001
            pass
        try:
            q.put(dict(wid=wid, ok=False, err=traceback.format_exc()))
        except Exception:                          # noqa: BLE001
            pass


def _worker_ceiling(wid, segments, barrier, q):
    """The node-ceiling worker: the SAME bigdata files, but a dumb sequential
    8 MiB-block ``os.pread`` loop over its own disjoint byte range.  No psdata, no
    parsing -- this is the bandwidth the filesystem will give this node, i.e. the
    number psdata's aggregate GB/s is graded against."""
    try:
        barrier.wait()
        io0 = proc_io()
        t0 = time.monotonic()
        total = 0
        for path, off, length in segments:
            fd = os.open(path, os.O_RDONLY)
            try:
                done = 0
                while done < length:
                    b = os.pread(fd, min(CEIL_BLOCK, length - done), off + done)
                    if not b:
                        break
                    done += len(b)
                total += done
            finally:
                os.close(fd)
        t1 = time.monotonic()
        io1 = proc_io()
        d = io_delta(io0, io1)                     # probe overhead subtracted
        q.put(dict(wid=wid, ok=True, t0=t0, t1=t1, secs=t1 - t0, nbytes=total,
                   rchar=d["rchar"], syscr=d["syscr"],
                   read_bytes=d["read_bytes"],
                   peak_rss=peak_rss_bytes()))
    except BaseException:                          # noqa: BLE001
        try:
            barrier.abort()
        except Exception:                          # noqa: BLE001
            pass
        try:
            q.put(dict(wid=wid, ok=False, err=traceback.format_exc()))
        except Exception:                          # noqa: BLE001
            pass


def run_parallel(ctx, target, per_worker_args):
    """Spawn ``len(per_worker_args)`` workers, rendezvous them on a barrier, and
    time the region from the barrier release to the last worker's report.

    The barrier is what makes the AGGREGATE number honest: without it, worker 0
    would be half done reading before worker 15 had finished importing numpy, and
    'aggregate throughput' would silently be 'staggered throughput'.  The parent is
    the (W+1)-th party, so it starts its clock at the same instant the workers do.
    """
    W = len(per_worker_args)
    barrier = ctx.Barrier(W + 1)
    q = ctx.Queue()
    procs = [ctx.Process(target=target, args=(i,) + tuple(a) + (barrier, q),
                         daemon=False)
             for i, a in enumerate(per_worker_args)]
    for p in procs:
        p.start()
    try:
        barrier.wait(BARRIER_TIMEOUT)
    except threading.BrokenBarrierError:
        for p in procs:
            p.terminate()
        raise RuntimeError(
            "a worker aborted (or timed out) before the timed region started -- "
            "no number is emitted")
    t0 = time.monotonic()
    results = []
    try:
        for _ in range(W):
            results.append(q.get(timeout=BARRIER_TIMEOUT))
    except _queue.Empty:
        for p in procs:
            p.terminate()
        raise RuntimeError("a worker never reported -- no number is emitted")
    t1 = time.monotonic()
    for p in procs:
        p.join(timeout=120)
        if p.is_alive():
            p.terminate()
    bad = [r for r in results if not r.get("ok")]
    if bad:
        for r in bad:
            print(f"WORKER {r['wid']} FAILED:\n{r.get('err')}", file=sys.stderr)
        raise RuntimeError(f"{len(bad)}/{W} workers failed -- no number is emitted")
    return t1 - t0, sorted(results, key=lambda r: r["wid"])


# ==========================================================================
# Partitioning
# ==========================================================================
def partition(ks, n_parts):
    """Split ``ks`` into ``n_parts`` CONTIGUOUS, near-equal, DISJOINT slices --
    the same shape as examples/cube_ray_shared_index.py's ``_partition``.  Every
    event is read by exactly one worker; the union is the whole window.

    Contiguity is load-bearing for the hypothesis: consecutive events of one
    stream are adjacent on disk, so a contiguous slice is precisely the case
    PERF-01's coalescing can merge and the case PAR-03 measured.

    EXACTLY ``n_parts`` NON-EMPTY slices, or nothing: silently dropping an empty
    slice would let a cell labelled wc=16 actually run 15 workers, and the whole
    point of the harness is the number in that label."""
    n = len(ks)
    if not 1 <= n_parts <= n:
        raise ValueError(f"cannot split {n} events into {n_parts} non-empty "
                         f"slices -- a cell labelled wc={n_parts} would not run "
                         f"{n_parts} workers")
    edges = [round(i * n / n_parts) for i in range(n_parts + 1)]
    slices = [ks[edges[i]:edges[i + 1]] for i in range(n_parts)]
    assert len(slices) == n_parts and all(slices), (
        f"partition({n}, {n_parts}) produced {len([s for s in slices if s])} "
        f"non-empty slices, not {n_parts}")
    return slices


def ceiling_share(slot_bytes, n_files):
    """Bytes one ceiling slot draws from EACH chunk file (see ceiling_segments)."""
    return slot_bytes // max(1, n_files)


def ceiling_segments(files, slot_index, slot_bytes):
    """Map ceiling slot ``slot_index`` onto ``[(path, offset, length), ...]``.

    Every slot draws an EQUAL SHARE (``slot_bytes // n_files``) from EVERY chunk
    file, so a ceiling worker reads exactly the file set a psdata worker reads: all
    5 streams, interleaved.  Mapping the slot into the CONCATENATED space of
    ``sorted(chunks)`` instead (the old behaviour) meant the whole ceiling phase
    read stream 0 and part of stream 1 and NEVER touched streams 2/3/4 -- a 1-2
    file sequential number, presented as the interleaved bandwidth psdata is graded
    against.  It is the WRONG DENOMINATOR.

    Slot ``i`` takes byte range ``[i * share, (i + 1) * share)`` in each file, and
    VISITS the files round-robin from ``files[i % len(files)]`` so the workers of a
    cell do not all queue on file 0 first.  Distinct slots therefore read DISJOINT
    bytes, and nothing WRAPS: a wrap would make two slots overlap and inflate the
    ceiling with cache hits, so main() REFUSES up front (see the ceiling-capacity
    check) if the requested slots would run past the end of a chunk file."""
    n = len(files)
    share = ceiling_share(slot_bytes, n)
    off = slot_index * share
    segs = []
    for j in range(n):
        path, sz = files[(slot_index + j) % n]
        take = min(share, max(0, sz - off))        # 0 only if the guard was skipped
        if take > 0:
            segs.append((path, off, take))
    return segs


# ==========================================================================
# Stats -- NO point is ever printed without a spread
# ==========================================================================
def spread(xs):
    """(median, stdev, min, max) -- the FOUR numbers every point must carry.
    ``statistics.stdev`` is the sample (n-1) stdev; REPS >= 3 is enforced in main,
    so it is always defined."""
    return (statistics.median(xs), statistics.stdev(xs), min(xs), max(xs))


# ==========================================================================
# Main
# ==========================================================================
def main():
    host, xwhy = require_exclusive_batch_node()
    if REPS < 3:
        print(f"REFUSE: BENCH_REPS={REPS} < 3. A point without >= 3 reps has no "
              f"honest spread (stdev/min/max), and this campaign prints no point "
              f"without one.")
        sys.exit(2)
    spec = DETECTORS.get(DET_NAME)
    if spec is None:
        print(f"no detector spec named {DET_NAME!r} (have {sorted(DETECTORS)})")
        sys.exit(2)

    import psdata                                   # PYTHONPATH selects the end

    user = os.environ.get("USER", "nobody")
    jid = os.environ.get("SLURM_JOB_ID", "nojob")
    workdir = os.environ.get("BENCH_WORKDIR") or f"/lscratch/{user}/bench_workers_{jid}"
    try:
        os.makedirs(workdir, exist_ok=True)
    except OSError:
        workdir = tempfile.mkdtemp(prefix="bench_workers_")

    print("=" * 78, flush=True)
    print(f"=== bench_workers  host={host}  job={jid}  det={DET_NAME}  "
          f"workers={WORKERS}  reps={REPS}  events/cell={EVENTS}  "
          f"subbatch={SUBBATCH}  workdir={workdir} ===")
    print("=" * 78, flush=True)
    lib_sha = print_provenance(host, xwhy, psdata, spec)

    # ---- cell ORDER (Latin square) and disjointness (the cold discipline) ---
    # The worker order ROTATES by rep.  With a fixed order, cell i's window is
    # pinned to offset i*EVENTS forever, so every worker count sits at the SAME
    # position within each rep block -- and any position-in-run effect (client
    # warm-up, drift, a slow region of the run) lands entirely on the worker-count
    # axis, i.e. straight into speedup and efficiency.  Worse, W=1 -- the speedup
    # BASELINE -- would always be the FIRST cell of a block, so a per-block warm-up
    # would penalise the baseline and INFLATE every speedup above it.  Rotating
    # makes each worker count visit early/middle/late windows across the reps; the
    # windows stay disjoint because the (W, rep) cells are still distinct.
    cells = []
    for r in range(REPS):
        k = r % len(WORKERS)
        cells.extend((W, r) for W in WORKERS[k:] + WORKERS[:k])   # rep-major
    if EVENTS < max(WORKERS):
        print(f"REFUSE: BENCH_EVENTS={EVENTS} < max(BENCH_WORKERS)="
              f"{max(WORKERS)}: a cell labelled wc={max(WORKERS)} could not give "
              f"every worker a non-empty slice, so it would not be a "
              f"{max(WORKERS)}-worker point at all.")
        sys.exit(2)
    need = len(cells) * EVENTS
    if need > spec["N"]:
        print(f"REFUSE: {len(WORKERS)} worker-counts x {REPS} reps x {EVENTS} "
              f"events = {need} event slots, but the run has only {spec['N']} "
              f"events. Cells would OVERLAP, so a later cell would re-read a "
              f"warm cell's bytes and the numbers would be cache artefacts. "
              f"Lower BENCH_EVENTS (<= {spec['N'] // len(cells)}) or BENCH_REPS.")
        sys.exit(2)

    # ---- build the shared index ONCE, persist it (US-008) -------------------
    print("#### Shared index (built ONCE in the parent, then persisted)",
          flush=True)
    r = psdata.open(exp=spec["exp"], run=spec["run"], dir=DIR)
    t0 = time.monotonic()
    ridx = r.build_index(rebuild=True, source="smd")
    build_s = time.monotonic() - t0
    # Name the artifact by (detector, library SHA, job) so a BEFORE-written index
    # can NEVER be read by an AFTER job or vice versa: BEFORE's save() writes a
    # pickle, AFTER's writes the IDX-02 PSDATIDX container -- the same path must
    # not be shared between the ends.  It is rebuilt every job anyway.
    tag = (lib_sha or "nosha")[:12]
    idx_path = os.path.join(workdir, f"{spec['name']}-{tag}-{jid}.idx")
    t0 = time.monotonic()
    ridx.save(idx_path)
    save_s = time.monotonic() - t0
    print(f"  build (smd scan) : {build_s:.2f}s, n_events={ridx.n_events}, "
          f"smd_read={ridx.smd_bytes_read / 1e6:.1f} MB, "
          f"source={ridx.scan_source!r}")
    print(f"  save             : {save_s:.3f}s -> {idx_path} "
          f"({os.path.getsize(idx_path) / 1e6:.2f} MB)")
    print(f"  workers will RunIndex.load() this artifact -- ZERO SMD rescan per "
          f"worker (the shared-index pattern of examples/cube_ray_shared_index.py)",
          flush=True)
    if ridx.n_events < need:
        print(f"REFUSE: index has {ridx.n_events} events, need {need} disjoint "
              f"slots. Lower BENCH_EVENTS/BENCH_REPS.")
        sys.exit(2)

    chunks = all_chunk_paths(ridx)
    chunk_sizes = [(p, os.path.getsize(p)) for p in chunks]
    if not chunk_sizes:
        print("REFUSE: the index exposes no chunk files (chunk_files/bd_files), so "
              "nothing can be evicted and no ceiling can be measured -- every "
              "number below would be silently WARM and ungraded.")
        sys.exit(2)
    n_streams = len(ridx.entries[0])
    b_e = bytes_per_event(ridx, range(min(512, ridx.n_events)))
    print(f"  streams/event    : {n_streams}   bytes/event: {b_e / 1e6:.2f} MB   "
          f"chunk files: {len(chunks)} "
          f"({sum(s for _, s in chunk_sizes) / 1e9:.1f} GB)", flush=True)

    # ---- the memory bound, stated and enforced BEFORE the clock starts ------
    per_worker_bytes, agg_bytes, budget = memory_bound(b_e, max(WORKERS))

    # ---- the node ceiling must FIT, or it re-reads its own bytes ------------
    # Every ceiling slot takes `share` bytes from EVERY chunk file (ceiling_segments),
    # so the phase needs n_slots x share bytes of EACH file.  If that runs past the
    # end of a file the old code WRAPPED, two slots would overlap, and the second
    # would be served from page cache -- inflating the very ceiling psdata is graded
    # against.  Refuse instead.
    n_slots = REPS * sum(WORKERS)
    ceil_share = ceiling_share(CEIL_SLOT, len(chunk_sizes))
    ceil_need = n_slots * ceil_share
    smallest = min(sz for _, sz in chunk_sizes)
    if not SKIP_CEILING and ceil_need > smallest:
        print(f"REFUSE: the ceiling phase needs {n_slots} disjoint slots x "
              f"{ceil_share / 1e9:.2f} GB per chunk file = {ceil_need / 1e9:.1f} GB "
              f"of EACH of the {len(chunk_sizes)} chunk files, but the smallest is "
              f"only {smallest / 1e9:.1f} GB. Slots would WRAP and overlap, so a "
              f"later slot would re-read a warm slot's bytes and the 'ceiling' "
              f"would be a page-cache artefact -- the one number psdata is graded "
              f"against. Lower BENCH_REPS/BENCH_WORKERS, or set "
              f"BENCH_SKIP_CEILING=1.")
        sys.exit(2)
    if not SKIP_CEILING:
        print(f"#### Ceiling capacity: {n_slots} slots x {ceil_share / 1e6:.0f} MB "
              f"per file = {ceil_need / 1e9:.1f} GB of each chunk file "
              f"(smallest chunk {smallest / 1e9:.1f} GB) -- disjoint, no wrap\n",
              flush=True)

    # ---- disjoint contiguous window per cell -------------------------------
    windows = {cell: i * EVENTS for i, cell in enumerate(cells)}
    print(f"#### Cells: {len(cells)} disjoint contiguous windows of {EVENTS} "
          f"events ({EVENTS * b_e / 1e9:.1f} GB each; "
          f"{need * b_e / 1e9:.0f} GB total, none re-read); worker order ROTATED "
          f"per rep (Latin square), so no worker count is pinned to a fixed "
          f"position in the run\n", flush=True)

    ctx = mp.get_context("spawn")     # a FRESH worker: no inherited fds, no
    #                                   inherited index object, no forked state --
    #                                   each worker's /proc/self/io is its own.
    ridx_entries = ridx.entries       # keep for the useful-bytes accounting
    ridx.close()
    r.close()

    # ---- COLDPROOF: prove DONTNEED actually evicts, before phase 1 ----------
    # Prime a KNOWN number of bytes from the TAIL of the chunk files (a region no
    # cell and no ceiling slot ever reads), then evict.  If Cached does not fall by
    # at least half of what we just read, DONTNEED is a no-op on this filesystem and
    # every number below is a WARM number -- which the COLDPROOF line then says out
    # loud instead of leaving the reader to assume otherwise.
    print("#### Cold discipline, PROVEN (prime a known volume, then evict)",
          flush=True)
    primed = prime_page_cache(chunk_sizes)
    fadvise_dontneed(chunks, expect_drop=primed)   # <-- the FIRST fadvise call
    print(flush=True)

    # ======================================================================
    # PHASE 1 -- psdata multi-worker scaling
    # ======================================================================
    print("=" * 78)
    print("PHASE 1 -- psdata aggregate read scaling (shared persisted index)")
    print("=" * 78, flush=True)
    per_cell = {}                     # (W, rep) -> metrics
    max_seen_rss = 0
    for (W, rep) in cells:
        start = windows[(W, rep)]
        ks = list(range(start, start + EVENTS))
        useful = sum(rec[2] for k in ks for rec in ridx_entries[k].values())
        slices = partition(ks, W)
        fadvise_dontneed(chunks)                       # cold
        wall, res = run_parallel(
            ctx, _worker_read,
            [(idx_path, spec["det"], sl, SUBBATCH) for sl in slices])

        n_del = sum(x["n_delivered"] for x in res)
        n_req = sum(x["n_requested"] for x in res)
        rchar = sum(x["rchar"] for x in res)
        syscr = sum(x["syscr"] for x in res)
        rbyte = sum(x["read_bytes"] for x in res)
        rss = sum(x["peak_rss"] for x in res)
        max_seen_rss = max(max_seen_rss, rss)
        skew = max(x["t0"] for x in res) - min(x["t0"] for x in res)
        per_cell[(W, rep)] = dict(
            wall=wall, evt_s=n_del / wall, GB_s=rchar / wall / 1e9,
            n_delivered=n_del, n_requested=n_req, rchar=rchar, syscr=syscr,
            read_bytes=rbyte, useful=useful, peak_rss=rss, skew=skew,
            window=start)
        for x in res:
            print(f"WORKER wc={W} rep={rep} w={x['wid']} evt={x['n_delivered']}"
                  f"/{x['n_requested']} s={x['secs']:.2f} "
                  f"evt_s={x['n_delivered'] / max(x['secs'], 1e-9):.1f} "
                  f"rchar_MB={x['rchar'] / 1e6:.0f} syscr={x['syscr']} "
                  f"MB_per_read={x['rchar'] / max(x['syscr'], 1) / 1e6:.1f} "
                  f"read_bytes_MB={x['read_bytes'] / 1e6:.0f} "
                  f"rss_MB={x['peak_rss'] / 1e6:.0f}", flush=True)
        c = per_cell[(W, rep)]
        print(f"CELL   wc={W} rep={rep} window=[{start},{start + EVENTS}) "
              f"wall={wall:.2f}s evt_s={c['evt_s']:.1f} GB_s={c['GB_s']:.3f} "
              f"syscr={syscr} syscr_per_evt={syscr / max(n_del, 1):.2f} "
              f"MB_per_read={rchar / max(syscr, 1) / 1e6:.1f} "
              f"read_amp={rchar / max(useful, 1):.4f} "
              f"rss_GB={rss / 1e9:.1f} start_skew_ms={skew * 1e3:.0f}\n",
              flush=True)
        if n_del != n_req:
            print(f"  NOTE: {n_req - n_del} of {n_req} events delivered no "
                  f"{spec['det']} stack (missing-segment rule); evt/s uses the "
                  f"DELIVERED count.", flush=True)

    # ======================================================================
    # PHASE 2 -- the node ceiling (same files, dumb 8 MiB preads)
    # ======================================================================
    ceil = {}
    if not SKIP_CEILING:
        print("=" * 78)
        print(f"PHASE 2 -- node read-bandwidth CEILING (raw {CEIL_BLOCK // 1024**2} "
              f"MiB preads over the same bigdata files, {CEIL_SLOT // 1024**3} GiB "
              f"per worker per rep)")
        print("=" * 78, flush=True)
        slot = 0
        for rep in range(REPS):
            for W in WORKERS:
                args = []
                for _ in range(W):
                    args.append((ceiling_segments(chunk_sizes, slot, CEIL_SLOT),))
                    slot += 1
                fadvise_dontneed(chunks)
                wall, res = run_parallel(ctx, _worker_ceiling, args)
                nb = sum(x["nbytes"] for x in res)
                sc = sum(x["syscr"] for x in res)
                ceil.setdefault(W, []).append(
                    dict(wall=wall, GB_s=nb / wall / 1e9, nbytes=nb, syscr=sc))
                print(f"CEILCELL wc={W} rep={rep} wall={wall:.2f}s "
                      f"GB_s={nb / wall / 1e9:.3f} GB={nb / 1e9:.1f} syscr={sc} "
                      f"MB_per_read={nb / max(sc, 1) / 1e6:.1f}", flush=True)
        print(flush=True)

    # ======================================================================
    # SUMMARY -- every point carries stdev + min + max.  Greppable.
    # ======================================================================
    print("=" * 78)
    print("SUMMARY (median over reps; every point printed with its spread)")
    print("=" * 78, flush=True)
    # The scaling baseline is the SMALLEST worker count actually measured (1 in
    # the default grid), so speedup/efficiency mean what they say even if the
    # grid is overridden with BENCH_WORKERS.
    #
    # speedup is PAIRED, rep by rep: rep r of W is divided by rep r of the BASELINE,
    # giving REPS speedup samples that carry a spread of their own.  A ratio of two
    # medians over INDEPENDENT cells is the one number everybody quotes, and printed
    # bare it hides exactly the variance that decides whether it means anything.
    base_W = min(WORKERS)
    base_evt = [per_cell[(base_W, rep)]["evt_s"] for rep in range(REPS)]
    rows = []
    shortfall = []
    for W in WORKERS:
        cs = [per_cell[(W, rep)] for rep in range(REPS)]
        e_med, e_sd, e_lo, e_hi = spread([c["evt_s"] for c in cs])
        g_med, g_sd, g_lo, g_hi = spread([c["GB_s"] for c in cs])
        sp = [cs[r]["evt_s"] / base_evt[r] for r in range(REPS)]     # PAIRED
        s_med, s_sd, s_lo, s_hi = spread(sp)
        f_med, f_sd, f_lo, f_hi = spread([x / (W / base_W) for x in sp])
        syscr = sum(c["syscr"] for c in cs)
        n_del = sum(c["n_delivered"] for c in cs)
        n_req = sum(c["n_requested"] for c in cs)
        rchar = sum(c["rchar"] for c in cs)
        useful = sum(c["useful"] for c in cs)
        rss = max(c["peak_rss"] for c in cs)
        deliv = n_del / max(n_req, 1)
        if deliv < 0.99:
            shortfall.append((W, n_del, n_req, deliv))
        cg = None
        if W in ceil:
            cg = statistics.median([c["GB_s"] for c in ceil[W]])
        rows.append(dict(W=W, e_med=e_med, e_sd=e_sd, e_lo=e_lo, e_hi=e_hi,
                         g_med=g_med, g_sd=g_sd, g_lo=g_lo, g_hi=g_hi,
                         syscr=syscr, n_del=n_del, n_req=n_req, rchar=rchar,
                         useful=useful, rss=rss, ceil=cg,
                         s_med=s_med, s_sd=s_sd, s_lo=s_lo, s_hi=s_hi,
                         f_med=f_med, f_sd=f_sd, f_lo=f_lo, f_hi=f_hi))
        # delivered=/requested= travel WITH the number: evt/s divides by the
        # DELIVERED count while wall covers reading ALL the REQUESTED bytes, so a
        # shortfall silently flatters evt/s -- and it must not be invisible in the
        # one line that gets grepped and published.
        print(f"WORKERS={W} evt_s={e_med:.1f} GB_s={g_med:.3f} "
              f"stdev_evt_s={e_sd:.1f} min={e_lo:.1f} max={e_hi:.1f} "
              f"reps={REPS} "
              f"stdev_GB_s={g_sd:.3f} GB_s_min={g_lo:.3f} GB_s_max={g_hi:.3f} "
              f"delivered={n_del} requested={n_req} "
              f"delivered_frac={deliv:.4f} "
              f"syscr={syscr} syscr_per_evt={syscr / max(n_del, 1):.3f} "
              f"MB_per_read={rchar / max(syscr, 1) / 1e6:.1f} "
              f"read_amp={rchar / max(useful, 1):.4f} "
              f"speedup={s_med:.2f} stdev_speedup={s_sd:.2f} "
              f"speedup_min={s_lo:.2f} speedup_max={s_hi:.2f} "
              f"eff={f_med:.3f} stdev_eff={f_sd:.3f} eff_min={f_lo:.3f} "
              f"eff_max={f_hi:.3f} base_workers={base_W} "
              f"peak_rss_GB={rss / 1e9:.1f} "
              f"psdata_sha={(lib_sha or 'nosha')[:12]}", flush=True)

    for W in WORKERS:
        if W in ceil:
            c_med, c_sd, c_lo, c_hi = spread([c["GB_s"] for c in ceil[W]])
            g_med = statistics.median([per_cell[(W, r)]["GB_s"]
                                       for r in range(REPS)])
            print(f"CEILING WORKERS={W} GB_s={c_med:.3f} stdev_GB_s={c_sd:.3f} "
                  f"min={c_lo:.3f} max={c_hi:.3f} reps={REPS} "
                  f"psdata_GB_s={g_med:.3f} "
                  f"frac_of_ceiling={g_med / max(c_med, 1e-9):.3f}", flush=True)

    # ---- markdown table ----------------------------------------------------
    print("\n### Table W -- aggregate read scaling "
          f"({spec['name']}, {EVENTS} events/cell, {REPS} reps, "
          f"psdata {(lib_sha or 'nosha')[:12]})\n")
    print("| workers | delivered | requested | evt/s (med) | stdev | min | max | "
          "GB/s (med) | stdev | min | max | syscr/evt | MB/read | read amp | "
          "speedup (med) | stdev | min | max | eff (med) | stdev | min | max | "
          "ceiling GB/s | % of ceiling |")
    print("|--------:|----------:|----------:|------------:|------:|----:|----:|"
          "-----------:|------:|----:|----:|----------:|--------:|---------:|"
          "-------------:|------:|----:|----:|----------:|------:|----:|----:|"
          "-------------:|-------------:|")
    for x in rows:
        cg = "n/a" if x["ceil"] is None else f"{x['ceil']:.3f}"
        pc = "n/a" if x["ceil"] is None else f"{x['g_med'] / x['ceil']:.1%}"
        print(f"| {x['W']} | {x['n_del']} | {x['n_req']} | "
              f"{x['e_med']:.1f} | {x['e_sd']:.1f} | {x['e_lo']:.1f} "
              f"| {x['e_hi']:.1f} | {x['g_med']:.3f} | {x['g_sd']:.3f} | "
              f"{x['g_lo']:.3f} | {x['g_hi']:.3f} | "
              f"{x['syscr'] / max(x['n_del'], 1):.3f} | "
              f"{x['rchar'] / max(x['syscr'], 1) / 1e6:.1f} | "
              f"{x['rchar'] / max(x['useful'], 1):.4f} | "
              f"{x['s_med']:.2f}x | {x['s_sd']:.2f} | {x['s_lo']:.2f}x | "
              f"{x['s_hi']:.2f}x | {x['f_med']:.2f} | {x['f_sd']:.2f} | "
              f"{x['f_lo']:.2f} | {x['f_hi']:.2f} | "
              f"{cg} | {pc} |")
    print()

    # ---- the memory bound, VERIFIED ----------------------------------------
    # The GATE is the node budget (25% of MemTotal); the MODEL is a prediction and
    # is reported beside it, so a model that under-states the real peak is visible
    # even in a run that passes.  Both numbers, both verdicts, one line.
    vs_model = "OK" if max_seen_rss <= agg_bytes else "OVER"
    vs_budget = "OK" if max_seen_rss <= budget else "VIOLATED"
    print(f"MEMBOUND measured peak_rss_sum={max_seen_rss / 1e9:.2f} GB "
          f"model={agg_bytes / 1e9:.2f} GB vs_model={vs_model} "
          f"budget={budget / 1e9:.1f} GB ({MEM_FRACTION:.0%} of MemTotal) "
          f"vs_budget={vs_budget} verdict={vs_budget} "
          f"(the FAIL gate is the budget; the model is reported, not gated)",
          flush=True)

    # ---- purity: this harness never imported psana / mpi4py ----------------
    impure = [m for m in ("psana", "mpi4py") if m in sys.modules]
    print(f"PURITY psana_imported={int('psana' in sys.modules)} "
          f"mpi4py_imported={int('mpi4py' in sys.modules)} "
          f"(the read path is framework-free; psana was probed only in a clean "
          f"subprocess for provenance)", flush=True)

    if max_seen_rss > budget:
        print("FAIL: the memory bound was VIOLATED -- the numbers above stand, "
              "but the configuration is not safe to repeat.")
        sys.exit(1)
    if shortfall:
        for (W, d, rq, fr) in shortfall:
            print(f"FAIL: WORKERS={W} DELIVERED only {d} of {rq} requested events "
                  f"({fr:.2%} < 99%). evt/s divides by the DELIVERED count while "
                  f"wall covers reading ALL the REQUESTED bytes, so this point "
                  f"OVERSTATES throughput and must not be published.")
        sys.exit(1)
    if impure:
        print(f"FAIL: the benchmark process imported {impure} -- it must stay "
              f"framework-free.")
        sys.exit(1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
