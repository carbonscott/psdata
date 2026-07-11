#!/usr/bin/env python3
"""IDX-04 -- the persisted index must be INVALIDATED when its xtc files change.

Bug (from the project bug matrix):

    IDX-04 -- Nothing invalidates the index when the xtc files change under it.
    (Safe-by-accident: the read-time ts re-check raises at read time rather than
    returning garbage.)  Evidence: index.py:253-257 (the read-time ts re-check,
    now in ``RunIndex._assemble_stream_dgram``).

A persisted index records byte OFFSETS into the run's bigdata chunk files.  If
those files are modified/replaced/regrown after the index is built (a
re-transfer, a truncation, a different run reusing the path), the offsets no
longer correspond -- yet on the pre-fix code NOTHING proactively refuses the
stale index at load time.  The only backstop is the read-time timestamp
re-check, which raises only when (and if) a caller happens to read an affected
event -- an accidental safety net, not a deliberate invalidation.

The fix records a cheap ``(size, mtime_ns)`` fingerprint of every indexed file
at build time and verifies it up front in :meth:`RunIndex.load` (one ``os.stat``
per chunk file, never a re-read), raising a clear error naming the changed
file(s) BEFORE serving any offset -- with ``verify_files=False`` to bypass a
change known to be benign.

This suite is the pre-fix / post-fix discriminator.  It is **fully
self-contained**: stdlib + numpy only, NO psana, NO SLAC data.  It constructs
the index's data structures directly (as the IDX-02 / IDX-03 tests do) and
materializes tiny stand-in files, so it runs anywhere.  It contains NO part of
the fix -- the invalidation logic under test lives entirely inside the library's
``load`` / build-time fingerprinting; the test only exercises the public
``RunIndex.save`` / ``RunIndex.load`` contract and observes the outcome.

What it proves:

  1. A CHANGED (truncated) indexed file makes ``load`` RAISE up front, naming the
     changed file -- BEFORE any offset is served.  On the pre-fix code no
     fingerprint is recorded and ``load`` has no such check, so the stale index
     loads without complaint -> the "must raise up front" assertion FAILS.
  2. An UNCHANGED run loads clean -- no false invalidation (byte-exact behavior
     preserved).
  3. ``verify_files=False`` BYPASSES the check, loading the (known-benign) stale
     index without raising.

Run (numpy only; no psana needed):
    python3 tests/test_idx04_invalidation.py
"""

import os
import sys
import tempfile

import numpy as np  # noqa: F401  (index/format construct numpy dtypes)

# --- locate the package (parent of this tests dir), robust to cwd -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")  # .../<repo>/src (holds psdata/)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import psdata.format as F
import psdata.index as IX


# ==========================================================================
# Synthetic index -- built directly over materialized stand-in files.
# ==========================================================================
# A tiny but structurally-complete run: one multi-chunk stream (s000 rolls
# c000 -> c001) and one single-chunk stream (s004).  The stand-in files carry
# real bytes so a truncation genuinely changes their size.
_LAYOUT = (
    "exp-r0042-s000-c000.xtc2",
    "exp-r0042-s000-c001.xtc2",
    "exp-r0042-s004-c000.xtc2",
)
_FILE_BYTES = 8192   # each stand-in chunk file's size before any change


def _materialize(root):
    """Create the run's bigdata chunk files as stand-ins with real content."""
    os.makedirs(root, exist_ok=True)
    for rel in _LAYOUT:
        with open(os.path.join(root, rel), "wb") as fh:
            fh.write(b"\x00" * _FILE_BYTES)


def _build_synthetic_index(root):
    """A small :class:`RunIndex` whose offsets point into the stand-in chunk
    files under ``root``, then have its build-time file fingerprints recorded
    exactly as the real build path does.

    On the FIX, ``_record_file_fingerprints`` exists and captures the
    ``(size, mtime_ns)`` baseline (so a later changed file is detectable); on the
    PRE-FIX code the method is absent, so no baseline is recorded -- which is
    precisely why the pre-fix ``load`` cannot detect the change.
    """
    c000 = os.path.join(root, _LAYOUT[0])
    c001 = os.path.join(root, _LAYOUT[1])
    c400 = os.path.join(root, _LAYOUT[2])

    rc = F.RunConfig()
    rc.stream_files = {0: c000, 4: c400}
    table = {"det_type": "jungfrau", "det_name": "jungfrau", "det_id": "SN0001",
             "alg_name": "raw", "alg_version": 1, "segment": 0, "num_arrays": 1,
             "names": [{"name": "raw", "type": 1, "rank": 3}]}
    rc.raw_tables = {0: {(2, 1): table}, 4: {}}
    det = F.DetectorInfo("jungfrau", "jungfrau", "SN0001")
    det._add_table("raw", table, 0, (2, 1))
    rc.detectors = {"jungfrau": det}

    idx = IX.RunIndex(rc)
    idx.timestamps = [0x1234_5678_9abc, 0x1234_5678_9abd, 0x1234_5678_9abe]
    idx.entries = [
        {0: (c000, 0, 4096), 4: (c400, 0, 128)},
        {0: (c001, 0, 4096)},
        {0: (c001, 4096, 4096), 4: (c400, 128, 128)},
    ]
    idx.bd_files = dict(rc.stream_files)
    idx.chunk_files = {0: [c000, c001], 4: [c400]}
    idx.multichunk_streams = {0}
    idx.scan_source = "smd"

    # Record the build-time file fingerprints exactly as build()/build_from_bigdata
    # do.  Guarded so this same test file also runs against the PRE-FIX library
    # (where the method does not exist): there, no baseline is recorded, and the
    # pre-fix load has nothing to check -- which is the behavior the discriminator
    # exposes below.
    rec = getattr(idx, "_record_file_fingerprints", None)
    if rec is not None:
        rec()
    return idx


def _load_raised_invalidation(idx_path, **kw):
    """Call ``RunIndex.load(idx_path, **kw)`` and return the raised message if it
    raised an invalidation error, else ``None`` (it loaded).  ``TypeError`` from a
    pre-fix ``load`` that lacks a passed keyword is treated as "did not
    invalidate" so the discriminator reports the real pre-fix behavior."""
    try:
        IX.RunIndex.load(idx_path, **kw)
    except TypeError:
        return None
    except ValueError as e:
        return str(e)
    return None


# ==========================================================================
# 1. A changed (truncated) indexed file invalidates load UP FRONT.
# ==========================================================================
def test_changed_file_invalidates_up_front():
    """Build + save an index, then TRUNCATE one indexed bigdata file (its size --
    and mtime -- change).  ``load`` (default strict) must RAISE up front, naming
    the changed file, BEFORE serving any offset.

    Pre-fix: no fingerprint is recorded and ``load`` has no such check, so the
    stale index loads silently -> this assertion FAILS (the bug)."""
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "xtc")
        _materialize(root)
        idx = _build_synthetic_index(root)
        idx_path = os.path.join(td, "run0042.pidx")
        idx.save(idx_path)

        # Mutate ONE underlying file under the index: truncate c001 to half.
        victim = os.path.join(root, _LAYOUT[1])
        with open(victim, "r+b") as fh:
            fh.truncate(_FILE_BYTES // 2)
        # Bump mtime as well, so the change is caught even on a filesystem whose
        # size somehow matched (belt and suspenders; truncation already differs).
        st = os.stat(victim)
        os.utime(victim, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

        msg = _load_raised_invalidation(idx_path)   # default: verify_files=True
        assert msg is not None, (
            "load() must REFUSE a stale index up front when an indexed file "
            "changed under it -- pre-fix nothing invalidates the index, so it "
            "loaded silently and only the read-time ts re-check might catch it "
            "later (IDX-04)")
        assert os.path.basename(victim) in msg, (
            "the invalidation error must NAME the changed file %r; got:\n%s"
            % (os.path.basename(victim), msg))
    print("[changed] a truncated indexed file makes load() raise up front, "
          "naming the file")


# ==========================================================================
# 2. An unchanged run loads clean -- no false invalidation.
# ==========================================================================
def test_unchanged_file_loads_clean():
    """Nothing is modified between save and load: the fingerprints match, so
    ``load`` must NOT raise (byte-exact behavior preserved -- the check never
    fires on an untouched run)."""
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "xtc")
        _materialize(root)
        idx = _build_synthetic_index(root)
        idx_path = os.path.join(td, "run0042.pidx")
        idx.save(idx_path)

        back = IX.RunIndex.load(idx_path)           # default strict; must not raise
        assert back.n_events == 3, \
            "an unchanged index must load with all its events intact"
    print("[unchanged] an untouched run loads clean -- no false invalidation")


# ==========================================================================
# 3. verify_files=False bypasses the check for a known-benign change.
# ==========================================================================
def test_bypass_loads_stale_index():
    """A caller who KNOWS a change is benign can pass ``verify_files=False`` to
    load the (technically stale) index without raising.  This is a FIX-only
    contract: the pre-fix ``load`` has no such parameter."""
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "xtc")
        _materialize(root)
        idx = _build_synthetic_index(root)
        idx_path = os.path.join(td, "run0042.pidx")
        idx.save(idx_path)

        # Change a file so strict load WOULD raise ...
        victim = os.path.join(root, _LAYOUT[1])
        with open(victim, "r+b") as fh:
            fh.truncate(_FILE_BYTES // 2)

        # ... strict raises (sanity), bypass does not.
        assert _load_raised_invalidation(idx_path) is not None, \
            "sanity: strict load must raise on the changed file"
        back = IX.RunIndex.load(idx_path, verify_files=False)
        assert back.n_events == 3, \
            "verify_files=False must bypass the check and load the index"
    print("[bypass] verify_files=False loads a known-benign stale index")


def main():
    test_changed_file_invalidates_up_front()
    print("[ok] changed file invalidates load up front")
    test_unchanged_file_loads_clean()
    print("[ok] unchanged run loads clean (no false invalidation)")
    test_bypass_loads_stale_index()
    print("[ok] verify_files=False bypass works")
    print("\nALL IDX-04 TESTS PASSED")


if __name__ == "__main__":
    main()
