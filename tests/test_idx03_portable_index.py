#!/usr/bin/env python3
"""IDX-03 -- the persisted index must be RELOCATABLE, not tied to absolute paths.

Bug (from the project bug matrix):

    IDX-03 -- The persisted index hard-codes absolute file paths -- not portable
    across mounts, hosts, or containers.

psdata's headline feature is a persisted, SHAREABLE index artifact: built once
off ``/sdf`` and re-read by many jobs.  If the on-disk index stores the run's
xtc2 files by ABSOLUTE path (``/sdf/data/lcls/ds/<exp>/xtc/...-c000.xtc2``),
then copying that index to another mount, host, or container -- where the SAME
data lives under a DIFFERENT prefix -- makes ``load`` point at absent or wrong
files.  A shareable artifact must be relocatable.

This suite is the pre-fix / post-fix discriminator for path portability.  It is
**fully self-contained**: stdlib + numpy only, NO psana, NO SLAC data -- it
constructs the index's data structures directly and materializes tiny empty
stand-in files, so it runs anywhere.  It contains NO part of the fix; it only
exercises the public ``RunIndex.save`` / ``RunIndex.load`` contract.

What it proves:

  1. RELOCATE VIA dir= OVERRIDE -- an index built with its data under root A is
     loaded with ``dir=<root B>``, and every persisted path (bd_files,
     chunk_files, per-event entries, env_records, RunConfig.stream_files)
     resolves under root B, never under the original hard-coded root A.  On the
     pre-fix ``load`` there is no ``dir=`` parameter (TypeError) -> this fails.
  2. RELOCATE BESIDE THE INDEX -- an index whose original root no longer exists,
     shipped together with its data into a new directory, resolves (with NO
     override) to the data next to the index.  On the pre-fix ``load`` the
     stored absolute paths come back verbatim (still pointing at the vanished
     original root) -> the "resolves under the new dir" assertion fails.
  3. ORIGINAL LOCATION IS BYTE-EXACT -- loaded from where it was built (data
     present), every path equals the original absolute string exactly, so
     downstream byte-for-byte reads are unaffected.  (A guard: passes pre- and
     post-fix; it must never regress.)
  4. A WRONG dir= FAILS LOUDLY -- pointing ``dir=`` at a directory that lacks the
     data raises a clear error naming the missing file, so a mis-relocation is
     never silently loaded as a wrong/absent file.

Run (numpy only; no psana needed):
    python3 tests/test_idx03_portable_index.py
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
# Synthetic index -- built directly under an arbitrary ``root``, NO xtc2 read.
# ==========================================================================
# The run's files, RELATIVE to whatever root the index is built with.  Includes
# a rolled second chunk (s000-c001) and a nested smalldata sidecar, so the
# portable form must round-trip both a flat and a sub-directory relative path.
_REL_LAYOUT = (
    "exp-r0042-s000-c000.xtc2",
    "exp-r0042-s000-c001.xtc2",
    "exp-r0042-s004-c000.xtc2",
    os.path.join("smalldata", "exp-r0042-s004.smd.xtc2"),
)


def _p(root, rel):
    return os.path.normpath(os.path.join(root, rel))


def _synthetic_index(root):
    """A small but structurally-complete :class:`RunIndex` whose every file path
    lives under ``root`` -- exercising the RAGGED per-event ``entries``, a
    multi-chunk stream, ``env_records`` (nested sidecar path), and a nested
    ``RunConfig`` (tuple-keyed tables, segment sets, a ``FieldInfo``)."""
    c000 = _p(root, _REL_LAYOUT[0])   # stream 0, chunk 0
    c001 = _p(root, _REL_LAYOUT[1])   # stream 0, chunk 1 (rolled)
    c400 = _p(root, _REL_LAYOUT[2])   # stream 4, chunk 0
    smd4 = _p(root, _REL_LAYOUT[3])   # stream 4 env sidecar (sub-dir)

    rc = F.RunConfig()
    rc.stream_files = {0: c000, 4: c400}
    rc.stream_configs = {
        0: ({"service": 2, "env": 0, "ts": 0x0001_0000_0002, "sec": 1,
             "nsec": 2, "src": 7, "damage": 0, "typeid": 0, "extent": 100},
            4096),
    }
    table = {"det_type": "jungfrau", "det_name": "jungfrau", "det_id": "SN0001",
             "alg_name": "raw", "alg_version": 1, "segment": 0, "num_arrays": 1,
             "names": [{"name": "raw", "type": 1, "rank": 3},
                       {"name": "intOffset", "type": 3, "rank": 0}]}
    rc.raw_tables = {0: {(2, 1): table}, 4: {}}
    det = F.DetectorInfo("jungfrau", "jungfrau", "SN0001")
    det._add_table("raw", table, 0, (2, 1))
    rc.detectors = {"jungfrau": det}

    idx = IX.RunIndex(rc)
    idx.timestamps = [0x1234_5678_9abc, 0x1234_5678_9abd, 0x1234_5678_9abe]
    idx.entries = [
        {0: (c000, 4096, 5_000_000), 4: (c400, 4096, 128)},   # two streams
        {0: (c001, 128, 5_000_000)},                          # ragged + rolled
        {0: (c001, 5_000_128, 5_000_000), 4: (c400, 4224, 128)},
    ]
    idx.env_records = {"epics": {4: [(0x1234_5678_9abc, smd4, 64, 128),
                                     (0x1234_5678_9abd, smd4, 192, 128)]}}
    idx.bd_files = dict(rc.stream_files)
    idx.chunk_files = {0: [c000, c001], 4: [c400]}
    idx.multichunk_streams = {0}
    idx.build_seconds = 2.5
    idx.smd_bytes_read = 4_242
    idx.scan_source = "smd"
    idx.scan_bytes_read = 4_242
    idx.include_shutdown_tail = False
    return idx


def _all_persisted_paths(idx):
    """Every distinct file path the (in-memory) index references."""
    paths = set(idx.bd_files.values())
    for lst in idx.chunk_files.values():
        paths.update(lst)
    for ev in idx.entries:
        for (p, _off, _sz) in ev.values():
            paths.add(p)
    for streams in idx.env_records.values():
        for recs in streams.values():
            for (_ts, p, _off, _sz) in recs:
                paths.add(p)
    paths.update(idx.run_config.stream_files.values())
    return paths


def _materialize(root):
    """Create the run's files as tiny empty stand-ins under ``root`` (enough for
    load's existence check; no bytes are ever read)."""
    for rel in _REL_LAYOUT:
        p = _p(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb"):
            pass


# ==========================================================================
# 1. Relocate via an explicit dir= override.
# ==========================================================================
def test_relocate_via_dir_override():
    """Built with data under a (fabricated, ABSENT) root A, loaded with
    ``dir=<root B>`` where the data actually lives -> every path resolves under
    root B, none under the hard-coded root A.  Pre-fix load has no ``dir=``."""
    with tempfile.TemporaryDirectory() as td:
        # Root A never exists on disk -- it stands for "the /sdf prefix on the
        # build host", absent on this one.
        root_a = os.path.join(td, "hostA_sdf", "xtc")
        root_b = os.path.join(td, "hostB_data", "xtc")   # where the data is now
        _materialize(root_b)

        idx = _synthetic_index(root_a)
        idx_path = os.path.join(td, "run0042.pidx")       # index off to the side
        idx.save(idx_path)

        back = IX.RunIndex.load(idx_path, dir=root_b)

        for p in _all_persisted_paths(back):
            assert p.startswith(os.path.abspath(root_b) + os.sep), \
                "path %r did not resolve under the override dir %r" % (p, root_b)
            assert root_a not in p, \
                "path %r still points at the hard-coded original root %r" \
                % (p, root_a)
        # the relocated files exist; the original hard-coded ones never did.
        for s, p in back.bd_files.items():
            assert os.path.exists(p), "relocated bd_file %r missing" % (p,)
    print("[dir-override] index built under root A loads under dir=<root B>")


# ==========================================================================
# 2. Relocate by shipping the index together with its data (no override).
# ==========================================================================
def test_relocate_beside_index_no_dir():
    """Original root is gone; the index is shipped INTO the same directory as
    its data.  With NO override, load resolves to the data beside the index.
    Pre-fix load returns the stored absolute paths verbatim -> they still point
    at the vanished original root, so the 'resolves under new dir' check fails."""
    with tempfile.TemporaryDirectory() as td:
        root_a = os.path.join(td, "gone_original_root", "xtc")   # never created
        new_home = os.path.join(td, "shipped")                   # data + index
        _materialize(new_home)

        idx = _synthetic_index(root_a)
        idx_path = os.path.join(new_home, "run0042.pidx")        # beside the data
        idx.save(idx_path)

        back = IX.RunIndex.load(idx_path)                        # no dir=

        for p in _all_persisted_paths(back):
            assert p.startswith(os.path.abspath(new_home) + os.sep), \
                "path %r did not resolve beside the index at %r" % (p, new_home)
            assert root_a not in p, \
                "path %r still points at the vanished original root %r" \
                % (p, root_a)
    print("[beside-index] index shipped with its data resolves without dir=")


# ==========================================================================
# 3. Loaded from its ORIGINAL location, paths are byte-exact (guard).
# ==========================================================================
def test_original_location_byteexact():
    """When the data is still at the build location, load (no override)
    reproduces the ORIGINAL absolute path strings exactly -- downstream reads
    open the very same files, byte-for-byte.  Also proves save does not mutate
    the live index's paths.  Passes pre- and post-fix; must never regress."""
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "orig_root", "xtc")
        _materialize(root)

        idx = _synthetic_index(root)
        before = _all_persisted_paths(idx)          # snapshot the live strings
        idx_path = os.path.join(td, "run0042.pidx")
        idx.save(idx_path)
        # save must not have mutated the in-memory index.
        assert _all_persisted_paths(idx) == before, \
            "save() mutated the live index's paths"

        back = IX.RunIndex.load(idx_path)

        # every field reconstructs to the exact original absolute strings.
        assert back.bd_files == idx.bd_files, (back.bd_files, idx.bd_files)
        assert back.chunk_files == idx.chunk_files
        assert back.entries == idx.entries
        assert back.env_records == idx.env_records
        assert back.run_config.stream_files == idx.run_config.stream_files
        assert _all_persisted_paths(back) == before
    print("[original] load from the build location reproduces exact paths")


# ==========================================================================
# 4. A wrong dir= fails loudly, naming the missing file.
# ==========================================================================
def test_wrong_dir_raises_clear_error():
    """Pointing ``dir=`` at a directory that does NOT hold the data must raise a
    clear error naming a missing file -- never silently load a wrong/absent
    path.  (Pre-fix load has no ``dir=`` and raises TypeError instead.)"""
    with tempfile.TemporaryDirectory() as td:
        root_a = os.path.join(td, "hostA", "xtc")
        empty = os.path.join(td, "empty_dir")            # exists, but no data
        os.makedirs(empty, exist_ok=True)

        idx = _synthetic_index(root_a)
        idx_path = os.path.join(td, "run0042.pidx")
        idx.save(idx_path)

        raised = None
        try:
            IX.RunIndex.load(idx_path, dir=empty)
        except FileNotFoundError as e:
            raised = str(e)
        assert raised is not None, \
            "load(dir=<dir lacking the data>) must raise, not load a bad path"
        # the error names a concrete missing file (one of the run's files).
        assert any(os.path.basename(r) in raised for r in _REL_LAYOUT), \
            "the error must name the missing file: %r" % (raised,)
    print("[wrong-dir] a dir= lacking the data raises a clear, naming error")


def main():
    test_relocate_via_dir_override()
    print("[ok] relocate via dir= override")
    test_relocate_beside_index_no_dir()
    print("[ok] relocate beside the index (no override)")
    test_original_location_byteexact()
    print("[ok] original-location load is byte-exact")
    test_wrong_dir_raises_clear_error()
    print("[ok] wrong dir= fails loudly")
    print("\nALL IDX-03 TESTS PASSED")


if __name__ == "__main__":
    main()
