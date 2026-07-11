#!/usr/bin/env python3
"""IDX-02 -- the persisted index must be a SAFE on-disk format, not a pickle.

Bug (from the project bug matrix):

    IDX-02 -- The persisted index is a bare ``pickle`` -- no version, no magic,
    no checksum, and it executes code on load (an RCE vector on a shared
    analysis filesystem).

psdata's headline feature is a persisted, shareable index artifact that is
written ONCE and re-read by MANY worker processes off ``/sdf`` -- a shared
multi-user analysis filesystem.  ``pickle.load`` executes arbitrary code baked
into the file it reads, so a bare-pickle index lets any user who can write a
directory a victim's job loads from run code inside that victim's job.  A bare
pickle also has no magic, no format version, and no integrity check, so a
truncated / corrupt / format-drifted index crashes inscrutably or -- worse --
loads something wrong.

This suite is the pre-fix/post-fix discriminator for the disk format.  It is
**fully self-contained**: stdlib + numpy only, NO psana, NO SLAC data -- it
constructs the index's data structures directly, so it runs anywhere.  It
contains NO part of the fix; it only exercises the public
``RunIndex.save`` / ``RunIndex.load`` contract.

What it proves:

  1. THE MONEY ASSERTION -- ``load`` REFUSES a file whose pickled payload would
     execute code, and that code does NOT run.  On the pre-fix (bare-pickle)
     ``load`` this gadget fires (a sentinel file is written) -> this test fails;
     after the fix ``load`` checks magic before interpreting anything and never
     unpickles -> the sentinel is never created -> this test passes.
  2. A benign old-format bare pickle (the exact dict the pre-fix ``save`` wrote)
     is likewise refused -- silently trusting a pickle is the vulnerability, so
     backward-compat is deliberately NOT offered; the error tells the user to
     rebuild.
  3. Magic + format version are enforced (wrong magic / bumped version -> clear
     error naming the problem).
  4. Integrity is enforced (flipping one payload byte is detected).
  5. A save/load round-trip of a synthetic index is faithful -- every persisted
     field, including the ragged per-event ``entries`` and the nested
     ``RunConfig`` (tuple-keyed tables, segment sets, ``FieldInfo`` objects),
     survives byte-for-byte.

Run (numpy only; no psana needed):
    python3 tests/test_index_format_idx02.py
"""

import os
import pickle
import struct
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
# Synthetic index -- built directly, NO xtc2 file is read.
# ==========================================================================
def _synthetic_index():
    """A small but structurally-complete :class:`RunIndex`, constructed in
    memory.  Exercises every shape the persisted state can take: 64-bit
    timestamps, the RAGGED per-event ``entries`` (different stream sets per
    event -- the object-array trap), ``env_records`` (nested dict -> list of
    tuples), tuple-keyed config tables, segment *sets*, and a ``FieldInfo``."""
    rc = F.RunConfig()
    rc.stream_files = {0: "/data/exp-r0007-s000-c000.xtc2",
                       4: "/data/exp-r0007-s004-c000.xtc2"}
    # stream_configs: {stream: (parse_dgram_header-style dict, cfg_end)}
    rc.stream_configs = {
        0: ({"service": 2, "env": 0, "ts": 0x0001_0000_0002, "sec": 1,
             "nsec": 2, "src": 7, "damage": 0, "typeid": 0, "extent": 100},
            4096),
    }
    # raw_tables: {stream: {(nodeId, namesId): names_table}} -- tuple keys +
    # a names list of dicts (one rank-3 array field, one rank-0 scalar field).
    table = {"det_type": "jungfrau", "det_name": "jungfrau", "det_id": "SN0001",
             "alg_name": "raw", "alg_version": 1, "segment": 0, "num_arrays": 1,
             "names": [{"name": "raw", "type": 1, "rank": 3},
                       {"name": "intOffset", "type": 3, "rank": 0}]}
    rc.raw_tables = {0: {(2, 1): table}, 4: {}}
    det = F.DetectorInfo("jungfrau", "jungfrau", "SN0001")
    det._add_table("raw", table, 0, (2, 1))     # builds algs/segments/seg_to_stream
    rc.detectors = {"jungfrau": det}

    idx = IX.RunIndex(rc)
    idx.timestamps = [0x1234_5678_9abc, 0x1234_5678_9abd, 0x1234_5678_9abe]
    cp0 = "/data/exp-r0007-s000-c000.xtc2"
    cp4 = "/data/exp-r0007-s004-c000.xtc2"
    idx.entries = [
        {0: (cp0, 4096, 5_000_000), 4: (cp4, 4096, 128)},   # two streams
        {0: (cp0, 5_004_096, 5_000_000)},                   # ragged: one stream
        {0: (cp0, 10_004_096, 5_000_000), 4: (cp4, 4224, 128)},
    ]
    idx.env_records = {"epics": {4: [(0x1234_5678_9abc, cp4, 4224, 128),
                                     (0x1234_5678_9abd, cp4, 4352, 128)]}}
    idx.bd_files = dict(rc.stream_files)
    idx.chunk_files = {0: [cp0], 4: [cp4]}
    idx.multichunk_streams = set()
    idx.build_seconds = 2.5
    idx.smd_bytes_read = 4_242
    idx.scan_source = "smd"
    idx.scan_bytes_read = 4_242
    idx.include_shutdown_tail = False
    return idx


def _save_valid(td, name="idx.bin"):
    """Save a synthetic index in the current (fixed) on-disk format and return
    (path, raw_bytes)."""
    idx = _synthetic_index()
    path = os.path.join(td, name)
    idx.save(path)
    with open(path, "rb") as fh:
        raw = fh.read()
    return path, raw


# ==========================================================================
# 1. THE money assertion: no arbitrary code execution on load.
# ==========================================================================
class _ExecGadget:
    """Its pickle, when loaded, calls ``exec`` on an attacker-chosen string --
    the classic ``__reduce__`` RCE gadget.  Here the payload merely writes a
    sentinel file, so the test can PROVE whether code ran without doing harm."""

    def __init__(self, sentinel_path):
        self._sentinel = sentinel_path

    def __reduce__(self):
        return (exec, ("open(%r, 'w').write('pwned')" % self._sentinel,))


def test_load_refuses_malicious_pickle_and_runs_no_code():
    """A file whose pickled payload would execute code must be REFUSED by
    ``load``, and the code must NOT run.  This is the whole point of IDX-02:
    on the pre-fix bare-pickle ``load`` this gadget fires (sentinel created)."""
    with tempfile.TemporaryDirectory() as td:
        sentinel = os.path.join(td, "PWNED")
        malicious = os.path.join(td, "malicious.index")
        with open(malicious, "wb") as fh:
            fh.write(pickle.dumps(_ExecGadget(sentinel)))
        assert not os.path.exists(sentinel), "precondition: sentinel absent"

        refused = False
        try:
            IX.RunIndex.load(malicious)
        except Exception:
            refused = True   # any refusal is fine; the sentinel is the proof

        # THE money assertion: the embedded code did NOT execute.
        assert not os.path.exists(sentinel), (
            "RCE: RunIndex.load EXECUTED code embedded in the index file "
            "(the __reduce__ gadget created the sentinel) -- the persisted "
            "index must not be a pickle")
        assert refused, "load must refuse a non-psdata-format index file"
    print("[rce] load refuses a malicious pickle; embedded code did not run")


def test_load_refuses_benign_old_bare_pickle():
    """An old-format index (the exact dict the pre-fix ``save`` wrote, as a
    plain pickle) is REFUSED with a clear, actionable error -- backward-compat
    is deliberately not offered, because silently trusting a pickle is the
    vulnerability."""
    with tempfile.TemporaryDirectory() as td:
        idx = _synthetic_index()
        old = os.path.join(td, "old_format.index")
        with open(old, "wb") as fh:
            pickle.dump(idx._persist_state(), fh, protocol=4)  # the OLD format
        try:
            IX.RunIndex.load(old)
        except ValueError as e:
            msg = str(e).lower()
            assert "magic" in msg or "pickle" in msg or "rebuild" in msg, \
                "the refusal must be a clear, actionable error: %r" % (e,)
        else:
            raise AssertionError(
                "load accepted an old bare-pickle index -- it must refuse it")
    print("[compat] old bare-pickle index refused with an actionable error")


# ==========================================================================
# 2. Magic + format version are enforced.
# ==========================================================================
def test_load_rejects_bad_magic():
    with tempfile.TemporaryDirectory() as td:
        path, raw = _save_valid(td)
        bad = os.path.join(td, "badmagic.index")
        with open(bad, "wb") as fh:
            fh.write(b"NOTMAGIC" + raw[8:])   # clobber the 8-byte magic
        try:
            IX.RunIndex.load(bad)
        except ValueError as e:
            assert "magic" in str(e).lower(), \
                "wrong-magic error must name the magic mismatch: %r" % (e,)
        else:
            raise AssertionError("load accepted a file with wrong magic")
    print("[magic] a file with the wrong magic bytes is rejected")


def test_load_rejects_bumped_version():
    with tempfile.TemporaryDirectory() as td:
        path, raw = _save_valid(td)
        # layout: magic(8) | version uint32 LE | ...  -- bump the version.
        cur = struct.unpack("<I", raw[8:12])[0]
        bumped = raw[:8] + struct.pack("<I", cur + 1) + raw[12:]
        bad = os.path.join(td, "badver.index")
        with open(bad, "wb") as fh:
            fh.write(bumped)
        try:
            IX.RunIndex.load(bad)
        except ValueError as e:
            m = str(e).lower()
            assert "version" in m and str(cur + 1) in str(e), (
                "version error must name the found vs expected version: %r"
                % (e,))
        else:
            raise AssertionError("load accepted a bumped-version file")
    print("[version] a file with an unknown format version is rejected")


# ==========================================================================
# 3. Integrity is enforced.
# ==========================================================================
def test_load_detects_payload_corruption():
    with tempfile.TemporaryDirectory() as td:
        path, raw = _save_valid(td)
        assert IX.RunIndex.load(path).n_events == 3, "sanity: valid file loads"
        # flip one bit in the LAST byte (always inside the payload region).
        corrupt = bytearray(raw)
        corrupt[-1] ^= 0x01
        bad = os.path.join(td, "corrupt.index")
        with open(bad, "wb") as fh:
            fh.write(bytes(corrupt))
        try:
            IX.RunIndex.load(bad)
        except ValueError as e:
            m = str(e).lower()
            assert "integrity" in m or "corrupt" in m or "payload" in m, \
                "a corrupted payload must raise a clear integrity error: %r" \
                % (e,)
        else:
            raise AssertionError("load accepted a payload with a flipped byte")
    print("[integrity] a single flipped payload byte is detected on load")


# ==========================================================================
# 4. Faithful round-trip.
# ==========================================================================
def test_save_load_roundtrip_faithful():
    """save() then load() reconstructs every persisted field exactly, and the
    two per-process caches (_bd_fds / _ts_to_k) come back empty."""
    with tempfile.TemporaryDirectory() as td:
        idx = _synthetic_index()
        path = os.path.join(td, "roundtrip.index")
        idx.save(path)
        back = IX.RunIndex.load(path)

    # the whole persisted state, field by field
    assert back.n_events == idx.n_events == 3
    assert back.timestamps == idx.timestamps
    assert back.entries == idx.entries, (back.entries, idx.entries)
    assert back.env_records == idx.env_records
    assert back.bd_files == idx.bd_files
    assert back.chunk_files == idx.chunk_files
    assert back.multichunk_streams == idx.multichunk_streams
    assert back.build_seconds == idx.build_seconds
    assert back.smd_bytes_read == idx.smd_bytes_read
    assert back.scan_source == idx.scan_source
    assert back.scan_bytes_read == idx.scan_bytes_read
    assert back.include_shutdown_tail == idx.include_shutdown_tail
    # per-process caches reset (never carry stale fds across a reload)
    assert back._bd_fds == {} and back._ts_to_k is None

    # the nested RunConfig survives with the right TYPES, not just values
    brc = back.run_config
    assert isinstance(brc, F.RunConfig)
    assert brc.stream_files == idx.run_config.stream_files
    assert brc.stream_configs == idx.run_config.stream_configs
    assert brc.raw_tables == idx.run_config.raw_tables      # tuple keys intact
    assert (2, 1) in brc.raw_tables[0], "tuple config-table key lost"
    bdet = brc.detector("jungfrau")
    odet = idx.run_config.detector("jungfrau")
    assert isinstance(bdet, F.DetectorInfo)
    assert bdet.seg_to_stream == odet.seg_to_stream          # tuple keys
    assert bdet.names_id == odet.names_id
    assert bdet.segments == odet.segments                    # sets
    assert bdet.seg_detids == odet.seg_detids
    fi = bdet.algs["raw"]["raw"]
    assert isinstance(fi, F.FieldInfo)
    assert (fi.name, fi.type_code, fi.rank) == ("raw", 1, 3)
    assert fi.np_dtype == np.dtype(F.DTYPE_NP[1]), "FieldInfo dtype not restored"
    print("[roundtrip] every persisted field + nested RunConfig survives faithfully")


def main():
    test_load_refuses_malicious_pickle_and_runs_no_code()
    print("[ok] no code execution on load (RCE gone)")
    test_load_refuses_benign_old_bare_pickle()
    print("[ok] old bare-pickle index refused")
    test_load_rejects_bad_magic()
    print("[ok] bad magic rejected")
    test_load_rejects_bumped_version()
    print("[ok] bumped format version rejected")
    test_load_detects_payload_corruption()
    print("[ok] payload corruption detected")
    test_save_load_roundtrip_faithful()
    print("[ok] round-trip faithful")
    print("\nALL IDX-02 TESTS PASSED")


if __name__ == "__main__":
    main()
