#!/usr/bin/env python3
"""US-004 acceptance: robustness -- multi-chunk roll, damage, missing segment.

Run on sdfiana025 with the production psana for cross-checks:

    source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
    bash psdata/run_tests.sh <abs path to this file>

What this verifies (US-004 acceptance criteria):

  1. **Multi-chunk roll** -- on each Enable the reader follows ``chunkinfo`` and
     rolls the bigdata file from ``c000`` to ``c001+``; events spanning the
     chunk boundary read **byte-identical** to psana across it.  Verified on a
     real multi-chunk run (mfx101589626 / mfx101343025), comparing
     random-access ``RunIndex.read_event`` raw arrays to psana ground truth
     captured around the boundary.

  2. **Missing segment -> detector None** -- a detector whose received segment
     set differs from its declared set is returned as ``None`` for that event
     (psana ``detector_impl.py:_segments`` rule), exercised by synthetically
     dropping a segment.

  3. **Damage surfaced, not dropped** -- per-segment ``Xtc.damage`` is decoded
     (id = low 12 bits, userbits = value>>12) and surfaced via
     ``Event.damage`` / ``decode_damage``; the raw array is still returned.

The multi-chunk ground truth is produced by ``tools/gt_capture.py`` (psana) and
saved as ``gt_mc35_*.npy`` + ``gt_mc35_manifest.json``; this test loads those and
compares.  Capture it once, from the repo root, in the psana env::

    source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
    PYTHONPATH=src python3 tools/gt_capture.py --out-dir .

If the ground truth is absent the multi-chunk psana comparison emits a SKIP
record -- which is NOT in ``tests/skips_allowed.txt`` and therefore FAILS the
suite (HYG-03: a skipped oracle is not a passing oracle).  The SMD-only
structural roll checks still run and report independently.
"""

import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for _skips

import psdata
from psdata import format as psf
from psdata import index as psindex
from psdata import stream as psstream

from _skips import skip   # machine-readable skip records (HYG-03)

# ---- multi-chunk reference run (located & verified for US-004) -------------
MC_DIR = "/sdf/data/lcls/ds/mfx/mfx101343025/xtc"
MC_EXP = "mfx101343025"
MC_RUN = 35
# psana ground-truth captured around the s007 chunk boundary by gt_capture.py.
RUNDIR = os.path.dirname(os.path.abspath(__file__)) + "/.."  # not used; GT in cwd
GT_MANIFEST = os.environ.get(
    "PSDATA_GT_MANIFEST",
    os.path.join(os.getcwd(), "gt_mc35_manifest.json"))


def _mc_stream_files():
    """The multi-chunk run's per-stream c000 bigdata files, keyed by real
    stream index (so jungfrau segments map to the right streams)."""
    import glob
    paths = sorted(glob.glob(os.path.join(
        MC_DIR, f"{MC_EXP}-r{MC_RUN:04d}-s*-c000.xtc2")))
    paths = psdata.filter_c000(paths)
    return {psdata.stream_index_of(p): p for p in paths}


def _mc_smd_files(stream_files):
    return psindex.smd_files_for(stream_files)


# ==========================================================================
# 0. import purity
# ==========================================================================
def test_import_purity():
    psf.assert_no_framework_imports()
    code = (
        "import sys; import psdata; "
        "bad=[m for m in ('psana','mpi4py','h5py') if m in sys.modules]; "
        "assert not bad, bad; print('clean')")
    env = dict(os.environ)
    out = subprocess.check_output([sys.executable, "-c", code], env=env)
    assert out.strip() == b"clean", out
    print("[ok] import purity (no psana/mpi4py/h5py)")


# ==========================================================================
# 1. damage decode + Event.damage surfacing (unit + structural)
# ==========================================================================
def test_damage_decode_unit():
    # id = low 12 bits, userbits = value>>12 (DAMAGE_USERBITSHIFT=12)
    assert psdata.decode_damage(0) == (0, 0)
    assert psf.DAMAGE_USERBITSHIFT == 12
    assert psf.DAMAGE_VALUEBITMASK == 0x0FFF
    # damage id 6 (MissingData), userbits 0
    assert psdata.decode_damage(6) == (6, 0)
    # userbits 1 (bit 12), id 4 (Corrupted): value = (1<<12)|4
    assert psdata.decode_damage((1 << 12) | 4) == (4, 1)
    # full: userbits 0xF, id 0xFFF
    assert psdata.decode_damage(0xFFFF) == (0x0FFF, 0xF)
    print("[ok] decode_damage: id=val&0xfff, userbits=val>>12")


def test_event_damage_surface():
    """Event.damage returns {seg: (id, userbits)} for a contributing detector,
    and a clean event reports id 0 for every segment (damage surfaced, raw
    still returned -- not dropped)."""
    files = _mc_stream_files()
    rc = psdata.discover(files)
    jdet = "jungfrau"
    n = 0
    for evt in psdata.events(files, run_config=rc):
        dmg = evt.damage(jdet)
        if dmg is None:
            continue
        raw = evt.raw(jdet)
        # damage is surfaced per segment; raw is still present (not dropped)
        assert raw is not None
        for seg, (did, ub) in dmg.items():
            assert isinstance(did, int) and isinstance(ub, int)
        # is_damaged consistent with per-seg ids
        assert evt.is_damaged(jdet) == any(d != 0 for (d, _u) in dmg.values())
        n += 1
        if n >= 3:
            break
    assert n > 0, "no jungfrau events seen to check damage surface"
    print(f"[ok] Event.damage surfaces per-segment (id,userbits); "
          f"raw still returned (checked {n} events)")


# ==========================================================================
# 2. missing-segment -> detector None (synthetic, psana rule)
# ==========================================================================
def test_missing_segment_none():
    """Drop one declared segment from an assembled event's seg-index and verify
    the detector is reported None (received set != declared set), matching
    psana detector_impl.py:_segments."""
    files = _mc_stream_files()
    rc = psdata.discover(files)
    jdet = "jungfrau"
    declared = sorted(rc.detector(jdet).segments["raw"])
    for evt in psdata.events(files, run_config=rc):
        full = evt.raw(jdet)
        if full is None:
            continue
        assert sorted(full) == declared
        # now drop one segment from the captured index and re-check
        key = (jdet, "raw")
        captured = evt._seg_index[key]
        victim = sorted(captured)[0]
        saved = captured.pop(victim)
        evt._pulseid_cache = None
        assert evt.raw(jdet) is None, \
            "dropping a segment must make the detector None"
        captured[victim] = saved  # restore
        assert evt.raw(jdet) is not None
        print(f"[ok] missing-segment -> None (dropped seg {victim} of "
              f"{len(declared)} -> detector None; restored -> present)")
        return
    raise AssertionError("no complete jungfrau event found to test")


# ==========================================================================
# 3. multi-chunk roll: structural + byte-identical to psana across boundary
# ==========================================================================
def test_chunk_roll_structural():
    """The index follows the chunkinfo roll: at least one jungfrau stream rolls
    through >1 chunk file, and the per-event entry's chunk path switches from
    c000 to c001 exactly where intOffset resets."""
    files = _mc_stream_files()
    rc = psdata.discover(files)
    smd = _mc_smd_files(files)
    ridx = psindex.RunIndex.build(smd, rc)
    print(f"[build] {ridx!r}")
    print(f"[build] chunk_files per stream: "
          f"{ {s: [os.path.basename(p) for p in c] for s, c in ridx.chunk_files.items()} }")
    assert ridx.multichunk_streams, \
        "expected at least one multi-chunk stream in this run"
    # for a rolled stream, find the event index where its chunk path changes
    rolled_stream = sorted(ridx.multichunk_streams)[0]
    chunks = ridx.chunk_files[rolled_stream]
    assert len(chunks) >= 2
    prev = None
    switch_k = None
    for k, entry in enumerate(ridx.entries):
        if rolled_stream not in entry:
            continue
        path = entry[rolled_stream][0]
        if prev is not None and path != prev:
            switch_k = k
            break
        prev = path
    assert switch_k is not None, "no chunk path switch found in entries"
    # at the switch, offset resets to a small value (new chunk starts at 0)
    _p, off_after, _s = ridx.entries[switch_k][rolled_stream]
    assert off_after < 1 << 30, \
        f"offset after roll should be near start of new chunk, got {off_after}"
    print(f"[ok] chunk roll: stream {rolled_stream} switches "
          f"{os.path.basename(chunks[0])} -> {os.path.basename(chunks[1])} "
          f"at event k={switch_k}, offset resets to {off_after}")
    ridx.close()
    return ridx, switch_k


def test_chunk_roll_vs_psana():
    """Byte-identical to psana across the chunk boundary: load the psana ground
    truth captured around the boundary and compare random-access raw arrays."""
    if not os.path.exists(GT_MANIFEST):
        return skip(
            "multichunk_psana_oracle",
            f"multi-chunk psana ground truth not found at {GT_MANIFEST}; "
            f"the byte-exactness of reads ACROSS the chunk boundary is "
            f"therefore unverified. Capture it with: "
            f"PYTHONPATH=src python3 tools/gt_capture.py --out-dir . "
            f"(needs psana; run from the dir this test reads the manifest "
            f"from, or point PSDATA_GT_MANIFEST at it).")
    with open(GT_MANIFEST) as f:
        man = json.load(f)
    gt_dir = os.path.dirname(os.path.abspath(GT_MANIFEST))
    files = _mc_stream_files()
    rc = psdata.discover(files)
    smd = _mc_smd_files(files)
    ridx = psindex.RunIndex.build(smd, rc)

    checked = 0
    spanned_boundary = set()
    for i, ts, pid, shape, dtype in man["events"]:
        gt_path = os.path.join(gt_dir, f"gt_mc35_{i}.npy")
        if not os.path.exists(gt_path):
            continue
        gt = np.load(gt_path, allow_pickle=True)
        evt = ridx.read_event(ts)
        # which chunk did each contributing stream read from?
        entry = ridx.entries[ridx._position_of(ts)]
        chunks = {s: os.path.basename(entry[s][0]) for s in entry}
        spanned_boundary |= set(chunks.values())
        got = evt.stack("jungfrau")
        if gt is None or (hasattr(gt, "dtype") and gt.dtype == object):
            # psana returned None (missing) -> our reader should agree
            assert got is None, f"event {i}: psana None but psdata not None"
        else:
            assert got is not None, f"event {i}: psdata None but psana not"
            assert np.array_equal(got, gt), \
                f"event {i} ts={ts}: jungfrau raw != psana"
        # pulseId match
        assert int(evt.pulseId) == int(pid), \
            f"event {i}: pulseId {evt.pulseId} != psana {pid}"
        checked += 1
    ridx.close()
    assert checked > 0, "no ground-truth events matched the index"
    print(f"[ok] multi-chunk: {checked} events around the boundary "
          f"byte-identical to psana (raw + pulseId); chunk files spanned: "
          f"{sorted(spanned_boundary)}")


# ==========================================================================
# 4. c000-only file filter (mirrors ds_base.py initial filter)
# ==========================================================================
def test_filter_c000():
    sample = [
        "/x/mfx101343025-r0035-s000-c000.xtc2",
        "/x/mfx101343025-r0035-s007-c000.xtc2",
        "/x/mfx101343025-r0035-s007-c001.xtc2",
        "/x/mfx101343025-r0035-s008-c002.xtc2",
        "/x/tmoc00118-r0222-s000-c000.xtc2",   # 'c001' substring trap-free
    ]
    kept = psdata.filter_c000(sample)
    assert kept == [
        "/x/mfx101343025-r0035-s000-c000.xtc2",
        "/x/mfx101343025-r0035-s007-c000.xtc2",
        "/x/tmoc00118-r0222-s000-c000.xtc2",
    ], kept
    # stream index extraction
    assert psdata.stream_index_of("/x/mfx101343025-r0035-s007-c000.xtc2") == 7
    assert psdata.stream_index_of("/x/foo-c001.xtc2") is None
    print("[ok] filter_c000 keeps only -c000 files; stream_index_of parses s###")


def main():
    print("=" * 72)
    print("US-004 acceptance: multi-chunk roll, damage, missing segment")
    print("=" * 72)
    test_import_purity()
    test_damage_decode_unit()
    test_filter_c000()
    test_missing_segment_none()
    test_event_damage_surface()
    test_chunk_roll_structural()
    test_chunk_roll_vs_psana()
    print()
    print("ALL US-004 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
