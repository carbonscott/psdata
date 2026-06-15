#!/usr/bin/env python3
"""US-006 acceptance test: one-time calibration-constant snapshot + pinning.

Verifies the US-006 acceptance criteria for the reference Jungfrau dataset
(exp=mfx100848724, run=51, dir=/sdf/data/lcls/ds/prj/public01/xtc, det=jungfrau):

  1. ``psdata.calib.snapshot_calib`` caches, per (detector, run): pedestals,
     pixel_gain, pixel_offset, mask (from ``det.raw._calibconst`` and
     ``det.raw._mask(status=True)``) plus the geometry text.

  2. Snapshots are pinned by ``(detector_uniqueid, run)`` and retain the
     per-ctype validity metadata (run / run_end / version).  Reloading a
     snapshot reproduces the EXACT arrays psana's ``det.raw._calibconst``
     returns (``np.array_equal``), and the pin records which run the constants
     were taken for.

  3. For the reference Jungfrau dataset the snapshot contains:
       pedestals     (3,32,512,1024) f32
       pixel_gain    (3,32,512,1024) f32
       pixel_offset  (3,32,512,1024) f32
       mask          (32,512,1024)   u8
       geometry      ~5-8 KB text
     (leading 3 axis = the three gain stages, the HDR signature).

  4. The reload path (load_snapshot / CalibSnapshot) is pure numpy: in a fresh
     interpreter, loading a snapshot pulls in NO psana / mpi4py / h5py.

This test needs the PRODUCTION psana env (the psconda.sh install) to GENERATE
the snapshot and the ground truth -- run it on sdfiana025 via
``psdata/run_tests.sh psdata/tests/test_calib_us006.py``.  The byte-exact +
shape checks are skipped cleanly (with a message) if psana is not importable,
so the offline-reload import-purity check still runs without the prod env.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# --- locate the package (parent of this tests dir) --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../<repo>/src (holds psdata/)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# Reference dataset -- lives in the TEST, never in the library.
EXP = "mfx100848724"
RUN = 51
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
DET = "jungfrau"

# Per-acceptance expected shapes/dtypes of the named HDR constants.
EXPECT = {
    "pedestals":    ((3, 32, 512, 1024), np.float32),
    "pixel_gain":   ((3, 32, 512, 1024), np.float32),
    "pixel_offset": ((3, 32, 512, 1024), np.float32),
    "mask":         ((32, 512, 1024),    np.uint8),
}


def _have_psana():
    try:
        import psana  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# 1. import purity of the RELOAD path -- pure numpy, no psana
# --------------------------------------------------------------------------
def test_reload_import_purity_in_proc():
    """Importing psdata.calib + the reload API must not pull in a framework
    (snapshot_calib imports psana lazily, on call only)."""
    import psdata.calib as pc
    # touch the reload surface
    _ = pc.load_snapshot, pc.CalibSnapshot
    pc.assert_no_framework_imports()
    for m in ("psana", "mpi4py", "h5py"):
        assert m not in sys.modules, f"{m} leaked into sys.modules on import"


def test_reload_import_purity_subprocess(snapshot_dir=None):
    """In a fresh interpreter, importing psdata.calib AND reloading a snapshot
    leaves sys.modules psana-free and pulls in numpy."""
    load_stmt = (
        f"snap=psdata.calib.load_snapshot({snapshot_dir!r}); "
        "assert snap.pedestals is not None or snap.run is not None; "
    ) if snapshot_dir else ""
    code = (
        "import sys, psdata.calib; "
        + load_stmt +
        "psdata.calib.assert_no_framework_imports(); "
        "bad=[m for m in ('psana','mpi4py','h5py') if m in sys.modules]; "
        "assert not bad, bad; "
        "assert 'numpy' in sys.modules, 'numpy should be imported'; "
        "print('CLEAN')"
    )
    env = dict(os.environ, PYTHONPATH=_PKG_PARENT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "CLEAN" in out.stdout, out.stdout


# --------------------------------------------------------------------------
# 2 + 3. snapshot, reload, byte-exact vs psana; shapes; pin + validity
# --------------------------------------------------------------------------
def test_snapshot_reload_byte_exact(out_dir):
    """Snapshot the reference Jungfrau run, reload offline, and assert the
    reloaded arrays are byte-identical to psana's _calibconst / _mask, with the
    expected shapes, a correct pin, and retained validity metadata.

    Writes the snapshot under ``out_dir`` (owned/cleaned by the caller) and
    returns the snapshot directory path so the caller can reuse it for the
    fresh-interpreter reload-purity check."""
    import psdata.calib as pc

    # --- regenerate psana ground truth ourselves -------------------------
    from psana import DataSource
    ds = DataSource(exp=EXP, run=RUN, dir=DIR)
    myrun = next(ds.runs())
    det = myrun.Detector(DET)
    gt_cc = det.raw._calibconst                  # {ctype:(ndarray|str, meta)}
    gt_mask = np.asarray(det.raw._mask(status=True))
    gt_uniqueid = det.raw._uniqueid

    assert gt_cc is not None, "psana _calibconst is None (DB unreachable?)"
    for ctype in ("pedestals", "pixel_gain", "pixel_offset"):
        assert ctype in gt_cc, f"psana did not return {ctype!r}"

    # --- take the snapshot (the only psana-using step) -------------------
    snap_dir = pc.snapshot_calib(exp=EXP, run=RUN, dir=DIR, detname=DET,
                                 out_dir=out_dir)
    assert os.path.isdir(snap_dir)
    print(f"[snapshot] wrote {snap_dir}")

    # --- everything below is the pure-numpy reload + cross-check ---------
    if True:
        # manifest sanity: pinned dir name
        assert os.path.basename(snap_dir) == f"{DET}_r{RUN:04d}", snap_dir

        # --- reload offline ----------------------------------------------
        snap = pc.load_snapshot(snap_dir)
        print(f"[reload] {snap!r}")

        # --- 2: pin records (detector_uniqueid, run) ---------------------
        assert snap.run == RUN, snap.run
        assert snap.detname == DET, snap.detname
        assert snap.detector_uniqueid == gt_uniqueid, (
            "pin uniqueid != psana _uniqueid")
        assert snap.exp == EXP, snap.exp
        print(f"[pin] (uniqueid={snap.detector_uniqueid[:24]}..., run={snap.run})")

        # --- 2: byte-exact reload of EVERY ndarray ctype -----------------
        rebuilt = snap.calibconst()
        for ctype, (gt_arr, _gt_meta) in gt_cc.items():
            if isinstance(gt_arr, np.ndarray):
                got = snap.array(ctype)
                assert got is not None, f"snapshot dropped ndarray ctype {ctype!r}"
                assert got.shape == gt_arr.shape, (ctype, got.shape, gt_arr.shape)
                assert got.dtype == gt_arr.dtype, (ctype, got.dtype, gt_arr.dtype)
                assert np.array_equal(got, gt_arr), (
                    f"byte mismatch for {ctype!r}: "
                    f"max|diff|={np.abs(got.astype('float64') - gt_arr.astype('float64')).max()}")
                # and via the reconstructed calibconst dict
                assert np.array_equal(rebuilt[ctype][0], gt_arr)
                print(f"[byte-exact] {ctype:14s} {got.shape} {got.dtype} array_equal=True")
            elif isinstance(gt_arr, str):
                # geometry text: byte-exact
                assert snap.geometry == gt_arr, "geometry text mismatch"
                assert rebuilt[ctype][0] == gt_arr
                print(f"[byte-exact] {ctype:14s} text len={len(snap.geometry)} equal=True")

        # --- 2: mask byte-exact ------------------------------------------
        assert snap.mask is not None, "snapshot did not cache the mask"
        assert np.array_equal(snap.mask, gt_mask), "mask byte mismatch"
        print(f"[byte-exact] {'mask':14s} {snap.mask.shape} {snap.mask.dtype} array_equal=True")

        # --- 3: required shapes/dtypes for the reference Jungfrau --------
        for ctype, (shape, dtype) in EXPECT.items():
            arr = snap.array(ctype) if ctype != "mask" else snap.mask
            assert arr is not None, f"required ctype {ctype!r} missing"
            assert arr.shape == shape, (ctype, arr.shape, shape)
            assert arr.dtype == dtype, (ctype, arr.dtype, dtype)
        # leading 3 axis == 3 gain stages (HDR signature)
        assert snap.pedestals.shape[0] == 3, snap.pedestals.shape
        # geometry text present and of the expected order of magnitude
        assert snap.geometry is not None and 1000 < len(snap.geometry) < 20000, (
            f"geometry text len={None if snap.geometry is None else len(snap.geometry)}")
        print("[shapes] pedestals/pixel_gain/pixel_offset (3,32,512,1024) f32, "
              "mask (32,512,1024) u8, geometry text -- OK")

        # --- 2: validity metadata retained per ctype --------------------
        with open(os.path.join(snap_dir, pc.MANIFEST_NAME)) as fh:
            manifest = json.load(fh)
        for ctype, (gt_arr, gt_meta) in gt_cc.items():
            v = snap.validity(ctype)
            assert "run" in v, f"validity missing 'run' for {ctype!r}"
            assert "run_end" in v, f"validity missing 'run_end' for {ctype!r}"
            assert "version" in v, f"validity missing 'version' for {ctype!r}"
            # the retained validity run must match what psana reported
            if isinstance(gt_meta, dict) and "run" in gt_meta:
                assert int(v["run"]) == int(gt_meta["run"]), (
                    ctype, v["run"], gt_meta["run"])
        print("[validity] per-ctype run/run_end/version retained and match psana")

        # the pin run (51) differs from a constant's validity run (e.g. 49) --
        # this is exactly the silent-staleness trap the acceptance calls out.
        peds_valid_run = int(snap.validity("pedestals")["run"])
        print(f"[pin vs validity] snapshot taken for run {snap.run}; "
              f"pedestals valid from run {peds_valid_run} "
              f"(<= {snap.run} -> pin run is covered)")
        assert snap.is_valid_for_run(RUN), (
            "every constant's validity range should cover the pin run")

    # The snapshot dir lives under the caller-owned out_dir; return it so the
    # caller can reuse it for the fresh-interpreter reload-purity check, then
    # clean the whole out_dir.
    return snap_dir


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("US-006 acceptance: calibration-constant snapshot + pinning")
    print("=" * 72)

    # reload-path import purity always runs (no psana needed)
    test_reload_import_purity_in_proc()
    print("[ok] reload import purity (in-proc)")
    test_reload_import_purity_subprocess()
    print("[ok] reload import purity (subprocess, no snapshot)")

    if not _have_psana():
        print("\n[skip] psana not importable -- snapshot/byte-exact checks "
              "skipped. Source psconda.sh on sdfiana025 to run them.")
        print("\nUS-006 reload-purity checks PASSED (psana-dependent checks "
              "skipped)")
        return

    tmp_parent = tempfile.mkdtemp(prefix="psdata_calib_us006_outer_")
    try:
        snap_dir = test_snapshot_reload_byte_exact(out_dir=tmp_parent)
        print("[ok] snapshot -> reload byte-exact vs psana (np.array_equal)")
        # reload import purity WITH an actual snapshot load in a fresh interp
        test_reload_import_purity_subprocess(snapshot_dir=snap_dir)
        print("[ok] reload import purity (subprocess, loads the snapshot)")
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)

    print("\nALL US-006 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
