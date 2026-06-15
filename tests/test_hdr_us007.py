#!/usr/bin/env python3
"""US-007 acceptance test: standalone offline calibrated 2-D HDR image render.

Verifies the US-007 acceptance criteria for the reference Jungfrau dataset
(exp=mfx100848724, run=51, dir=/sdf/data/lcls/ds/prj/public01/xtc, det=jungfrau):

  1. Given a raw (N,512,1024) uint16 stack plus US-006's cached constants, the
     module produces a calibrated 2-D image fully offline (no web DB, no MPI):
       * gain decode vendored from psana UtilsJungfrau.calib_jungfrau*;
       * geometry index maps from GeometryAccess (one-time, snapshot-time);
       * remap vendored from psana UtilsAreaDetector (mapmode=2 / fillholes).

  2. The produced calib (32,512,1024) f32 and image (4216,4432) f32 match
     psana's det.raw.calib(evt) / det.raw.image(evt) with max|diff| == 0
     (np.array_equal).

  3. The render APPLY path is fully standalone: in a fresh interpreter, loading
     a snapshot with cached index maps and rendering raw->calib->image pulls in
     NO psana / mpi4py / h5py (only numpy).  (The one psana touch -- deriving
     the index maps from geometry text -- is a one-time snapshot-time prep.)

This test needs the PRODUCTION psana env (psconda.sh install) to GENERATE the
snapshot + ground truth -- run on sdfiana025 via
``psdata/run_tests.sh psdata/tests/test_hdr_us007.py``.  The pure-numpy import
checks run without psana; the byte-exact checks skip cleanly (with a message)
if psana is not importable.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

# --- locate the package (parent of this tests dir) --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(os.path.dirname(_HERE))   # .../<repo> (holds psdata/)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# Reference dataset -- lives in the TEST, never in the library.
EXP = "mfx100848724"
RUN = 51
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
DET = "jungfrau"


def _have_psana():
    try:
        import psana  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# 1. import purity of the RENDER (apply) path -- pure numpy, no psana
# --------------------------------------------------------------------------
def test_render_import_purity_in_proc():
    """Importing psdata.hdr (gain decode + image remap + render engine) must
    not pull in a framework.  geometry derivation imports psana lazily, on
    call only."""
    import psdata.hdr as hdr
    _ = hdr.HDRImager, hdr.calib_jungfrau, hdr.assemble_image
    for m in ("psana", "mpi4py", "h5py"):
        assert m not in sys.modules, f"{m} leaked into sys.modules on import"


def test_render_import_purity_subprocess(snapshot_dir=None):
    """In a fresh interpreter, importing psdata.hdr AND (optionally) doing a
    full offline render leaves sys.modules psana-free and pulls in numpy."""
    render_stmt = ""
    if snapshot_dir:
        render_stmt = (
            "import numpy as np; "
            f"from psdata.calib import load_snapshot; "
            f"snap=load_snapshot({snapshot_dir!r}); "
            "im=psdata.hdr.HDRImager(snap, derive_geometry_if_missing=False); "
            # synthetic raw: shape from the cached mask, dtype uint16
            "nseg=snap.mask.shape[0]; "
            "raw=np.zeros((nseg,512,1024), dtype=np.uint16); "
            "calib=im.calib(raw); img=im.image(calib); "
            "assert calib.shape==(nseg,512,1024) and calib.dtype==np.float32; "
            "assert img.ndim==2 and img.dtype==np.float32; "
        )
    code = (
        "import sys, psdata.hdr; "
        + render_stmt +
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
# 2. byte-exact calib + image vs psana (regenerates GT itself)
# --------------------------------------------------------------------------
def test_render_byte_exact(out_dir):
    """Snapshot the reference run + cache index maps (psana), then render
    raw->calib->image fully offline and assert byte-identical to psana.

    Returns the snapshot dir so the caller can reuse it for the fresh-interp
    offline-render purity check."""
    import psdata.calib as pc
    import psdata.hdr as hdr
    import psdata.hdr.geometry as hgeo

    # --- regenerate psana ground truth ourselves ------------------------
    from psana import DataSource
    ds = DataSource(exp=EXP, run=RUN, dir=DIR)
    myrun = next(ds.runs())
    det = myrun.Detector(DET)
    evt = next(myrun.events())
    gt_raw = np.asarray(det.raw.raw(evt))
    gt_calib = np.asarray(det.raw.calib(evt))
    gt_image = np.asarray(det.raw.image(evt))
    assert gt_raw.shape == (32, 512, 1024) and gt_raw.dtype == np.uint16
    assert gt_calib.shape == (32, 512, 1024) and gt_calib.dtype == np.float32
    print(f"[gt] raw {gt_raw.shape} calib {gt_calib.shape} image {gt_image.shape}")

    # --- one-time snapshot of constants (US-006) ------------------------
    snap_dir = pc.snapshot_calib(exp=EXP, run=RUN, dir=DIR, detname=DET,
                                 out_dir=out_dir)
    # --- one-time geometry index-map cache (the single psana touch) -----
    ix_path, iy_path = hgeo.cache_pixel_indexes_for_snapshot(snap_dir)
    print(f"[prep] cached index maps:\n  {ix_path}\n  {iy_path}")

    # --- everything below is the pure-numpy offline render --------------
    snap = pc.load_snapshot(snap_dir)
    imager = hdr.HDRImager(snap, derive_geometry_if_missing=False)
    print(f"[render] {imager!r}")

    my_calib = imager.calib(gt_raw)
    assert my_calib.shape == (32, 512, 1024), my_calib.shape
    assert my_calib.dtype == np.float32, my_calib.dtype
    dcal = np.abs(np.nan_to_num(my_calib) - np.nan_to_num(gt_calib))
    assert np.array_equal(my_calib, gt_calib), (
        f"calib not byte-exact: max|diff|={dcal.max()}")
    print(f"[byte-exact] calib {my_calib.shape} {my_calib.dtype} "
          f"max|diff|={dcal.max()} array_equal=True")

    my_image = imager.image(my_calib)
    assert my_image.ndim == 2 and my_image.dtype == np.float32
    di = np.abs(np.nan_to_num(my_image) - np.nan_to_num(gt_image))
    assert np.array_equal(my_image, gt_image), (
        f"image not byte-exact: max|diff|={di.max()}")
    print(f"[byte-exact] image {my_image.shape} {my_image.dtype} "
          f"max|diff|={di.max()} array_equal=True")

    # render() convenience == the two steps
    c2, i2 = imager.render(gt_raw)
    assert np.array_equal(c2, my_calib) and np.array_equal(i2, my_image)

    # index maps derived from geometry text == psana _pixel_coord_indexes
    pix = det.raw._pixel_coord_indexes()
    gt_ix, gt_iy = np.asarray(pix[0]), np.asarray(pix[1])
    assert np.array_equal(imager.ix, gt_ix), "ix != psana _pixel_coord_indexes"
    assert np.array_equal(imager.iy, gt_iy), "iy != psana _pixel_coord_indexes"
    print("[geo] cached ix/iy == det.raw._pixel_coord_indexes() (byte-exact)")

    return snap_dir


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("US-007 acceptance: standalone offline calibrated 2-D HDR image")
    print("=" * 72)

    # render-path import purity always runs (no psana needed)
    test_render_import_purity_in_proc()
    print("[ok] render import purity (in-proc)")
    test_render_import_purity_subprocess()
    print("[ok] render import purity (subprocess, no snapshot)")

    if not _have_psana():
        print("\n[skip] psana not importable -- snapshot/byte-exact checks "
              "skipped. Source psconda.sh on sdfiana025 to run them.")
        print("\nUS-007 render-purity checks PASSED (psana-dependent checks "
              "skipped)")
        return

    import shutil
    tmp_parent = tempfile.mkdtemp(prefix="psdata_hdr_us007_outer_")
    try:
        snap_dir = test_render_byte_exact(out_dir=tmp_parent)
        print("[ok] offline render calib + image byte-exact vs psana "
              "(max|diff| == 0)")
        # full offline render in a fresh interpreter -> stays psana-free
        test_render_import_purity_subprocess(snapshot_dir=snap_dir)
        print("[ok] render import purity (subprocess, full offline render)")
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)

    print("\nALL US-007 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
