#!/usr/bin/env python3
"""US-001 acceptance test for psdata.format.

Verifies the four acceptance criteria:

  1. Pure-Python parse core reusing psdata.py's functions/constants, ZERO
     psana import.  (Checked structurally + by import-purity.)
  2. Generic discovery from the Configure Names tables -- no hardcoded
     detector name, stream list, file path, or timestamp in the library.
  3. For the reference dataset (exp=mfx100848724, run=51,
     dir=/sdf/data/lcls/ds/prj/public01/xtc), extracting detector 'jungfrau'
     field 'raw' for one event yields a (32,512,1024) uint16 stack
     byte-identical (np.array_equal) to psana's det.raw.raw(evt).  The event
     timestamp is obtained from psana itself (evt.timestamp), not hardcoded.
  4. Importing psdata.format does not import psana, mpi4py, or h5py.

This test needs the production psana env to generate ground truth:
    source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
Run it on host sdfiana025.

The package import-purity (criterion 4) is asserted *before* psana is
imported, and is also independently checked in a fresh subprocess
(test_import_purity_subprocess) so a caller's later psana import can't mask it.
"""

import os
import subprocess
import sys

import numpy as np

# --- locate the package (parent of this tests dir) --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(os.path.dirname(_HERE))  # .../<repo>  (holds psdata/)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# Reference dataset -- lives in the TEST, never in the library.
EXP = "mfx100848724"
RUN = 51
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
DET = "jungfrau"
ALG = "raw"
FIELD = "raw"
EXPECTED_SHAPE = (32, 512, 1024)


def _stream_files():
    """Resolve the run's per-stream xtc2 files by globbing -- the test, not the
    library, knows the file-naming pattern."""
    import glob
    paths = sorted(glob.glob(f"{DIR}/{EXP}-r{RUN:04d}-s*-c000.xtc2"))
    assert paths, f"no stream files found under {DIR}"
    # map stream index from the -sNNN- field
    files = {}
    for p in paths:
        base = os.path.basename(p)
        sidx = int(base.split("-s")[1].split("-")[0])
        files[sidx] = p
    return files


def test_import_purity_before_psana():
    """psdata.format import must NOT have pulled in a framework."""
    # Importing here (psana not yet imported in this process at module top).
    from psdata import format as psformat
    psformat.assert_no_framework_imports()
    for m in ("psana", "mpi4py", "h5py"):
        assert m not in sys.modules, f"{m} leaked into sys.modules on import"


def test_import_purity_subprocess():
    """In a fresh interpreter, importing psdata leaves sys.modules clean."""
    code = (
        "import sys; import psdata; import psdata.format as f; "
        "f.assert_no_framework_imports(); "
        "bad=[m for m in ('psana','mpi4py','h5py') if m in sys.modules]; "
        "assert not bad, bad; print('CLEAN')"
    )
    env = dict(os.environ, PYTHONPATH=_PKG_PARENT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "CLEAN" in out.stdout, out.stdout


def test_reuses_psdata_functions_and_constants():
    """Sanity: the named functions/constants exist in psdata.format with the
    documented byte-layout values (verbatim from psdata.py)."""
    from psdata import format as f
    for fn in ("parse_dgram_header", "iter_xtc_children", "parse_names_block",
               "parse_configure", "extract_field"):
        assert callable(getattr(f, fn)), f"missing function {fn}"
    assert f.DGRAM_HDR == 24 and f.XTC_HDR == 12
    assert f.NAME_SZ == 524 and f.NAMEINFO_SZ == 1036 and f.SHAPE_SZ == 20
    assert f.TID_NAMES == 4 and f.TID_SHAPESDATA == 1 and f.TID_DATA == 3
    assert f.DTYPE_SIZE[1] == 2 and f.DTYPE_NP[1] is np.uint16


def test_generic_discovery():
    """discover() finds jungfrau and its 32 segments from the Names tables,
    with no hardcoded detector/stream knowledge in the library."""
    from psdata import format as psformat
    rc = psformat.discover(_stream_files())

    assert DET in rc.detector_names(), \
        f"{DET} not discovered; have {rc.detector_names()}"
    det = rc.detector(DET)
    assert ALG in det.alg_names(), f"alg {ALG} not found; have {det.alg_names()}"
    assert FIELD in det.field_names(ALG), \
        f"field {FIELD} not found; have {det.field_names(ALG)}"
    fi = det.algs[ALG][FIELD]
    assert fi.np_dtype == np.uint16, fi.np_dtype
    assert fi.rank == 3, fi.rank  # raw field declared rank 3

    seg_ids = det.segment_ids(ALG)
    assert len(seg_ids) == 32, f"expected 32 segments, got {len(seg_ids)}"
    assert seg_ids == list(range(32)), seg_ids
    # segments are spread across multiple streams (discovered, not hardcoded)
    streams = det.streams_for(ALG)
    assert len(streams) >= 2, streams
    return rc


def test_byte_exact_against_psana():
    """The pure-Python (32,512,1024) raw stack is byte-identical to psana's
    det.raw.raw(evt) for the same event; timestamp comes FROM psana."""
    from psana import DataSource  # ground-truth oracle (production env)
    from psdata import format as psformat

    ds = DataSource(exp=EXP, run=RUN, dir=DIR)
    run = next(ds.runs())
    pdet = run.Detector(DET)
    gt_raw = None
    gt_ts = None
    for evt in run.events():
        r = pdet.raw.raw(evt)
        if r is not None:
            gt_raw = np.asarray(r)
            ts = evt.timestamp
            gt_ts = int(ts() if callable(ts) else ts)
            break
    assert gt_raw is not None, "psana produced no jungfrau frame"
    assert gt_raw.shape == EXPECTED_SHAPE, gt_raw.shape

    # pure-Python extraction at the SAME (psana-derived) timestamp
    rc = psformat.discover(_stream_files())
    res = psformat.extract_detector_event(rc, DET, ALG, FIELD, gt_ts)
    raw = res["stack"]

    assert raw.shape == EXPECTED_SHAPE, raw.shape
    assert raw.dtype == np.uint16, raw.dtype
    assert np.array_equal(raw, gt_raw), (
        f"byte mismatch: max|diff|="
        f"{np.abs(raw.astype('int64') - gt_raw.astype('int64')).max()}")
    print(f"[byte-exact] ts={gt_ts} shape={raw.shape} "
          f"np.array_equal=True bytes_read={res['bytes_read']}")
    print(f"[byte-exact] segments={res['seg_ids']}")


def main():
    print("=" * 72)
    print("US-001 acceptance: psdata.format pure-Python xtc2 parse + discovery")
    print("=" * 72)
    test_import_purity_before_psana(); print("[ok] import purity (in-proc)")
    test_import_purity_subprocess();   print("[ok] import purity (subprocess)")
    test_reuses_psdata_functions_and_constants()
    print("[ok] reuses psdata.py functions/constants")
    rc = test_generic_discovery()
    print(f"[ok] generic discovery: detectors={rc.detector_names()}")
    print(f"     jungfrau: {rc.detector('jungfrau')!r}")
    test_byte_exact_against_psana()
    print("[ok] byte-exact vs psana (np.array_equal)")
    print("\nALL US-001 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
