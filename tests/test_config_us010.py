#!/usr/bin/env python3
"""US-010 acceptance test: per-segment CONFIGURE-object accessor.

Verifies the US-003 (ralph) acceptance criteria for psdata's new
``Run.seg_configs`` / ``Run.config_object`` accessor, which exposes the
per-segment CONFIGURE-block fields the epix10ka gain-range decode needs
(``trbit`` / ``asicPixelConfig``).  These live in the ``config`` alg of the
Configure dgram (offset 0), not in any L1Accept -- psdata previously exposed
only L1Accept event fields.

  1. Byte-exact gate (exp=ued1010667, run=177,
     dir=/sdf/data/lcls/ds/prj/public01/xtc, det='epixquad', segs {0,1,2,3}):
     the accessor's trbit (4,) and asicPixelConfig (4,176,192) uint8 arrays are
     np.array_equal to psana's
     det.raw._seg_configs()[seg].config.{trbit,asicPixelConfig} for every
     segment.

  2. Coverage (not epix-specific): the accessor also reads a non-epix detector
     that carries a 'config' alg -- jungfrau (exp=mfx100848724, run=51) -- for
     all 32 segments, exposing its config-alg fields.

  3. Purity: importing psdata with the new accessor stays numpy-only --
     ('psana','mpi4py','h5py') absent from sys.modules (asserted in-proc and in
     a fresh subprocess).

This test needs the PRODUCTION psana env to GENERATE the _seg_configs() ground
truth -- run it on sdfiana025 via psdata's run_tests.sh:
    source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
    bash run_tests.sh tests/test_config_us010.py
The byte-exact checks skip cleanly (with a message) if psana is unavailable, so
the import-purity check still runs without the prod env.
"""

import os
import subprocess
import sys

import numpy as np

# --- locate the package (parent of this tests dir) --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../<repo>/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# Reference datasets -- live in the TEST, never in the library.
EPIX_EXP = "ued1010667"
EPIX_RUN = 177
EPIX_DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
EPIX_DET = "epixquad"
EPIX_SEGS = [0, 1, 2, 3]
# Reference shapes/dtypes (HANDOFF datasets section).
TRBIT_SHAPE = (4,)
ASICPC_SHAPE = (4, 176, 192)

JF_EXP = "mfx100848724"
JF_RUN = 51
JF_DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
JF_DET = "jungfrau"
JF_NSEG = 32


def _have_psana():
    try:
        import psana  # noqa: F401
        return True
    except Exception:
        return False


def _open(exp, run, dir):
    import psdata
    return psdata.open(exp=exp, run=run, dir=dir)


# --------------------------------------------------------------------------
# 3a. import purity in-process -- the accessor must not pull in a framework
# --------------------------------------------------------------------------
def test_import_purity_in_proc():
    """Importing psdata + touching the new accessor surface must not pull in
    psana / mpi4py / h5py.

    A *sibling* test in this same process generates ground truth by importing
    psana, which would then show up in ``sys.modules`` here through no fault of
    psdata's import chain (see the note in ``psdata.format``).  So the in-proc
    check only asserts what psdata's own import could have introduced: if psana
    was already loaded by a sibling we skip the sys.modules assertion and defer
    to the authoritative fresh-interpreter check
    (:func:`test_import_purity_subprocess`).
    """
    import psdata
    assert hasattr(psdata.Run, "seg_configs"), "Run.seg_configs missing"
    assert hasattr(psdata.Run, "config_object"), "Run.config_object missing"
    assert callable(psdata.format.read_config_object), \
        "format.read_config_object missing"
    if any(m in sys.modules for m in ("psana", "mpi4py", "h5py")):
        print("SKIP in-proc purity: a sibling test already imported psana; "
              "the fresh-interpreter subprocess check is authoritative")
        return
    psdata.format.assert_no_framework_imports()
    for m in ("psana", "mpi4py", "h5py"):
        assert m not in sys.modules, f"{m} leaked into sys.modules on import"


# --------------------------------------------------------------------------
# 3b. import purity in a fresh interpreter, exercising the accessor offline
# --------------------------------------------------------------------------
def test_import_purity_subprocess():
    """In a fresh interpreter, importing psdata AND reading a config object
    (numpy-only, no psana DataSource) leaves sys.modules framework-free."""
    code = (
        "import sys; import psdata; "
        "r = psdata.open(exp=%r, run=%d, dir=%r); "
        "sc = r.seg_configs(%r); "
        "assert sorted(sc) == %r, sorted(sc); "
        "tr = sc[0].config.trbit; "
        "apc = sc[0].config.asicPixelConfig; "
        "assert tr.shape == %r and apc.shape == %r, (tr.shape, apc.shape); "
        "psdata.format.assert_no_framework_imports(); "
        "bad=[m for m in ('psana','mpi4py','h5py','dgram','pymongo') "
        "if m in sys.modules]; assert not bad, bad; "
        "print('CLEAN')"
        % (EPIX_EXP, EPIX_RUN, EPIX_DIR, EPIX_DET, EPIX_SEGS,
           TRBIT_SHAPE, ASICPC_SHAPE)
    )
    env = dict(os.environ, PYTHONPATH=_PKG_PARENT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "CLEAN" in out.stdout, out.stdout


# --------------------------------------------------------------------------
# 1. byte-exact gate vs psana det.raw._seg_configs() -- epix10ka (4 segments)
# --------------------------------------------------------------------------
def test_epix_seg_configs_byte_exact():
    """epixquad trbit (4,) + asicPixelConfig (4,176,192) u8 are np.array_equal
    to psana's _seg_configs() for every segment."""
    if not _have_psana():
        print("SKIP byte-exact: psana not importable (source psconda.sh)")
        return

    # psdata accessor (numpy only)
    r = _open(EPIX_EXP, EPIX_RUN, EPIX_DIR)
    scfg = r.seg_configs(EPIX_DET)
    assert sorted(scfg) == EPIX_SEGS, f"segments {sorted(scfg)} != {EPIX_SEGS}"

    # psana ground truth
    from psana import DataSource
    ds = DataSource(exp=EPIX_EXP, run=EPIX_RUN, dir=EPIX_DIR)
    myrun = next(ds.runs())
    det = myrun.Detector(EPIX_DET)
    psana_sc = det.raw._seg_configs()
    assert sorted(psana_sc) == EPIX_SEGS, f"psana segs {sorted(psana_sc)}"

    for seg in EPIX_SEGS:
        cob = psana_sc[seg].config
        ps_trbit = np.asarray(cob.trbit)
        ps_apc = np.asarray(cob.asicPixelConfig)

        my_trbit = np.asarray(scfg[seg].config.trbit)
        my_apc = np.asarray(scfg[seg].config.asicPixelConfig)

        assert my_trbit.shape == TRBIT_SHAPE, (seg, my_trbit.shape)
        assert my_apc.shape == ASICPC_SHAPE, (seg, my_apc.shape)
        assert my_trbit.dtype == np.uint8, (seg, my_trbit.dtype)
        assert my_apc.dtype == np.uint8, (seg, my_apc.dtype)

        assert np.array_equal(my_trbit, ps_trbit), \
            f"seg {seg} trbit differs: psdata {my_trbit} vs psana {ps_trbit}"
        assert np.array_equal(my_apc, ps_apc), \
            (f"seg {seg} asicPixelConfig differs "
             f"(max|diff|={int(np.abs(my_apc.astype(int) - ps_apc.astype(int)).max())})")
        print(f"seg {seg}: trbit + asicPixelConfig byte-exact vs psana")

    # config_object alias returns the same thing
    scfg2 = r.config_object(EPIX_DET)
    assert sorted(scfg2) == EPIX_SEGS
    assert np.array_equal(np.asarray(scfg2[0].config.trbit),
                          np.asarray(scfg[0].config.trbit))
    print("epix10ka byte-exact gate PASS (all 4 segments)")


# --------------------------------------------------------------------------
# 2. coverage on a non-epix detector that carries a 'config' alg (jungfrau)
# --------------------------------------------------------------------------
def test_jungfrau_config_coverage():
    """The accessor is not epix-specific: it reads jungfrau's config alg for
    all 32 segments (jungfrau carries a 'config' alg of static settings)."""
    r = _open(JF_EXP, JF_RUN, JF_DIR)

    det = r.detector(JF_DET)
    assert "config" in det.alg_names(), \
        f"jungfrau has no config alg; algs={det.alg_names()}"

    scfg = r.seg_configs(JF_DET)
    assert len(scfg) == JF_NSEG, f"expected {JF_NSEG} segments, got {len(scfg)}"
    assert sorted(scfg) == list(range(JF_NSEG)), sorted(scfg)

    # Each segment exposes the 'config' alg with the discovered fields.
    cfg_fields = det.field_names("config")
    assert cfg_fields, "no config-alg fields discovered for jungfrau"
    for seg, seg_cfg in scfg.items():
        assert "config" in seg_cfg.alg_names(), (seg, seg_cfg.alg_names())
        ns = seg_cfg.config
        assert sorted(ns.field_names()) == sorted(cfg_fields), \
            (seg, set(ns.field_names()) ^ set(cfg_fields))
        # a plain scalar field is reachable by attribute syntax
        assert "firmwareVersion" in cfg_fields
        fv = getattr(ns, "firmwareVersion")
        assert np.isscalar(fv) or np.ndim(fv) == 0, (seg, type(fv))
        # a dotted field is reachable through the fields dict
        assert "user.bias_voltage_v" in ns.fields, seg

    # cross-check ONE jungfrau field against psana (coverage, not the gate)
    if _have_psana():
        from psana import DataSource
        ds = DataSource(exp=JF_EXP, run=JF_RUN, dir=JF_DIR)
        myrun = next(ds.runs())
        pdet = myrun.Detector(JF_DET)
        ps_sc = pdet.raw._seg_configs()
        ps_fv = int(ps_sc[0].config.firmwareVersion)
        my_fv = int(scfg[0].config.firmwareVersion)
        assert my_fv == ps_fv, f"jungfrau firmwareVersion {my_fv} != psana {ps_fv}"
        print(f"jungfrau coverage: config alg read for 32 segs; "
              f"firmwareVersion {my_fv} matches psana")
    else:
        print(f"jungfrau coverage: config alg read for {JF_NSEG} segments "
              f"({len(cfg_fields)} fields) -- psana cross-check skipped")


if __name__ == "__main__":
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures.append((name, e))
                print(f"FAIL {name}: {e}")
            except Exception as e:  # pragma: no cover
                failures.append((name, e))
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{len(failures)} test(s) failed")
        sys.exit(1)
    print("\nALL US-010 TESTS PASSED")
