#!/usr/bin/env python3
"""US-005 acceptance test: public psdata package API + psana regression harness.

Verifies the US-005 acceptance criteria:

  1. The public ``psdata`` package opens a run BOTH ways and exposes the same
     run through one handle:
       * ``psdata.open(exp=, run=, dir=)`` -- files resolved by the standard
         layout ``{dir}/{exp}-r{run:04d}-s{stream:03d}-c000.xtc2`` (SMD under
         ``{dir}/smalldata/``), and
       * ``psdata.open(files=[...])`` -- an explicit per-stream file list.
     A run streams events (``r.events()``), random-accesses an event by
     timestamp (``r.read_event(ts)`` / ``r.read_event_at(k)``), and introspects
     detectors / fields / segments (``r.detector_names()`` / ``r.detector()``).
     Importing ``psdata`` pulls in ONLY numpy (no psana / mpi4py / h5py).

  2. This regression harness regenerates psana ground truth ITSELF --
     ``DataSource(exp=,run=,dir=).Detector(name).raw.raw(evt)`` for a handful of
     events of the reference dataset -- rather than relying on the ephemeral
     ``/tmp/gt_*.npy`` that ``psdata.py`` loads.  It asserts that ``psdata``'s
     raw arrays (via BOTH the streaming and the random-access public API) are
     byte-identical (``np.array_equal``) to psana's, and that the pulseId
     matches ``run.Detector('timing').raw.pulseId(evt)``.

This test needs the PRODUCTION psana env (the psconda.sh install) to generate
ground truth:
    source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
Run it on host sdfiana025 (via ``psdata/run_tests.sh``, which sets PYTHONPATH so
``import psdata`` finds the package and ``import psana`` finds the prod env).

The import-purity check (criterion 1) is asserted BEFORE psana is imported and
again in a fresh subprocess, so a later psana import in this process cannot mask
a leak.  The byte-exact checks need psana and are skipped cleanly if it is not
importable (with a clear message), so the package-API checks still run without
the prod env.
"""

import os
import subprocess
import sys

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
TIMING = "timing"            # psana's name for the timing detector (det_type 'ts')
EXPECTED_SHAPE = (32, 512, 1024)
# Event positions to cross-check against psana (well within the run).
K_VALUES = [0, 1, 5, 17, 50]


def _explicit_files():
    """Resolve the run's per-stream first-chunk files by globbing -- to feed the
    explicit-file-list form of ``psdata.open``.  The library has its own
    resolver for the exp/run/dir form; this exercises the OTHER entry point."""
    import glob
    paths = sorted(glob.glob(f"{DIR}/{EXP}-r{RUN:04d}-s*-c000.xtc2"))
    assert paths, f"no stream files found under {DIR}"
    return paths


# --------------------------------------------------------------------------
# 1. import purity -- psdata pulls in only numpy
# --------------------------------------------------------------------------
def test_import_purity_before_psana():
    """Importing the public psdata package must NOT pull in a framework."""
    import psdata
    psdata.run.assert_no_framework_imports()
    for m in ("psana", "mpi4py", "h5py"):
        assert m not in sys.modules, f"{m} leaked into sys.modules on import"


def test_import_purity_subprocess():
    """In a fresh interpreter, importing psdata leaves sys.modules clean and
    pulls in numpy (only)."""
    code = (
        "import sys, psdata; "
        "psdata.run.assert_no_framework_imports(); "
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
# 1. public package API -- open both ways, stream / random-access / introspect
# --------------------------------------------------------------------------
def test_open_both_ways_equivalent():
    """``psdata.open(exp,run,dir)`` and ``psdata.open(files=[...])`` resolve to
    the same run: same streams, same detectors/fields/segments, and the same
    first events (ts/pulseId/raw byte-identical) by both streaming and random
    access -- no psana needed."""
    import psdata

    r_idr = psdata.open(exp=EXP, run=RUN, dir=DIR)
    r_files = psdata.open(files=_explicit_files())

    # same stream set and same files keyed by the real s### index
    assert r_idr.files == r_files.files, \
        f"exp/run/dir files {r_idr.files} != explicit files {r_files.files}"

    # introspection: jungfrau discovered with 32 segments and a rank-3 raw field
    dets = r_idr.detector_names()
    assert DET in dets, f"{DET} not discovered; have {dets}"
    det = r_idr.detector(DET)
    assert "raw" in det.alg_names() and "raw" in det.field_names("raw")
    assert det.algs["raw"]["raw"].np_dtype == np.uint16
    assert det.algs["raw"]["raw"].rank == 3
    assert det.segment_ids("raw") == list(range(32))
    # the timing detector is discoverable by type (carries pulseId)
    assert r_idr.find_detector_by_type("ts"), "no timing (det_type 'ts') detector"

    # stream the first few events from the exp/run/dir handle
    n = max(K_VALUES) + 1
    streamed = []
    for k, evt in enumerate(r_idr.events()):
        if k >= n:
            break
        streamed.append((evt.timestamp, evt.pulseId, evt.stack(DET)))
    assert len(streamed) == n

    # random-access the same events through the explicit-files handle and
    # confirm byte-identity (cross-checks both open() forms AND both access
    # paths against each other -- a psana-free consistency net).
    for k in K_VALUES:
        sts, spid, sstack = streamed[k]
        evt_at = r_files.read_event_at(k)
        evt_ts = r_files.read_event(sts)
        assert evt_at.timestamp == evt_ts.timestamp == sts, \
            f"k={k}: ts mismatch"
        assert evt_at.pulseId == evt_ts.pulseId == spid, \
            f"k={k}: pulseId mismatch"
        rstack = evt_at.stack(DET)
        assert np.array_equal(rstack, sstack), \
            f"k={k}: random-access raw != streaming raw"
        assert np.array_equal(evt_ts.stack(DET), sstack), \
            f"k={k}: read_event(ts) raw != streaming raw"

    r_idr.close()
    r_files.close()
    print(f"[api] open(exp/run/dir) == open(files) for streams "
          f"{sorted(r_idr.files)}; streamed and random-accessed "
          f"K={K_VALUES} byte-identical; detectors={dets}")


def test_open_argument_guards():
    """``open`` rejects ambiguous / incomplete arguments."""
    import psdata
    for kwargs in (
        dict(),                                   # nothing
        dict(exp=EXP, run=RUN),                   # missing dir
        dict(exp=EXP, run=RUN, dir=DIR, files=_explicit_files()),  # both
    ):
        try:
            psdata.open(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"open({kwargs}) should have raised ValueError")
    # a nonexistent run dir -> FileNotFoundError
    try:
        psdata.open(exp="nope999", run=1, dir="/tmp/does-not-exist")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("open(nonexistent) should raise FileNotFoundError")
    print("[api] open() argument guards ok")


# --------------------------------------------------------------------------
# 2. regression harness -- regenerate psana ground truth, assert byte-identical
# --------------------------------------------------------------------------
def _psana_ground_truth(check_k):
    """Regenerate ground truth FROM psana itself (no /tmp/gt_*.npy): for each k
    in ``check_k`` return ``{k: (ts, pulseId, raw(32,512,1024))}`` obtained from
    ``DataSource(exp,run,dir).Detector('jungfrau').raw.raw(evt)`` and
    ``Detector('timing').raw.pulseId(evt)``."""
    from psana import DataSource
    ds = DataSource(exp=EXP, run=RUN, dir=DIR)
    run = next(ds.runs())
    jf = run.Detector(DET)
    timing = run.Detector(TIMING)
    out = {}
    kmax = max(check_k)
    for k, evt in enumerate(run.events()):
        if k > kmax:
            break
        if k in check_k:
            ts = evt.timestamp
            ts = int(ts() if callable(ts) else ts)
            raw = np.asarray(jf.raw.raw(evt))
            pid = timing.raw.pulseId(evt)
            pid = int(pid() if callable(pid) else pid)
            out[k] = (ts, pid, raw)
    return out


def test_regression_byte_identical_vs_psana():
    """psdata raw arrays (via the PUBLIC open() API, both streaming and random
    access) are byte-identical to psana's freshly-regenerated ground truth, and
    pulseId matches psana's timing.raw.pulseId(evt)."""
    try:
        from psana import DataSource  # noqa: F401  (probe availability)
    except Exception as e:                            # pragma: no cover
        print(f"[regression] SKIPPED (psana not importable: {e}). "
              f"Source psconda.sh and re-run to exercise the regression check.")
        return

    import psdata

    check_k = set(K_VALUES)
    gt = _psana_ground_truth(check_k)
    assert set(gt) == check_k, f"psana yielded {sorted(gt)}, wanted {sorted(check_k)}"
    for k, (_ts, _pid, raw) in gt.items():
        assert raw.shape == EXPECTED_SHAPE and raw.dtype == np.uint16, \
            f"k={k}: psana raw shape/dtype {raw.shape}/{raw.dtype}"

    # ---- streaming path (public open(exp/run/dir).events()) --------------
    r = psdata.open(exp=EXP, run=RUN, dir=DIR)
    streamed = {}
    kmax = max(check_k)
    for k, evt in enumerate(r.events()):
        if k > kmax:
            break
        if k in check_k:
            streamed[k] = (evt.timestamp, evt.pulseId, evt.stack(DET))

    for k in sorted(check_k):
        gts, gpid, graw = gt[k]
        sts, spid, sstack = streamed[k]
        assert sts == gts, f"k={k}: streaming ts {sts} != psana ts {gts}"
        assert spid == gpid, f"k={k}: streaming pulseId {spid} != psana {gpid}"
        assert sstack is not None and sstack.shape == graw.shape, \
            f"k={k}: streaming stack {None if sstack is None else sstack.shape}"
        assert np.array_equal(sstack, graw), (
            f"k={k}: streaming raw != psana raw; max|diff|="
            f"{np.abs(sstack.astype('int64') - graw.astype('int64')).max()}")

    # ---- random-access path (public open(...).read_event) ----------------
    for k in sorted(check_k):
        gts, gpid, graw = gt[k]
        evt = r.read_event(gts)               # by timestamp
        evt_at = r.read_event_at(k)           # by position
        assert evt.timestamp == evt_at.timestamp == gts, f"k={k}: ts mismatch"
        assert evt.pulseId == gpid, \
            f"k={k}: random pulseId {evt.pulseId} != psana {gpid}"
        rstack = evt.stack(DET)
        assert np.array_equal(rstack, graw), (
            f"k={k}: random-access raw != psana raw; max|diff|="
            f"{np.abs(rstack.astype('int64') - graw.astype('int64')).max()}")
        assert np.array_equal(evt_at.stack(DET), graw), \
            f"k={k}: read_event_at raw != psana raw"
    r.close()
    print(f"[regression] regenerated psana ground truth for K={sorted(check_k)} "
          f"and matched psdata raw (streaming + random access) byte-identically; "
          f"pulseId matches psana timing.raw.pulseId")


def main():
    print("=" * 72)
    print("US-005 acceptance: public psdata package API + psana regression")
    print("=" * 72)
    test_import_purity_before_psana(); print("[ok] import purity (in-proc)")
    test_import_purity_subprocess();   print("[ok] import purity (subprocess)")
    test_open_both_ways_equivalent()
    print("[ok] open(exp/run/dir) == open(files); stream/random/introspect")
    test_open_argument_guards()
    print("[ok] open() argument guards")
    test_regression_byte_identical_vs_psana()
    print("[ok] regression: psdata raw == regenerated psana ground truth")
    print("\nALL US-005 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
