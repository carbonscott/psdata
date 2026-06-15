#!/usr/bin/env python3
"""US-002 acceptance test for psdata.stream.

Verifies the acceptance criteria:

  1. Multi-stream event assembly is an exact 64-bit-timestamp k-way merge
     (event ts = min head ts; matching heads join; non-matching are not
     consumed) -- exercised structurally and end-to-end.
  2. ``psdata.events(stream_files)`` yields L1Accept events in ascending
     timestamp order; each exposes ``timestamp``, ``pulseId``, and
     ``{detname: {seg: ndarray}}``.
  3. For the reference dataset (exp=mfx100848724 run=51
     dir=/sdf/data/lcls/ds/prj/public01/xtc), the first 20 L1Accept events
     match psana's ``run.events()`` order by timestamp exactly; each
     detector's raw arrays are byte-identical to psana's ``det.raw.raw(evt)``;
     pulseId matches psana's ``run.Detector('timing').raw.pulseId(evt)``.
  4. The missing-segment -> detector ``None`` rule is implemented (the
     reference run is clean, so a synthetic missing-segment case proves the
     logic; US-004 exercises it on a dataset that triggers it naturally).
  5. Importing psdata / psdata.stream does not import psana, mpi4py, or h5py.

Needs the production psana env (psconda.sh) on host sdfiana025 to generate
ground truth.
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
N_EVENTS = 20
JUNGFRAU = "jungfrau"


def _stream_files():
    """Resolve the run's per-stream xtc2 files by globbing -- the test, not the
    library, knows the file-naming pattern."""
    import glob
    paths = sorted(glob.glob(f"{DIR}/{EXP}-r{RUN:04d}-s*-c000.xtc2"))
    assert paths, f"no stream files found under {DIR}"
    files = {}
    for p in paths:
        base = os.path.basename(p)
        sidx = int(base.split("-s")[1].split("-")[0])
        files[sidx] = p
    return files


# --------------------------------------------------------------------------
# 5. import purity
# --------------------------------------------------------------------------
def test_import_purity_before_psana():
    import psdata
    from psdata import stream as psstream
    psstream.assert_no_framework_imports()
    for m in ("psana", "mpi4py", "h5py"):
        assert m not in sys.modules, f"{m} leaked into sys.modules on import"


def test_import_purity_subprocess():
    code = (
        "import sys, psdata, psdata.stream as s; "
        "s.assert_no_framework_imports(); "
        "bad=[m for m in ('psana','mpi4py','h5py') if m in sys.modules]; "
        "assert not bad, bad; print('CLEAN')"
    )
    env = dict(os.environ, PYTHONPATH=_PKG_PARENT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "CLEAN" in out.stdout, out.stdout


# --------------------------------------------------------------------------
# 1./2. ascending-order streaming + identity fields (no psana needed)
# --------------------------------------------------------------------------
def test_streaming_ascending_and_fields():
    import psdata
    rc = psdata.discover(_stream_files())
    last = None
    n = 0
    for evt in psdata.events(_stream_files(), run_config=rc):
        assert evt.service in (11, 12), f"non-L1Accept yielded: {evt.service}"
        assert isinstance(evt.timestamp, int)
        if last is not None:
            assert evt.timestamp > last, (
                f"timestamps not strictly ascending: {last} -> {evt.timestamp}")
        last = evt.timestamp
        # pulseId is readable (timing detector exists in this run)
        assert evt.pulseId is not None
        # detector dict view works
        d = evt.as_dict()  # {detname: {seg: ndarray}}
        assert JUNGFRAU in d, f"jungfrau missing from event dict: {list(d)}"
        n += 1
        if n >= N_EVENTS:
            break
    assert n == N_EVENTS, n
    print(f"[ascending] streamed {n} events, strictly ascending ts, "
          f"pulseId+detector dict present")


# --------------------------------------------------------------------------
# 4. missing-segment -> None (synthetic; reference run is clean)
# --------------------------------------------------------------------------
def test_missing_segment_returns_none():
    """Drop one captured segment of jungfrau from an event and confirm the
    detector reports None (psana _segments rule), while the full event does
    not."""
    import psdata
    from psdata.stream import Event
    rc = psdata.discover(_stream_files())
    gen = psdata.events(_stream_files(), run_config=rc)
    evt = next(gen)
    full = evt.stack(JUNGFRAU)
    assert full is not None and full.shape[0] == 32, \
        f"expected complete jungfrau, got {None if full is None else full.shape}"

    # forge a damaged event: same seg index minus one jungfrau segment
    key = (JUNGFRAU, "raw")
    forged_index = {k: dict(v) for k, v in evt._seg_index.items()}
    a_seg = sorted(forged_index[key])[0]
    del forged_index[key][a_seg]
    forged = Event(evt.timestamp, evt.service, rc, forged_index)
    assert forged.stack(JUNGFRAU) is None, \
        "missing-segment jungfrau should be None"
    assert forged.raw(JUNGFRAU) is None
    # a single-segment detector in the forged event is unaffected
    # (epix100_0 declares one segment and has a 'raw' field).
    other = forged.raw("epix100_0", field="raw", alg="raw")
    assert other is not None, "unrelated detector should be unaffected"
    gen.close()
    print("[missing-seg] dropping one jungfrau segment -> detector None (ok)")


# --------------------------------------------------------------------------
# 3. byte-exact vs psana for the first 20 events
# --------------------------------------------------------------------------
def test_byte_exact_first_20_vs_psana():
    from psana import DataSource
    import psdata

    ds = DataSource(exp=EXP, run=RUN, dir=DIR)
    run = next(ds.runs())
    timing = run.Detector("timing")

    # Every detector in this run that exposes a raw array (det.raw.raw).
    # Discovered from psana so the test stays honest about "each detector".
    raw_dets = {}
    for name in sorted(run.detnames):
        try:
            d = run.Detector(name)
        except Exception:
            continue
        if hasattr(d, "raw") and hasattr(d.raw, "raw"):
            raw_dets[name] = d
    assert JUNGFRAU in raw_dets, sorted(raw_dets)
    assert len(raw_dets) >= 2, ("expected >=2 raw-array detectors to prove "
                                f"'each detector', got {sorted(raw_dets)}")

    # ground truth: first N events' (ts, pulseId, {det: raw}) from psana
    gt = []
    for i, evt in enumerate(run.events()):
        if i >= N_EVENTS:
            break
        ts = evt.timestamp
        ts = int(ts() if callable(ts) else ts)
        pid = timing.raw.pulseId(evt)
        raws = {}
        for name, d in raw_dets.items():
            r = d.raw.raw(evt)
            raws[name] = None if r is None else np.asarray(r)
        gt.append((ts, int(pid), raws))
    assert len(gt) == N_EVENTS, len(gt)

    # psdata streaming
    rc = psdata.discover(_stream_files())
    ours = []
    for k, evt in enumerate(psdata.events(_stream_files(), run_config=rc)):
        if k >= N_EVENTS:
            break
        ours.append(evt)
    assert len(ours) == N_EVENTS, len(ours)

    for i, ((gts, gpid, graws), evt) in enumerate(zip(gt, ours)):
        assert evt.timestamp == gts, \
            f"event {i}: ts mismatch psdata={evt.timestamp} psana={gts}"
        assert evt.pulseId == gpid, \
            f"event {i}: pulseId mismatch psdata={evt.pulseId} psana={gpid}"
        for name, graw in graws.items():
            my = evt.stack(name)
            if graw is None:
                assert my is None, f"event {i} {name}: psdata !None, psana None"
                continue
            assert my is not None, f"event {i} {name}: psdata None, psana !None"
            # psana stacks per-segment frames; a single-segment detector comes
            # back as (1, H, W).  Compare ignoring a unit leading axis.
            a, b = np.asarray(my), np.asarray(graw)
            if a.shape != b.shape and a.reshape(-1).shape == b.reshape(-1).shape:
                a = a.reshape(b.shape)
            assert a.shape == b.shape, \
                f"event {i} {name}: shape {a.shape} vs {b.shape}"
            assert a.dtype == b.dtype, \
                f"event {i} {name}: dtype {a.dtype} vs {b.dtype}"
            assert np.array_equal(a, b), (
                f"event {i} {name}: raw byte mismatch; max|diff|="
                f"{np.abs(a.astype('int64') - b.astype('int64')).max()}")
    jshape = gt[0][2][JUNGFRAU].shape
    assert jshape == (32, 512, 1024), jshape
    print(f"[byte-exact] first {N_EVENTS} events: ts + pulseId + raw arrays of "
          f"{sorted(raw_dets)} all byte-identical to psana (np.array_equal)")


def main():
    print("=" * 72)
    print("US-002 acceptance: psdata.stream multi-stream event assembly")
    print("=" * 72)
    test_import_purity_before_psana(); print("[ok] import purity (in-proc)")
    test_import_purity_subprocess();   print("[ok] import purity (subprocess)")
    test_streaming_ascending_and_fields()
    print("[ok] ascending streaming + pulseId + detector dict")
    test_missing_segment_returns_none()
    print("[ok] missing-segment -> detector None")
    test_byte_exact_first_20_vs_psana()
    print("[ok] byte-exact vs psana (ts + pulseId + raw, first 20)")
    print("\nALL US-002 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
