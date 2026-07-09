#!/usr/bin/env python3
"""US-008 -- serializable + disk-persisted random-access index.

psdata ALREADY has random access (``RunIndex.read_event`` /
``read_event_at``, backed by an SMD-scan index).  This story makes that
once-built index:

  1. **persistable to disk** -- ``RunIndex.save(path)`` / ``RunIndex.load(path)``
     write/reload the built index to ONE file WITHOUT rescanning the SMD files,
     so random access is instant on reopen (a single-process benefit);
  2. **serializable in-memory** -- ``to_dict`` / ``from_dict`` and the pickle
     protocol (``__getstate__`` / ``__setstate__``) ship the once-built index to
     parallel workers with no per-worker rescan.

THE gotcha being guarded (verified live, see the handoff): ``RunIndex._bd_fds``
caches raw OS file-descriptor integers; ``pickle`` does NOT refuse them, so an
index serialized *after a read* would carry stale fds and, on reload, raise
``OSError(9)`` or silently ``pread`` the WRONG file.  The disk and in-memory
paths therefore share ONE state-stripping helper that EXCLUDES ``_bd_fds`` /
``_ts_to_k`` and re-inits them empty on reconstruction.

This suite is **self-contained** -- the oracle is psdata's own ``read_event_at``
path (the index built in-process), so no psana is needed.  The cross-process
checks use ``subprocess`` so a reload genuinely happens in a fresh interpreter.

Run (on sdfiana025, from the repo root):
    PYTHONPATH=src .venv/bin/python tests/test_persist_us008.py
or via the suite:
    bash run_tests.sh tests/test_persist_us008.py
"""

import glob
import json
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

# --- locate the package (parent of this tests dir) --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")  # .../<repo>/src (holds psdata/)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Reference datasets -- live in the TEST, never in the library.
# Single-chunk primary run.
EXP = "mfx100848724"
RUN = 51
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
JUNGFRAU = "jungfrau"

# Multi-chunk run (streams roll c000 -> c001): exercises c001+ chunk paths
# stored INSIDE entries surviving persistence.  Heavier (~48k events), so the
# multi-chunk round-trips spot-check a few positions including the post-roll
# region rather than the whole run.
MC_EXP = "mfx101343025"
MC_RUN = 35
MC_DIR = "/sdf/data/lcls/ds/MFX/mfx101343025/xtc"

# Event positions to spot-check (single-chunk run has ~17,872 events).
K_VALUES = [0, 1, 5, 17, 100, 999, 17871]   # includes 0 and the last event

# An epics variable verified to resolve to a real (non-None) value at the
# last sampled event of this run -- probes that env_records CONTENT (not just
# its presence in the persisted field set) survives a round-trip.
ENV_PROBE_VAR = "BeamMonitor_diode_x"


def _stream_files(directory, exp, run):
    """Resolve a run's per-stream bigdata c000 xtc2 files by globbing -- the
    test (not the library) knows the file-naming pattern.  The index follows the
    ``chunkinfo`` roll to later chunks itself."""
    paths = sorted(glob.glob(f"{directory}/{exp}-r{run:04d}-s*-c000.xtc2"))
    assert paths, f"no stream files found under {directory} for {exp} r{run}"
    files = {}
    for p in paths:
        base = os.path.basename(p)
        sidx = int(base.split("-s")[1].split("-")[0])
        files[sidx] = p
    return files


def _build_index(directory, exp, run):
    import psdata
    from psdata import index as psindex
    files = _stream_files(directory, exp, run)
    rc = psdata.discover(files)
    return psindex.build_index(files, run_config=rc)


def _event_fingerprint(evt, det_names):
    """A hashable, byte-exact fingerprint of an event: (ts, pulseId, and for
    each detector a sha-free exact array snapshot via .tobytes())."""
    parts = {"ts": int(evt.timestamp), "pid": int(evt.pulseId)}
    for name in det_names:
        st = evt.stack(name)
        if st is None:
            parts[name] = None
        else:
            parts[name] = (st.shape, str(st.dtype), st.tobytes())
    return parts


def _fingerprints_equal(a, b):
    if a["ts"] != b["ts"] or a["pid"] != b["pid"]:
        return False
    if set(a) != set(b):
        return False
    for k in a:
        if k in ("ts", "pid"):
            continue
        if a[k] is None or b[k] is None:
            if a[k] is not b[k] and not (a[k] is None and b[k] is None):
                return False
            continue
        if a[k] != b[k]:
            return False
    return True


def _raw_det_names(rc):
    """Real (non-bookkeeping) detectors that expose a raw.raw field."""
    out = []
    for name in rc.detector_names():
        det = rc.detector(name)
        if "raw" in det.algs and "raw" in det.algs["raw"]:
            out.append(name)
    return out


# ==========================================================================
# 1. persisted state is EXACTLY the specified fields; fds/ts cache excluded
# ==========================================================================
def test_persist_state_fields_exact():
    """``_persist_state`` (the single source of truth for save / to_dict /
    __getstate__) must carry exactly the specified fields and MUST NOT leak the
    fd cache or the ts->k cache."""
    import psdata  # noqa: F401  (ensure import purity baseline)
    ridx = _build_index(DIR, EXP, RUN)
    try:
        # force the fd cache to be populated so we'd catch a leak.
        ridx.read_event_at(0)
        assert ridx._bd_fds, "expected fds open after a read"

        state = ridx._persist_state()
        expected = {
            "timestamps", "entries", "env_records", "bd_files", "chunk_files",
            "multichunk_streams", "run_config", "build_seconds",
            "smd_bytes_read", "scan_source", "scan_bytes_read",
            "include_shutdown_tail",
        }
        assert set(state) == expected, \
            f"persisted fields {set(state)} != required {expected}"
        # the two per-process caches MUST be absent.
        assert "_bd_fds" not in state and "_ts_to_k" not in state, \
            "persisted state leaked a per-process cache"
        # to_dict / __getstate__ route through the SAME helper -> same keys.
        assert set(ridx.to_dict()) == expected
        assert set(ridx.__getstate__()) == expected
    finally:
        ridx.close()
    print(f"[state] persisted fields == {sorted(expected)}; "
          f"_bd_fds/_ts_to_k excluded")


# ==========================================================================
# 2. in-memory round-trip (to_dict/from_dict, pickle) -- byte-identical reads
# ==========================================================================
def test_to_dict_from_dict_byte_identical():
    import psdata
    ridx = _build_index(DIR, EXP, RUN)
    det_names = _raw_det_names(ridx.run_config)
    try:
        # READ FIRST so _bd_fds is populated -- the gotcha only bites a
        # serialized-after-read index.
        for k in K_VALUES:
            ridx.read_event_at(k)

        payload = ridx.to_dict()
        clone = psdata.RunIndex.from_dict(payload)
        try:
            assert clone._bd_fds == {}, "from_dict must start with empty fds"
            assert clone._ts_to_k is None, "from_dict must reset _ts_to_k"
            assert clone.n_events == ridx.n_events
            assert clone.smd_bytes_read == ridx.smd_bytes_read, \
                "build provenance (smd_bytes_read) must survive round-trip"

            # env_records CONTENT round-trip.  to_dict/from_dict hands back the
            # SAME live dict by reference (no copy), so this equality is
            # nearly tautological here -- the real coverage of a genuine
            # serialize/deserialize is the disk save/load test's pickle path
            # below.  Still assert it (plus an as-of lookup through the
            # reloaded index), so a future to_dict/from_dict change that drops
            # or copies env_records gets caught here too.
            assert ridx.env_records, "expected non-empty env_records for this run"
            assert clone.env_records == ridx.env_records, \
                "env_records content differs after to_dict/from_dict"
            from psdata import envstore as _envstore
            probe_ts = int(ridx.timestamps[K_VALUES[-1]])
            orig_val = _envstore.EnvStoreManager(
                ridx.run_config, ridx.env_records
            ).store("epics").value(ENV_PROBE_VAR, probe_ts)
            clone_val = _envstore.EnvStoreManager(
                clone.run_config, clone.env_records
            ).store("epics").value(ENV_PROBE_VAR, probe_ts)
            assert orig_val is not None, \
                f"probe var {ENV_PROBE_VAR!r} must resolve to a real value"
            assert clone_val == orig_val, \
                f"env as_of({ENV_PROBE_VAR!r}) differs after to_dict/from_dict"

            for k in K_VALUES:
                a = _event_fingerprint(ridx.read_event_at(k), det_names)
                b = _event_fingerprint(clone.read_event_at(k), det_names)
                assert _fingerprints_equal(a, b), \
                    f"to_dict/from_dict read at k={k} not byte-identical"
        finally:
            clone.close()

        # pickle.dumps(idx) of an index that has READ must ALSO be safe.
        blob = pickle.dumps(ridx, protocol=4)
        un = pickle.loads(blob)
        try:
            assert un._bd_fds == {} and un._ts_to_k is None
            for k in K_VALUES:
                a = _event_fingerprint(ridx.read_event_at(k), det_names)
                b = _event_fingerprint(un.read_event_at(k), det_names)
                assert _fingerprints_equal(a, b), \
                    f"pickle round-trip read at k={k} not byte-identical"
        finally:
            un.close()
    finally:
        ridx.close()
    print(f"[mem] to_dict/from_dict + pickle round-trip byte-identical at "
          f"K={K_VALUES} ({len(det_names)} dets); blob "
          f"{len(pickle.dumps(ridx.to_dict(), protocol=4))/1e6:.2f} MB")


# ==========================================================================
# 3. disk save/load reloads in a SEPARATE PROCESS with NO SMD I/O
# ==========================================================================
# A small driver script run by subprocess: it loads a saved index and an
# expected-fingerprints file, asserts byte-identity, and -- crucially -- spies
# os.open / os.pread to prove the load+read path touches ZERO smalldata (SMD)
# files (the "no rescan; reload opens only the index file" guarantee).  It
# writes a JSON result to stdout's last line.
_RELOAD_DRIVER = r'''
import json, os, pickle, sys
sys.path.insert(0, sys.argv[1])          # src/ dir
idx_path = sys.argv[2]
expect_path = sys.argv[3]
k_values = json.loads(sys.argv[4])
det_names = json.loads(sys.argv[5])

# spy: record any open / pread that hits a smalldata (SMD) file.
_real_open = os.open
_real_pread = os.pread
smd_opens = []
smd_bytes = [0]
def _is_smd(path):
    p = str(path)
    return ".smd.xtc2" in p or "/smalldata/" in p
def spy_open(path, *a, **k):
    fd = _real_open(path, *a, **k)
    if _is_smd(path):
        smd_opens.append(str(path))
    spy_open._fd2path[fd] = str(path)
    return fd
spy_open._fd2path = {}
def spy_pread(fd, n, off):
    path = spy_open._fd2path.get(fd, "")
    if _is_smd(path):
        smd_bytes[0] += n
    return _real_pread(fd, n, off)
os.open = spy_open
os.pread = spy_pread

import psdata
# load opens ONLY the index file; no SMD scan.
ridx = psdata.RunIndex.load(idx_path)

# baseline provenance survives, but the LOAD itself read no SMD.
result = {
    "n_events": ridx.n_events,
    "smd_bytes_read_attr": int(ridx.smd_bytes_read),
    "smd_files_opened_during_reload": list(smd_opens),
    "smd_bytes_read_during_reload": int(smd_bytes[0]),
    "bd_fds_empty_on_load": (ridx._bd_fds == {}),
    "ts_to_k_none_on_load": (ridx._ts_to_k is None),
    "reads_match": True,
}

with open(expect_path, "rb") as fh:
    expect = pickle.load(fh)   # {k: (ts, pid, {det: (shape,dtype,bytes)|None})}

# env_records content + as-of probe must survive the disk round-trip too
# (not just be present in the persisted field set).
from psdata import envstore as _envstore
result["env_records_match"] = (ridx.env_records == expect["env_records"])
probe_var, probe_ts, probe_val = expect["env_probe"]
reloaded_val = _envstore.EnvStoreManager(
    ridx.run_config, ridx.env_records
).store("epics").value(probe_var, probe_ts)
result["env_probe_match"] = (
    reloaded_val is not None and reloaded_val == probe_val)

for k in k_values:
    evt = ridx.read_event_at(k)
    ets, epid, edets = expect[k]
    if int(evt.timestamp) != ets or int(evt.pulseId) != epid:
        result["reads_match"] = False
        result["fail_k"] = k
        break
    ok = True
    for name in det_names:
        st = evt.stack(name)
        ex = edets[name]
        if ex is None:
            if st is not None:
                ok = False; break
        else:
            shp, dt, raw = ex
            if st is None or st.shape != tuple(shp) or str(st.dtype) != dt \
               or st.tobytes() != raw:
                ok = False; break
    if not ok:
        result["reads_match"] = False
        result["fail_k"] = k
        break

ridx.close()
print("RESULT " + json.dumps(result))
'''


def _expected_for_subprocess(ridx, det_names, k_values, path):
    expect = {}
    for k in k_values:
        evt = ridx.read_event_at(k)
        dets = {}
        for name in det_names:
            st = evt.stack(name)
            dets[name] = None if st is None else (
                list(st.shape), str(st.dtype), st.tobytes())
        expect[k] = (int(evt.timestamp), int(evt.pulseId), dets)
    # env_records content + one as-of probe, keyed by string (never collides
    # with the integer k keys above) -- lets the reload subprocess confirm
    # slow (env) data survives the disk round-trip, not just its presence.
    from psdata import envstore as _envstore
    expect["env_records"] = ridx.env_records
    probe_ts = int(ridx.timestamps[k_values[-1]])
    probe_val = _envstore.EnvStoreManager(
        ridx.run_config, ridx.env_records
    ).store("epics").value(ENV_PROBE_VAR, probe_ts)
    expect["env_probe"] = (ENV_PROBE_VAR, probe_ts, probe_val)
    with open(path, "wb") as fh:
        pickle.dump(expect, fh, protocol=4)


def _run_reload_subprocess(idx_path, expect_path, k_values, det_names):
    drv = os.path.join(tempfile.gettempdir(), "psdata_us008_reload_driver.py")
    with open(drv, "w") as fh:
        fh.write(_RELOAD_DRIVER)
    proc = subprocess.run(
        [sys.executable, drv, _SRC, idx_path, expect_path,
         json.dumps(list(k_values)), json.dumps(list(det_names))],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"reload subprocess failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, f"no RESULT line in subprocess output:\n{proc.stdout}"
    return json.loads(line[-1][len("RESULT "):])


def test_save_load_separate_process_no_rescan():
    """Build, READ (populate fds), save; reload in a SEPARATE process and assert
    reads are byte-identical AND the reload opened only the index file -- no SMD
    file reopened, ZERO SMD bytes read on the load+read path."""
    ridx = _build_index(DIR, EXP, RUN)
    det_names = _raw_det_names(ridx.run_config)
    try:
        # read several positions FIRST (incl. 0 and the last) so the saved
        # index would carry stale fds if the helper failed to strip them.
        for k in K_VALUES:
            ridx.read_event_at(k)
        assert ridx._bd_fds, "fds should be open after reads"
        orig_smd = ridx.smd_bytes_read

        with tempfile.TemporaryDirectory() as td:
            idx_path = os.path.join(td, "r51.index")
            expect_path = os.path.join(td, "expect.pkl")
            ridx.save(idx_path)
            _expected_for_subprocess(ridx, det_names, K_VALUES, expect_path)

            res = _run_reload_subprocess(idx_path, expect_path,
                                         K_VALUES, det_names)

        assert res["reads_match"], \
            f"reloaded reads differ at k={res.get('fail_k')}"
        assert res["bd_fds_empty_on_load"], \
            "reloaded index must start with an EMPTY fd cache (gotcha guard)"
        assert res["ts_to_k_none_on_load"], "reloaded _ts_to_k must be None"
        # the load+read path opened NO smalldata file and read 0 SMD bytes:
        # reload is instant, no SMD rescan.
        assert res["smd_files_opened_during_reload"] == [], (
            "reload opened SMD files -- it must open ONLY the index file: "
            f"{res['smd_files_opened_during_reload']}")
        assert res["smd_bytes_read_during_reload"] == 0, (
            "reload read SMD bytes -- expected 0 (no rescan): "
            f"{res['smd_bytes_read_during_reload']}")
        # build provenance is preserved across the disk round-trip.
        assert res["smd_bytes_read_attr"] == orig_smd, \
            "persisted smd_bytes_read (build provenance) should survive save"
        assert res["n_events"] == ridx.n_events
        assert res["env_records_match"], \
            "reloaded env_records content must equal the original " \
            "(disk round-trip via pickle)"
        assert res["env_probe_match"], \
            "reloaded env as_of probe must match the original " \
            "(disk round-trip via pickle)"
    finally:
        ridx.close()
    print(f"[disk] save -> separate-process load: byte-identical reads at "
          f"K={K_VALUES}; 0 SMD files opened, 0 SMD bytes read on reload "
          f"(build read {orig_smd/1e6:.1f} MB SMD, preserved)")


# ==========================================================================
# 4. multi-chunk coverage: c001+ chunk paths inside entries survive
# ==========================================================================
def test_multichunk_roundtrip_preserves_chunk_paths():
    """On the multi-chunk run, several entries reference c001 bigdata chunk
    files (after the chunk roll).  The save/reload AND to_dict/from_dict
    round-trips must preserve those per-entry chunk paths byte-for-byte and read
    post-roll events identically."""
    import psdata
    ridx = _build_index(MC_DIR, MC_EXP, MC_RUN)
    det_names = _raw_det_names(ridx.run_config)
    try:
        assert ridx.multichunk_streams, \
            "expected multichunk_streams on the multi-chunk run"
        # find a few event positions whose entry references a c001 chunk path.
        post_roll_ks = []
        for k, entry in enumerate(ridx.entries):
            if any("c001" in cp for (cp, _o, _s) in entry.values()):
                post_roll_ks.append(k)
            if len(post_roll_ks) >= 3:
                break
        assert post_roll_ks, "no c001 chunk paths found in entries"
        # also include 0 and the last event.
        ks = sorted(set([0, ridx.n_events - 1] + post_roll_ks))

        # read first so fds populate, then both round-trips.
        for k in ks:
            ridx.read_event_at(k)

        # --- in-memory round-trip
        clone = psdata.RunIndex.from_dict(ridx.to_dict())
        # --- disk round-trip (separate process)
        with tempfile.TemporaryDirectory() as td:
            idx_path = os.path.join(td, "mc.index")
            expect_path = os.path.join(td, "expect.pkl")
            ridx.save(idx_path)
            _expected_for_subprocess(ridx, det_names, ks, expect_path)
            res = _run_reload_subprocess(idx_path, expect_path, ks, det_names)

        try:
            # chunk paths preserved exactly (multichunk_streams + per-entry).
            assert clone.multichunk_streams == ridx.multichunk_streams
            assert clone.chunk_files == ridx.chunk_files
            for k in ks:
                assert clone.entries[k] == ridx.entries[k], \
                    f"entry chunk path/offset/size at k={k} not preserved"
                a = _event_fingerprint(ridx.read_event_at(k), det_names)
                b = _event_fingerprint(clone.read_event_at(k), det_names)
                assert _fingerprints_equal(a, b), \
                    f"multichunk to_dict read k={k} not byte-identical"
        finally:
            clone.close()

        assert res["reads_match"], \
            f"multichunk disk reload differs at k={res.get('fail_k')}"
        assert res["smd_files_opened_during_reload"] == []
        assert res["smd_bytes_read_during_reload"] == 0
    finally:
        ridx.close()
    print(f"[multichunk] {MC_EXP}/r{MC_RUN}: c001 chunk paths in entries "
          f"survive save/reload + to_dict/from_dict; post-roll reads "
          f"byte-identical (ks={ks}, multichunk_streams="
          f"{sorted(ridx.multichunk_streams) if False else 'preserved'})")


# ==========================================================================
# 5. import purity unchanged (numpy-only; pickle is stdlib)
# ==========================================================================
def test_import_purity_subprocess():
    code = (
        "import sys, psdata, psdata.index as i; "
        "i.RunIndex.save; i.RunIndex.load; i.RunIndex.to_dict; "
        "bad=[m for m in ('psana','mpi4py','h5py','torch','ray') "
        "if m in sys.modules]; "
        "print('BAD' if bad else 'OK', bad)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": _SRC},
    )
    assert proc.returncode == 0, \
        f"import-purity subprocess failed:\n{proc.stderr}"
    assert proc.stdout.startswith("OK"), \
        f"import psdata leaked a framework: {proc.stdout.strip()}"
    print(f"[purity] import psdata stays numpy-only with US-008 "
          f"({proc.stdout.strip()})")


if __name__ == "__main__":
    test_import_purity_subprocess()
    print("[ok] import purity (subprocess)")
    test_persist_state_fields_exact()
    print("[ok] persisted state fields exact")
    test_to_dict_from_dict_byte_identical()
    print("[ok] in-memory round-trip byte-identical")
    test_save_load_separate_process_no_rescan()
    print("[ok] disk save/load separate process, no SMD rescan")
    test_multichunk_roundtrip_preserves_chunk_paths()
    print("[ok] multi-chunk c001 chunk paths survive persistence")
    print("\nALL US-008 TESTS PASSED")
