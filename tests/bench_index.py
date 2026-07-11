#!/usr/bin/env python3
"""bench_index.py -- psdata-vs-psana random-access benchmark (docs/benchmark-plan.md).

Measures the ONE thing the plan exists to measure: the frames psana reads that
psdata skips, along two axes -- event-count K in {100,1000,10000} and per-event
size (jungfrau big frame vs epixquad small frame).

  * A  (Scan Amplification, the HEADLINE, un-fakeable, hardware-independent):
        A = datagrams_iterated_to_deliver_K / K.
        psdata A == 1.000 by construction (read_events touches only the K
        targeted frames via os.pread; ZERO scan at query time -- proven here by
        asserting smd_bytes_read / scan_bytes_read are unchanged across a query).
        psana A is MEASURED: the forward events() loop is instrumented to count
        events stepped until it breaks after max(ks). byteratio (rchar) == A.
  * S  (wall-clock speedup, COLD, corroboration only): S = t_psana / t_psdata
        for the SAME ks, psdata-first so psana stays cold, fadvise(DONTNEED)
        between passes + disjoint draws for the cold discipline. Reported and
        explained, NEVER the gate (A gates; S < 1 is real at small scan-depth).

A byte-exact CORRECTNESS GATE runs FIRST and gates the clock (fail-closed): if
the single suspect table is non-empty, NO timing number is emitted and the
process exits non-zero. The gate consolidates every divergence at once --
raw (byte-exact) /\ NaN-aware calib /\ ts round-trip /\ count equality /\ epix
active seg-config /\ import purity -- reusing the proven oracles
stress/stress_randaccess.py and stress/stress_xcheck.py (pscalib repo) as the
method (re-implemented here as one consolidated, memory-bounded pass; their
shape is scripts, not import-clean libraries).

Timing is valid ONLY on a milano compute node: a self-guard refuses to emit any
S / latency / build number off a batch node (see require_batch_node).

Memory guardrail: never materialize K x frame (335 GB at large-K jungfrau).
Every read walks ks in <= SUBBATCH-frame sub-batches, read-and-discard,
accumulating only scalars.

Run (one detector per node, two nodes in parallel) -- see tests/bench_index.sbatch:
    source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
    PYTHONPATH=<pscalib>/src:<psdata>/src BENCH_DET=jungfrau python tests/bench_index.py

Env overrides (run_tests.sh style; CONFIG is otherwise hardcoded):
    BENCH_DET     -- restrict to one detector spec ("jungfrau" | "epixquad")
    BENCH_KS      -- comma list overriding K in {100,1000,10000}
    BENCH_SEED    -- override SEED (default 1337)
    BENCH_MAXPOS  -- SMOKE ONLY: cap the sampling universe to the first M
                     positions (probe-style Mcap); relaxes the hard A-expectation
                     gate (A becomes M/K, not N/K) and labels rows SMOKE.
    BENCH_SKIP_BIGDATA_BUILD -- skip the ~5 min bigdata-scan build KPI (smoke).
    BENCH_WORKDIR -- scratch dir for the saved index + constants npz
                     (default /lscratch/<user>/bench_<jobid>).
    BENCH_ALLOW_NONBATCH -- dev escape hatch: run the gate/A anywhere, still
                     refuses to TIME off a batch node unless =1.
"""
import hashlib
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

# ==========================================================================
# CONFIG -- hardcoded module-level block (docs/benchmark-plan.md Sections 1, 5)
# ==========================================================================
SEED = int(os.environ.get("BENCH_SEED", "1337"))
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
SUBBATCH = 32                      # memory guardrail: <= 32 frames materialized
ADV_K = 1000                       # K used for the adversarial / psana-best rows
LAT_N = 50                         # single-event-latency sample size

# Two detectors, ~31x apart in bytes/event (the load-bearing size axis).
DETECTORS = [
    dict(name="jungfrau", exp="mfx100848724", run=51, det="jungfrau",
         N=17872, ev_mb=33.5),
    dict(name="epixquad", exp="ued1010667", run=177, det="epixquad",
         N=73800, ev_mb=1.08),
]

_ks_env = os.environ.get("BENCH_KS")
KS = ([int(x) for x in _ks_env.split(",") if x.strip()] if _ks_env
      else [100, 1000, 10000])
MAXPOS = int(os.environ.get("BENCH_MAXPOS", "0"))        # 0 => full N
SMOKE = MAXPOS > 0
SKIP_BIGDATA_BUILD = os.environ.get("BENCH_SKIP_BIGDATA_BUILD") == "1"
# Gate does RAW byte-exact on ALL timing positions (cheap), but NaN-aware calib
# on a large EVENLY-SPREAD subsample: calib is a deterministic per-pixel map, so
# once raw is verified everywhere and calib on this spread (+ the seg-config and
# image checks), every-single-frame calib (~600 ms/frame pure-Python decode x2)
# adds negligible assurance for hours of wall-clock. 0 => calib on every target.
GATE_CALIB_CAP = int(os.environ.get("BENCH_GATE_CALIB_CAP", "3000"))


# ==========================================================================
# Self-guard -- timing is invalid off a milano compute node
# ==========================================================================
def on_batch_node():
    """(ok, host, why). A batch node has $SLURM_JOB_ID set and is NOT in the
    shared iana/psana interactive pool nor a login node (milano nodes are
    sdfmilanNNN)."""
    host = socket.gethostname()
    jid = os.environ.get("SLURM_JOB_ID")
    low = host.lower()
    if not jid:
        return False, host, "no $SLURM_JOB_ID (not under Slurm)"
    if low.startswith("sdfiana") or low.startswith("psana") or "login" in low:
        return False, host, f"interactive/login pool host {host!r}"
    return True, host, "ok"


def require_batch_node():
    """Refuse to emit any timing / latency / build number off a batch node."""
    ok, host, why = on_batch_node()
    if not ok and os.environ.get("BENCH_ALLOW_NONBATCH") != "1":
        print(f"REFUSE: timing is invalid off a milano compute node ({why}); "
              f"host={host}. Submit tests/bench_index.sbatch, or set "
              f"BENCH_ALLOW_NONBATCH=1 to override (numbers will be noise).")
        sys.exit(2)
    return host


# ==========================================================================
# Low-level helpers
# ==========================================================================
def rchar():
    """Bytes this process has read (node-independent byte accounting)."""
    try:
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("rchar:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def all_chunk_paths(ridx):
    """Every distinct bigdata chunk file the index reads (rolled chunks too)."""
    paths = []
    cf = getattr(ridx, "chunk_files", None)
    if cf:
        for chunks in cf.values():
            paths.extend(chunks)
    else:
        paths.extend(getattr(ridx, "bd_files", {}).values())
    return sorted(set(paths))


def fadvise_dontneed(paths):
    """Best-effort page-cache eviction (POSIX_FADV_DONTNEED) for each path, so a
    re-read starts cold. 'only advice the kernel may decline' -- combined with
    --exclusive + disjoint draws this is the cold discipline (Section 2)."""
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


def derive_rng(detector, K, row_type):
    """Per-cell rng, derived (not drawn from a shared stream) so each cell is
    reproducible in isolation regardless of grid-traversal order. Stable across
    processes (sha1, NOT Python's salted hash())."""
    h = hashlib.sha1(f"{SEED}-{detector}-{K}-{row_type}".encode()).hexdigest()
    return np.random.default_rng(int(h[:8], 16))


def universe(N):
    """Sampling universe: full N, or the first MAXPOS positions in smoke mode."""
    return min(N, MAXPOS) if MAXPOS else N


def sample_random_disjoint(detector, K, N):
    """R = min(3, floor(U/K)) DISJOINT sorted draws (no frame re-read across
    reps -> doubles as cold discipline). No forced boundary (kept OUT of the
    timed random draw). Returns list[np.ndarray]."""
    U = universe(N)
    R = max(1, min(3, U // K))
    rng = derive_rng(detector, K, "random")
    pool = rng.choice(U, R * K, replace=False)
    return [np.sort(pool[i * K:(i + 1) * K]) for i in range(R)]


def sample_adversarial(detector, N):
    """ks forced to include N-1 -> scan-fraction == 1.0, deterministic worst
    case. One labeled row, never blended into the headline."""
    U = universe(N)
    rng = derive_rng(detector, ADV_K, "adversarial")
    head = rng.choice(U - 1, ADV_K - 1, replace=False)
    return np.sort(np.append(head, U - 1))


def sample_psana_best(detector, N):
    """ks confined so max(ks) ~ N/2 -> psana scans only ~half the run. One
    labeled row: proves the win survives psana's most favorable case."""
    U = universe(N)
    rng = derive_rng(detector, ADV_K, "psana-best")
    return np.sort(rng.choice(U // 2, ADV_K, replace=False))


# ==========================================================================
# psdata / psana timed read primitives (both <= SUBBATCH frames at a time)
# ==========================================================================
def psdata_read_timed(ridx, ks, det):
    """Deliver+decode the K frames via read_events in <= SUBBATCH sub-batches,
    read-and-discard. Returns (secs, bytes, scan_clean) where scan_clean asserts
    the query did ZERO SMD/bigdata scan (A_psdata == 1.000, by construction)."""
    ks = list(int(k) for k in ks)
    smd0, scan0 = ridx.smd_bytes_read, ridx.scan_bytes_read
    acc = 0.0
    r0 = rchar()
    t0 = time.monotonic()
    for i in range(0, len(ks), SUBBATCH):
        sub = ks[i:i + SUBBATCH]
        for evt in ridx.read_events(sub):
            st = evt.stack(det)
            if st is not None:
                acc += float(st.shape[0])         # touch -> force decode
        # sub + its events drop out of scope here (read-and-discard)
    secs = time.monotonic() - t0
    nbytes = rchar() - r0
    scan_clean = (ridx.smd_bytes_read == smd0 and ridx.scan_bytes_read == scan0)
    return secs, nbytes, scan_clean


def psana_forward_timed(exp, run, det, ks):
    """Steelman psana retrieval: ONE sorted forward pass of prun.events(),
    break after max(ks), raw() only on the K sampled positions. The forward
    pass reads bigdata for ALL max(ks)+1 events (no random seek). DataSource /
    Detector construction is NOT timed (warmup symmetry). Returns
    (secs, bytes, iterated, got)."""
    from psana import DataSource
    ks = sorted(int(k) for k in ks)
    kset = set(ks)
    kmax = ks[-1]
    ds = DataSource(exp=exp, run=run, dir=DIR)
    prun = next(ds.runs())
    pdet = prun.Detector(det)
    r0 = rchar()
    t0 = time.monotonic()                          # time from the FIRST step
    i = got = 0
    for evt in prun.events():
        if i in kset:
            if pdet.raw.raw(evt) is not None:
                got += 1
        if i >= kmax:
            break
        i += 1
    secs = time.monotonic() - t0
    nbytes = rchar() - r0
    return secs, nbytes, i + 1, got


# ==========================================================================
# Amortized one-time KPIs (Section 2) -- reported separately, never folded into S
# ==========================================================================
def measure_kpis(spec, workdir):
    import psdata
    name = spec["name"]
    kpis = {}

    # SMD index build (the fast path; target <= 30 s)
    r = psdata.open(exp=spec["exp"], run=spec["run"], dir=DIR)
    t0 = time.monotonic()
    ridx = r.build_index(rebuild=True, source="smd")
    kpis["smd_build_s"] = time.monotonic() - t0
    kpis["n_events"] = ridx.n_events

    # persist round-trip (save / load) + on-disk size
    path = os.path.join(workdir, f"{name}.idx")
    t0 = time.monotonic()
    ridx.save(path)
    kpis["save_s"] = time.monotonic() - t0
    kpis["idx_mb"] = os.path.getsize(path) / 1e6
    t0 = time.monotonic()
    psdata.RunIndex.load(path)
    kpis["load_s"] = time.monotonic() - t0

    # bigdata index build (no SMD; target <= 5 min) -- skippable in smoke
    if SKIP_BIGDATA_BUILD:
        kpis["bigdata_build_s"] = float("nan")
    else:
        t0 = time.monotonic()
        rb = r.build_index(rebuild=True, source="bigdata")
        kpis["bigdata_build_s"] = time.monotonic() - t0
        kpis["bigdata_n_events"] = rb.n_events

    r.close()
    return kpis, path


# ==========================================================================
# Correctness gate (Section 3) -- ONE suspect table; gates the clock
# ==========================================================================
def eq_raw(a, b):
    return a is not None and b is not None and a.shape == b.shape \
        and np.array_equal(a, b)


def eq_calib(a, b):
    # NaN-aware: jungfrau calib masks bad pixels to NaN; zero tolerance on
    # finite pixels (no rtol/atol ever).
    return a is not None and b is not None and a.shape == b.shape \
        and np.array_equal(a, b, equal_nan=True)


def run_gate(spec, ridx, gate_ks, workdir):
    """Consolidated byte-exact gate on the EXACT timing ks. Returns (suspects,
    dettype, npz_path). suspects == [] means the clock may start."""
    import psdata
    import pscalib
    from psana import DataSource

    det = spec["det"]
    suspects = []                          # (cell, suspect, fix)

    ds = DataSource(exp=spec["exp"], run=spec["run"], dir=DIR)
    prun = next(ds.runs())
    pdet = prun.Detector(det)
    constants = None
    try:
        constants = pdet.raw._calibconst
    except Exception as e:                  # noqa: BLE001
        print(f"[gate:{spec['name']}] no _calibconst: {e}")
    dettype = getattr(pdet.raw, "_dettype", None) or det
    is_epix = "epix" in str(dettype).lower()

    # seg_configs comes from the opened Run, not the index; open one cheap Run.
    run_obj = psdata.open(exp=spec["exp"], run=spec["run"], dir=DIR)
    seg_cfg = None
    if is_epix:
        try:
            seg_cfg = run_obj.seg_configs(det)
        except Exception as e:              # noqa: BLE001
            suspects.append((f"{spec['name']}/seg-config",
                             f"run.seg_configs({det!r}) raised {e!r}",
                             "expose the BeginStep-active config (run.py seg_configs)"))

    # ---- ONE full psana forward pass: count + ts-set + raw + NaN-aware calib
    # on the exact timing ks (compare-and-discard; O(1) memory). In SMOKE mode
    # we break after the last target (the whole-run count/ts-set checks need a
    # full pass, so they are skipped under smoke). ----
    targets = set(int(k) for k in gate_ks)
    kmax = max(targets) if targets else 0
    # RAW on every target; NaN-aware calib on an evenly-spread subsample.
    st = sorted(targets)
    if GATE_CALIB_CAP and len(st) > GATE_CALIB_CAP:
        step = len(st) / GATE_CALIB_CAP
        calib_targets = set(st[int(j * step)] for j in range(GATE_CALIB_CAP))
    else:
        calib_targets = set(st)
    raw_mism = cal_mism = cal_err = checked = calib_checked = 0
    psana_count = 0
    ts_psana = []
    smd0, scan0 = ridx.smd_bytes_read, ridx.scan_bytes_read
    for i, evt in enumerate(prun.events()):
        if SMOKE and i > kmax:
            break
        ts_psana.append(int(evt.timestamp))
        psana_count += 1
        if i in targets:
            praw = pdet.raw.raw(evt)
            try:
                draw = ridx.read_event_at(i).stack(det)
            except Exception as e:          # noqa: BLE001
                suspects.append((f"{spec['name']}/raw@{i}",
                                 f"psdata read_event_at({i}) raised {e!r}",
                                 "read_event_at (index.py:401)"))
                continue
            checked += 1
            if not eq_raw(draw, praw):
                raw_mism += 1
                if raw_mism <= 3:
                    ds_ = None if draw is None else draw.shape
                    suspects.append((f"{spec['name']}/raw@{i}",
                                     f"raw mismatch: psdata {ds_} vs psana "
                                     f"{None if praw is None else praw.shape}",
                                     "read_events (index.py:453) byte path"))
                continue
            if constants is not None and i in calib_targets:
                calib_checked += 1
                pcal = pdet.raw.calib(evt)
                try:
                    mcal = pscalib.calib(dettype, draw, constants, config=seg_cfg)
                except Exception as e:      # noqa: BLE001
                    cal_err += 1
                    if cal_err <= 3:
                        suspects.append((f"{spec['name']}/calib@{i}",
                                         f"pscalib.calib raised {e!r}",
                                         "pscalib registry.py:267 / apply plugin"))
                    continue
                if not eq_calib(mcal, pcal):
                    cal_mism += 1
                    if cal_mism <= 3:
                        suspects.append((f"{spec['name']}/calib@{i}",
                                         f"NaN-aware calib mismatch "
                                         f"({None if mcal is None else mcal.shape} "
                                         f"vs {None if pcal is None else pcal.shape})",
                                         "Imager.calib (render.py:122) / active seg-config"))

    # query did zero scan (A_psdata == 1.000 by construction)
    if not (ridx.smd_bytes_read == smd0 and ridx.scan_bytes_read == scan0):
        suspects.append((f"{spec['name']}/A", "query mutated scan counters "
                         "(read path is NOT scan-free)",
                         "read_events must not rescan SMD/bigdata"))

    # ---- count equality + ts SETS equal (catches ragged-shutdown-tail off-by-N)
    # Needs a full forward pass, so skipped under SMOKE (partial pass).
    if not SMOKE:
        if psana_count != ridx.n_events:
            suspects.append((f"{spec['name']}/count",
                             f"psana {psana_count} != psdata {ridx.n_events} L1 events",
                             "index completeness (index.py build / SMD gate)"))
        if set(ts_psana) != set(ridx.timestamps):
            d1 = len(set(ts_psana) - set(ridx.timestamps))
            d2 = len(set(ridx.timestamps) - set(ts_psana))
            suspects.append((f"{spec['name']}/ts-set",
                             f"timestamp sets differ (+{d1} psana-only, +{d2} psdata-only)",
                             "build_from_bigdata shutdown-tail clamp"))
    else:
        print(f"[gate:{spec['name']}] SMOKE: count/ts-set checks skipped "
              f"(need a full forward pass)")

    # ---- ts round-trip + absent-ts KeyError
    for k in sorted(targets)[:5]:
        ts = ridx.timestamps[k]
        try:
            a = ridx.read_event(ts).stack(det)
            b = ridx.read_event_at(k).stack(det)
            if not eq_raw(a, b):
                suspects.append((f"{spec['name']}/ts-roundtrip@{k}",
                                 "read_event(ts) != read_event_at(k)",
                                 "_position_of (index.py:363)"))
        except Exception as e:              # noqa: BLE001
            suspects.append((f"{spec['name']}/ts-roundtrip@{k}",
                             f"read_event(ts) raised {e!r}", "index.py:388"))
    try:
        ridx.read_event(1)                  # ts=1 cannot exist
        suspects.append((f"{spec['name']}/absent-ts",
                         "read_event(absent ts) did NOT raise KeyError",
                         "_position_of must raise KeyError"))
    except KeyError:
        pass

    # ---- request-order de-permute (descending + duplicate ks)
    tl = sorted(targets)
    if len(tl) >= 3:
        bks = [tl[-1], tl[0], tl[len(tl) // 2], tl[0]]   # descending-ish + dup
        bevts = ridx.read_events(bks)
        for j, k in enumerate(bks):
            single = ridx.read_event_at(k).stack(det)
            if not eq_raw(bevts[j].stack(det), single):
                suspects.append((f"{spec['name']}/depermute[{j}]",
                                 f"read_events returned read-order at request {j} "
                                 f"(k={k})",
                                 "de-permute coalesced reads by request index "
                                 "(index.py:453)"))

    # ---- epix active seg-config (the psana _seg_configs statefulness pitfall)
    if is_epix and seg_cfg is not None:
        try:
            pseg = pdet.raw._seg_configs()
            for s in sorted(set(seg_cfg) & set(pseg)):
                pt = np.asarray(pseg[s].config.trbit)
                mt = np.asarray(seg_cfg[s].config.trbit)
                if not np.array_equal(pt, mt):
                    suspects.append((f"{spec['name']}/seg-config[{s}]",
                                     "trbit != psana active _seg_configs",
                                     "expose BeginStep-active config (run.py)"))
                    break
        except Exception as e:              # noqa: BLE001
            print(f"[gate:{spec['name']}] seg-config compare skipped: {e!r}")

    # ---- import purity: assert in a CLEAN subprocess (psana is the in-proc
    # oracle, so the assert cannot run here). Dump unwrapped constants for it.
    npz_path = os.path.join(workdir, f"{spec['name']}_constants.npz")
    _dump_constants(constants, npz_path)
    ok, why = purity_attest(spec, dettype, npz_path)
    if not ok:
        suspects.append((f"{spec['name']}/purity",
                         f"clean-room import attest failed: {why}",
                         "psdata/pscalib must import none of the framework"))

    run_obj.close()
    print(f"[gate:{spec['name']}] raw_checked={checked} (all targets) "
          f"calib_checked={calib_checked}/{len(calib_targets)} "
          f"raw_mism={raw_mism} calib_mism={cal_mism} calib_err={cal_err} "
          f"psana_count={psana_count} n_events={ridx.n_events} dettype={dettype}")
    return suspects, dettype, npz_path


def _dump_constants(constants, npz_path):
    """Dump psana _calibconst (unwrapped ndarrays) to npz so the purity
    subprocess can apply calib WITHOUT importing psana."""
    if constants is None:
        np.savez(npz_path)
        return
    out = {}
    for k in constants:
        v = constants[k]
        if isinstance(v, (tuple, list)) and v and isinstance(v[0], np.ndarray):
            v = v[0]
        if isinstance(v, np.ndarray):
            out[k] = v
    np.savez(npz_path, **out)


_PURITY_SRC = r'''
import sys, numpy as np, psdata, pscalib
exp, run, dir_, det, dettype, npz, is_epix = (
    {exp!r}, {run!r}, {dir_!r}, {det!r}, {dettype!r}, {npz!r}, {is_epix!r})
r = psdata.open(exp=exp, run=run, dir=dir_)
raw = r.read_event_at(0).stack(det)
psdata.format.assert_no_framework_imports()          # read path clean
cons = {{k: np.load(npz)[k] for k in np.load(npz).files}}
cfg = r.seg_configs(det) if is_epix else None
cal = pscalib.calib(dettype, raw, cons, config=cfg)  # apply path
pscalib.assert_no_framework_imports()                # apply path clean
print("PURITY=PASS", None if cal is None else cal.shape)
'''


def purity_attest(spec, dettype, npz_path):
    """Run the import-purity assert in a FRESH interpreter that never imports
    psana (Section 3: 'asserted on imports... fresh interpreter')."""
    src = _PURITY_SRC.format(exp=spec["exp"], run=spec["run"], dir_=DIR,
                             det=spec["det"], dettype=dettype, npz=npz_path,
                             is_epix="epix" in str(dettype).lower())
    try:
        p = subprocess.run([sys.executable, "-c", src], capture_output=True,
                           text=True, timeout=600)
    except Exception as e:                  # noqa: BLE001
        return False, f"subprocess error {e!r}"
    if p.returncode == 0 and "PURITY=PASS" in p.stdout:
        return True, p.stdout.strip().splitlines()[-1]
    return False, (p.stdout + p.stderr).strip()[-300:]


def print_suspect_table(suspects):
    print("\n#### Suspect table (Section 3 -- empty == gate PASS)\n")
    print("| cell | suspect | likely fix |")
    print("|------|---------|------------|")
    if not suspects:
        print("| *(empty)* | -- | -- |")
    for cell, suspect, fix in suspects:
        print(f"| {cell} | {suspect} | {fix} |")
    print()


# ==========================================================================
# Timing (Section 2) -- runs ONLY after the gate passes, ONLY on a batch node
# ==========================================================================
def time_cell(spec, ridx, ks_draws, row, K):
    """Median over the disjoint draws of (A, byteratio, S, scanfrac, t_*).
    psdata-first per draw so psana stays cold; fadvise(DONTNEED) before each
    pass for the cold start."""
    bd_files = all_chunk_paths(ridx)
    A_l, br_l, S_l, sf_l, tpa_l, tpd_l, mbpa_l, mbpd_l = ([] for _ in range(8))
    for ks in ks_draws:
        K_eff = len(ks)
        fadvise_dontneed(bd_files)
        t_pd, b_pd, clean = psdata_read_timed(ridx, ks, spec["det"])
        fadvise_dontneed(bd_files)
        t_pa, b_pa, iterated, got = psana_forward_timed(
            spec["exp"], spec["run"], spec["det"], ks)
        A = iterated / K_eff
        br = b_pa / max(b_pd, 1)
        S = (t_pa / t_pd) if t_pd > 0 else float("nan")
        A_l.append(A); br_l.append(br); S_l.append(S)
        sf_l.append((iterated) / spec["N"])
        tpa_l.append(t_pa); tpd_l.append(t_pd)
        mbpa_l.append(b_pa / 1e6); mbpd_l.append(b_pd / 1e6)
    med = statistics.median
    return dict(det=spec["name"], row=row, K=K, R=len(ks_draws),
                A=med(A_l), byteratio=med(br_l), S=med(S_l), scanfrac=med(sf_l),
                t_psana=med(tpa_l), t_psdata=med(tpd_l),
                MB_psana=med(mbpa_l), MB_psdata=med(mbpd_l),
                Amatch=abs(med(br_l) - med(A_l)) <= 0.05 * max(med(A_l), 1),
                scan_clean=clean)


def latency_us(spec, ridx):
    """Corroborative single-event latency: median us/ev of read_event_at over
    scrambled ks (the random primitive psana cannot do without re-iterating)."""
    rng = derive_rng(spec["name"], 0, "latency")
    ks = rng.choice(universe(spec["N"]), min(LAT_N, universe(spec["N"])),
                    replace=False).tolist()
    fadvise_dontneed(all_chunk_paths(ridx))
    samples = []
    for k in ks:
        t0 = time.monotonic()
        ridx.read_event_at(int(k)).stack(spec["det"])
        samples.append((time.monotonic() - t0) * 1e6)
    return statistics.median(samples)


# ==========================================================================
# Main
# ==========================================================================
def emit_result_line(r):
    print(f"RESULT det={r['det']:<9s} row={r['row']:<11s} K={r['K']:<6d} "
          f"R={r['R']} | A_psana={r['A']:8.1f} A_psdata=1.000 "
          f"byteratio={r['byteratio']:8.1f} Amatch={int(r['Amatch'])} "
          f"scanfrac={r['scanfrac']:.3f} | "
          f"t_psana={r['t_psana']:7.2f}s t_psdata={r['t_psdata']:7.3f}s "
          f"S={r['S']:6.2f} | MB_psana={r['MB_psana']:8.0f} "
          f"MB_psdata={r['MB_psdata']:7.0f}", flush=True)


def print_markdown_table(rows):
    print("\n### Table A -- random-access grid (A headline; byteratio==A; cold S)\n")
    print("| detector | row | K | R | A_psana | byteratio | A==byteratio | "
          "scan-frac | cold S | t_psana (s) | t_psdata (s) |")
    print("|----------|-----|--:|--:|--------:|----------:|:------------:|"
          "---------:|-------:|------------:|-------------:|")
    for r in rows:
        print(f"| {r['det']} | {r['row']} | {r['K']} | {r['R']} | "
              f"{r['A']:.1f} | {r['byteratio']:.1f} | "
              f"{'yes' if r['Amatch'] else 'NO'} | {r['scanfrac']:.3f} | "
              f"{r['S']:.2f} | {r['t_psana']:.2f} | {r['t_psdata']:.3f} |")
    print()


def main():
    import psdata
    host = require_batch_node()
    specs = [s for s in DETECTORS
             if not os.environ.get("BENCH_DET")
             or s["name"] == os.environ["BENCH_DET"]]
    if not specs:
        print(f"no detector matches BENCH_DET={os.environ.get('BENCH_DET')!r}")
        sys.exit(2)

    user = os.environ.get("USER", "nobody")
    jid = os.environ.get("SLURM_JOB_ID", "nojob")
    workdir = os.environ.get("BENCH_WORKDIR") or f"/lscratch/{user}/bench_{jid}"
    try:
        os.makedirs(workdir, exist_ok=True)
    except OSError:
        workdir = tempfile.mkdtemp(prefix="bench_")

    print(f"=== bench_index  host={host}  job={jid}  smoke={SMOKE} "
          f"maxpos={MAXPOS or 'fullN'}  K={KS}  seed={SEED}  workdir={workdir} ===",
          flush=True)

    all_rows = []
    for spec in specs:
        print(f"\n========== {spec['name']} "
              f"(exp={spec['exp']} run={spec['run']} N={spec['N']} "
              f"~{spec['ev_mb']} MB/ev) ==========", flush=True)

        # ----- KPIs (one-time, timed apart) -----
        kpis, idx_path = measure_kpis(spec, workdir)
        print(f"[kpi:{spec['name']}] smd_build={kpis['smd_build_s']:.1f}s "
              f"(<=30s) bigdata_build={kpis['bigdata_build_s']:.1f}s (<=300s) "
              f"save={kpis['save_s']:.3f}s load={kpis['load_s']:.3f}s "
              f"(<=0.5s) idx={kpis['idx_mb']:.2f}MB (<=8MB) "
              f"n_events={kpis['n_events']}", flush=True)

        # ----- reuse check: load (no rebuild, no rescan) then query -----
        ridx = psdata.RunIndex.load(idx_path)
        _ = ridx.read_event_at(0)               # discard first (lazy fd open)

        # ----- build the timing ks ONCE (gate runs on the exact timing ks) -----
        random_draws = {K: sample_random_disjoint(spec["name"], K, spec["N"])
                        for K in KS}
        adv_ks = sample_adversarial(spec["name"], spec["N"])
        best_ks = sample_psana_best(spec["name"], spec["N"])
        gate_ks = set()
        for K in KS:
            gate_ks |= set(int(x) for x in random_draws[K][0])  # first draw
        gate_ks |= set(int(x) for x in adv_ks)
        gate_ks |= set(int(x) for x in best_ks)

        # ----- GATE (Section 3) -- runs FIRST, gates the clock -----
        print(f"\n--- correctness gate: {spec['name']} "
              f"({len(gate_ks)} timing positions) ---", flush=True)
        suspects, dettype, _npz = run_gate(spec, ridx, gate_ks, workdir)
        print_suspect_table(suspects)
        if suspects:
            print(f"GATE FAIL [{spec['name']}]: {len(suspects)} suspect(s) -- "
                  f"NO timing emitted (fail-closed).", flush=True)
            ridx.close()
            sys.exit(1)
        print(f"GATE PASS [{spec['name']}]: suspect table empty -- "
              f"raw /\\ NaN-aware calib /\\ ts /\\ count /\\ seg-config /\\ purity "
              f"clean. Clock starts.", flush=True)

        # ===== TIMING (clock starts; A headline, cold S corroboration) =====
        print(f"\n=== TIMING (clock starts): {spec['name']} ===", flush=True)
        for K in KS:
            r = time_cell(spec, ridx, random_draws[K], "random", K)
            all_rows.append(r); emit_result_line(r)
        r = time_cell(spec, ridx, [adv_ks], "adversarial", ADV_K)
        all_rows.append(r); emit_result_line(r)
        r = time_cell(spec, ridx, [best_ks], "psana-best", ADV_K)
        all_rows.append(r); emit_result_line(r)

        lat = latency_us(spec, ridx)
        print(f"RESULT det={spec['name']:<9s} single_event_latency_us={lat:.0f} "
              f"(read_event_at, scrambled, cold)", flush=True)
        ridx.close()

    print_markdown_table(all_rows)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
