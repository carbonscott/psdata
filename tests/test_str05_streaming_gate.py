#!/usr/bin/env python3
"""STR-05 regression: ``Run.events(gate=True)`` must be an O(1)-memory streaming
merge-join, not a build-the-whole-index-then-filter.

The bug (STR-05)
----------------
``Run.events(gate=True)`` gates the forward bigdata merge to the canonical
(SMD-equivalent) event set so the ragged DAQ-shutdown tail is dropped
(FAIL-01/GATE-02).  It used to obtain that set by building the ENTIRE
random-access index first::

    merged = _s.events(self.files, run_config=self.config)   # O(1) forward stream
    index  = self.build_index()                              # <-- builds ALL N entries
    valid  = frozenset(index.timestamps)                     # <-- materializes ALL N ts
    return (evt for evt in merged if evt.timestamp in valid)

So although the bigdata merge streams, the *gate* forced the whole index --
every one of the run's N timestamps and the per-event x per-stream
``RunIndex.entries`` -- to be materialized BEFORE event 0 was yielded.  At 1e7
events that is minutes of index build and tens of GB resident before ``evt[0]``.
psana, by contrast, streams the first event immediately in bounded memory.

The fix (STR-05)
----------------
``events(gate=True)`` now merge-joins two ASCENDING timestamp streams -- the
forward bigdata merge and a STREAMING gate source
(:func:`psdata.index.iter_gate_timestamps`) that yields the canonical timestamps
incrementally, reading only a bounded window of the SMD (or bigdata-header)
source at a time.  The first event is yielded after reading only the first gate
entry and the first bigdata event; peak memory is one bigdata event + O(1)
cursor state.  The yielded SEQUENCE is byte-identical to the old
``frozenset(build_index().timestamps)`` filter, and ``build_index`` / random
access are unchanged.

What this test asserts (all four, on a synthetic on-disk xtc2 run with a bigdata
c000->c001 CHUNK ROLL mid-run AND a ragged TAIL of L1Accepts past the last
gated event):

  1. **O(1) startup (the discriminator).**  ``events(gate=True)`` yields event 0
     WITHOUT calling ``build_index`` (so without materializing all N entries) --
     spied via ``Run.build_index``.  On the PARENT ``events()`` calls
     ``build_index()`` (building the full index) before it even returns the
     generator, so the spy count is nonzero at ``evt[0]`` and THIS assertion
     fails.  On the FIX ``build_index`` is never called and the streaming gate
     yields <= a small constant timestamps before ``evt[0]`` (also checked, when
     the fix's ``iter_gate_timestamps`` is present) -- a genuine O(1)-vs-O(N)
     discriminator that cannot pass vacuously (a real ``evt[0]`` IS produced).
  2. **Correctness -- identical gated set.**  ``list(events(gate=True))``
     timestamps == ``build_index().timestamps`` EXACTLY (same order): the ragged
     tail is dropped and events on BOTH sides of the chunk roll appear
     (anti-regression for GATE-02/FAIL-01/STR-01), and coverage reaches the LAST
     gated event (the merge-join must not stop short).
  3. **gate=False still ungated.**  ``list(events(gate=False))`` INCLUDES the
     ragged-tail phantoms -- a strict superset of the gated set -- proving the
     two modes still differ and gate=False is unchanged.
  4. **Fail-closed preserved.**  When the gate source cannot be built (absent
     files), ``events(gate=True)`` raises ``GateBuildError`` (chaining the
     cause), while ``events(gate=False)`` still returns the ungated merge.
  5. **Fail-closed parity on a truncated sidecar (F2).**  The streaming SMD gate
     must RAISE on a sub-24-byte trailing remnant on the sidecar -- exactly as
     ``_scan_smd_stream`` (what ``build_index`` uses) does -- never silently
     yield a short gate that would make the merge-join drop the corresponding
     trailing bigdata events as if they were the ragged tail.

Self-contained: stdlib + numpy only -- NO psana, NO SLAC data.  It hand-builds
synthetic xtc2 dgrams byte-for-byte to ``src/psdata/format.py`` (the same idiom
as ``tests/test_write02_dgram_damage.py`` / ``tests/test_stream_us002.py``).
The run has NO smalldata/ sidecars, so ``build_index(source="auto")`` and the
gate source both take the bigdata-scan path (timing/master-stream clamp), which
is what drops the tail.  Contains no part of the fix; cwd-robust;
``main()`` + ``__main__``.
"""

import os
import struct
import sys
import tempfile
import types

# --- locate the package under test (parent-of-tests/src), cwd-robust ---------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")  # .../<repo>/src (holds psdata/)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import psdata                       # noqa: E402  (pulls numpy, psdata's dep)
import psdata.format as fmt         # noqa: E402
import psdata.index as pidx         # noqa: E402
from psdata.run import Run, GateBuildError  # noqa: E402


# ===========================================================================
# Hand-build synthetic xtc2 dgrams (byte-for-byte to src/psdata/format.py):
#   Transition = uint32 nsec, uint32 sec, uint32 env   (service=(env>>24)&0xf)
#   Xtc        = uint32 src, uint16 damage, uint16 typeid, uint32 extent
# A minimal empty-payload L1Accept is all the GATE needs: the canonical/tail
# split is decided by WHICH STREAM carries each timestamp (the timing clamp),
# not by any detector payload, so no ShapesData is required.
# ===========================================================================
def _pack_xtc(src, damage, tid, payload):
    extent = fmt.XTC_HDR + len(payload)
    return struct.pack("<IHHI", src, damage, tid, extent) + payload


def _dgram_svc(service, ts, top_payload):
    """One on-disk dgram: Transition(12) + a top PARENT Xtc wrapping
    ``top_payload``, with ``service`` in the top 4 bits of env (matching
    ``parse_dgram_header``'s ``service = (env >> 24) & 0xf``)."""
    nsec = ts & 0xFFFFFFFF
    sec = (ts >> 32) & 0xFFFFFFFF
    env = (service & 0xf) << 24
    return (struct.pack("<III", nsec, sec, env)
            + _pack_xtc(0, 0, fmt.TID_PARENT, top_payload))


def _make_l1(ts):
    """A minimal ``L1Accept`` dgram carrying timestamp ``ts`` and an empty
    top-Xtc payload (service 12, no ShapesData)."""
    return _dgram_svc(fmt.SERVICE_L1ACCEPT, ts, b"")


def _write(path, ts_list):
    with open(path, "wb") as fh:
        for ts in ts_list:
            fh.write(_make_l1(ts))


# --- SMD sidecar byte builders (used only by the F2 fail-closed parity check):
#     a Configure declaring the smdinfo/offsetAlg Names table plus smdinfo-bearing
#     L1Accepts, byte-for-byte to format.parse_configure / parse_names_block (the
#     same layout as tests/test_fail04_truncated_index.py, replicated here so this
#     file stays self-contained). ------------------------------------------------
_UINT64 = 3                # DataType.UINT64 (the rank-0 smdinfo fields)
_SMDINFO_SRC = 1           # namesid_of(1) == (nodeId=0, namesId=1)


def _cstr(s, n=256):
    b = s.encode("latin-1")[:n]
    return b + b"\x00" * (n - len(b))


def _smd_configure():
    """A Configure dgram declaring the ``smdinfo``/``offsetAlg`` Names table
    (fields ``intOffset``, ``intDgramSize``) the SMD scan requires."""
    fields = [("intOffset", _UINT64, 0), ("intDgramSize", _UINT64, 0)]
    p = struct.pack("<I", len(fields))                       # num_arrays
    p += _cstr("offset") + _cstr("smdinfo") + _cstr("detid-serial")
    p += _cstr("offsetAlg") + struct.pack("<I", 1) + struct.pack("<I", 0)
    assert len(p) == fmt.NAMEINFO_SZ, (len(p), fmt.NAMEINFO_SZ)
    for fname, ftype, frank in fields:
        nm = b"\x00" * fmt.ALG_SZ + _cstr(fname) + struct.pack("<II", ftype, frank)
        assert len(nm) == fmt.NAME_SZ, (len(nm), fmt.NAME_SZ)
        p += nm
    return _dgram_svc(fmt.SERVICE_CONFIGURE, 1,
                      _pack_xtc(_SMDINFO_SRC, 0, fmt.TID_NAMES, p))


def _smd_l1(ts, intoff, intsize=96):
    """An L1Accept carrying an ``smdinfo`` ShapesData (rank-0 ``intOffset`` /
    ``intDgramSize``) -- what ``_scan_smd_stream`` records a ts for."""
    data = _pack_xtc(0, 0, fmt.TID_DATA, struct.pack("<QQ", intoff, intsize))
    sd = _pack_xtc(_SMDINFO_SRC, 0, fmt.TID_SHAPESDATA, data)
    return _dgram_svc(fmt.SERVICE_L1ACCEPT, ts, sd)


def _write_bytes(path, data):
    with open(path, "wb") as fh:
        fh.write(data)


def _det(name, det_type, stream, alg="raw", seg=0):
    """A one-segment DetectorInfo on ``stream`` (fabricated directly, as
    test_write02_dgram_damage._det / test_bigdata_scan_us012 do)."""
    d = fmt.DetectorInfo(name, det_type, f"{name}-id")
    d.algs[alg] = {}
    d.segments[alg] = {seg}
    d.seg_to_stream[(alg, seg)] = stream
    return d


# Run shape: NG canonical events (timing present), a bigdata chunk roll on the
# detector stream mid-run, and NTAIL ragged-tail L1Accepts past the last gated
# event (detector stream only -- no timing/master -> clamped out).  NG is made
# comfortably large so "materialize ALL NG before evt0" (parent) vs "<= a small
# constant before evt0" (fix) is a crisp, non-flaky discriminator.
NG = 120                 # canonical (gated) events
ROLL = 60                # detector stream rolls c000 -> c001 at this event index
NTAIL = 4                # ragged shutdown-tail L1Accepts (no timing stream)
_BASE = 1_000_000_000_000
_STEP = 10
CANONICAL = [_BASE + _STEP * i for i in range(NG)]
TAIL = [_BASE + _STEP * (NG + i) for i in range(NTAIL)]

TIMING_STREAM = 0
DET_STREAM = 1


def _build_synth_run(tmpdir):
    """A synthetic two-stream run with a chunk roll AND a ragged tail; NO SMD
    sidecars (so auto -> bigdata scan + timing clamp).  Returns a fresh
    :class:`Run` and the shared RunConfig.

      * stream 0 (timing, det_type 'ts')  -- single chunk, the CANONICAL ts only;
      * stream 1 (detector)               -- c000 = first ROLL canonical events,
        c001 = the rest of the canonical events PLUS the ragged TAIL (which,
        lacking the timing stream, the clamp drops).
    """
    s0c0 = os.path.join(tmpdir, "synth-r0001-s000-c000.xtc2")
    s1c0 = os.path.join(tmpdir, "synth-r0001-s001-c000.xtc2")
    s1c1 = os.path.join(tmpdir, "synth-r0001-s001-c001.xtc2")
    _write(s0c0, CANONICAL)                       # timing: canonical only
    _write(s1c0, CANONICAL[:ROLL])                # detector c000 (pre-roll)
    _write(s1c1, CANONICAL[ROLL:] + TAIL)         # detector c001 (post-roll + tail)

    rc = fmt.RunConfig()
    rc.stream_files = {TIMING_STREAM: s0c0, DET_STREAM: s1c0}
    # cfg_end == 0 -> the forward cursors read dgrams straight from offset 0
    # (these synthetic files carry no Configure prefix).
    rc.stream_configs = {TIMING_STREAM: (None, 0), DET_STREAM: (None, 0)}
    rc.raw_tables = {TIMING_STREAM: {}, DET_STREAM: {}}
    rc.detectors = {
        "timing": _det("timing", "ts", TIMING_STREAM),
        "det": _det("det", "epix", DET_STREAM),
    }
    return Run(rc.stream_files, rc), rc


def _fresh(rc):
    """A Run over the same files/config but with NO cached index -- so each
    check observes ``events()`` from a clean slate."""
    return Run(dict(rc.stream_files), rc)


# ===========================================================================
# 1. THE DISCRIMINATOR: O(1) startup -- evt0 without building the full index.
# ===========================================================================
def test_o1_startup_no_full_index_before_first_event():
    """``events(gate=True)`` must yield event 0 WITHOUT calling ``build_index``
    (hence without materializing all N entries).

    PARENT: ``events()`` calls ``self.build_index()`` eagerly -- before it even
    returns the generator -- so the spy count is nonzero at ``evt[0]`` and this
    assertion FAILS (nonzero exit).  FIX: ``build_index`` is never called and
    the streaming gate yields only a small constant number of timestamps before
    ``evt[0]``, so it PASSES."""
    with tempfile.TemporaryDirectory() as td:
        run, rc = _build_synth_run(td)

        # --- spy build_index on the class; delegate so it still works ---------
        calls = {"n": 0}
        orig = Run.build_index

        def spy(self, *a, **k):
            calls["n"] += 1
            return orig(self, *a, **k)

        Run.build_index = spy
        try:
            r = _fresh(rc)
            gen = r.events(gate=True)
            # On the PARENT the full index is already built by the time the
            # generator is handed back; on the FIX nothing is built yet.
            after_call = calls["n"]
            evt0 = next(gen)                     # the first gated event
            at_first = calls["n"]
        finally:
            Run.build_index = orig

        # non-vacuous: a REAL first event must be produced (the first canonical
        # timestamp), so the assertion below cannot pass on an empty stream.
        assert evt0.timestamp == CANONICAL[0], (
            f"STR-05: events(gate=True) yielded {evt0.timestamp} as evt0, "
            f"expected the first canonical ts {CANONICAL[0]}")

        # THE O(1)-STARTUP ASSERTION (fails on the parent, passes on the fix).
        assert at_first == 0, (
            f"STR-05: Run.events(gate=True) called build_index {at_first} time(s) "
            f"to yield event 0 -- it materialized the whole index (all {NG} "
            f"timestamps and the per-event x per-stream entries) BEFORE the first "
            f"event, instead of streaming a merge-join.  A streaming reader must "
            f"yield evt[0] in O(1) memory; the parent builds all N first "
            f"(build_index calls at/after the events() call: "
            f"{after_call}/{at_first}).")

        # Reinforce non-vacuously on the fix: the streaming gate source must have
        # yielded only a small constant number of timestamps before evt0 (not
        # all NG).  Guarded so a tree WITHOUT the fix (no iter_gate_timestamps)
        # simply skips this half -- the build_index assertion above already
        # reddens the parent.
        real_iter = getattr(pidx, "iter_gate_timestamps", None)
        if real_iter is not None:
            C = 8                               # << NG (=120)
            counted = {"n": 0}

            def counting(*a, **k):
                inner = real_iter(*a, **k)

                def wrap():
                    for ts in inner:
                        counted["n"] += 1
                        yield ts
                return wrap()

            pidx.iter_gate_timestamps = counting
            try:
                r2 = _fresh(rc)
                g2 = r2.events(gate=True)
                e0 = next(g2)
                assert e0.timestamp == CANONICAL[0]
                assert counted["n"] <= C, (
                    f"STR-05: the streaming gate yielded {counted['n']} timestamps "
                    f"before evt0 (must be <= {C} << {NG}); the gate is not "
                    f"O(1)-streamed")
                g2.close()
            finally:
                pidx.iter_gate_timestamps = real_iter
            gate_note = f"gate yielded {counted['n']} ts before evt0 (<= {C})"
        else:
            gate_note = "iter_gate_timestamps absent (pre-fix tree)"

    print(f"[ok] STR-05 O(1) startup: evt0 produced with build_index called "
          f"{at_first}x (== 0); {gate_note}")


# ===========================================================================
# 2. CORRECTNESS: the streamed gated set == build_index().timestamps EXACTLY.
# ===========================================================================
def test_gated_set_identical_to_index_drops_tail_spans_roll():
    """``list(events(gate=True))`` timestamps must equal ``build_index().
    timestamps`` exactly (same order): the ragged tail is dropped, events on both
    sides of the chunk roll appear, and coverage reaches the LAST gated event."""
    with tempfile.TemporaryDirectory() as td:
        run, rc = _build_synth_run(td)

        # the reference the old gate filtered against -- the whole random-access
        # index (unchanged by the fix); auto -> bigdata scan (no SMD sidecars).
        ridx = _fresh(rc).build_index()
        assert ridx.scan_source == "bigdata", ridx.scan_source
        ref = list(ridx.timestamps)
        assert ref == CANONICAL, (
            f"precondition: the index's canonical set must be the {NG} "
            f"timing-carrying events, got {len(ref)}")
        assert DET_STREAM in ridx.multichunk_streams, (
            "precondition: the detector stream must roll c000->c001 to exercise "
            f"the chunk-roll span; multichunk_streams={ridx.multichunk_streams}")

        gated = [e.timestamp for e in _fresh(rc).events(gate=True)]

    # byte-identical to the index event set, in order.
    assert gated == ref, (
        f"STR-05: gated forward set != build_index().timestamps.\n"
        f"  len(gated)={len(gated)} len(index)={len(ref)}\n"
        f"  first diff at "
        f"{next((i for i in range(max(len(gated), len(ref))) if gated[i:i+1] != ref[i:i+1]), None)}")

    # the ragged tail is excluded (anti-regression for GATE-02/FAIL-01).
    assert not (set(TAIL) & set(gated)), (
        f"STR-05: ragged shutdown-tail timestamps leaked past the gate: "
        f"{sorted(set(TAIL) & set(gated))}")

    # events on BOTH sides of the chunk roll appear (anti-regression for STR-01):
    # the pre-roll boundary event and every post-roll canonical event, up to and
    # including the LAST -- so the merge-join did not stop short at the roll or
    # before the final gated event.
    assert CANONICAL[ROLL - 1] in gated and CANONICAL[ROLL] in gated, (
        "STR-05: events straddling the chunk roll are missing from the gated set")
    assert gated[-1] == CANONICAL[-1], (
        f"STR-05: gated coverage stopped short of the last gated event "
        f"(last yielded {gated[-1]}, expected {CANONICAL[-1]})")

    print(f"[ok] STR-05 correctness: gated forward == index event set "
          f"({len(gated)} events); ragged tail dropped; spans the c000->c001 "
          f"roll through to the last gated event")


# ===========================================================================
# 3. gate=False still ungated -- a strict superset that includes the tail.
# ===========================================================================
def test_gate_false_is_strict_superset_with_tail():
    """``events(gate=False)`` yields the raw bigdata merge, INCLUDING the ragged
    tail -- a strict superset of the gated set, proving the two modes differ and
    gate=False is unchanged."""
    with tempfile.TemporaryDirectory() as td:
        run, rc = _build_synth_run(td)
        gated = [e.timestamp for e in _fresh(rc).events(gate=True)]
        ungated = [e.timestamp for e in _fresh(rc).events(gate=False)]

    assert ungated == CANONICAL + TAIL, (
        f"STR-05: ungated merge must be the full physical L1Accept set "
        f"(canonical + tail = {NG + NTAIL}); got {len(ungated)}")
    assert set(gated) < set(ungated), (
        "STR-05: the gated set must be a STRICT subset of the ungated set")
    assert set(ungated) - set(gated) == set(TAIL), (
        f"STR-05: the extra ungated events must be exactly the ragged tail "
        f"{TAIL}; got {sorted(set(ungated) - set(gated))}")

    print(f"[ok] STR-05 gate=False: ungated merge is a strict superset "
          f"({len(ungated)} vs {len(gated)}) including the {NTAIL} tail phantoms")


# ===========================================================================
# 4. Fail-closed preserved: an unbuildable gate source raises GateBuildError;
#    gate=False still returns the ungated merge without raising.
# ===========================================================================
def test_fail_closed_when_gate_source_unbuildable():
    """When the gate source cannot be built (absent files), ``events(gate=True)``
    raises ``GateBuildError`` (chaining the cause) rather than silently degrading
    to the ungated merge; ``events(gate=False)`` still returns the (lazy) ungated
    merge without raising and without building the gate."""
    with tempfile.TemporaryDirectory() as td:
        _run, rc = _build_synth_run(td)
    # rc/files are valid in shape, but point the Run at a directory that no
    # longer holds the data (the tempdir is gone): the gate source's eager setup
    # (source select -> open the bigdata c000) fails closed.
    absent = {TIMING_STREAM: "/no/such/dir/synth-r0001-s000-c000.xtc2",
              DET_STREAM: "/no/such/dir/synth-r0001-s001-c000.xtc2"}
    bad = Run(absent, rc)

    try:
        bad.events(gate=True)
    except GateBuildError as exc:
        assert exc.__cause__ is not None, (
            "STR-05: GateBuildError must chain the underlying cause "
            "('raise ... from')")
        assert isinstance(exc.__cause__, OSError), (
            f"STR-05: expected the chained cause to be the file OSError; got "
            f"{type(exc.__cause__).__name__}")
    else:
        raise AssertionError(
            "STR-05: events(gate=True) did NOT fail closed on an unbuildable "
            "gate source -- it must raise GateBuildError, not silently degrade "
            "to the ungated merge")

    # gate=False is the explicit ungated opt-out: a lazy generator, no raise.
    g = bad.events(gate=False)
    assert isinstance(g, types.GeneratorType), (
        "STR-05: events(gate=False) must return the (lazy) ungated merge "
        f"generator; got {type(g).__name__}")

    print("[ok] STR-05 fail-closed: unbuildable gate source -> GateBuildError "
          "(cause chained); gate=False still returns the ungated merge")


# ===========================================================================
# 5. F2 -- the streaming SMD gate must FAIL CLOSED on a truncated sidecar,
#    exactly as _scan_smd_stream (what build_index uses) does.
# ===========================================================================
def test_smd_gate_fails_closed_on_trailing_remnant():
    """A ``_StreamCursor`` treats a sub-24-byte trailing remnant on a
    single-chunk SMD sidecar as a CLEAN end (``peek()`` -> ``None``) -- but
    :func:`psdata.index._scan_smd_stream`, which ``build_index`` uses, FAILS
    CLOSED on exactly that ("trailing N byte(s) too few").  If the streaming gate
    accepted it as EOF it would yield a SHORT gate, and the merge-join would then
    silently drop the corresponding trailing bigdata event(s) as if they were the
    ragged tail -- a silent divergence from ``build_index`` and a silent breach of
    the fail-closed contract.

    This locks the parity: the streaming SMD gate must RAISE on the same corrupt
    bytes ``_scan_smd_stream`` raises on.  The oracle half (``_scan_smd_stream``
    raises) holds on both parent and fix; the gate half is a fix-only lock
    (``iter_gate_timestamps`` exists only on the fix) and is reached only after
    check 1, which already reddens the parent."""
    with tempfile.TemporaryDirectory() as td:
        sd = os.path.join(td, "smalldata")
        os.makedirs(sd)
        bd = os.path.join(td, "synth-r0002-s000-c000.xtc2")
        smd = os.path.join(sd, "synth-r0002-s000-c000.smd.xtc2")
        body = b"".join(_smd_l1(500 + i, 10 * i) for i in range(5))
        # a VALID sidecar, then a 7-byte remnant (< the 24-byte dgram header):
        # _StreamCursor stops here treating it as EOF; _scan_smd_stream raises.
        _write_bytes(smd, _smd_configure() + body + b"\x07" * 7)
        _write_bytes(bd, b"\x00" * 16)          # bd never opened on the SMD path
        rc = fmt.RunConfig()
        rc.stream_files = {0: bd}
        smd_map = pidx.smd_files_for(rc.stream_files)

        # ORACLE (parent AND fix): the index-build SMD scan fails closed here.
        try:
            pidx._scan_smd_stream(smd, bd, 1 << 22)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "precondition: _scan_smd_stream must RAISE on a sub-header "
                "trailing remnant -- the parity oracle is broken; a full "
                "synthetic SMD sidecar was not built correctly")

        # THE F2 LOCK (fix): the streaming SMD gate must ALSO raise on the same
        # bytes, not silently yield a short gate.
        gate_iter = getattr(pidx, "iter_gate_timestamps", None)
        if gate_iter is None:                   # pre-fix tree: nothing to lock
            print("[ok] STR-05 F2: iter_gate_timestamps absent (pre-fix tree); "
                  "the oracle _scan_smd_stream fails closed as required")
            return
        try:
            list(gate_iter(rc.stream_files, run_config=rc, smd_files=smd_map,
                           source="smd"))
        except RuntimeError as exc:
            assert "truncat" in str(exc).lower(), (
                f"STR-05 F2: the gate raised, but not a truncation error: "
                f"{exc!r}")
        else:
            raise AssertionError(
                "STR-05 F2: iter_gate_timestamps SILENTLY accepted a truncated "
                "SMD sidecar (a sub-header trailing remnant) as a clean end and "
                "yielded a short gate -- it must fail closed like "
                "_scan_smd_stream, or the merge-join drops the trailing bigdata "
                "event(s) as if they were the ragged shutdown tail")

    print("[ok] STR-05 F2: the streaming SMD gate fails closed on a sub-header "
          "trailing remnant -- parity with _scan_smd_stream / build_index")


def main():
    print("=" * 72)
    print("STR-05 regression: Run.events(gate=True) is an O(1) streaming "
          "merge-join")
    print("=" * 72)
    test_o1_startup_no_full_index_before_first_event()
    test_gated_set_identical_to_index_drops_tail_spans_roll()
    test_gate_false_is_strict_superset_with_tail()
    test_fail_closed_when_gate_source_unbuildable()
    test_smd_gate_fails_closed_on_trailing_remnant()
    print()
    print("ALL STR-05 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
