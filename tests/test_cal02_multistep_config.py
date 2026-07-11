#!/usr/bin/env python3
"""CAL-02 regression: the ACTIVE detector config must be addressable PER STEP.

An epix10ka detector's CONFIGURE-block settings -- notably ``trbit``, which
selects the gain-range decode branch -- can CHANGE across DAQ steps: a fresh
config rides on each ``BeginStep`` transition and, exactly like psana's
*stateful* ``det.raw._seg_configs()``, overrides the value in effect for that
step's events (last-wins per segment).  Evidence (EXEC:31300389): on
``uedcom103/r7`` psana's active ``trbit`` takes two values across five steps
``[(0,0,0,0),(1,1,1,1)]``.

The bug (CAL-02): ``psdata``'s ``Run.seg_configs(det)`` returned ONE config for
the whole run -- the one active up to the FIRST L1Accept (step 0's).  A
calibration consumer that decoded a LATER step's events with that single value
used the wrong ``trbit`` -> wrong gain decode, SILENTLY.

The fix exposes the config ACTIVE for a given step/event, matching psana's
stateful as-of semantics (the most recent ``BeginStep`` override at/before the
event, ``searchsorted(begin_ts, ts, 'right') - 1``):

  * ``run.seg_configs(det, step=k)`` / ``run.seg_configs_at(det, k)`` -- step k
  * ``run.seg_configs(det, evt=<Event or 64-bit ts>)``            -- as-of event
  * ``run.n_config_steps(det)``                                   -- step count

while the default single-value ``run.seg_configs(det)`` is UNCHANGED (byte-exact
for a single-step run, where the config is constant).

This file has TWO complementary legs:

A. A LOCAL SYNTHETIC leg (stdlib + numpy only -- NO psana, NO SLAC data) that
   pins the per-step active-config SELECTION logic.  Building a multi-step xtc2
   file byte-for-byte (Configure Names tables + per-BeginStep config ShapesData
   + interleaved L1Accepts) is out of proportion for a tight probe, so -- per
   the task's stated fallback -- it constructs a ``Run`` and pre-populates its
   per-step config table (the exact structure ``run.py`` builds from the run's
   BeginStep dgrams -- ``[(begin_step_ts, {seg: {field: value}}), ...]`` ordered
   by step), then drives the PUBLIC accessors and asserts they pick the right
   step.  This leg exercises the SELECTION only, NOT the table-BUILDING I/O.

B. A PSANA-ORACLE leg (``test_cal02_psana_oracle_uedcom103_r7``) that exercises
   the table-BUILDING I/O end-to-end -- ``_seg_config_steps`` /
   ``_read_config_dgram``, the ``env_records`` BeginStep read, the overlay
   accumulation, the ``is_begin_step`` gate, and ``n_config_steps``' real count
   -- by opening the KNOWN multi-step run ``uedcom103 / r7`` and comparing
   ``seg_configs(det, step=k)`` (built from the REAL BeginStep dgrams, cache NOT
   pre-populated) against psana's stateful ``det.raw._seg_configs()`` advanced to
   each step.  It is gated on psana availability via the suite skip protocol
   (``tests/_skips.py``): it SKIPS cleanly when psana is absent (local runs) and
   RUNS on the milano gate / full suite, where psana + the r7 data are present.
   (Leg A does NOT cover this I/O, and the single-value ``seg_configs`` oracle in
   ``test_config_us010.py`` does not touch the per-step path -- so without leg B
   a BeginStep dgram that failed to carry this detector's config ShapesData would
   silently collapse every step to step 0's config, uncaught.  Leg B closes
   that gap: the run's two distinct ``trbit`` values would expose the collapse.)

cwd-robust; ``main()`` / ``__main__``.

Parent vs fix (the probe's discriminating power, leg A):
  * On the FIX, ``seg_configs(det, step=2)`` / ``seg_configs_at`` /
    ``seg_configs(det, evt=...)`` return step 2's ``trbit=(1,1,1,1)`` -- NOT
    step 1's ``(0,0,0,0)`` -- and the single-step case returns its one config
    unchanged.  Every assertion holds -> the test PASSES.
  * On the PARENT, ``Run.seg_configs`` has no ``step``/``evt`` parameter and
    there is no ``seg_configs_at`` / ``n_config_steps`` at all: the run exposes
    only ONE collapsed config for the whole run, so a step-2 query cannot be
    made -- the calls raise ``TypeError`` / ``AttributeError`` -> the test
    exits NONZERO (FAIL).  That is the bug: later steps are silently
    unaddressable.
"""

import os
import sys

import numpy as np

# -- cwd-robust import of the psdata package from the sibling src/ tree --------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _HERE not in sys.path:                      # for _skips (sibling module)
    sys.path.insert(0, _HERE)

from psdata.run import Run  # noqa: E402
from _skips import skip     # noqa: E402  (machine-readable skip records, HYG-03)

_DET = "epixquad"
_ALG = "config"

# Reference multi-step dataset -- lives in the TEST, never in the library.
# uedcom103/r7: an epixquad run whose active trbit takes TWO distinct values
# across its DAQ steps ([(0,0,0,0),(1,1,1,1)]); the same run test_config_us010
# uses for the single-value BeginStep-override regression (EPIX_BS_*), here
# driven PER STEP.  (Matrix / EXEC:31300389.)
ORACLE_EXP = "uedcom103"
ORACLE_RUN = 7
ORACLE_DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
ORACLE_DET = "epixquad"
ORACLE_SEGS = [0, 1, 2, 3]


def _u8(values):
    return np.array(values, dtype=np.uint8)


def _seg_cfg(trbit, apc_fill):
    """One segment's config field map, shaped like a real epix10ka segment:
    ``trbit`` (4,) u8 and ``asicPixelConfig`` (4,176,192) u8 (small stand-in
    dims here -- the selection logic is shape-agnostic)."""
    return {
        "trbit": _u8(trbit),
        "asicPixelConfig": np.full((4, 3, 3), apc_fill, dtype=np.uint8),
    }


def _make_run(steps):
    """A ``Run`` whose per-step config table is pre-populated (bypassing file
    I/O), so the PUBLIC accessors exercise ONLY the per-step active-config
    SELECTION logic.  ``steps`` is ``[(begin_step_ts, {seg: field_map}), ...]``
    ordered by step -- exactly the structure ``Run._seg_config_steps`` builds
    from the run's BeginStep dgrams.  ``files``/``run_config`` are unused on the
    (cache-hit) selection path, so a stub config suffices."""
    run = Run(files={}, run_config=object())
    # Assign the whole cache dict (not index into it): this succeeds even on the
    # PARENT (where __init__ never created the attribute), so the parent's
    # failure surfaces at the PUBLIC per-step accessor -- which does not exist
    # there -- rather than here.
    run._seg_cfg_steps = {(_DET, _ALG): steps}
    return run


class _FakeEvent:
    """The minimum an ``evt=`` argument needs: a ``.timestamp`` (as a real
    :class:`psdata.stream.Event` exposes)."""

    def __init__(self, ts):
        self.timestamp = ts


# A multi-step run: trbit is (0,0,0,0) for steps 0 and 1, then CHANGES to
# (1,1,1,1) at step 2 (the CAL-02 shape: two distinct values across the steps).
# BeginStep timestamps 100 < 200 < 300 delimit the steps.
_MULTI_STEPS = [
    (100, {0: _seg_cfg((0, 0, 0, 0), 10)}),   # step 0
    (200, {0: _seg_cfg((0, 0, 0, 0), 10)}),   # step 1 (unchanged)
    (300, {0: _seg_cfg((1, 1, 1, 1), 20)}),   # step 2 (trbit + apc changed)
]

# A single-step run: exactly one config for the whole run.
_SINGLE_STEP = [
    (500, {0: _seg_cfg((0, 1, 0, 1), 7)}),
]


# ==========================================================================
# 1. THE CAL-02 probe: a step-2 event gets step 2's trbit, NOT step 1's
# ==========================================================================
def test_step2_event_uses_step2_trbit():
    run = _make_run(_MULTI_STEPS)

    # The two steps that straddle the change carry DIFFERENT trbit -- so a
    # single whole-run value (the parent's behaviour) cannot be correct for
    # both.  This is exactly why per-step addressing is required.
    step1 = run.seg_configs_at(_DET, 1)[0].config.trbit
    step2 = run.seg_configs_at(_DET, 2)[0].config.trbit
    assert not np.array_equal(step1, step2), (
        "test setup broken: steps 1 and 2 must have distinct trbit")

    # By EVENT (as-of): an event inside step 2 (ts >= 300) must resolve to step
    # 2's config, NOT step 1's.  On the parent this call cannot even be made.
    got = run.seg_configs(_DET, evt=_FakeEvent(350))[0].config.trbit
    assert np.array_equal(got, _u8((1, 1, 1, 1))), (
        f"CAL-02: a step-2 event must decode with step 2's trbit (1,1,1,1); "
        f"got {got!r} (step 1's is {step1!r})")
    assert not np.array_equal(got, step1), (
        "CAL-02: a step-2 event must NOT get step 1's trbit -- that is the "
        "silent wrong-gain-decode bug this guards against")
    print("[ok] step-2 event -> step 2's trbit (1,1,1,1), not step 1's (0,0,0,0)")


# ==========================================================================
# 2. Per-step access (step=k / seg_configs_at) picks each step's config
# ==========================================================================
def test_per_step_access():
    run = _make_run(_MULTI_STEPS)
    expected = {0: (0, 0, 0, 0), 1: (0, 0, 0, 0), 2: (1, 1, 1, 1)}
    for k, want in expected.items():
        via_kw = run.seg_configs(_DET, step=k)[0].config.trbit
        via_at = run.seg_configs_at(_DET, k)[0].config.trbit
        assert np.array_equal(via_kw, _u8(want)), (k, via_kw, want)
        assert np.array_equal(via_at, _u8(want)), (k, via_at, want)
    # A second field (asicPixelConfig) flows through the same path -- the fix is
    # generic over the segment's config fields, not special-cased to trbit.
    assert int(run.seg_configs_at(_DET, 0)[0].config.asicPixelConfig.flat[0]) == 10
    assert int(run.seg_configs_at(_DET, 2)[0].config.asicPixelConfig.flat[0]) == 20
    assert run.n_config_steps(_DET) == 3, run.n_config_steps(_DET)
    print("[ok] step=k / seg_configs_at pick each of the 3 steps; "
          "n_config_steps == 3; second field flows through")


# ==========================================================================
# 3. As-of semantics (side='right'-1): map each event ts to its owning step
# ==========================================================================
def test_asof_event_to_step():
    run = _make_run(_MULTI_STEPS)
    cases = [
        (150, (0, 0, 0, 0)),   # inside step 0
        (200, (0, 0, 0, 0)),   # exactly AT step 1's BeginStep (side='right')
        (299, (0, 0, 0, 0)),   # last ts of step 1
        (300, (1, 1, 1, 1)),   # exactly AT step 2's BeginStep -> step 2
        (350, (1, 1, 1, 1)),   # inside step 2
        (10_000, (1, 1, 1, 1)),  # far past the last step -> stays step 2
    ]
    for ts, want in cases:
        # accept both a raw 64-bit ts and an Event-like object
        by_ts = run.seg_configs(_DET, evt=ts)[0].config.trbit
        by_ev = run.seg_configs(_DET, evt=_FakeEvent(ts))[0].config.trbit
        assert np.array_equal(by_ts, _u8(want)), (ts, by_ts, want)
        assert np.array_equal(by_ev, _u8(want)), (ts, by_ev, want)
    # An event that precedes the first BeginStep clamps to step 0 (there are no
    # L1Accepts there on a real run, but the accessor must not blow up).
    early = run.seg_configs(_DET, evt=1)[0].config.trbit
    assert np.array_equal(early, _u8((0, 0, 0, 0))), early
    print("[ok] as-of (searchsorted right-1) maps ts 150/200/299 -> step<=1, "
          "300/350/10000 -> step 2; pre-first-step clamps to step 0")


# ==========================================================================
# 4. Single-step run: the one config is returned unchanged (back-compat)
# ==========================================================================
def test_single_step_unchanged():
    run = _make_run(_SINGLE_STEP)
    assert run.n_config_steps(_DET) == 1
    want = _u8((0, 1, 0, 1))
    # step 0, seg_configs_at, and any event ts must all yield the sole config.
    assert np.array_equal(run.seg_configs(_DET, step=0)[0].config.trbit, want)
    assert np.array_equal(run.seg_configs_at(_DET, 0)[0].config.trbit, want)
    for ts in (0, 500, 999_999):
        got = run.seg_configs(_DET, evt=ts)[0].config.trbit
        assert np.array_equal(got, want), (ts, got)
    print("[ok] single-step run -> the one config, unchanged, for step 0 / "
          "seg_configs_at / any event")


# ==========================================================================
# 5. The multi-step return shape == the single-value shape (byte-exact wrap)
# ==========================================================================
def test_return_shape_matches_single_value():
    """The per-step return is wrapped with the SAME ``_SegConfig`` /
    ``_AlgNamespace`` machinery as the default single-value accessor, so a
    consumer reads ``scfg[seg].config.<field>`` identically either way -- the
    per-step path changes WHICH config is returned, never its shape."""
    run = _make_run(_MULTI_STEPS)
    scfg = run.seg_configs(_DET, step=2)
    seg0 = scfg[0]
    # attribute access path (valid-identifier fields)
    assert np.array_equal(seg0.config.trbit, _u8((1, 1, 1, 1)))
    # and the fields-dict path both work, exactly like the single-value form
    assert np.array_equal(seg0.config.fields["trbit"], _u8((1, 1, 1, 1)))
    assert set(scfg) == {0}
    print("[ok] per-step return has the same seg_cfg.<alg>.<field> shape as the "
          "single-value accessor")


# ==========================================================================
# 6. Bad-argument hygiene
# ==========================================================================
def test_argument_hygiene():
    run = _make_run(_MULTI_STEPS)
    # step out of range -> IndexError (not a silent wrong config)
    try:
        run.seg_configs(_DET, step=99)
    except IndexError:
        pass
    else:
        raise AssertionError("step out of range must raise IndexError")
    # can't ask for both step= and evt=
    try:
        run.seg_configs(_DET, step=0, evt=_FakeEvent(150))
    except ValueError:
        pass
    else:
        raise AssertionError("passing both step= and evt= must raise ValueError")
    print("[ok] out-of-range step raises IndexError; step= + evt= raises "
          "ValueError")


# ==========================================================================
# 7. PSANA ORACLE (milano gate / full suite only -- SKIPS cleanly if psana is
#    absent): the REAL per-step table, built end-to-end from the run's BeginStep
#    dgrams, must match psana's stateful _seg_configs advanced to each step.
#
#    This is the ONLY end-to-end check of CAL-02's table BUILD -- it drives
#    _seg_config_steps / _read_config_dgram / the env_records BeginStep read /
#    the overlay accumulation / the is_begin_step gate / n_config_steps, none of
#    which the synthetic legs above (which pre-populate the cache) or the
#    single-value oracle in test_config_us010.py touch.  The cache is NOT
#    pre-populated here: the public accessors do the real file I/O.
# ==========================================================================
def test_cal02_psana_oracle_uedcom103_r7():
    """Per-step active config built from REAL BeginStep dgrams == psana's
    stateful ``det.raw._seg_configs()`` advanced to each step, for every segment.

    ``_seg_configs()`` is stateful (see test_config_us010): it starts at the
    Configure default and each ``BeginStep`` the DataSource crosses overrides it
    (last-wins).  So the config ACTIVE for step k -- the value psana's per-event
    ``det.raw.calib`` uses -- is what ``_seg_configs()`` reports once advanced to
    a step-k event.  uedcom103/r7's active ``trbit`` takes two distinct values
    across its steps, so this genuinely distinguishes a correct per-step build
    from a table that silently collapsed every step to step 0's config.
    """
    try:
        import psana  # noqa: F401
    except Exception:
        return skip(
            "cal02_multistep_psana_oracle",
            "psana is not importable in this environment (source psconda.sh); "
            "the per-step table-BUILD oracle for CAL-02 -- the REAL "
            "_seg_config_steps / _read_config_dgram / env_records path vs "
            "psana's stateful _seg_configs advanced per step, the only "
            "end-to-end check of the multi-step config build -- cannot run and "
            "must not silently pass")

    import psdata
    from psana import DataSource

    # -- psana ground truth: the ACTIVE (trbit, asicPixelConfig) per DAQ step --
    # Fully iterate each step's events (the documented psana2 nesting) but pin
    # the active config on the FIRST event of the step; draining keeps the step
    # boundaries clean rather than breaking mid-step.
    ds = DataSource(exp=ORACLE_EXP, run=ORACLE_RUN, dir=ORACLE_DIR)
    prun = next(ds.runs())
    det = prun.Detector(ORACLE_DET)

    psana_steps = []            # per step: (first_event_ts, {seg: (trbit, apc)})
    for pstep in prun.steps():
        active = None
        first_ts = None
        for i, pevt in enumerate(pstep.events()):
            if i == 0:
                det.raw.calib(pevt)        # advance stateful _seg_configs here
                sc = det.raw._seg_configs()
                active = {seg: (np.asarray(sc[seg].config.trbit).copy(),
                                np.asarray(sc[seg].config.asicPixelConfig).copy())
                          for seg in sorted(sc)}
                first_ts = int(pevt.timestamp)
            # keep draining this step's events (no work) to stay step-aligned
        if active is not None:
            psana_steps.append((first_ts, active))

    assert psana_steps, "psana yielded no steps/events for uedcom103/r7"

    # The run MUST show a real config change across steps, else the oracle can't
    # tell a correct per-step build from a collapse-to-step-0 (all identical).
    distinct_trbit = {tuple(int(x) for x in active[ORACLE_SEGS[0]][0])
                      for _ts, active in psana_steps}
    assert len(distinct_trbit) >= 2, (
        f"uedcom103/r7 no longer shows >1 distinct trbit across steps "
        f"(saw {sorted(distinct_trbit)}); the per-step oracle is toothless -- "
        f"choose a run whose config actually changes across steps")

    # -- psdata: build the REAL per-step table via the PUBLIC accessors --------
    # (cache is NOT pre-populated -- this drives _seg_config_steps end-to-end.)
    r = psdata.open(exp=ORACLE_EXP, run=ORACLE_RUN, dir=ORACLE_DIR)
    n = r.n_config_steps(ORACLE_DET)
    assert n == len(psana_steps), (
        f"psdata found {n} config step(s) but psana iterated "
        f"{len(psana_steps)} -- the BeginStep gate/count disagree")

    for k, (ts, active) in enumerate(psana_steps):
        by_step = r.seg_configs(ORACLE_DET, step=k)          # by step index
        by_evt = r.seg_configs(ORACLE_DET, evt=ts)           # as-of a real event
        assert sorted(by_step) == ORACLE_SEGS, (k, sorted(by_step))
        assert sorted(by_evt) == ORACLE_SEGS, (k, sorted(by_evt))
        for seg in ORACLE_SEGS:
            ps_trbit, ps_apc = active[seg]
            for label, scfg in (("step=%d" % k, by_step), ("evt@ts", by_evt)):
                my_trbit = np.asarray(scfg[seg].config.trbit)
                my_apc = np.asarray(scfg[seg].config.asicPixelConfig)
                assert np.array_equal(my_trbit, ps_trbit), (
                    f"step {k} seg {seg} [{label}]: psdata trbit {my_trbit} != "
                    f"psana active {ps_trbit} -- the per-step table build is "
                    f"WRONG (this is the CAL-02 collapse-to-step-0 bug)")
                assert np.array_equal(my_apc, ps_apc), (
                    f"step {k} seg {seg} [{label}]: asicPixelConfig differs "
                    f"from psana's active config")
        print(f"[ok] step {k}: trbit "
              f"{tuple(int(x) for x in active[ORACLE_SEGS[0]][0])} matches psana "
              f"active (step= and evt= agree, all {len(ORACLE_SEGS)} segs)")

    print(f"[ok] uedcom103/r7 PSANA ORACLE: {n} steps, "
          f"{len(distinct_trbit)} distinct trbit value(s) across steps; the "
          f"REAL per-step build is byte-exact vs psana _seg_configs "
          f"(step= and evt= paths)")


def main():
    print("=" * 72)
    print("CAL-02: per-step ACTIVE detector config must be addressable")
    print("=" * 72)
    # Local synthetic leg -- pins the SELECTION logic (no psana, no data).
    test_step2_event_uses_step2_trbit()
    test_per_step_access()
    test_asof_event_to_step()
    test_single_step_unchanged()
    test_return_shape_matches_single_value()
    test_argument_hygiene()
    # Cluster oracle leg -- pins the real table BUILD (skips cleanly w/o psana).
    test_cal02_psana_oracle_uedcom103_r7()
    print()
    print("ALL CAL-02 CHECKS PASSED")


if __name__ == "__main__":
    main()
