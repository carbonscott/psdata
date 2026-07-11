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

SELF-CONTAINED: stdlib + numpy only -- NO psana, NO SLAC data.  Building a
multi-step xtc2 file byte-for-byte (Configure Names tables + per-BeginStep
config ShapesData + interleaved L1Accepts) is out of proportion for a tight,
targeted probe, so -- per the task's stated fallback -- this exercises the
per-step active-config SELECTION logic DIRECTLY: it constructs a ``Run`` and
pre-populates its per-step config table (the exact structure ``run.py`` builds
from the run's BeginStep dgrams -- ``[(begin_step_ts, {seg: {field: value}}),
...]`` ordered by step), then drives the PUBLIC accessors and asserts they pick
the right step.  The file I/O that fills that table on a real run is covered by
the psana oracle (``tests/test_config_us010.py``); here we pin the selection.

cwd-robust; ``main()`` / ``__main__``.

Parent vs fix (the probe's discriminating power):
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

from psdata.run import Run  # noqa: E402

_DET = "epixquad"
_ALG = "config"


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


def main():
    print("=" * 72)
    print("CAL-02: per-step ACTIVE detector config must be addressable")
    print("=" * 72)
    test_step2_event_uses_step2_trbit()
    test_per_step_access()
    test_asof_event_to_step()
    test_single_step_unchanged()
    test_return_shape_matches_single_value()
    test_argument_hygiene()
    print()
    print("ALL CAL-02 CHECKS PASSED")


if __name__ == "__main__":
    main()
