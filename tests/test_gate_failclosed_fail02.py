#!/usr/bin/env python3
"""FAIL-02 regression: ``Run.events(gate=True)`` must FAIL CLOSED.

Bug (FAIL-02): the SMD gate that ``Run.events()`` builds to filter out the
ragged DAQ-shutdown-tail phantom events (FAIL-01: 17982 raw vs 17872 real on
mfx100848724/r51 -> 110 extras) was wrapped in a bare ``except Exception`` that,
on ANY index-build failure, *silently* degraded to the **ungated** bigdata
merge -- the unsafe path -- emitting at most a default-suppressed
``RuntimeWarning``.  So the one invariant protecting the reader's event set
could evaporate with no signal.

The contract this probe pins:

  * With ``gate=True`` (the default), if the gate index cannot be built,
    ``events()`` must **raise** -- not return an (ungated) generator.  The
    reader must never silently produce ungated events.
  * The old "warn once, then hand back the ungated merge" path must be gone:
    there must be no outcome where the caller is left holding ungated events
    accompanied only by a warning.
  * The ungated event set must remain reachable only through the explicit,
    loudly-named opt-out ``events(gate=False)`` -- which does NOT build the
    gate and does NOT raise.

Discriminator.  On the PARENT commit ``events()`` swallows the build failure,
warns, and returns the ungated merge, so the "must raise" assertion below fails
and this probe exits non-zero.  On the fixed commit ``events()`` raises, so the
probe passes.

Self-contained: no psana, no SLAC data, no real xtc2 file.  We construct a
``Run`` with dummy files/config and monkeypatch ``build_index`` on the instance
to raise -- the point is the control flow in ``events()``, not any data.  (It
imports :mod:`psdata`, which imports numpy, since numpy is psdata's declared
dependency; nothing else is required.)
"""

import inspect
import os
import sys
import warnings

# --- locate the package under test (parent-of-tests/src), cwd-robust --------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../<repo>/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from psdata.run import Run  # noqa: E402  (after sys.path shim)


def _make_run():
    """A Run over dummy inputs.  ``events()`` builds its merge generator lazily
    (psdata.stream.events is a generator function), so no file or numpy array is
    ever touched -- construction and the gate=False path work with placeholders.
    """
    return Run(files={0: "/nonexistent-s000-c000.xtc2"}, run_config=object())


class _BuildIndexRaiser:
    """Instance-level stand-in for ``Run.build_index`` that always raises the
    same ``OSError`` (a realistic index-build failure: an unreadable/truncated
    sidecar).  Counts its calls so we can prove the gate=False path never builds
    the index."""

    def __init__(self):
        self.calls = 0
        self.error = OSError("simulated: SMD/index build failure (FAIL-02 probe)")

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise self.error


def _call_events(run, **kwargs):
    """Call ``run.events(**kwargs)`` capturing whether it raised (and what) and
    any warnings emitted, so we can distinguish 'raised' from 'returned + warned'
    (the old silent-degrade shape)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            returned = run.events(**kwargs)
        except BaseException as exc:      # noqa: BLE001 -- probe records anything
            return {"raised": exc, "warnings": list(caught)}
        return {"returned": returned, "warnings": list(caught)}


def check_gate_true_fails_closed():
    """gate=True + a broken index build => events() must RAISE, not degrade."""
    run = _make_run()
    raiser = _BuildIndexRaiser()
    run.build_index = raiser                       # shadow the bound method

    result = _call_events(run)                      # gate defaults to True

    # (1) Core discriminator: it must have raised, not returned a generator.
    if "raised" not in result:
        returned = result["returned"]
        kind = "generator" if inspect.isgenerator(returned) else type(returned).__name__
        raise AssertionError(
            "FAIL-02: Run.events(gate=True) did NOT raise when the gate index "
            f"build failed -- it returned a {kind}. The gate silently degraded "
            "to the ungated bigdata merge (which would resurface unindexed "
            "DAQ-shutdown-tail phantom events). It must fail closed.")

    exc = result["raised"]

    # (2) The old warn-only degrade path must be gone: a caller must never be
    # left holding ungated events with merely a warning to show for it. Here
    # 'raised' already implies no generator was handed back; also assert no
    # 'degrade to ungated' warning was emitted as the (former) coping mechanism.
    degrade_warns = [w for w in result["warnings"]
                     if "ungated" in str(w.message).lower()
                     or "degrad" in str(w.message).lower()]
    if degrade_warns:
        raise AssertionError(
            "FAIL-02: Run.events() emitted a degrade-to-ungated warning "
            f"({degrade_warns[0].message!r}) instead of failing closed -- the "
            "warning-only path must not exist.")

    # (3) The failure must be surfaced with its cause intact. Two shapes are
    # acceptable: the original error propagates raw, or it is re-wrapped with
    # 'raise ... from' (cause preserved). Reject a re-wrap that drops the cause.
    if exc is not raiser.error:
        if exc.__cause__ is not raiser.error and not isinstance(exc.__cause__, OSError):
            raise AssertionError(
                "FAIL-02: Run.events() re-wrapped the build failure but did not "
                "chain the original cause (expected 'raise ... from e'); got "
                f"__cause__={exc.__cause__!r}.")

    print("OK: gate=True fails closed by raising "
          f"{type(exc).__name__} (cause chained: {exc.__cause__ is raiser.error})")


def check_gate_false_is_the_only_ungated_path():
    """gate=False is the explicit, loud opt-out: it returns the ungated merge,
    does NOT raise, and does NOT even build the gate index."""
    run = _make_run()
    raiser = _BuildIndexRaiser()
    run.build_index = raiser

    result = _call_events(run, gate=False)

    if "raised" in result:
        raise AssertionError(
            "Run.events(gate=False) must NOT raise -- it is the explicit "
            f"ungated opt-out; got {result['raised']!r}.")
    if not inspect.isgenerator(result["returned"]):
        raise AssertionError(
            "Run.events(gate=False) must return the (lazy) ungated merge "
            f"generator; got {type(result['returned']).__name__}.")
    if raiser.calls != 0:
        raise AssertionError(
            "Run.events(gate=False) must not build the gate index at all "
            f"(build_index called {raiser.calls}x) -- ungated means ungated.")

    print("OK: gate=False returns the ungated merge without building the gate")


def main():
    check_gate_true_fails_closed()
    check_gate_false_is_the_only_ungated_path()
    print("PASS: FAIL-02 gate fails closed; ungated only via explicit gate=False")
    return 0


if __name__ == "__main__":
    sys.exit(main())
