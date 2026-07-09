"""psana ground-truth oracle for psdata's env store.

This module is an *oracle*: it is the ONLY psdata-repo file allowed to import
psana, and it must never be imported by any psdata runtime module. A future
psdata test compares psdata's pure-python EnvStore against the values captured
here.

Why iterating in order matters
------------------------------
psana populates its EnvStore *online*, as a side effect of walking
``run.events()`` in stream order. Reading ``det(evt)`` dispatches to
``EnvStore.values()`` (psana/psexp/envstore.py), which does

    found_pos = np.searchsorted(env_ts, evt_ts)        # DEFAULT side='left'
    if found_pos == n_items: found_pos -= 1            # only clamp the tail
    ...scan backward up to PS_N_STEP_SEARCH_STEPS...

That is a correct as-of lookup *only while every ingested env timestamp is
<= the event's* -- i.e. exactly the streaming situation. If the store is fully
populated and then queried out of order, ``side='left'`` returns the FIRST env
timestamp that is ``>= evt_ts`` (the next *future* SlowUpdate), not ``None``.
So this oracle iterates ``run.events()`` in order and reads ``det(evt)`` as each
target event passes -- it does NOT pre-scan then random-query.

(psdata itself must instead mirror ``get_step_dgrams_of_event``'s as-of rule --
``searchsorted(ts, evt_ts, side='right') - 1`` -- which is order-independent.
The ``__main__`` block below proves the trap empirically so the two rules are
not conflated.)
"""

import os


class PsanaUnavailable(ImportError):
    """Raised when psana cannot be imported (the oracle needs a psconda env).

    A test may ``try: ... except PsanaUnavailable: pytest.skip(...)``.
    """


def _import_psana():
    """Lazily import psana with PS_PARALLEL forced to 'none'.

    Set the env var BEFORE the import (psana reads it at import time). Raise a
    single, catchable ``PsanaUnavailable`` so callers can cleanly SKIP.
    """
    os.environ.setdefault("PS_PARALLEL", "none")
    try:
        import psana  # noqa: F401
    except Exception as exc:  # ImportError, or a psana-internal import failure
        raise PsanaUnavailable(
            "psana is not importable in this environment; source psconda "
            f"(e.g. /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh). Cause: {exc!r}"
        ) from exc
    return psana


def _to_python(value):
    """Convert a psana env value to a plain, JSON-serializable Python object.

    - ``None``                       -> ``None``
    - numpy scalar (np.generic)      -> python int/float/bool via ``.item()``
    - numpy array (rank>=1)          -> nested list via ``.tolist()``
      (e.g. StaleFlags is a rank-1 uint32 array)
    - ``bytes``/``bytearray``        -> str, NUL-terminated, latin-1 decoded
    - str / int / float / bool       -> unchanged (psana pre-decodes charstr
      fields such as ``step_docstring`` to native ``str``)
    """
    import numpy as np

    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).split(b"\x00")[0].decode("latin-1")
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    # Last resort: try numpy-ish .item(), else stringify (keeps output JSON-safe).
    try:
        return value.item()
    except Exception:
        return str(value)


def psana_env_ground_truth(exp, run, dir, event_ks, epics_vars, scan_vars):
    """Iterate psana's ``run.events()`` in order; capture env values at the
    given event indices.

    Returns a plain-Python (JSON-serializable) dict::

        {
          "event_ks":   [k, ...],                 # as requested, sorted+deduped
          "timestamps": {k: int(evt.timestamp)},  # psana event ts at index k
          "epics":      {k: {var: value_or_None}},
          "scan":       {k: {var: value_or_None}},
          "epicsinfo":  {var: pv_name},           # from run.epicsinfo
          "scaninfo":   {var: alg},               # from run.scaninfo
          "n_events_iterated": int,
        }

    Stops iterating after ``max(event_ks)`` L1Accepts (the golden run has 10000
    L1Accepts / 2.3 GB of bigdata -- do NOT walk the whole run).

    Raises ``PsanaUnavailable`` if psana cannot be imported.
    """
    _import_psana()
    from psana import DataSource

    event_ks = sorted({int(k) for k in event_ks})
    epics_vars = list(epics_vars)
    scan_vars = list(scan_vars)

    result = {
        "event_ks": event_ks,
        "timestamps": {},
        "epics": {},
        "scan": {},
        "epicsinfo": {},
        "scaninfo": {},
        "n_events_iterated": 0,
    }

    if not event_ks:
        # Nothing to iterate for; still expose the info tables below.
        ds = DataSource(exp=exp, run=run, dir=dir)
        psrun = next(ds.runs())
        result["epicsinfo"] = {k[0]: v for k, v in psrun.epicsinfo.items()}
        result["scaninfo"] = {k[0]: v for k, v in psrun.scaninfo.items()}
        return result

    kmax = max(event_ks)
    targets = set(event_ks)

    ds = DataSource(exp=exp, run=run, dir=dir)
    psrun = next(ds.runs())

    # Build the env detectors ONCE, before iterating. Each holds a live
    # reference to the same EnvStore object the online walk mutates, so reading
    # det(evt) mid-iteration sees only the timestamps ingested so far.
    # run.Detector accepts either the internal var name ("AT1K0_photon_energy")
    # or the raw PV name ("AT1K0:GAS:PhotonEnergy_RBV"). It does NOT accept the
    # store keys 'epics'/'scan'.
    epics_dets = {var: psrun.Detector(var) for var in epics_vars}
    scan_dets = {var: psrun.Detector(var) for var in scan_vars}

    n_iterated = 0
    for i, evt in enumerate(psrun.events()):
        n_iterated = i + 1
        if i in targets:
            result["timestamps"][i] = int(evt.timestamp)
            result["epics"][i] = {
                var: _to_python(det(evt)) for var, det in epics_dets.items()
            }
            result["scan"][i] = {
                var: _to_python(det(evt)) for var, det in scan_dets.items()
            }
        if i >= kmax:
            break

    result["n_events_iterated"] = n_iterated

    # info tables come from the config (not from ingested dgrams), so they are
    # valid regardless of how far we iterated.
    #   run.epicsinfo -> {(var, pv_name): pv_name}   (pv_name '' if unmapped)
    #   run.scaninfo  -> {(var, alg): alg}
    result["epicsinfo"] = {k[0]: v for k, v in psrun.epicsinfo.items()}
    result["scaninfo"] = {k[0]: v for k, v in psrun.scaninfo.items()}

    return result


def out_of_order_trap(exp, run, dir, var="AT1K0_photon_energy", kmax=3000):
    """Empirically prove psana's ``values()`` out-of-order trap.

    Streaming: read ``det(evt0)`` as evt#0 passes (store still empty for `var`
    -> ``None``). Then keep iterating so the store fills, and re-query the SAME
    captured evt#0 object. With ``side='left'`` and a now-populated store, the
    lookup returns the FIRST env ts >= evt0_ts -- the next *future* SlowUpdate --
    instead of ``None``.

    Returns a dict of the numbers involved.
    """
    _import_psana()
    from psana import DataSource

    ds = DataSource(exp=exp, run=run, dir=dir)
    psrun = next(ds.runs())
    det = psrun.Detector(var)

    evt0 = None
    streaming_val = "<not captured>"
    ts0 = None
    for i, evt in enumerate(psrun.events()):
        if i == 0:
            evt0 = evt
            ts0 = int(evt.timestamp)
            streaming_val = _to_python(det(evt))  # store empty for `var` -> None
        if i >= kmax:
            break

    # Store is now populated up to ~kmax. Inspect the owning env_manager.
    epics_store = psrun.esm.stores["epics"]
    owner_ts0 = owner_n = None
    for em in epics_store.env_managers:
        if em.n_items > 0 and em.locate_variable(var) is not None:
            owner_n = int(em.n_items)
            owner_ts0 = int(em.timestamps[0])  # first ingested SlowUpdate ts
            break

    out_of_order_val = _to_python(det(evt0))  # re-query evt#0 out of order

    return {
        "var": var,
        "evt0_ts": ts0,
        "streaming_answer": streaming_val,
        "out_of_order_answer": out_of_order_val,
        "owner_first_env_ts": owner_ts0,
        "owner_n_items": owner_n,
        "evt0_precedes_first_env_ts": (ts0 is not None and owner_ts0 is not None and ts0 < owner_ts0),
    }


if __name__ == "__main__":
    import json

    EXP = "rixx45619"
    RUN = 122
    DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
    EVENT_KS = [0, 1, 500, 3000]
    EPICS_VARS = ["AT1K0_photon_energy", "GMD_IonMesh", "StaleFlags"]
    SCAN_VARS = ["step_value", "step_docstring"]

    gt = psana_env_ground_truth(
        exp=EXP, run=RUN, dir=DIR,
        event_ks=EVENT_KS, epics_vars=EPICS_VARS, scan_vars=SCAN_VARS,
    )

    print("=" * 72)
    print("psana_env_ground_truth(exp=%r, run=%d, dir=%r)" % (EXP, RUN, DIR))
    print("  event_ks=%r" % (EVENT_KS,))
    print("  epics_vars=%r" % (EPICS_VARS,))
    print("  scan_vars=%r" % (SCAN_VARS,))
    print("=" * 72)
    print(json.dumps(gt, indent=2, sort_keys=True, default=str))

    print("\n" + "-" * 72)
    print("SPOT CHECKS")
    print("-" * 72)
    print("timestamps[0] == 4305271747828686475 ->",
          gt["timestamps"][0] == 4305271747828686475, "(%r)" % gt["timestamps"][0])
    print("epics[0]['AT1K0_photon_energy'] is None ->",
          gt["epics"][0]["AT1K0_photon_energy"] is None)
    print("epics[500]['AT1K0_photon_energy'] == 1000.294677734375 ->",
          gt["epics"][500]["AT1K0_photon_energy"] == 1000.294677734375,
          "(%r)" % gt["epics"][500]["AT1K0_photon_energy"])
    print("epics[3000]['AT1K0_photon_energy'] == 1000.3060302734375 ->",
          gt["epics"][3000]["AT1K0_photon_energy"] == 1000.3060302734375,
          "(%r)" % gt["epics"][3000]["AT1K0_photon_energy"])
    print("scan[0]['step_value'] ->", repr(gt["scan"][0]["step_value"]), "(expect 10000.0)")
    print("scan[3000]['step_value'] ->", repr(gt["scan"][3000]["step_value"]), "(expect 30000.0)")
    print("len(epicsinfo) == 88 ->", len(gt["epicsinfo"]) == 88, "(%d)" % len(gt["epicsinfo"]))
    print("epicsinfo['StaleFlags'] == '' ->",
          gt["epicsinfo"].get("StaleFlags") == "", "(%r)" % gt["epicsinfo"].get("StaleFlags"))
    print("StaleFlags @evt0  ->", repr(gt["epics"][0]["StaleFlags"]))
    print("StaleFlags @evt500->", repr(gt["epics"][500]["StaleFlags"]),
          "(rank-1 uint32 array -> serialized as python list)")
    print("scaninfo ->", gt["scaninfo"])
    print("n_events_iterated ->", gt["n_events_iterated"])

    print("\n" + "-" * 72)
    print("OUT-OF-ORDER TRAP (the value psdata must NOT reproduce)")
    print("-" * 72)
    trap = out_of_order_trap(exp=EXP, run=RUN, dir=DIR)
    print(json.dumps(trap, indent=2, default=str))
    print(
        "\nSUMMARY: streaming det(evt0)=%r  vs  out-of-order det(evt0)=%r"
        % (trap["streaming_answer"], trap["out_of_order_answer"])
    )
    print(
        "The out-of-order answer is the FIRST SlowUpdate (ts=%r), a FUTURE value; "
        "streaming correctly returns None." % (trap["owner_first_env_ts"],)
    )
