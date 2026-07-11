#!/usr/bin/env python3
"""psdata.run -- the public "open a run" surface of psdata.

This is the US-005 packaging layer: a single, documented entry point that ties
the parse core (:mod:`psdata.format`), the streaming event-assembly layer
(:mod:`psdata.stream`), and the random-access index (:mod:`psdata.index`) into
one object you open once and then stream, random-access, or introspect.

Open a run **either** by ``exp`` / ``run`` / ``dir`` (the standard LCLS-II file
layout)::

    import psdata
    r = psdata.open(exp="mfx100848724", run=51,
                    dir="/sdf/data/lcls/ds/prj/public01/xtc")

    for evt in r.events():                 # forward streaming, ts order
        ts  = evt.timestamp
        pid = evt.pulseId
        jf  = evt.stack("jungfrau")        # (32,512,1024) uint16 or None

    ridx = r.build_index()                 # random access (scans SMD only)
    evt  = ridx.read_event(ts)             # by 64-bit timestamp
    evt  = ridx.read_event_at(1000)        # ... or by event position

**or** by an explicit list of per-stream xtc2 files::

    r = psdata.open(files=["/path/...-s000-c000.xtc2",
                           "/path/...-s001-c000.xtc2", ...])

The file-layout convention is the *only* place that knows path patterns:

  * bigdata stream files  ``{dir}/{exp}-r{run:04d}-s{stream:03d}-c000.xtc2``
  * SMD (smalldata) index ``{dir}/smalldata/{base}.smd.xtc2`` for each bigdata
    ``{dir}/{base}.xtc2`` (resolved by :func:`psdata.index.smd_files_for`).

Only the first chunk (``c000``) of each stream is opened; later chunks are
reached by following the ``chunkinfo`` roll (see :mod:`psdata.index`).

This module is **raw arrays only** -- no calibration, no geometry, no MPI -- and
imports **no** psana / mpi4py / h5py (only the standard library and numpy; see
:func:`assert_no_framework_imports`).
"""

import bisect
import glob
import os
import struct

from . import format as _f
from . import stream as _s
from . import index as _i
from . import envstore as _e


class GateBuildError(RuntimeError):
    """The SMD gate index could not be built, so ``Run.events()`` refuses to
    stream.

    Raised by :meth:`Run.events` (default ``gate=True``) when
    :meth:`Run.build_index` fails.  The gate is the reader's single safety net:
    it filters the ungated bigdata k-way merge down to the SMD-defined
    (psana-equivalent) event set, dropping the ragged DAQ shutdown-tail
    L1Accepts that the smalldata writer never indexed (observed on
    mfx100848724/r51: 17982 raw vs 17872 real -> 110 phantom events).

    If that net cannot be built the reader **fails closed**: it raises this
    instead of silently degrading to the unsafe ungated merge.  A caller who
    genuinely wants the ungated stream (e.g. a run with no SMD sidecars *and*
    no bigdata-scan gate) must ask for it explicitly with
    ``Run.events(gate=False)``.  The original cause is chained
    (``raise ... from``) so it is never lost.
    """


# Index-build failures we know how to explain -- i.e. EXPECTED ways a gate
# build can fail, each re-wrapped into a GateBuildError that names the
# gate=False remedy.  This spans: I/O (OSError), malformed values / bad
# ``source`` (ValueError), config/stream lookup mismatches (KeyError,
# IndexError), a present-but-malformed SMD sidecar (index.py raises
# ``RuntimeError`` when a sidecar has no smdinfo table), and a corrupt Configure
# whose header unpack fails (``struct.error``).  ``RuntimeError`` also covers a
# genuine programming bug, but that stays fully debuggable: it is re-raised as a
# GateBuildError *chained* to the original (``raise ... from exc``), so the
# traceback and cause survive.  Anything outside this set propagates raw --
# still fail-closed, just unwrapped.
_INDEX_BUILD_ERRORS = (
    OSError, ValueError, KeyError, IndexError, RuntimeError, struct.error)


class _AlgNamespace:
    """Attribute view over one algorithm's CONFIGURE-block fields of a segment.

    Mirrors the leaf of psana's ``det.raw._seg_configs()[seg].config`` object:
    each field is reachable as an attribute (``cfg.config.trbit``).  Field names
    that are not valid Python identifiers (e.g. jungfrau's dotted
    ``user.bias_voltage_v`` or enum-suffixed ``DYNAMIC:gainModeEnum``) cannot be
    reached by attribute syntax -- read those from the :attr:`fields` dict or
    via ``getattr(cfg.config, name)`` with the literal name.
    """

    __slots__ = ("fields",)

    def __init__(self, fields):
        self.fields = dict(fields)

    def __getattr__(self, name):
        if name.startswith("__"):
            # Dunder probes (e.g. __setstate__/__deepcopy__/__reduce_ex__)
            # during pickle/deepcopy fire __getattr__ before the slot is set;
            # short-circuit them so we don't recurse on the missing slot.
            raise AttributeError(name)
        try:
            return self.fields[name]
        except KeyError:
            raise AttributeError(name) from None

    def field_names(self):
        return sorted(self.fields)

    def __repr__(self):
        return f"_AlgNamespace(fields={self.field_names()})"


class _SegConfig:
    """One segment's CONFIGURE object: ``seg_cfg.<alg>.<field>``.

    The segment-level namespace returned per segment by
    :meth:`Run.seg_configs`.  Exposes one attribute per algorithm (today only
    ``config``), matching psana's ``det.raw._seg_configs()[seg].config.<field>``
    access pattern.
    """

    __slots__ = ("_algs",)

    def __init__(self, algs):
        self._algs = dict(algs)            # alg name -> _AlgNamespace

    def __getattr__(self, name):
        if name.startswith("__"):
            # Dunder probes (e.g. __setstate__/__deepcopy__/__reduce_ex__)
            # during pickle/deepcopy fire __getattr__ before the slot is set;
            # short-circuit them so we don't recurse on the missing slot.
            raise AttributeError(name)
        try:
            return object.__getattribute__(self, "_algs")[name]
        except KeyError:
            raise AttributeError(name) from None

    def alg_names(self):
        return sorted(self._algs)

    def __repr__(self):
        return f"_SegConfig(algs={self.alg_names()})"

# Stream/chunk file-name pattern: ``<exp>-r<run4>-s<stream3>-c<chunk3>.xtc2``.
# psana opens only the first chunk (``c000``) of each stream and rolls forward
# from there; :func:`psdata.format.filter_c000` reproduces that filter.
_BIGDATA_GLOB = "{exp}-r{run:04d}-s*-c000.xtc2"


def _resolve_run_files(exp, run, dir):
    """Resolve a run's per-stream first-chunk (``c000``) bigdata files.

    Globs ``{dir}/{exp}-r{run:04d}-s*-c000.xtc2`` and keys each file by its real
    ``s###`` stream index (so the mapping matches the SMD stream indices, which
    psana indexes 1:1).  Returns ``{stream_index: path}``.

    Raises ``FileNotFoundError`` if no files match -- the caller passed an
    exp/run/dir that resolves to nothing.
    """
    pattern = os.path.join(dir, _BIGDATA_GLOB.format(exp=exp, run=run))
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"no xtc2 stream files match {pattern!r} -- check exp/run/dir")
    files = {}
    for p in paths:
        sidx = _f.stream_index_of(p)
        if sidx is None:                          # pragma: no cover (glob guard)
            raise ValueError(f"cannot parse stream index from {p!r}")
        files[sidx] = p
    return files


def _files_to_mapping(files):
    """Normalize an explicit file list/mapping to ``{stream_index: path}``.

    Accepts the same forms as :func:`psdata.format.discover` -- a ``{index:
    path}`` dict, ``(index, path)`` pairs, or a plain list of paths.  For a plain
    list of paths the real ``s###`` index is recovered from each file name when
    possible (so an explicit c000 list keys by its true stream id, matching the
    SMD index); files that don't match the ``-s###-c000`` pattern fall back to
    positional indices.
    """
    if isinstance(files, dict):
        return {int(i): p for i, p in files.items()}
    items = list(files)
    if items and isinstance(items[0], (tuple, list)):
        return {int(i): p for i, p in items}
    # plain list of paths: recover the real stream index where the name allows.
    out = {}
    for pos, p in enumerate(items):
        sidx = _f.stream_index_of(p)
        out[sidx if sidx is not None else pos] = p
    return out


class Run:
    """A handle to one opened run -- stream, random-access, or introspect it.

    Construct via :func:`psdata.open`, not directly.  A :class:`Run` is a thin,
    documented facade over the three psdata layers; it owns the discovered
    :class:`~psdata.format.RunConfig` and lazily builds the random-access
    :class:`~psdata.index.RunIndex` on first use.

    Attributes
    ----------
    exp, run, dir : str | int | None
        Identify the run when opened by exp/run/dir (``None`` for an explicit
        file list).
    files : dict
        ``{stream_index: bigdata_c000_path}`` -- the first-chunk files opened.
    config : psdata.format.RunConfig
        The run's discovered detector / field / segment configuration.
    """

    def __init__(self, files, run_config, exp=None, run=None, dir=None,
                 smd_files=None):
        self.exp = exp
        self.run = run
        self.dir = dir
        self.files = dict(files)
        self.config = run_config
        self._smd_files = smd_files            # explicit SMD map, or None
        self._index = None                     # lazily built RunIndex
        self._env = None                       # lazily built EnvStoreManager
        self._seg_cfg_steps = {}               # (det, alg) -> per-step configs

    # -- streaming ---------------------------------------------------------
    def events(self, gate=True):
        """Yield assembled :class:`~psdata.stream.Event` objects in ascending
        timestamp order (forward streaming).  Each event exposes ``timestamp``,
        ``pulseId``, and lazy raw detector arrays (``evt.stack(name)`` /
        ``evt.raw(name)`` / ``evt.as_dict()``).

        By default (``gate=True``) the event set is **gated to the SMD-defined
        index**, so forward streaming yields exactly the events psana and
        :meth:`read_event_at` do.  Why: on a run with a ragged DAQ-shutdown tail,
        some bigdata streams carry trailing L1Accepts the SMD writer never
        indexed (it stopped first); the raw k-way merge over the bigdata would
        surface those extras, making ``events()`` disagree with the index and
        with psana (observed: jungfrau mfx100848724/r51 -> 17982 raw vs 17872
        indexed).  Gating filters the merge to the indexed timestamps so the
        forward, random-access, and psana event sets coincide.

        **Fail-closed.**  The gate is this reader's single safety net.  If it
        cannot be built -- :meth:`build_index` raises -- ``events()`` does NOT
        fall back to the ungated merge; it raises :class:`GateBuildError`
        (chaining the original cause).  A silent degrade would let the phantom
        shutdown-tail events leak back in with no signal, which is exactly the
        defect this guards against.

        Opting out is explicit and loud: pass ``gate=False`` for the ungated,
        SMD-independent merge straight over the bigdata (the same as the
        low-level :func:`psdata.events`).  That may surface unindexed
        shutdown-tail events on a truncated run -- you are asking for the raw
        bigdata set, on purpose.  There is no automatic, silent path to an
        ungated stream: the only way to get one is to name ``gate=False``.

        Note on the gate's *source*.  ``build_index(source="auto")`` uses the
        SMD sidecars when present and otherwise reconstructs the same canonical
        event set by scanning the bigdata dgram headers directly
        (``scan_source == "bigdata"``).  The bigdata scan still applies the
        timing/master-stream clamp that drops the shutdown tail, so the gate
        remains effective against the FAIL-01 phantom-event bug even with no
        SMD; it is weaker only in that it is no longer an *independent* witness
        of the bigdata (a truncation shared by scan and merge would not be
        caught).  Inspect ``run.build_index().scan_source`` if that distinction
        matters to you.

        **O(1)-memory streaming (STR-05).**  The gate is applied by a streaming
        *merge-join*, not by pre-building the whole index.  Both the forward
        bigdata merge and the gate source (:func:`psdata.index.iter_gate_
        timestamps`) are ascending in timestamp, so this method walks them in
        lockstep: it advances the gate cursor to each bigdata event's timestamp
        and yields the event iff the gate carries it (a bigdata timestamp past
        the gate cursor is a phantom -> skipped; when the gate exhausts, every
        remaining bigdata event is the ragged shutdown tail -> dropped).  So the
        first event is yielded after reading only the first gate entry and the
        first bigdata event, and peak memory is one bigdata event plus O(1)
        cursor state -- rather than materializing all N timestamps (and the
        per-event x per-stream ``RunIndex.entries``) before event 0.  The yielded
        SEQUENCE is byte-identical to the old ``frozenset(build_index().
        timestamps)`` filter; :meth:`build_index` and random access are
        unchanged and still available.
        """
        merged = _s.events(self.files, run_config=self.config)
        if not gate:
            return merged
        try:
            gate_ts = _i.iter_gate_timestamps(
                self.files, run_config=self.config, smd_files=self._smd_files,
                source="auto")
        except _INDEX_BUILD_ERRORS as exc:
            # Fail closed: the safety net could not be built, so refuse to
            # stream rather than silently degrade to the ungated merge (which
            # would resurrect the phantom shutdown-tail events with no signal).
            # iter_gate_timestamps(source="auto") already covers a *legitimate*
            # SMD absence by scanning the bigdata headers, so reaching here is a
            # real build failure.  A caller who truly wants the ungated set must
            # say so explicitly with gate=False.  Setup (source selection, file
            # opens, Configure parses) runs eagerly inside iter_gate_timestamps,
            # so a failure surfaces HERE at call time, not mid-stream.
            raise GateBuildError(
                f"Run.events(gate=True): could not build the SMD gate source "
                f"for this run ({type(exc).__name__}: {exc}). Refusing to "
                f"stream: an ungated bigdata merge may surface unindexed "
                f"DAQ-shutdown-tail events that psana and read_event_at() do "
                f"not have. Fix the gate source, or -- if you genuinely want "
                f"the raw, ungated bigdata event set -- call "
                f"Run.events(gate=False) explicitly."
            ) from exc
        return self._gated_stream(merged, gate_ts)

    @staticmethod
    def _gated_stream(merged, gate_ts):
        """Merge-join the forward bigdata merge ``merged`` against the ascending
        gate timestamps ``gate_ts`` (STR-05), yielding exactly the events whose
        timestamp is in the gate -- byte-identical to the old
        ``(evt for evt in merged if evt.timestamp in frozenset(index.
        timestamps))`` filter, but in O(1) memory with the first event immediate.

        Both inputs ascend in timestamp.  For each bigdata event ts ``b``:
        advance the gate cursor over every gate ts ``< b`` (gate entries with no
        bigdata match -- normal); if the gate head ``== b`` the event is gated in
        (yield, leaving the head so a repeated ``b`` from a ts collision still
        matches); if the head is ``> b`` the event is a phantom absent from the
        gate (skip); and when the gate EXHAUSTS every remaining bigdata event is
        the ragged shutdown tail past the last gated event (stop).  Both open
        cursor sets are released on stop / early consumer break."""
        gate = iter(gate_ts)
        try:
            try:
                head = next(gate)
            except StopIteration:
                return                     # empty gate -> no gated events
            for evt in merged:
                b = evt.timestamp
                while head is not None and head < b:
                    try:
                        head = next(gate)
                    except StopIteration:
                        head = None
                if head is None:
                    return                 # gate exhausted -> rest is the tail
                if head == b:
                    yield evt              # gated in (head kept for a ts dup)
                # head > b: a phantom not in the gate -> skip this event
        finally:
            for gen in (merged, gate_ts):
                close = getattr(gen, "close", None)
                if close is not None:
                    close()

    # -- random access -----------------------------------------------------
    def build_index(self, rebuild=False, source="auto"):
        """Build (or return the cached) random-access
        :class:`~psdata.index.RunIndex` for this run.

        The index is built by scanning only the small SMD files -- the GB-scale
        bigdata is never read during the build.  Cached on the run after the
        first call; pass ``rebuild=True`` to force a fresh build.

        ``source`` selects where the index comes from ({"auto","smd","bigdata"},
        passed through to :func:`psdata.index.build_index`).
        """
        if self._index is None or rebuild:
            self._index = _i.build_index(
                self.files, run_config=self.config, smd_files=self._smd_files,
                source=source)
        return self._index

    # convenience alias matching the noun used in the docs/README
    index = build_index

    def read_event(self, ts):
        """Random-access the event at exact 64-bit timestamp ``ts`` (builds the
        index on first use).  Returns a :class:`~psdata.stream.Event`; raises
        ``KeyError`` if ``ts`` is not an indexed L1Accept event."""
        return self.build_index().read_event(ts)

    def read_event_at(self, k):
        """Random-access the ``k``-th L1Accept event (0-based, ascending ts;
        builds the index on first use).  Returns a
        :class:`~psdata.stream.Event`."""
        return self.build_index().read_event_at(k)

    def read_events(self, ks):
        """Batch random-access the events at positions ``ks`` in one coalesced
        call (builds the index on first use).  Equivalent to
        ``[read_event_at(k) for k in ks]`` but issues its ``pread``s grouped per
        chunk file in ascending-offset order; returns the events in ``ks``
        order.  See :meth:`psdata.index.RunIndex.read_events`."""
        return self.build_index().read_events(ks)

    def read_stack(self, ks, det, field="raw", alg="raw"):
        """Batch-read events ``ks`` and stack one detector into a single
        preallocated ``(len(ks), n_seg, *seg_shape)`` ndarray (builds the index
        on first use).  See :meth:`psdata.index.RunIndex.read_stack`."""
        return self.build_index().read_stack(ks, det, field=field, alg=alg)

    # -- env / slow data (epics, scan) ------------------------------------
    @property
    def env(self):
        """The run's :class:`~psdata.envstore.EnvStoreManager` -- as-of random
        access to the ``epics`` and ``scan`` env stores.

        Lazily built on first use; building it first builds the random-access
        index (whose header walk already recorded the env dgram offsets, so no
        extra I/O), then wraps the index's ``env_records``.  Cached on the run.
        """
        if self._env is None:
            idx = self.build_index()
            self._env = _e.EnvStoreManager(self.config, idx.env_records)
        return self._env

    def env_store(self, name):
        """Return the :class:`~psdata.envstore.EnvStore` named ``name``
        (``"epics"`` or ``"scan"``)."""
        return self.env.store(name)

    def epics(self, var, ts):
        """As-of value of epics variable ``var`` at event timestamp ``ts`` (or
        ``None`` if no SlowUpdate at/before ``ts`` carries it).  Random access:
        ``searchsorted(env_ts, ts, 'right')-1`` then backward-skip, exactly as
        psana's env store -- see :meth:`psdata.envstore.EnvStore.as_of`."""
        return self.env.store("epics").value(var, ts)

    def scan(self, var, ts):
        """As-of value of scan field ``var`` (``step_value`` / ``step_docstring``)
        at event timestamp ``ts`` (or ``None``).  Fed by BeginStep; the backward
        scan skips the containerless BeginRun."""
        return self.env.store("scan").value(var, ts)

    def epicsinfo(self):
        """``{(var, pv): pv}`` mapping every epics variable to its real PV name
        (``''`` if unmapped) -- byte-identical to psana's ``run.epicsinfo``."""
        return self.env.epicsinfo()

    def scaninfo(self):
        """``{(var, alg): alg}`` for the scan store -- byte-identical to psana's
        ``run.scaninfo``."""
        return self.env.scaninfo()

    # -- introspection -----------------------------------------------------
    def detector_names(self, include_bookkeeping=False):
        """Sorted detector names discovered from the Configure Names tables.  By
        default container-bookkeeping pseudo-detectors (runinfo / chunkinfo /
        ...) are omitted."""
        return self.config.detector_names(
            include_bookkeeping=include_bookkeeping)

    def detector(self, name):
        """Return the :class:`~psdata.format.DetectorInfo` for ``name`` -- its
        algs, fields (name / dtype / rank), segments, and segment->stream map."""
        return self.config.detector(name)

    def find_detector_by_type(self, det_type):
        """Sorted names of detectors whose ``det_type`` matches (e.g. ``'ts'``
        for the timing detector that carries ``pulseId``)."""
        return self.config.find_detector_by_type(det_type)

    def uniqueid(self, detname):
        """Long hardware unique-id of ``detname`` -- byte-identical to psana's
        ``det.raw._uniqueid``.

        Built purely from the Configure Names tables (det_type + each
        segment's serial, in segment order) -- no event read, no psana.  This
        is the long composite id used to address a detector's calibration
        constants, so callers (e.g. a web-DB constant fetch) can derive it
        from the data instead of pinning it as a literal::

            uid = run.uniqueid("jungfrau")
            cons = webdb.get_constants(uid, exp=..., run=...)
        """
        return self.config.uniqueid(detname)

    # -- CONFIGURE-block accessor -----------------------------------------
    @staticmethod
    def _wrap_seg_configs(per_seg, alg):
        """Wrap ``{segment_id: {field: value}}`` as ``{segment_id: seg_cfg}``
        with ``seg_cfg.<alg>.<field>`` access (the public shape of
        :meth:`seg_configs`)."""
        return {seg: _SegConfig({alg: _AlgNamespace(fld_map)})
                for seg, fld_map in per_seg.items()}

    def seg_configs(self, detname, alg="config", step=None, evt=None):
        """Per-segment CONFIGURE-block object for ``detname``.

        Returns ``{segment_id: seg_cfg}`` where ``seg_cfg.<alg>.<field>`` reads
        a static settings field written into the Configure/BeginStep dgram (not
        an L1Accept event field) -- e.g. for epix10ka::

            scfg = run.seg_configs("epixquad")          # {0: ..., 1: ..., ...}
            trbit = scfg[0].config.trbit                # (4,)   uint8
            apc   = scfg[0].config.asicPixelConfig      # (4,176,192) uint8

        These per-ASIC fields are exactly what the epix gain-range decode needs
        and are byte-identical to psana's
        ``det.raw._seg_configs()[seg].config.{trbit,asicPixelConfig}``.  The
        accessor is generic -- it works for any detector whose Names tables
        declare a ``config`` algorithm (jungfrau, epix10ka, ...), reading the
        fields with the same DescData decoder used for event data.

        Field names that are not valid Python identifiers (jungfrau's dotted /
        enum-suffixed config fields) are reachable from
        ``seg_cfg.<alg>.fields[name]`` rather than by attribute syntax.

        **Multi-step runs (CAL-02).**  An epix10ka config field -- notably
        ``trbit``, which selects the gain-decode branch -- can CHANGE across DAQ
        steps: a fresh config rides on each ``BeginStep`` transition and, like
        psana's *stateful* ``det.raw._seg_configs()``, overrides the value in
        effect for that step's events (last-wins per segment).  The default,
        single-value form (``step``/``evt`` both ``None``) returns the config
        active up to the FIRST L1Accept -- i.e. step 0's -- and is **byte-exact
        for a single-step run** (the config is constant, so there is nothing to
        pick).  On a MULTI-step run that one value is WRONG for later steps, so a
        calibration consumer must address the active config per step/event:

        * ``run.seg_configs(det, step=k)`` -- the config ACTIVE during DAQ step
          ``k`` (0-based); equivalently :meth:`seg_configs_at`.
        * ``run.seg_configs(det, evt=e)`` -- the config active AT event ``e``
          (an :class:`~psdata.stream.Event` or a raw 64-bit timestamp),
          resolved with psana's as-of rule: the most recent ``BeginStep``
          override at/before the event (``searchsorted(begin_ts, ts, 'right') -
          1``).  This is the value psana's per-event ``det.raw.calib`` uses.

        See :meth:`n_config_steps` for the number of steps.
        """
        if step is None and evt is None:
            # Backward-compatible single-value path -- UNCHANGED (byte-exact for
            # a single-step run, where the config is constant across the run).
            per_seg = _f.read_config_object(self.config, detname, alg=alg)
            return self._wrap_seg_configs(per_seg, alg)
        if step is not None and evt is not None:
            raise ValueError("pass at most one of step= or evt=")

        steps = self._seg_config_steps(detname, alg=alg)
        if step is not None:
            if not 0 <= step < len(steps):
                raise IndexError(
                    f"config step {step} out of range for {detname!r} "
                    f"(run has {len(steps)} config step(s))")
            return self._wrap_seg_configs(steps[step][1], alg)

        # evt given: resolve to the active step by the as-of rule (side='right'
        # then -1), matching psana's stateful _seg_configs / EnvStore semantics.
        ts = getattr(evt, "timestamp", evt)
        if ts is None:
            raise ValueError(
                "seg_configs(evt=...): event has no timestamp (an unindexed "
                "shutdown-tail event); cannot resolve its DAQ step")
        ts = int(ts)
        begin_ts = [bts for bts, _cfg in steps if bts is not None]
        if not begin_ts:
            pos = 0                            # no BeginStep -> the sole step
        else:
            pos = bisect.bisect_right(begin_ts, ts) - 1
            if pos < 0:
                pos = 0                        # event precedes the first step
        return self._wrap_seg_configs(steps[pos][1], alg)

    def _seg_config_steps(self, detname, alg="config"):
        """Ordered per-step ACTIVE config for ``(detname, alg)``.

        Returns a list ``[(begin_step_ts, {segment: {field: value}}), ...]``
        ordered by step, where each entry is the config ACTIVE for that step --
        psana's stateful ``_seg_configs`` as of that step's ``BeginStep`` (the
        Configure default, then every ``BeginStep`` override up to and including
        that step, last-wins per segment).  Element 0 is byte-identical to the
        single-value :meth:`seg_configs`; each later element folds in that
        step's override.  Cached on the run.

        The BeginStep dgrams were already located, with zero extra I/O, by the
        index build and bucketed into the ``scan`` env store as ``{stream:
        [(ts, path, offset, size), ...]}``; each holds the whole broadcast
        BeginStep dgram (byte-identical whether sourced from the SMD sidecar or
        the bigdata), so the config ShapesData that rides on a BeginStep is read
        straight from there -- no walk of the GB-scale bigdata.
        """
        key = (detname, alg)
        cached = self._seg_cfg_steps.get(key)
        if cached is not None:
            return cached

        det = self.config.detector(detname)
        if alg not in det.algs:
            raise KeyError(f"detector {detname!r} has no alg {alg!r} "
                           f"(have {det.alg_names()})")
        want_fields = det.field_names(alg)

        # Base = the step-0 / Configure-default active config (walks the front
        # transitions up to the first L1Accept, last-wins).  Byte-identical to
        # the single-value accessor; the running state the later steps fold into.
        active = {seg: dict(fld_map)
                  for seg, fld_map in _f.read_config_object(
                      self.config, detname, alg=alg).items()}

        streams = det.streams_for(alg)
        nkey_to_seg = {}
        for stream in streams:
            tables = self.config.raw_tables[stream]
            nkey_to_seg[stream] = {
                nkey: tbl["segment"] for nkey, tbl in tables.items()
                if tbl["det_name"] == detname and tbl["alg_name"] == alg}

        # A BeginStep is broadcast to every stream with the SAME timestamp, so
        # one step == one ts, merging each owning stream's segments.
        scan_recs = self.build_index().env_records.get("scan", {})
        per_ts = {}                            # ts -> {stream: (path, off, sz)}
        for stream in streams:
            for (ts, path, offset, size) in scan_recs.get(stream, ()):
                per_ts.setdefault(int(ts), {})[stream] = (path, offset, size)

        steps = []
        for ts in sorted(per_ts):
            is_begin_step = False
            for stream, (path, offset, size) in per_ts[ts].items():
                overlay = self._read_config_dgram(
                    path, offset, size, self.config.raw_tables[stream],
                    nkey_to_seg[stream], want_fields)
                if overlay is None:
                    continue                   # a BeginRun the scan store holds
                is_begin_step = True
                active.update(overlay)         # stateful last-wins per segment
            if is_begin_step:
                steps.append((ts, {seg: dict(m) for seg, m in active.items()}))

        if not steps:
            # No BeginStep at all (unusual) -> the run is one step whose config
            # is the Configure default.
            steps = [(None, {seg: dict(m) for seg, m in active.items()})]

        self._seg_cfg_steps[key] = steps
        return steps

    @staticmethod
    def _read_config_dgram(path, offset, size, tables, nkey_to_seg,
                           want_fields):
        """Extract this detector's per-segment config from ONE BeginStep dgram.

        Reads the dgram bytes at ``(path, offset, size)`` and decodes each
        ``want_fields`` value for every segment whose config ShapesData rides on
        it -- the same primitives (:func:`~psdata.format.iter_shapesdata` /
        :func:`~psdata.format.extract_field`) the single-value accessor uses.
        Returns ``{segment: {field: value}}`` (``{}`` if a BeginStep carries no
        config for this detector), or ``None`` if the dgram is not a
        ``BeginStep`` (a ``BeginRun`` the scan store also holds -- it never
        overrides config)."""
        fd = os.open(path, os.O_RDONLY)
        try:
            buf = bytearray(os.pread(fd, size, offset))
        finally:
            os.close(fd)
        hdr = _f.parse_dgram_header(buf, 0)
        if hdr["service"] != _f.SERVICE_BEGINSTEP:
            return None
        out = {}
        for xoff, xh in _f.iter_shapesdata(buf, 0, hdr):
            nkey = _f.namesid_of(xh["src"])
            seg = nkey_to_seg.get(nkey)
            if seg is None:
                continue
            table = tables[nkey]
            out[seg] = {fld: _f.extract_field(buf, xoff, xh, table, fld)[0]
                        for fld in want_fields}
        return out

    def seg_configs_at(self, detname, step, alg="config"):
        """The per-segment CONFIGURE object ACTIVE during DAQ ``step`` (0-based)
        -- ``seg_configs(detname, alg, step=step)``.

        For a multi-step run whose config changes across steps (e.g. epix10ka
        ``trbit`` differing per step), this is the config a calibration consumer
        must apply to that step's events; the single-arg :meth:`seg_configs`
        returns only step 0's, which silently mis-decodes later steps' gain."""
        return self.seg_configs(detname, alg=alg, step=step)

    def n_config_steps(self, detname, alg="config"):
        """Number of DAQ steps that carry a (possibly re-overridden) ``(detname,
        alg)`` config -- ``1`` for a single-step run, ``>1`` when the config is
        re-emitted on later ``BeginStep`` transitions."""
        return len(self._seg_config_steps(detname, alg=alg))

    def config_object(self, detname, alg="config"):
        """Alias for :meth:`seg_configs` -- the per-segment CONFIGURE object
        ``{segment_id: seg_cfg}`` (``seg_cfg.<alg>.<field>``)."""
        return self.seg_configs(detname, alg=alg)

    # -- resource management ----------------------------------------------
    def close(self):
        """Release the random-access index's and env store's open file
        descriptors (if built).  Streaming opens/closes its own per-call
        cursors."""
        if self._env is not None:
            self._env.close()
            self._env = None
        if self._index is not None:
            self._index.close()
            self._index = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        who = (f"exp={self.exp!r} run={self.run!r}"
               if self.exp is not None else f"{len(self.files)} files")
        return (f"Run({who}, streams={sorted(self.files)}, "
                f"detectors={self.detector_names()})")


def open(exp=None, run=None, dir=None, files=None, run_config=None):
    """Open a run for reading -- the public entry point of psdata.

    Provide **either** ``exp`` + ``run`` + ``dir`` (the files are resolved by
    the standard layout ``{dir}/{exp}-r{run:04d}-s{stream:03d}-c000.xtc2`` with
    the SMD index under ``{dir}/smalldata/``) **or** an explicit ``files`` list
    (a ``{index: path}`` dict, ``(index, path)`` pairs, or a plain list of
    paths).

    Parameters
    ----------
    exp : str, optional
        Experiment id (e.g. ``"mfx100848724"``).
    run : int, optional
        Run number.
    dir : str, optional
        Directory holding the run's bigdata xtc2 files (with ``smalldata/``
        underneath).
    files : sequence | dict, optional
        Explicit per-stream xtc2 files, used instead of exp/run/dir.
    run_config : psdata.format.RunConfig, optional
        Pre-discovered config for the resolved files; discovered here if
        omitted.

    Returns
    -------
    Run
        A handle to stream events, random-access by timestamp/position, and
        introspect detectors / fields / segments.

    Raises
    ------
    ValueError
        If neither (exp, run, dir) nor files is given (or both are).
    FileNotFoundError
        If exp/run/dir resolves to no stream files.
    """
    by_idr = exp is not None or run is not None or dir is not None
    if by_idr and files is not None:
        raise ValueError("pass either exp/run/dir or files, not both")
    if not by_idr and files is None:
        raise ValueError("pass exp/run/dir (all three) or an explicit files list")

    if by_idr:
        if exp is None or run is None or dir is None:
            raise ValueError("exp, run, and dir are all required together")
        mapping = _resolve_run_files(exp, run, dir)
        smd_files = None                       # default smalldata/ layout
    else:
        mapping = _files_to_mapping(files)
        smd_files = None

    if run_config is None:
        run_config = _f.discover(mapping)
    return Run(mapping, run_config, exp=exp, run=run, dir=dir,
               smd_files=smd_files)


# ==========================================================================
# Import-purity self check (delegates to the format module's checker)
# ==========================================================================
def assert_no_framework_imports():
    """Raise AssertionError if a framework leaked into ``sys.modules``."""
    _f.assert_no_framework_imports()
