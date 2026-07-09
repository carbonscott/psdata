#!/usr/bin/env python3
"""psdata.envstore -- as-of random access to the run's slow (env) data.

This is the env-store layer on top of the parse core (:mod:`psdata.format`) and
the random-access index (:mod:`psdata.index`).  It serves the value of a slow
control variable (``epics``) or a scan step field (``scan``) *as of* any event
timestamp, by random access -- reproducing psana's ``EnvStore`` semantics
exactly, but with no psana, mpi4py, h5py, or MPI (numpy + stdlib only).

What an env store is
--------------------
LCLS-II writes slow / bookkeeping data on **transition** dgrams, not on the
per-event ``L1Accept``:

  * ``epics`` slow control values ride on every ``SlowUpdate`` (service 10).
  * ``scan`` step fields (``step_value`` / ``step_docstring``) ride on every
    ``BeginStep`` (service 6); psana also feeds the scan store the ``BeginRun``
    (service 4) -- which carries **no** scan container -- so the backward scan
    below must skip it.

psana keeps **one env manager per stream** per store (no cross-stream merge, no
timestamp dedup); this module mirrors that: :attr:`EnvStore._records` is keyed
by stream.  The per-event byte offsets of those transition dgrams are recorded
-- with **zero extra I/O** -- while the index build already walks every dgram
header (see :attr:`psdata.index.RunIndex.env_records`), so serving a value is a
single ``os.pread`` of the transition dgram, decoded lazily with the same
generic DescData decoder used for event data.

The as-of rule (random access)
-------------------------------
For a query timestamp ``ts`` and one stream's ascending env timestamps
``env_ts``::

    pos = np.searchsorted(env_ts, ts, side='right') - 1     # at-or-before
    if pos < 0:  ->  None (the event precedes every env update)

then scan **backward** up to :data:`N_STEP_SEARCH_STEPS` dgrams, skipping any
dgram that does not actually carry the store's container/alg for the variable
(this is how the ``scan`` store skips its ``BeginRun`` entry).  The first dgram
that yields the field wins; if none do, the value is ``None``.

``side='right'`` (an exactly-equal env timestamp counts as at-or-before) matches
psana's ``get_step_dgrams_of_event``; it is **not** psana's online ``values()``
(``side='left'``), which is only correct because psana feeds its store in order.
A variable with no update at/before the event returns ``None`` -- never raises,
never clamps.

Purity: like the rest of ``psdata``, this module imports no psana / mpi4py /
h5py -- only the standard library and numpy (see
:func:`assert_no_framework_imports`).
"""

import os

import numpy as np

from . import format as _f

# The two default env stores psana's EnvStoreManager always creates.
_STORE_NAMES = ("epics", "scan")

# The detector whose Configure container maps an epics variable -> its real PV
# name (psana reads this from ``config.epicsinfo``).  Only the epics store has
# one; the scan store's PV name is always ''.
_EPICS_INFO_DET = "epicsinfo"
# A spurious field in the epicsinfo container (value 'epicsname'); it is NOT an
# epics variable -- exclude it from the var -> PV mapper.
_EPICS_INFO_SKIP_FIELD = "keys"

# Default backward-scan depth (psana's PS_N_STEP_SEARCH_STEPS).  A module
# constant, overridable at call time via the PSDATA_N_STEP_SEARCH_STEPS env var.
N_STEP_SEARCH_STEPS = 10


def _search_steps():
    return int(os.environ.get("PSDATA_N_STEP_SEARCH_STEPS", N_STEP_SEARCH_STEPS))


class EnvStore:
    """As-of random access to one env store (``"epics"`` or ``"scan"``).

    Built from the run's :class:`~psdata.format.RunConfig` and the store's slice
    of :attr:`psdata.index.RunIndex.env_records` (``{stream: [(ts, path, offset,
    size), ...]}`` ascending by ts).  The store's *variables* and their alg come
    from the discovered detector of the same name (``epics`` / ``scan``); the
    values are decoded lazily from the transition dgram bytes at lookup time.
    """

    def __init__(self, name, run_config, stream_records):
        self.name = name
        self._run_config = run_config
        # {stream: [(ts, path, offset, size), ...]} ascending by ts
        self._records = {int(s): list(recs)
                         for s, recs in (stream_records or {}).items()}
        self._ts_cache = {}          # stream -> np.uint64 ascending array
        self._fds = {}               # path -> open O_RDONLY fd (never persisted)
        self._pv_mapper = None       # {var: pv} for epics; built lazily

    # -- detector / variable metadata --------------------------------------
    def _det(self):
        return self._run_config.detectors.get(self.name)

    def var_names(self):
        """Sorted names of every variable this store exposes (the fields of the
        discovered ``epics`` / ``scan`` detector)."""
        det = self._det()
        if det is None:
            return []
        names = set()
        for fields in det.algs.values():
            names.update(fields)
        return sorted(names)

    def alg_of(self, var):
        """Algorithm under which ``var`` is declared, or ``None``."""
        det = self._det()
        if det is None:
            return None
        for alg, fields in det.algs.items():
            if var in fields:
                return alg
        return None

    def _field_info(self, var):
        det = self._det()
        if det is None:
            return None
        for fields in det.algs.values():
            if var in fields:
                return fields[var]
        return None

    def owning_stream(self, var):
        """The (lowest) stream index that declares ``var``'s container, or
        ``None``.  Mirrors psana skipping every env manager whose config does
        not declare the variable (``EnvManager.locate_variable``)."""
        streams = self._owning_streams(var)
        return streams[0] if streams else None

    def _owning_streams(self, var):
        """All stream indices that declare ``var``'s (detector, alg), via
        :meth:`psdata.format.DetectorInfo.streams_for`."""
        det = self._det()
        alg = self.alg_of(var)
        if det is None or alg is None:
            return []
        return det.streams_for(alg)

    # -- timestamps --------------------------------------------------------
    def _ts_array(self, stream):
        arr = self._ts_cache.get(stream)
        if arr is None:
            recs = self._records.get(stream, ())
            arr = np.array([r[0] for r in recs], dtype=np.uint64)
            self._ts_cache[stream] = arr
        return arr

    def timestamps(self, stream=None):
        """Ascending ``uint64`` env timestamps.  For a given ``stream`` its own
        array; for ``stream=None`` the sorted unique union across streams (the
        broadcast transitions share timestamps across streams, so the union is
        one stream's set)."""
        if stream is not None:
            return self._ts_array(int(stream))
        if not self._records:
            return np.empty(0, dtype=np.uint64)
        allts = np.concatenate([self._ts_array(s)
                                for s in sorted(self._records)])
        return np.unique(allts)

    def n_items(self, stream=None):
        """Number of env dgrams -- for one ``stream``, or (``stream=None``) the
        total across all streams."""
        if stream is not None:
            return len(self._records.get(int(stream), ()))
        return sum(len(v) for v in self._records.values())

    # -- lazy dgram byte access --------------------------------------------
    def _read_dgram(self, path, offset, size):
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return bytearray(os.pread(fd, size, offset))

    def _convert(self, var, raw):
        """Coerce the raw DescData value to psana's ``_return_types`` shape:
        type 10 (CHARSTR) -> str; rank-0 type<8 -> int, type 8/9 -> float;
        rank>=1 numeric -> ndarray."""
        fi = self._field_info(var)
        if fi is None:
            return raw
        t, rank = fi.type_code, fi.rank
        if t == _f.TYPE_CHARSTR:
            return _f.decode_charstr(raw)
        if rank == 0:
            if t < 8:
                return int(raw)
            if t < 10:
                return float(raw)
        return np.asarray(raw)

    # -- the as-of lookup --------------------------------------------------
    def as_of(self, var, ts):
        """Return ``(value, env_ts)`` of ``var`` as of event timestamp ``ts``,
        or ``(None, None)`` if no update at/before ``ts`` carries it.

        Random-access reproduction of psana's env semantics: per owning stream,
        ``pos = searchsorted(env_ts, ts, 'right') - 1``; then scan backward up to
        :data:`N_STEP_SEARCH_STEPS` dgrams, skipping any that lack the store's
        container/alg for ``var`` (skips the scan store's ``BeginRun``).  The
        first stream to yield a value wins.
        """
        alg = self.alg_of(var)
        if alg is None:
            return (None, None)
        tq = np.uint64(int(ts))
        depth = _search_steps()
        tables_by_stream = self._run_config.raw_tables
        for stream in self._owning_streams(var):
            recs = self._records.get(stream)
            if not recs:
                continue
            env_ts = self._ts_array(stream)
            pos = int(np.searchsorted(env_ts, tq, side="right")) - 1
            if pos < 0:
                continue
            tables = tables_by_stream[stream]
            for p in range(pos, pos - depth, -1):
                if p < 0:
                    break
                rts, path, offset, size = recs[p]
                buf = self._read_dgram(path, offset, size)
                hdr = _f.parse_dgram_header(buf, 0)
                raw = _f.read_dgram_field(buf, 0, hdr, tables,
                                          self.name, alg, var)
                if raw is not None:
                    return (self._convert(var, raw), int(rts))
        return (None, None)

    def value(self, var, ts):
        """The as-of value of ``var`` at ``ts`` (``as_of(...)[0]``), or ``None``."""
        return self.as_of(var, ts)[0]

    # -- epics var -> PV name mapping --------------------------------------
    def _mapper(self):
        if self._pv_mapper is not None:
            return self._pv_mapper
        mapper = {}
        # Only the epics store has real PV names (from the epicsinfo container).
        if self.name == "epics":
            info_det = self._run_config.detectors.get(_EPICS_INFO_DET)
            if info_det is not None:
                for alg in info_det.alg_names():
                    streams = info_det.streams_for(alg)
                    if not streams:
                        continue
                    stream = streams[0]
                    path = self._run_config.stream_files[stream]
                    tables = self._run_config.raw_tables[stream]
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        buf = bytearray(os.pread(fd, _f._CONFIG_READ, 0))
                    finally:
                        os.close(fd)
                    hdr = _f.parse_dgram_header(buf, 0)
                    for fn in info_det.field_names(alg):
                        if fn == _EPICS_INFO_SKIP_FIELD:
                            continue
                        raw = _f.read_dgram_field(buf, 0, hdr, tables,
                                                  _EPICS_INFO_DET, alg, fn)
                        if raw is None:
                            continue
                        mapper[fn] = _f.decode_charstr(raw)
        self._pv_mapper = mapper
        return mapper

    def pv_name(self, var):
        """The real PV name of ``var`` -- ``''`` if unmapped (a var present in
        ``epics.raw`` but absent from ``epicsinfo``, e.g. ``StaleFlags``) and
        ``''`` for every scan var."""
        return self._mapper().get(var, "")

    # -- introspection dicts (mirror psana run.epicsinfo / run.scaninfo) ---
    def info(self):
        """psana-shaped introspection dict.

        epics: ``{(var, pv): pv}`` -- mirrors ``run.epicsinfo`` (unmapped ->
        ``('var',''): ''``).  scan: ``{(var, alg): alg}`` -- mirrors
        ``run.scaninfo``.
        """
        out = {}
        if self.name == "epics":
            for var in self.var_names():
                pv = self.pv_name(var)
                out[(var, pv)] = pv
        else:
            for var in self.var_names():
                alg = self.alg_of(var)
                out[(var, alg)] = alg
        return out

    # -- resource management -----------------------------------------------
    def close(self):
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()

    def __repr__(self):
        return (f"EnvStore(name={self.name!r}, streams={sorted(self._records)}, "
                f"n_items={self.n_items()}, nvars={len(self.var_names())})")


class EnvStoreManager:
    """Owns the run's env stores (``"epics"`` and ``"scan"``), mirroring psana's
    ``EnvStoreManager``.

    Built from the run's :class:`~psdata.format.RunConfig` and the index's
    :attr:`~psdata.index.RunIndex.env_records`.  Both stores are always created
    (as psana does); a store whose detector is absent simply has no variables.
    """

    def __init__(self, run_config, env_records):
        self._run_config = run_config
        env_records = env_records or {}
        self.stores = {
            name: EnvStore(name, run_config, env_records.get(name, {}))
            for name in _STORE_NAMES
        }

    def store(self, name):
        """Return the :class:`EnvStore` named ``name`` (``"epics"``/``"scan"``)."""
        return self.stores[name]

    def epicsinfo(self):
        """``run.epicsinfo`` -- the epics store's ``{(var, pv): pv}`` info dict."""
        return self.stores["epics"].info()

    def scaninfo(self):
        """``run.scaninfo`` -- the scan store's ``{(var, alg): alg}`` info dict."""
        return self.stores["scan"].info()

    def close(self):
        for store in self.stores.values():
            store.close()

    def __repr__(self):
        return f"EnvStoreManager(stores={sorted(self.stores)})"


# ==========================================================================
# Import-purity self check (delegates to the format module's checker)
# ==========================================================================
def assert_no_framework_imports():
    """Raise AssertionError if a framework leaked into ``sys.modules``."""
    _f.assert_no_framework_imports()
