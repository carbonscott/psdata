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

import glob
import os

from . import format as _f
from . import stream as _s
from . import index as _i

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

    # -- streaming ---------------------------------------------------------
    def events(self):
        """Yield assembled :class:`~psdata.stream.Event` objects in ascending
        timestamp order (forward streaming).  Each event exposes ``timestamp``,
        ``pulseId``, and lazy raw detector arrays (``evt.stack(name)`` /
        ``evt.raw(name)`` / ``evt.as_dict()``)."""
        return _s.events(self.files, run_config=self.config)

    # -- random access -----------------------------------------------------
    def build_index(self, rebuild=False):
        """Build (or return the cached) random-access
        :class:`~psdata.index.RunIndex` for this run.

        The index is built by scanning only the small SMD files -- the GB-scale
        bigdata is never read during the build.  Cached on the run after the
        first call; pass ``rebuild=True`` to force a fresh build.
        """
        if self._index is None or rebuild:
            self._index = _i.build_index(
                self.files, run_config=self.config, smd_files=self._smd_files)
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

    # -- resource management ----------------------------------------------
    def close(self):
        """Release the random-access index's open bigdata file descriptors (if
        an index was built).  Streaming opens/closes its own per-call cursors."""
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
