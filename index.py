#!/usr/bin/env python3
"""psdata.index -- random-access-by-event index and read-by-timestamp.

This is the US-003 layer on top of the US-001 parse core
(:mod:`psdata.format`) and the US-002 event-assembly layer
(:mod:`psdata.stream`).  It builds a compact ``timestamp -> {stream:
(offset, size)}`` index by scanning only the **small SMD files**
(``smalldata/{exp}-r{run:04d}-s###-c000.smd.xtc2``), then serves an arbitrary
event by timestamp with a single ``os.pread`` per contributing stream into the
GB-scale bigdata files -- **without** sequentially scanning them.

How it works
------------
Every SMD file carries, on each ``L1Accept`` dgram, a ``smdinfo`` detector
(``det_type='offset'``, ``alg='offsetAlg'``) with two rank-0 ``uint64`` fields:

  * ``intOffset``    -- byte offset of the matching bigdata dgram, and
  * ``intDgramSize`` -- its full on-disk size (``XTC_HDR + extent``),

both pointing into the **bigdata file of the same stream index** (psana
indexes SMD file and bigdata file 1:1 -- ``event_manager.py`` reads
``d.smdinfo[0].offsetAlg.intOffset`` / ``.intDgramSize`` ~lines 106-107 and
opens ``self.dm.xtc_files[i_smd]`` for stream ``i_smd``).  So scanning the tiny
SMD files yields, per event timestamp, exactly where to ``pread`` each big
dgram.

An event is assembled from the ``pread``-ed bigdata dgrams using the same
machinery as the streaming path (:func:`psdata.stream._index_dgram` +
:class:`psdata.stream.Event`), so a randomly-read event is byte-identical to
the same event obtained by streaming.

Scope (US-003): single-chunk runs (``c000`` only).  Multi-chunk roll (offsets
reset at chunk boundaries) is US-004; an ``intOffset`` that decreases relative
to the running maximum within a stream is detected and reported here, but
following the roll is deferred.

Like the rest of ``psdata``, this module imports **no** psana / mpi4py / h5py
-- only the standard library and numpy.
"""

import bisect
import os
import time

import numpy as np

from . import format as _f
from . import stream as _s

# The bookkeeping detector that carries the per-event bigdata offset/size.
_SMDINFO_DET = "smdinfo"
_SMDINFO_ALG = "offsetAlg"
_OFFSET_FIELD = "intOffset"
_SIZE_FIELD = "intDgramSize"


# ==========================================================================
# Locating the smdinfo table in an SMD stream's Configure
# ==========================================================================
def _find_smdinfo_key(tables):
    """Return the ``(nodeId, namesId)`` of the ``smdinfo`` Names table in one
    SMD stream's Configure ``tables`` (``{(nodeId,namesId): names_table}``), or
    ``None`` if this stream has no smdinfo (e.g. it is a bigdata, not an SMD,
    file)."""
    for key, t in tables.items():
        if t["det_name"] == _SMDINFO_DET and t["alg_name"] == _SMDINFO_ALG:
            if (_OFFSET_FIELD in {n["name"] for n in t["names"]}
                    and _SIZE_FIELD in {n["name"] for n in t["names"]}):
                return key
    return None


def _read_smdinfo(buf, dg_off, dg_hdr, smdinfo_key, table):
    """Find the ``smdinfo`` ShapesData in one SMD dgram and return
    ``(intOffset, intDgramSize)`` as Python ints, or ``None`` if absent."""
    top_payload = dg_off + _f.DGRAM_HDR
    top_end = dg_off + _f.XTC_HDR + dg_hdr["extent"]
    found = [None]

    def recurse(payload_off, payload_end):
        if found[0] is not None:
            return
        for xoff, xh in _f.iter_xtc_children(buf, payload_off, payload_end):
            t = _f.typeid_type(xh["typeid"])
            if t == _f.TID_SHAPESDATA:
                if _f.namesid_of(xh["src"]) == smdinfo_key:
                    off_arr, _s1, _m1 = _f.extract_field(
                        buf, xoff, xh, table, _OFFSET_FIELD)
                    sz_arr, _s2, _m2 = _f.extract_field(
                        buf, xoff, xh, table, _SIZE_FIELD)
                    found[0] = (int(off_arr), int(sz_arr))
                    return
            elif t == _f.TID_PARENT:
                recurse(xoff + _f.XTC_HDR, xoff + xh["extent"])
                if found[0] is not None:
                    return

    recurse(top_payload, top_end)
    return found[0]


# ==========================================================================
# The random-access index
# ==========================================================================
class RunIndex:
    """A timestamp -> ``{stream: (bigdata_offset, bigdata_size)}`` index for a
    run, built by scanning only the small SMD files.

    Attributes
    ----------
    timestamps : list[int]
        L1Accept event timestamps in ascending order (the index of an event in
        this list is its event number / position ``k``).
    entries : list[dict]
        ``entries[k]`` is ``{stream: (offset, size)}`` -- which bigdata streams
        carry event ``k`` and where their dgram lives.  Only streams that
        contributed a segment to event ``k`` are present (a stream missing for
        an event is simply absent, so the US-002 missing-segment rule applies).
    bd_files : dict
        ``{stream: bigdata_path}`` -- the bigdata file each stream reads from
        (from the run config, same stream index as the SMD file).
    run_config : psdata.format.RunConfig
        The run's discovered detector / segment configuration (shared with the
        streaming path so assembled events are identical).
    build_seconds : float
        Wall-clock time spent building the index (SMD scan only).
    smd_bytes_read : int
        Total bytes read from the SMD files during the build.
    multichunk_streams : set[int]
        Streams whose ``intOffset`` was observed to *decrease* (a chunk-roll
        signature).  Empty for a clean single-chunk run; following the roll is
        US-004.
    """

    def __init__(self, run_config):
        self.run_config = run_config
        self.timestamps = []          # ascending L1Accept ts
        self.entries = []             # entries[k] = {stream: (offset, size)}
        self.bd_files = dict(run_config.stream_files)
        self.build_seconds = 0.0
        self.smd_bytes_read = 0
        self.multichunk_streams = set()
        # private: open bigdata fds, lazily, cached for repeated reads
        self._bd_fds = {}
        self._ts_to_k = None          # built lazily for read_event(ts)

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, smd_files, run_config, smd_read_chunk=1 << 22):
        """Build a :class:`RunIndex` by scanning the SMD files only.

        Parameters
        ----------
        smd_files : dict | sequence
            The run's SMD files, same forms as
            :func:`psdata.format.discover` accepts (``{stream: path}``,
            ``(stream, path)`` pairs, or a plain path list keyed positionally).
            The stream index of each SMD file **must** match the bigdata stream
            index in ``run_config`` (they share ``s###`` by construction).
        run_config : psdata.format.RunConfig
            The run config discovered from the *bigdata* stream files (its
            ``stream_files`` map gives the bigdata path per stream).
        smd_read_chunk : int
            Read granularity for the growing SMD window.

        Returns
        -------
        RunIndex
        """
        items = _f._normalize_stream_files(smd_files)
        idx = cls(run_config)

        t0 = time.monotonic()
        # per-stream ordered lists of (ts, offset, size)
        per_stream = {}
        for stream, path in items:
            recs, nbytes, rolled = _scan_smd_stream(path, smd_read_chunk)
            per_stream[stream] = recs
            idx.smd_bytes_read += nbytes
            if rolled:
                idx.multichunk_streams.add(stream)

        idx._merge_streams(per_stream)
        idx.build_seconds = time.monotonic() - t0
        return idx

    def _merge_streams(self, per_stream):
        """Combine the per-stream ``(ts, off, size)`` records into the unified
        ascending ``timestamps`` / ``entries`` index by exact-timestamp merge
        (an event's identity is its 64-bit ts, shared across streams)."""
        # gather the union of timestamps, then for each ts collect the streams
        merged = {}   # ts -> {stream: (off, size)}
        for stream, recs in per_stream.items():
            for ts, off, size in recs:
                merged.setdefault(ts, {})[stream] = (off, size)
        for ts in sorted(merged):
            self.timestamps.append(ts)
            self.entries.append(merged[ts])

    # -- lookup ------------------------------------------------------------
    @property
    def n_events(self):
        return len(self.timestamps)

    def _position_of(self, ts):
        """Return the event position ``k`` whose timestamp is exactly ``ts``,
        or raise ``KeyError``."""
        i = bisect.bisect_left(self.timestamps, ts)
        if i < len(self.timestamps) and self.timestamps[i] == ts:
            return i
        raise KeyError(f"no indexed event with timestamp {ts}")

    def _bd_fd(self, stream):
        fd = self._bd_fds.get(stream)
        if fd is None:
            fd = os.open(self.bd_files[stream], os.O_RDONLY)
            self._bd_fds[stream] = fd
        return fd

    def read_event(self, ts):
        """Read the event at exact timestamp ``ts`` by random access.

        Performs one ``os.pread`` per contributing stream at the indexed
        ``(offset, size)`` -- never scans the bigdata file -- and assembles an
        :class:`psdata.stream.Event` identical to the one the streaming path
        would yield for the same event.

        Raises ``KeyError`` if ``ts`` is not an indexed L1Accept event.
        """
        k = self._position_of(ts)
        return self.read_event_at(k)

    def read_event_at(self, k):
        """Read the ``k``-th L1Accept event (0-based, ascending ts) by random
        access.  Returns a :class:`psdata.stream.Event`."""
        if not (0 <= k < len(self.timestamps)):
            raise IndexError(f"event position {k} out of range "
                             f"[0, {len(self.timestamps)})")
        ts = self.timestamps[k]
        entry = self.entries[k]

        seg_index = {}
        service = _s.SERVICE_L1ACCEPT
        for stream in sorted(entry):
            offset, size = entry[stream]
            fd = self._bd_fd(stream)
            raw = os.pread(fd, size, offset)
            if len(raw) != size:
                raise RuntimeError(
                    f"stream {stream}: short read at offset {offset} "
                    f"(wanted {size}, got {len(raw)})")
            buf = bytearray(raw)
            h = _f.parse_dgram_header(buf, 0)
            if h["ts"] != ts:
                raise RuntimeError(
                    f"stream {stream}: bigdata dgram ts {h['ts']} != indexed "
                    f"ts {ts} (offset {offset}); chunk boundary? (US-004)")
            service = h["service"]
            snap_h = dict(h)
            snap_h["_off"] = 0
            snap_h["_total"] = size
            tables = self.run_config.raw_tables[stream]
            _s._index_dgram(buf, 0, snap_h, tables, seg_index)

        return _s.Event(ts, service, self.run_config, seg_index)

    # -- resource management ----------------------------------------------
    def close(self):
        for fd in self._bd_fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._bd_fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return (f"RunIndex(n_events={self.n_events}, "
                f"streams={sorted(self.bd_files)}, "
                f"build_seconds={self.build_seconds:.3f}, "
                f"smd_MB={self.smd_bytes_read / 1e6:.1f})")


# ==========================================================================
# SMD scan: read one SMD file's L1Accept (ts, intOffset, intDgramSize) records
# ==========================================================================
def _scan_smd_stream(path, read_chunk):
    """Scan one SMD file and return ``(records, bytes_read, rolled)`` where
    ``records`` is a list of ``(ts, intOffset, intDgramSize)`` for every
    ``L1Accept`` and ``rolled`` flags a decrease in ``intOffset`` (the
    chunk-roll signature; following it is US-004).

    Reads the SMD file in a growing ``os.pread`` window -- the SMD files are
    small (a few MB), so this is the only I/O the index build does; the bigdata
    files are never opened here.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        # Configure sits at offset 0; read enough to parse it, then grow.
        buf = bytearray(os.pread(fd, max(read_chunk, _f._CONFIG_READ), 0))
        bytes_read = len(buf)
        _cfg, tables, cfg_end = _f.parse_configure(buf)

        smdinfo_key = _find_smdinfo_key(tables)
        if smdinfo_key is None:
            raise RuntimeError(
                f"{os.path.basename(path)} has no smdinfo table -- is it an "
                f"SMD (smalldata) file?")
        table = tables[smdinfo_key]

        records = []
        rolled = False
        max_off = -1
        off = cfg_end
        while True:
            if off + _f.DGRAM_HDR > len(buf):
                more = os.pread(fd, read_chunk, len(buf))
                if not more:
                    break
                buf.extend(more)
                bytes_read += len(more)
                continue
            h = _f.parse_dgram_header(buf, off)
            total = _f.XTC_HDR + h["extent"]
            if off + total > len(buf):
                more = os.pread(fd, max(read_chunk, total), len(buf))
                if not more:
                    break   # truncated final dgram -> stop (clean EOF)
                buf.extend(more)
                bytes_read += len(more)
                continue

            if h["service"] == _f.SERVICE_L1ACCEPT:
                info = _read_smdinfo(buf, off, h, smdinfo_key, table)
                if info is not None:
                    intoff, intsize = info
                    if intoff < max_off:
                        rolled = True
                    max_off = max(max_off, intoff)
                    records.append((h["ts"], intoff, intsize))
            off += total
        return records, bytes_read, rolled
    finally:
        os.close(fd)


# ==========================================================================
# Convenience: discover SMD files & build the index from a run directory
# ==========================================================================
def smd_files_for(stream_files):
    """Given a run's bigdata ``stream_files`` (``{stream: path}`` or the other
    accepted forms), return the matching SMD files ``{stream: smd_path}`` by
    the standard layout (``<dir>/smalldata/<base>.smd.xtc2`` for a bigdata
    ``<dir>/<base>.xtc2``).

    Resolving by filename convention keeps the path pattern in one place; the
    caller can also pass an explicit SMD-file mapping straight to
    :meth:`RunIndex.build`.
    """
    items = _f._normalize_stream_files(stream_files)
    out = {}
    for stream, bd_path in items:
        d = os.path.dirname(bd_path)
        base = os.path.basename(bd_path)
        root, ext = os.path.splitext(base)         # ('...-c000', '.xtc2')
        smd_path = os.path.join(d, "smalldata", root + ".smd" + ext)
        out[stream] = smd_path
    return out


def build_index(stream_files, run_config=None, smd_files=None):
    """Build a :class:`RunIndex` for a run.

    Parameters
    ----------
    stream_files : see :func:`psdata.format.discover`
        The run's bigdata xtc2 stream files.
    run_config : psdata.format.RunConfig, optional
        Pre-discovered config for ``stream_files``; discovered here if omitted.
    smd_files : optional
        Explicit SMD-file mapping.  If omitted, resolved from ``stream_files``
        via :func:`smd_files_for` (the standard ``smalldata/`` layout).

    Returns
    -------
    RunIndex
    """
    if run_config is None:
        run_config = _f.discover(stream_files)
    if smd_files is None:
        smd_files = smd_files_for(stream_files)
    return RunIndex.build(smd_files, run_config)


# ==========================================================================
# Import-purity self check (delegates to the format module's checker)
# ==========================================================================
def assert_no_framework_imports():
    """Raise AssertionError if a framework leaked into ``sys.modules``."""
    _f.assert_no_framework_imports()
