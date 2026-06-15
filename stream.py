#!/usr/bin/env python3
"""psdata.stream -- multi-stream event assembly (timestamp + pulseId + raw).

This is the US-002 streaming layer on top of the US-001 parse core
(:mod:`psdata.format`).  It assembles per-event detector data from a run's
several xtc2 stream files by replicating psana's event-builder rule exactly,
and yields :class:`Event` objects in ascending timestamp order.

The merge rule is a clean-room reimplementation of psana's
``EventBuilder._gather_event`` (``psana/psana/eventbuilder.pyx`` ~line 322):

  * Advance every stream to its head (next unconsumed) dgram.
  * The event timestamp is the **minimum** head timestamp across streams.
  * Every stream whose head timestamp **equals** that minimum joins the event;
    its head dgram is consumed.
  * Every non-matching head is **not** consumed -- it belongs to a later event
    (psana rewinds the offset; here the cursor simply does not advance).

Only ``L1Accept`` dgrams (``TransitionId.isEvent`` -- service 12 or 11) are
yielded as events; the surrounding transitions (Configure / BeginRun /
BeginStep / Enable / SlowUpdate / ...) are consumed by the merge but skipped
for yielding, matching ``psana`` ``run.events()``.

The missing-segment rule mirrors ``DetectorImpl._segments``
(``psana/psana/detector/detector_impl.py`` ~line 121): a detector whose
received segment set does not match the full set declared in its Configure
Names table is reported as ``None`` for that event.

Like :mod:`psdata.format`, this module imports **no** psana / mpi4py / h5py --
only the standard library and numpy.
"""

import os
import sys

import numpy as np

from . import format as _f

# Re-export the format constants used here for convenience / clarity.
DGRAM_HDR = _f.DGRAM_HDR
XTC_HDR = _f.XTC_HDR

# TransitionId.isEvent is True only for L1Accept (12) and L1Accept_EndOfBatch
# (11).  (psana/psana/psexp/transitionid.py:65.)
SERVICE_L1ACCEPT = 12
SERVICE_L1ACCEPT_EOB = 11
_EVENT_SERVICES = frozenset({SERVICE_L1ACCEPT, SERVICE_L1ACCEPT_EOB})

# Bit 63 of the timing pulseId flags an LCLS-1 origin; psana masks it off
# (psana/psana/detector/ts.py:139).
_PULSEID_LCLS1_BIT = 1 << 63

# det_type of the timing system detector that carries pulseId.
_TIMING_DET_TYPE = "ts"


# ==========================================================================
# Per-stream cursor: lazily reads dgrams from one xtc2 file in order
# ==========================================================================
class _StreamCursor:
    """A forward cursor over one xtc2 stream's dgrams.

    Reads the file in growing windows with ``os.pread`` (never loads the whole
    bigdata file) and exposes the *head* dgram (timestamp + service + byte
    span) without consuming it.  ``advance`` consumes the head.

    To bound memory, the read buffer is periodically compacted: once the
    consumed prefix exceeds ``_trim_threshold`` bytes it is dropped and
    ``base`` (the absolute file offset of ``buf[0]``) is bumped.
    """

    def __init__(self, stream, path, start_off, read_chunk=1 << 20,
                 trim_threshold=1 << 22):
        self.stream = stream
        self.path = path
        self.read_chunk = read_chunk
        self._trim_threshold = trim_threshold

        self._fd = os.open(path, os.O_RDONLY)
        self.base = start_off              # abs file offset of buf[0]
        self.buf = bytearray()             # window starting at `base`
        self.pos = 0                       # cursor within buf (relative)
        self.eof = False
        self.bytes_read = 0

        # cached head (None until peek populates it)
        self._head = None                  # dict from parse_dgram_header
        self._head_total = 0               # on-disk size of head dgram

    # -- low-level buffer management ---------------------------------------
    def _fill_to(self, need_rel_end):
        """Ensure ``buf`` holds at least up to relative offset ``need_rel_end``.
        Returns True if satisfied, False at EOF before that much is available."""
        while len(self.buf) < need_rel_end:
            chunk = os.pread(self._fd, self.read_chunk,
                             self.base + len(self.buf))
            if not chunk:
                self.eof = True
                return False
            self.buf.extend(chunk)
            self.bytes_read += len(chunk)
        return True

    def _maybe_trim(self):
        if self.pos >= self._trim_threshold:
            del self.buf[:self.pos]
            self.base += self.pos
            self.pos = 0

    # -- head access -------------------------------------------------------
    def peek(self):
        """Return the head dgram header dict (with absolute ``_off`` and
        on-disk ``_total``), or ``None`` at end of stream.  Does not consume."""
        if self._head is not None:
            return self._head
        if not self._fill_to(self.pos + DGRAM_HDR):
            return None
        h = _f.parse_dgram_header(self.buf, self.pos)
        total = XTC_HDR + h["extent"]
        if not self._fill_to(self.pos + total):
            raise RuntimeError(
                f"stream {self.stream}: truncated dgram at file offset "
                f"{self.base + self.pos}")
        h["_off"] = self.pos
        h["_total"] = total
        self._head = h
        self._head_total = total
        return h

    def head_ts(self):
        h = self.peek()
        return None if h is None else h["ts"]

    def advance(self):
        """Consume the head dgram; the next ``peek`` reads the following one."""
        if self._head is None:
            if self.peek() is None:
                return
        self.pos += self._head_total
        self._head = None
        self._head_total = 0
        self._maybe_trim()

    def head_view(self):
        """Return ``(buf, off)`` for the head dgram so its payload can be
        parsed in place.  ``off`` is relative to ``buf``."""
        h = self.peek()
        if h is None:
            return None
        return self.buf, h["_off"], h

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ==========================================================================
# Event -- one assembled L1Accept across all matching streams
# ==========================================================================
class Event:
    """One assembled event.

    Exposes the event identity (``timestamp``, ``pulseId``) and lazy access to
    each detector's raw segment arrays.  Detector data is parsed on demand from
    the per-stream dgram buffers captured at assembly time, so building an
    Event is cheap; the numpy arrays are only materialized when requested.

    Notes
    -----
    The captured buffers are *views into the stream cursors' rolling windows*
    valid only until the next event is pulled from the same generator.  The
    streaming API copies out arrays on access (``np.frombuffer`` returns a view;
    callers needing to retain data across iterations should ``.copy()`` -- the
    convenience ``raw`` / ``as_dict`` helpers already return fresh arrays).
    """

    __slots__ = ("timestamp", "service", "_run_config", "_seg_index",
                 "_pulseid_cache")

    def __init__(self, timestamp, service, run_config, seg_index):
        self.timestamp = timestamp
        self.service = service
        self._run_config = run_config
        # seg_index: {(det_name, alg): {segment: (buf, shapesdata_off, hdr,
        #                                          table)}}
        self._seg_index = seg_index
        self._pulseid_cache = None

    # -- identity ----------------------------------------------------------
    @property
    def pulseId(self):
        """The timing detector's ``pulseId`` field for this event (bit 63 --
        the LCLS-1 flag -- masked off, as psana does).  ``None`` if the run has
        no timing detector or it is absent this event."""
        if self._pulseid_cache is not None:
            return self._pulseid_cache
        pid = self._read_pulse_id()
        self._pulseid_cache = pid
        return pid

    def _read_pulse_id(self):
        rc = self._run_config
        timing_names = rc.find_detector_by_type(_TIMING_DET_TYPE)
        if not timing_names:
            return None
        det_name = timing_names[0]
        det = rc.detector(det_name)
        # pulseId lives in the 'raw' alg of the timing detector.
        alg = "raw" if "raw" in det.algs else None
        if alg is None:
            return None
        if "pulseId" not in det.algs[alg]:
            return None
        segs = self._extract_segments(det_name, alg, "pulseId")
        if segs is None or not segs:
            return None
        # timing data comes from one segment (psana assumes seg 0/first).
        raw_pid = int(segs[sorted(segs)[0]])
        return raw_pid & ~_PULSEID_LCLS1_BIT

    # -- detector access ---------------------------------------------------
    def detector_names(self, include_bookkeeping=False):
        """Detector names that contributed data to *this* event."""
        present = {d for (d, _a) in self._seg_index}
        out = []
        for name in present:
            det = self._run_config.detectors.get(name)
            if det is None:
                continue
            if include_bookkeeping or not det.is_bookkeeping:
                out.append(name)
        return sorted(out)

    def _expected_segments(self, det_name, alg):
        det = self._run_config.detectors.get(det_name)
        if det is None or alg not in det.segments:
            return None
        return sorted(det.segments[alg])

    def _extract_segments(self, det_name, alg, field):
        """Return ``{segment: ndarray-or-scalar}`` for ``field`` of
        ``(det_name, alg)`` across this event, or ``None`` if the detector's
        received segment set does not match the full declared set (the psana
        missing-segment rule)."""
        key = (det_name, alg)
        captured = self._seg_index.get(key)
        if not captured:
            return None
        expected = self._expected_segments(det_name, alg)
        received = sorted(captured)
        if expected is not None and received != expected:
            # missing-segment -> detector is None for this event
            return None
        out = {}
        for seg, (buf, sd_off, hdr, table) in captured.items():
            arr, _shape, _meta = _f.extract_field(buf, sd_off, hdr, table,
                                                   field)
            out[seg] = arr
        return out

    def raw(self, det_name, field="raw", alg="raw"):
        """Return ``{segment: ndarray}`` for ``(det_name, alg).field`` this
        event, or ``None`` if the detector is missing a segment (psana rule).

        Arrays are normalized so a per-segment array declared as ``(1, H, W)``
        is returned as ``(H, W)`` (matching the per-segment frame shape psana
        stacks)."""
        segs = self._extract_segments(det_name, alg, field)
        if segs is None:
            return None
        out = {}
        for seg, arr in segs.items():
            a = np.asarray(arr)
            if a.ndim >= 3 and a.shape[0] == 1:
                a = a.reshape(a.shape[1:])
            out[seg] = np.array(a, copy=True)
        return out

    def stack(self, det_name, field="raw", alg="raw"):
        """Return ``(n_segments, *seg_shape)`` stack ordered by segment id, or
        ``None`` if the detector is missing a segment.  This matches the layout
        psana's ``det.raw.raw(evt)`` returns."""
        segs = self.raw(det_name, field=field, alg=alg)
        if segs is None:
            return None
        seg_ids = sorted(segs)
        sample = segs[seg_ids[0]]
        out = np.empty((len(seg_ids),) + sample.shape, dtype=sample.dtype)
        for k, s in enumerate(seg_ids):
            out[k] = segs[s]
        return out

    def as_dict(self, field="raw", alg="raw", include_bookkeeping=False):
        """``{det_name: {segment: ndarray}}`` for every detector that declares
        ``alg``/``field`` and contributed this event.  A detector missing a
        segment maps to ``None`` (psana rule)."""
        out = {}
        for name in self.detector_names(include_bookkeeping=include_bookkeeping):
            det = self._run_config.detector(name)
            if alg not in det.algs or field not in det.algs[alg]:
                continue
            out[name] = self.raw(name, field=field, alg=alg)
        return out

    def __repr__(self):
        return (f"Event(timestamp={self.timestamp}, "
                f"pulseId={self.pulseId}, "
                f"detectors={self.detector_names()})")


# ==========================================================================
# Segment indexing: walk an event dgram and capture every ShapesData
# ==========================================================================
def _index_dgram(buf, dg_off, dg_hdr, tables, out):
    """Recursively walk one event dgram; for every ShapesData record its
    ``(buf, shapesdata_off, hdr, names_table)`` into ``out`` keyed by
    ``(det_name, alg) -> {segment: ...}``.  ``tables`` is the stream's
    ``{(nodeId,namesId): names_table}`` Configure map.

    ``buf`` must be a **stable** buffer (a snapshot of the dgram), not a
    cursor's rolling window -- the captured ``(buf, off)`` are read lazily when
    a detector field is requested, possibly after the cursor has advanced and
    compacted its window.  The caller snapshots the dgram's bytes and passes a
    ``dg_off`` relative to that snapshot.
    """
    top_payload = dg_off + DGRAM_HDR
    top_end = dg_off + XTC_HDR + dg_hdr["extent"]

    def recurse(payload_off, payload_end):
        for xoff, xh in _f.iter_xtc_children(buf, payload_off, payload_end):
            t = _f.typeid_type(xh["typeid"])
            if t == _f.TID_SHAPESDATA:
                key = _f.namesid_of(xh["src"])
                table = tables.get(key)
                if table is None:
                    continue
                det_alg = (table["det_name"], table["alg_name"])
                seg = table["segment"]
                out.setdefault(det_alg, {})[seg] = (buf, xoff, xh, table)
            elif t == _f.TID_PARENT:
                recurse(xoff + XTC_HDR, xoff + xh["extent"])

    recurse(top_payload, top_end)


# ==========================================================================
# Public streaming API
# ==========================================================================
def events(stream_files, run_config=None):
    """Yield assembled :class:`Event` objects in ascending timestamp order.

    Parameters
    ----------
    stream_files : see :func:`psdata.format.discover`
        The run's per-stream xtc2 files (``{index: path}``, ``(index, path)``
        pairs, or a plain path list).
    run_config : :class:`psdata.format.RunConfig`, optional
        A pre-discovered config for ``stream_files``; if omitted it is
        discovered here.  Reusing one avoids re-reading each Configure.

    Yields
    ------
    Event
        One per ``L1Accept`` (only ``isEvent`` services are yielded), in
        ascending 64-bit timestamp order, with ``timestamp``, ``pulseId``, and
        lazy raw detector access.

    Notes
    -----
    The Event's underlying buffers are views into rolling per-stream windows
    and are only valid until the next event is produced; the ``raw`` / ``stack``
    / ``as_dict`` accessors copy out arrays, so retained results are safe.
    """
    if run_config is None:
        run_config = _f.discover(stream_files)
    rc = run_config

    cursors = []
    for stream in sorted(rc.stream_files):
        _cfg, cfg_end = rc.stream_configs[stream]
        cursors.append(_StreamCursor(stream, rc.stream_files[stream], cfg_end))

    try:
        while True:
            # --- find min head ts across live streams (k-way merge key) ---
            min_ts = None
            for cur in cursors:
                ts = cur.head_ts()
                if ts is None:
                    continue
                if min_ts is None or ts < min_ts:
                    min_ts = ts
            if min_ts is None:
                return  # all streams exhausted

            # The service of the event is the head service of the min-ts
            # stream (psana: out_service = services[smd_id]); all matching
            # heads share the same ts and -- for a well-formed run -- service.
            service = None
            is_event = False
            for cur in cursors:
                ts = cur.head_ts()
                if ts == min_ts:
                    service = cur.peek()["service"]
                    is_event = service in _EVENT_SERVICES
                    break

            # --- collect matching heads; consume only those ----------------
            seg_index = {}
            for cur in cursors:
                hv = cur.head_view()
                if hv is None:
                    continue
                buf, off, h = hv
                if h["ts"] != min_ts:
                    continue  # belongs to a later event -- do NOT consume
                if is_event:
                    # Snapshot just this dgram's bytes so the captured offsets
                    # stay valid after the cursor advances and compacts its
                    # rolling window.  Offsets become relative to the snapshot.
                    total = h["_total"]
                    snap = bytes(buf[off:off + total])
                    snap_h = dict(h)
                    snap_h["_off"] = 0
                    tables = rc.raw_tables[cur.stream]
                    _index_dgram(snap, 0, snap_h, tables, seg_index)
                cur.advance()

            if is_event:
                yield Event(min_ts, service, rc, seg_index)
            # else: a transition (Configure/BeginRun/Enable/SlowUpdate/...) --
            # consumed across all matching streams above, but not yielded.
    finally:
        for cur in cursors:
            cur.close()


# ==========================================================================
# Import-purity self check (delegates to the format module's checker)
# ==========================================================================
def assert_no_framework_imports():
    """Raise AssertionError if a framework leaked into ``sys.modules``."""
    _f.assert_no_framework_imports()
