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

A detector can be ``None`` for two very different reasons, and STR-02 keeps them
apart.  A *genuine absence* is the missing-segment rule above -- the stream is
present but the detector did not contribute a (complete) frame this event.  A
*stream drop-out* is a data-quality fault: a bigdata stream that was producing
dgrams stops (hits EOF -- a killed DAQ / transport dropout) while other streams
carry the run on, so every detector on that stream is silently absent from then
on.  On the k-way merge the two are indistinguishable at the value level (both
yield ``None``), which is exactly psana's ``DroppedContribution`` condition
surfacing as an unnamed hole.  :func:`events` therefore records the streams that
had stopped by the time each event was assembled on the :class:`Event`
(``Event.dropped_streams``); :meth:`Event.detector_status` reports ``"dropped"``
vs ``"absent"`` so a consumer can tell the fault from a legitimate absence
instead of seeing an indistinguishable ``None``.  The frame VALUES are never
changed -- ``dropped_streams`` is a pure side-channel added to the Event, so
``stack`` / ``raw`` / ``as_dict`` return exactly what they did before, on any run.

The drop-out test is purely *structural* (a stream that WAS yielding has hit EOF
while others continue), so it cannot by itself distinguish a genuine mid-run
fault from a stream that simply stopped at the canonical end of the run while
other streams wrote a benign DAQ-shutdown tail (a ragged shutdown leaves some
streams with extra trailing L1Accepts past where others stopped --
mfx100848724/r51).  On the **canonical event set this never surfaces**: those
tail events are past every SMD-indexed timestamp, so the gated default
``Run.events()`` filters them out and ``dropped_streams`` is empty for a healthy
run.  It appears only on the explicitly opted-into **ungated** path
(``Run.events(gate=False)`` or the raw :func:`events` generator), where -- by the
structural definition of a drop-out -- an earlier-stopping stream is reported
``"dropped"`` on that shutdown tail (the same trailing events FAIL-01/STR-01
already treat as over-counted on that path).

Like :mod:`psdata.format`, this module imports **no** psana / mpi4py / h5py --
only the standard library and numpy.
"""

import os
import re
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
# Multi-chunk enumeration (STR-01): the -c00N chunk files one stream rolls
# through, by the on-disk filename convention -- the SAME set the random-access
# index path walks.
# ==========================================================================
def _enumerate_stream_chunks(c000_path):
    """Ordered bigdata chunk files for one stream, from its ``-c000`` path, by
    the ``-c000``/``-c001``/... filename convention (stopping at the first
    missing chunk id).  A single-chunk stream returns ``[c000_path]``.

    This mirrors :func:`psdata.index._enumerate_bd_chunks` **exactly** so the
    forward streaming cursor (:class:`_StreamCursor`) rolls through the identical
    set of chunk files the random-access index path walks in
    ``index._scan_bigdata_stream``.  The two paths must agree on which bytes back
    an event past a chunk roll (STR-01): the index followed the roll while the
    forward path did not, so ``stack()`` went ``None`` past the boundary while
    ``read_event_at`` returned full frames from the next chunk -- the reader
    contradicting itself.

    Enumerating by the on-disk ``-c00N`` convention -- rather than following the
    ``chunkinfo`` carried on each Enable (the SMD path's mechanism) -- keeps
    forward streaming self-contained: it needs only the bigdata files it already
    opens, with no SMD file and no ``chunkinfo`` decode.  The chunk filenames
    ``chunkinfo`` would name are exactly these ``-c00N`` siblings in the same
    directory (see ``index._enumerate_bd_chunks``), so the set walked here is
    identical to the one the SMD-following path rolls through.

    A path that does not match the ``-c00N.xtc2`` convention (e.g. a synthetic
    test fixture) is returned unchanged as a single-element list, so such runs
    stream exactly as before.
    """
    d = os.path.dirname(c000_path)
    base = os.path.basename(c000_path)
    m = re.search(r"-c(\d+)\.xtc2$", base)
    if not m:
        return [c000_path]
    prefix = base[:m.start()]            # '<exp>-rNNNN-sMMM'
    width = len(m.group(1))              # zero-pad width (3 -> c000)
    chunks = []
    cid = 0
    while True:
        cand = os.path.join(d, f"{prefix}-c{cid:0{width}d}.xtc2")
        if not os.path.exists(cand):
            break
        chunks.append(cand)
        cid += 1
    return chunks if chunks else [c000_path]


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

    Multi-chunk streams (STR-01): a long run's bigdata rolls from ``-c000`` to
    ``-c001``/... at 250-500 GB, and per-chunk dgram offsets restart at 0.  The
    cursor enumerates its stream's chunk files up front (by the same ``-c00N``
    filename convention the random-access index path uses) and, when it exhausts
    one chunk at a clean dgram boundary, transparently rolls to the next and
    continues from offset 0.  So the forward path keeps yielding events past the
    roll -- byte-identical to what the index path (``read_event_at``) serves for
    the same timestamps -- instead of silently going dead there.  A single-chunk
    stream has just one chunk and behaves exactly as before.
    """

    def __init__(self, stream, path, start_off, read_chunk=1 << 20,
                 trim_threshold=1 << 22):
        self.stream = stream
        self.read_chunk = read_chunk
        self._trim_threshold = trim_threshold

        # Chunk files this stream rolls through (STR-01).  ``path`` is the
        # ``-c000`` bigdata file; enumerate its ``-c00N`` siblings the SAME way
        # the index path does (:func:`_enumerate_stream_chunks` ==
        # ``index._enumerate_bd_chunks``) so forward streaming and random access
        # agree on the bytes past a roll.  A single-chunk stream yields
        # ``[path]``.
        self._chunks = _enumerate_stream_chunks(path)
        self._chunk_idx = 0
        self.path = self._chunks[0]        # the chunk file currently being read

        self._fd = os.open(self._chunks[0], os.O_RDONLY)
        self.base = start_off              # abs offset of buf[0] in `self.path`
        self.buf = bytearray()             # window starting at `base`
        self.pos = 0                       # cursor within buf (relative)
        self.eof = False
        self.bytes_read = 0

        # cached head (None until peek populates it)
        self._head = None                  # dict from parse_dgram_header
        self._head_total = 0               # on-disk size of head dgram

        # STR-01 roll guard: ts of the last dgram consumed (across all chunks),
        # and a one-shot flag set for the first peek after a roll -- so the
        # forward path can fail closed (mirroring the index path's "chunk roll
        # mismatch") rather than silently mis-decode if a roll is ill-formed.
        self._last_ts = None
        self._just_rolled = False

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

    # -- multi-chunk roll (STR-01) -----------------------------------------
    def _has_next_chunk(self):
        return self._chunk_idx + 1 < len(self._chunks)

    def _roll_to_next_chunk(self):
        """Advance to this stream's next bigdata chunk (``-c000`` -> ``-c001``
        -> ...) after the current one is exhausted at a clean dgram boundary,
        mirroring the index path's roll follow (``index._scan_bigdata_stream``
        walking ``_enumerate_bd_chunks``, whose per-chunk offsets restart at 0).
        Resets the intra-chunk offset to 0 and continues yielding from the new
        chunk.  Returns ``True`` if a next chunk was opened, ``False`` if this
        stream has no further chunk (a genuine end-of-stream).

        Only called from :meth:`peek` when the current window is fully consumed
        (``self.pos == len(self.buf)``), i.e. the chunk ended exactly on a dgram
        boundary -- guaranteed because the DAQ never splits a dgram across a
        chunk roll.
        """
        if not self._has_next_chunk():
            return False
        os.close(self._fd)
        self._chunk_idx += 1
        self.path = self._chunks[self._chunk_idx]
        self._fd = os.open(self.path, os.O_RDONLY)
        # Offsets restart at 0 in the new chunk; drop the (fully consumed)
        # window and rebase.
        self.base = 0
        del self.buf[:]
        self.pos = 0
        self.eof = False
        self._just_rolled = True
        return True

    # -- head access -------------------------------------------------------
    def peek(self):
        """Return the head dgram header dict (with absolute ``_off`` and
        on-disk ``_total``), or ``None`` at end of stream.  Does not consume.

        When the current chunk file is exhausted at a clean dgram boundary and
        the stream rolled to a further ``-c00N`` chunk, transparently follow the
        roll (STR-01) and continue from offset 0 of the next chunk -- so a
        multi-chunk run keeps yielding events past the roll instead of the
        cursor silently going dead (which left ``stack()`` returning ``None`` for
        detectors on rolled streams past the boundary, while ``read_event_at``
        returned full frames from the next chunk).  The k-way merge in
        :func:`events` sees an uninterrupted head stream and needs no change;
        single-chunk streams have no further chunk and behave exactly as before.
        """
        if self._head is not None:
            return self._head
        while not self._fill_to(self.pos + DGRAM_HDR):
            # No further full dgram header available in the current chunk.
            if self.pos != len(self.buf):
                # Partial header bytes remain at the chunk's physical end.  With
                # no further chunk this is a truncated tail (a killed DAQ) --
                # treat as end-of-stream, exactly as before.  With a further
                # chunk it means the chunk did NOT split on a dgram boundary,
                # which must never happen: fail closed (mirroring the index
                # path's "chunk roll mismatch") rather than silently mis-decode.
                if self._has_next_chunk():
                    raise RuntimeError(
                        f"stream {self.stream}: {os.path.basename(self.path)} "
                        f"ends {len(self.buf) - self.pos} byte(s) into a dgram "
                        f"header but a further chunk exists; chunk roll "
                        f"mismatch")
                return None
            # Clean dgram boundary: follow the roll to the next chunk if there
            # is one; else this is the genuine end of the stream.
            if not self._roll_to_next_chunk():
                return None
        h = _f.parse_dgram_header(self.buf, self.pos)
        total = XTC_HDR + h["extent"]
        if not self._fill_to(self.pos + total):
            raise RuntimeError(
                f"stream {self.stream}: truncated dgram at file offset "
                f"{self.base + self.pos} of {os.path.basename(self.path)}")
        if self._just_rolled:
            # First dgram of a freshly-rolled chunk: ts must not go backward
            # (dgrams stay in ascending ts across the roll -- the same ordering
            # the index path and the k-way merge rely on).  A backward ts means
            # the roll landed in the wrong file: fail closed rather than feed the
            # merge an out-of-order, mis-decoded event.
            self._just_rolled = False
            if self._last_ts is not None and h["ts"] < self._last_ts:
                raise RuntimeError(
                    f"stream {self.stream}: first dgram of "
                    f"{os.path.basename(self.path)} has ts {h['ts']} < last ts "
                    f"{self._last_ts} of the previous chunk; chunk roll "
                    f"mismatch")
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
        # Remember this dgram's ts so a subsequent chunk roll can verify the
        # next chunk continues in ascending ts order (STR-01 fail-closed guard).
        self._last_ts = self._head["ts"]
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
                 "_pulseid_cache", "_dropped_streams")

    def __init__(self, timestamp, service, run_config, seg_index,
                 dropped_streams=frozenset()):
        self.timestamp = timestamp
        self.service = service
        self._run_config = run_config
        # seg_index: {(det_name, alg): {segment: (buf, shapesdata_off, hdr,
        #                                          table)}}
        self._seg_index = seg_index
        self._pulseid_cache = None
        # STR-02: stream indices that were producing dgrams earlier in the run
        # but had stopped (hit EOF) by the time this event was assembled from
        # the streams still live.  Empty on the canonical (gated) event set of a
        # healthy run and for a random-access read (which observes one event in
        # isolation and cannot witness a run-scale drop-out); the one benign
        # exception is a ragged DAQ-shutdown tail on the ungated path -- see the
        # module docstring.  A detector carried on such a stream is absent for a
        # *data-quality* reason (a DAQ/transport dropout), distinct from a
        # detector legitimately not present this event -- see
        # :meth:`detector_status`.
        self._dropped_streams = frozenset(dropped_streams)

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

    # -- stream drop-out (STR-02) ------------------------------------------
    @property
    def dropped_streams(self):
        """The stream indices that had **dropped out** by the time this event
        was assembled: streams that were producing dgrams earlier in the run but
        had hit EOF while other streams carried the run on (a DAQ/transport
        dropout).  A ``frozenset`` -- empty on the canonical (gated) event set of
        a healthy run and for any random-access read.  (The test is structural,
        so on the explicitly ungated path a benign DAQ-shutdown tail can present
        an earlier-stopping stream as dropped on those trailing events; the gated
        ``Run.events()`` default filters them out -- see the module docstring.)
        Detectors carried on these streams are absent this event for a
        data-quality reason, not because they were legitimately not present;
        :meth:`detector_status` classifies which is which."""
        return self._dropped_streams

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

    def _alg_complete(self, det_name, alg):
        """True if ``(det_name, alg)`` contributed its FULL declared segment set
        this event (the same completeness test :meth:`_extract_segments` applies,
        without decoding any field bytes)."""
        captured = self._seg_index.get((det_name, alg))
        if not captured:
            return False
        expected = self._expected_segments(det_name, alg)
        return expected is not None and sorted(captured) == expected

    def detector_status(self, det_name, alg="raw"):
        """Classify why :meth:`raw`/:meth:`stack` returns data or ``None`` for
        ``(det_name, alg)`` this event -- the STR-02 discriminator.  One of:

          * ``"present"`` -- the detector contributed its full segment set (a
            non-``None`` frame).
          * ``"dropped"`` -- the detector is ``None`` because at least one of the
            streams that carry it had **dropped out** of the run by this event
            (a DAQ/transport dropout; see :attr:`dropped_streams`).  A
            *data-quality* fault.
          * ``"absent"`` -- the detector is ``None`` for a legitimate reason: its
            streams are all still live but it did not contribute a complete
            frame this event (missing-segment rule / simply not present).
          * ``"unknown"`` -- no such detector in the run config.

        This is what a healthy ``None`` (``"absent"``) and a drop-out ``None``
        (``"dropped"``) differ by: on the raw arrays alone they are
        indistinguishable, which is the bug STR-02 fixes."""
        det = self._run_config.detectors.get(det_name)
        if det is None:
            return "unknown"
        if self._alg_complete(det_name, alg):
            return "present"
        streams = set(det.streams_for(alg)) if alg in det.segments else set()
        if streams & self._dropped_streams:
            return "dropped"
        return "absent"

    def dropped_detectors(self, alg="raw", include_bookkeeping=False):
        """Sorted names of the detectors that are ``None`` this event **because a
        stream carrying them dropped out** (``detector_status == "dropped"``) --
        the run's data-quality casualties for this event, as opposed to detectors
        that are simply, legitimately absent.  Empty for a healthy event."""
        out = []
        for name, det in self._run_config.detectors.items():
            if not include_bookkeeping and det.is_bookkeeping:
                continue
            if self.detector_status(name, alg=alg) == "dropped":
                out.append(name)
        return sorted(out)

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

    def damage(self, det_name, alg="raw"):
        """Return ``{segment: (damage_id, userbits)}`` for ``(det_name, alg)``
        this event, decoded from each segment's ShapesData ``Xtc.damage``.

        ``damage_id`` is the low-12-bit damage code (0 == undamaged; nonzero is
        a :class:`DamageBitmask` value) and ``userbits`` the high-4-bit user
        field -- the same split psana applies (``DAMAGE_USERBITSHIFT=12``,
        ``detector/damage.py``).  psana does **not** drop damaged data, and
        neither does ``psdata``: the raw array is still returned by
        :meth:`raw`/:meth:`stack`; damage is surfaced here, not silently
        dropped.  ``None`` if the detector/alg did not contribute this event.

        Note the per-segment damage is read from the ShapesData (alg-level) Xtc,
        matching psana's ``segment._xtc.damage`` (it exposes damage only at the
        alg level -- ``detector/damage.py`` docstring).
        """
        key = (det_name, alg)
        captured = self._seg_index.get(key)
        if not captured:
            return None
        out = {}
        for seg, (_buf, _off, hdr, _table) in captured.items():
            out[seg] = _f.decode_damage(hdr["damage"])
        return out

    def is_damaged(self, det_name, alg="raw"):
        """True if any contributing segment of ``(det_name, alg)`` has a nonzero
        damage id this event.  ``False`` if undamaged; ``None`` if the
        detector/alg is absent this event."""
        dmg = self.damage(det_name, alg=alg)
        if dmg is None:
            return None
        return any(did != 0 for (did, _ub) in dmg.values())

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

    # STR-02 drop-out tracking.  ``ever_live`` = streams that have produced at
    # least one head; ``dropped`` = streams that were live but have since hit EOF
    # while the run keeps yielding events from other streams.  ``dropped`` grows
    # monotonically.  On the canonical event set of a clean run every stream
    # reaches the run's end together, so all heads go None on the same final
    # iteration (which yields nothing) and ``dropped`` stays empty.  The test is
    # purely structural, though: on a ragged DAQ-shutdown tail some streams write
    # extra trailing L1Accepts (mfx100848724/r51), so an earlier-stopping stream
    # is -- unavoidably -- flagged dropped on those tail events; that only shows
    # on the ungated path, since the gated ``Run.events()`` default filters those
    # events out.  Either way the frame VALUES are byte-unchanged: ``dropped``
    # only feeds the new side-channel, never a returned array.
    ever_live = set()
    dropped = set()
    try:
        while True:
            # --- find min head ts across live streams (k-way merge key) ---
            min_ts = None
            for cur in cursors:
                ts = cur.head_ts()
                if ts is None:
                    # A stream at EOF that WAS producing dgrams has dropped out
                    # (STR-02): it stopped before the run's end.  Recorded so the
                    # assembled event can distinguish this data-quality fault from
                    # a detector that is legitimately absent.  A stream that was
                    # never live (empty file) is not a drop-out.
                    if cur.stream in ever_live:
                        dropped.add(cur.stream)
                    continue
                ever_live.add(cur.stream)
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
                yield Event(min_ts, service, rc, seg_index,
                            dropped_streams=frozenset(dropped))
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
