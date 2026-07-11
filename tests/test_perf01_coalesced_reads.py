#!/usr/bin/env python3
"""PERF-01 -- the batch read must REALLY coalesce its ``pread`` syscalls.

Bug (from the project bug matrix):

    PERF-01 -- the "coalesced batch read" issues exactly as many syscalls as the
    serial path -- no merge, no ``readv``, no ``preadv``.  The pre-fix
    ``RunIndex.read_events`` sorts the per-(event, stream) reads by ascending
    offset and reuses fds, but then issues ONE ``os.pread`` per (event, stream):
    K events over S streams cost K x S syscalls, the SAME total as reading each
    event serially.  The headline "coalesced batch read" does not coalesce.

The fix merges, within each bigdata chunk file, the reads whose byte ranges are
contiguous / near-adjacent into a SINGLE ``os.pread`` over the covering span,
then slices that buffer back to each dgram's exact bytes.  So K CONTIGUOUS events
(consecutive events in a stream are contiguous on disk) over S streams cost far
FEWER than K x S syscalls, while every returned dgram is byte-for-byte identical
to the per-``pread`` serial read.

This suite is the pre-fix/post-fix discriminator and is **fully self-contained**:
stdlib + numpy only, NO psana, NO SLAC data.  It builds the index's data
structures directly (as ``tests/test_mem01_bounded_read.py`` and
``tests/test_index_format_idx02.py`` do) over a REAL temp file holding several
contiguous fixed-size "dgrams" of random bytes, stubs only the per-dgram
*assembly* step so no valid xtc2 structure is needed (the stub records the raw
bytes handed to it), and spies on ``os.pread`` / ``os.preadv`` to COUNT syscalls
and compare bytes.  It contains NO part of the fix -- it drives only the public
``read_events`` / ``read_event_at`` and never mentions the merge internals.

Discriminator (parent SHA d5ca72e6977c4560fcd7ea8a6db3f597fa647a9a):

  * PARENT ``read_events`` issues one ``os.pread`` per (event, stream) -> the
    batch total is exactly K x S, the same as the serial path.  Assertion (a)
    "strictly fewer than K x S" FAILS -> the test exits nonzero.
  * FIX merges the contiguous per-stream reads -> the batch issues far fewer
    syscalls than K x S while returning byte-identical dgrams -> all pass.

Run (numpy only; no psana needed):
    python3 tests/test_perf01_coalesced_reads.py
"""

import os
import sys
import tempfile

import numpy as np  # noqa: F401  (the index constructs numpy dtypes on its paths)

# --- locate the package (parent of this tests dir), robust to cwd -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")   # .../<repo>/src (holds psdata/)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import psdata.index as IX
import psdata.stream as ST


# ==========================================================================
# Synthetic reader -- a REAL temp file per stream, NO xtc2 structure needed.
# ==========================================================================
class _StubRunConfig:
    """Enough of a RunConfig for RunIndex construction + Event(...) (which only
    stores its args).  No detectors, so no assembly/pulseId work is triggered."""
    stream_files = {}
    raw_tables = {}


def _write_stream_file(path, k_count, size, gap, rng):
    """Write ``k_count`` dgrams of ``size`` random bytes each, separated by
    ``gap`` bytes of DISTINCT filler.  Returns ``{k: dgram_bytes}`` and the byte
    offset of each dgram.  The filler differs from every dgram, so a coalesced
    read that mistakenly returned gap bytes for a dgram would be caught."""
    stride = size + gap
    dgrams = {}
    offs = {}
    blob = bytearray()
    for k in range(k_count):
        payload = rng.bytes(size)
        dgrams[k] = bytes(payload)
        offs[k] = k * stride
        blob += payload
        if gap:
            blob += b"\xee" * gap        # filler that appears in NO dgram
    with open(path, "wb") as fh:
        fh.write(blob)
    return dgrams, offs


def _build_reader(tmpdir, n_streams, k_count, size, gap, capture):
    """Build a :class:`RunIndex` whose ``entries`` point into one real file per
    stream (each with ``k_count`` fixed-size dgrams laid out ``size``+``gap``
    apart).  ``_assemble_stream_dgram`` is stubbed to RECORD the exact bytes it
    is handed (keyed by ``(k, stream)``) into ``capture`` and return a service,
    so the read paths run byte-observably with no valid xtc2 payload.

    Returns ``(ridx, ground)`` where ``ground[(k, stream)]`` is the true dgram
    bytes on disk (the definition of what a per-dgram ``pread`` must return)."""
    rng = np.random.default_rng(1234)
    ground = {}
    per_stream_off = {}
    for s in range(n_streams):
        path = os.path.join(tmpdir, "fake-s%03d-c000.xtc2" % s)
        dgrams, offs = _write_stream_file(path, k_count, size, gap, rng)
        per_stream_off[s] = (path, offs)
        for k in range(k_count):
            ground[(k, s)] = dgrams[k]

    ridx = IX.RunIndex(_StubRunConfig())
    ridx.timestamps = list(range(k_count))
    ridx.entries = [
        {s: (per_stream_off[s][0], per_stream_off[s][1][k], size)
         for s in range(n_streams)}
        for k in range(k_count)
    ]

    def _capturing_assemble(stream, chunk_path, offset, size_, raw, ts, seg_index,
                            stream_damage=None):
        # snapshot the bytes actually handed to assembly for this (k, stream)
        # (stream_damage is the WRITE-02 side-channel arg; the byte-capture
        # oracle ignores it -- this stub only records the raw dgram bytes)
        k = int(ts)                      # timestamps are 0..k_count-1 here
        assert len(raw) == size_, (
            "assembly got %d bytes, indexed size is %d" % (len(raw), size_))
        capture[(k, stream)] = bytes(raw)
        return ST.SERVICE_L1ACCEPT

    ridx._assemble_stream_dgram = _capturing_assemble
    return ridx, ground


# ==========================================================================
# os.pread / os.preadv syscall spy
# ==========================================================================
class _SyscallSpy:
    """Count and delegate ``os.pread`` / ``os.preadv`` while installed."""

    def __init__(self):
        self.pread_calls = 0
        self.preadv_calls = 0
        self.bytes_read = 0
        self._real_pread = os.pread
        self._real_preadv = getattr(os, "preadv", None)

    def total(self):
        return self.pread_calls + self.preadv_calls

    def __enter__(self):
        def spy_pread(fd, n, off):
            self.pread_calls += 1
            buf = self._real_pread(fd, n, off)
            self.bytes_read += len(buf)
            return buf
        os.pread = spy_pread
        if self._real_preadv is not None:
            def spy_preadv(fd, buffers, off, *a, **kw):
                self.preadv_calls += 1
                n = self._real_preadv(fd, buffers, off, *a, **kw)
                self.bytes_read += n
                return n
            os.preadv = spy_preadv
        return self

    def __exit__(self, *exc):
        os.pread = self._real_pread
        if self._real_preadv is not None:
            os.preadv = self._real_preadv


# ==========================================================================
# 1. THE discriminator: a contiguous batch coalesces (fewer syscalls) AND
#    stays byte-identical to the per-dgram serial read.
# ==========================================================================
def test_batch_coalesces_and_is_byte_identical():
    S = 5                       # streams (contributing per event)
    K = 8                       # events
    SIZE = 1024                 # bytes per dgram
    serial_expected = K * S     # the pre-PERF-01 syscall total

    with tempfile.TemporaryDirectory() as tmp:
        cap_batch = {}
        ridx, ground = _build_reader(tmp, S, K, SIZE, gap=0, capture=cap_batch)

        # --- batch read: count the syscalls it issues -------------------
        ks = list(range(K))
        with _SyscallSpy() as spy_b:
            events = ridx.read_events(ks)
        assert len(events) == K, "read_events returned %d for %d ks" % (
            len(events), K)
        batch_syscalls = spy_b.total()

        # --- serial oracle: one pread per (event, stream) = K x S -------
        cap_serial = {}
        ridx._assemble_stream_dgram = _rebind_capture(ridx, cap_serial)
        with _SyscallSpy() as spy_s:
            for k in ks:
                ridx.read_event_at(k)
        serial_syscalls = spy_s.total()

    # The serial path is the K x S reference (unchanged by the fix): this anchors
    # what "K x S" means for THIS reader, so assertion (a) is not a bare literal.
    assert serial_syscalls == serial_expected, (
        "serial read issued %d syscalls, expected K*S=%d -- the per-dgram "
        "reference is off" % (serial_syscalls, serial_expected))

    # (a) REAL coalescing: the batch issues STRICTLY FEWER syscalls than the
    #     serial K x S.  The pre-PERF-01 batch issues exactly K x S -> FAILS here.
    assert batch_syscalls < serial_expected, (
        "PERF-01 not fixed: the batch issued %d syscalls for %d contiguous "
        "events x %d streams -- the SAME as the serial K*S=%d path (no merge, "
        "no coalescing)." % (batch_syscalls, K, S, serial_expected))

    # (b) byte-exactness: every dgram the batch handed to assembly is identical
    #     to the serial (per-dgram read_event_at) bytes AND to the true on-disk
    #     bytes.  Fewer syscalls must not change a single byte.
    assert set(cap_batch) == set(ground) == set(cap_serial), \
        "captured (k, stream) coverage differs between batch/serial/ground"
    for key in ground:
        assert cap_batch[key] == ground[key], (
            "coalesced read returned WRONG bytes for %r (byte-exactness "
            "violated)" % (key,))
        assert cap_serial[key] == ground[key], \
            "serial read returned wrong bytes for %r" % (key,)
        assert cap_batch[key] == cap_serial[key], \
            "batch bytes != serial bytes for %r" % (key,)

    print("[ok] contiguous batch: %d events x %d streams coalesced %d serial "
          "syscalls -> %d (%.1fx fewer), byte-identical to per-dgram reads"
          % (K, S, serial_expected, batch_syscalls,
             serial_expected / max(batch_syscalls, 1)))


def _rebind_capture(ridx, capture):
    """A fresh capturing assembler bound to ``capture`` (reused for the serial
    oracle so it observes the same per-dgram bytes the batch did)."""
    def _cap(stream, chunk_path, offset, size_, raw, ts, seg_index,
             stream_damage=None):
        capture[(int(ts), stream)] = bytes(raw)
        return ST.SERVICE_L1ACCEPT
    return _cap


# ==========================================================================
# 2. near-adjacent (small transition-like gap) reads still merge, and the
#    discarded gap bytes NEVER leak into a returned dgram.
# ==========================================================================
def test_small_gap_merges_and_excludes_gap_bytes():
    S = 2
    K = 6
    SIZE = 2048
    GAP = 4096                  # < RunIndex._COALESCE_MAX_GAP (bridged, discarded)
    serial_expected = K * S
    assert GAP <= IX.RunIndex._COALESCE_MAX_GAP, "gap must be bridgeable"

    with tempfile.TemporaryDirectory() as tmp:
        cap = {}
        ridx, ground = _build_reader(tmp, S, K, SIZE, gap=GAP, capture=cap)
        with _SyscallSpy() as spy:
            ridx.read_events(list(range(K)))
        merged_syscalls = spy.total()

    # a small gap between consecutive dgrams is bridged -> still fewer than K x S
    assert merged_syscalls < serial_expected, (
        "small-gap batch did not coalesce: %d syscalls == K*S=%d"
        % (merged_syscalls, serial_expected))
    # and the bridged (0xEE) filler must NOT appear in any returned dgram
    for key, dg in ground.items():
        assert cap[key] == dg, "gap bytes leaked into dgram %r" % (key,)
        assert b"\xee" * GAP not in cap[key], \
            "filler leaked into returned dgram %r" % (key,)
    print("[ok] near-adjacent reads (gap=%d B) merge to %d syscalls (< K*S=%d); "
          "bridged gap bytes excluded from every returned dgram"
          % (GAP, merged_syscalls, serial_expected))


# ==========================================================================
# 3. a WIDE gap (a sparse random batch) is NOT over-merged: reads separated by
#    more than the gap threshold stay separate, so a sparse batch never reads a
#    big hole.  (Guards against a naive "merge everything in the file" fix.)
# ==========================================================================
def test_wide_gap_is_not_over_merged():
    S = 1
    SIZE = 512
    # two dgrams separated by a hole far larger than the gap threshold.
    hole = IX.RunIndex._COALESCE_MAX_GAP * 4
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "fake-s000-c000.xtc2")
        rng = np.random.default_rng(7)
        d0, d1 = rng.bytes(SIZE), rng.bytes(SIZE)
        with open(path, "wb") as fh:
            fh.write(d0)
            fh.write(b"\x00" * hole)
            fh.write(d1)
        off1 = SIZE + hole

        cap = {}
        ridx = IX.RunIndex(_StubRunConfig())
        ridx.timestamps = [0, 1]
        ridx.entries = [{0: (path, 0, SIZE)}, {0: (path, off1, SIZE)}]
        ridx._assemble_stream_dgram = _rebind_capture(ridx, cap)

        with _SyscallSpy() as spy:
            ridx.read_events([0, 1])
        far_syscalls = spy.total()

    # two far-apart dgrams must NOT be merged across the big hole (that would
    # read ~4x the gap threshold of useless bytes) -> still 2 separate reads.
    assert far_syscalls == 2, (
        "a wide %d B hole was bridged (%d syscalls) -- a sparse batch must not "
        "read big holes" % (hole, far_syscalls))
    assert cap[(0, 0)] == d0 and cap[(1, 0)] == d1, "wide-gap bytes wrong"
    print("[ok] wide gap (%d B) NOT over-merged: 2 far dgrams stay 2 reads, "
          "byte-exact" % hole)


# ==========================================================================
# 4. coalescing survives round-trips through read_stack / iter_events (the
#    public batch surfaces all route through read_events).
# ==========================================================================
def test_coalescing_applies_through_iter_events():
    S = 3
    K = 9
    SIZE = 700
    with tempfile.TemporaryDirectory() as tmp:
        cap = {}
        ridx, ground = _build_reader(tmp, S, K, SIZE, gap=0, capture=cap)
        with _SyscallSpy() as spy:
            batches = list(ridx.iter_events(range(K), batch_size=K))
        total = sum(len(b) for b in batches)
        syscalls = spy.total()
    assert total == K, "iter_events streamed %d events, expected %d" % (total, K)
    # one batch of all K contiguous events -> coalesced well under K x S
    assert syscalls < K * S, (
        "iter_events (one full batch) did not coalesce: %d syscalls == K*S=%d"
        % (syscalls, K * S))
    for key, dg in ground.items():
        assert cap[key] == dg, "iter_events byte mismatch at %r" % (key,)
    print("[ok] iter_events routes through the coalesced read_events: %d "
          "syscalls for K*S=%d, byte-identical" % (syscalls, K * S))


def main():
    print("=" * 72)
    print("PERF-01: batch read must REALLY coalesce its pread syscalls")
    print("=" * 72)
    test_batch_coalesces_and_is_byte_identical()
    test_small_gap_merges_and_excludes_gap_bytes()
    test_wide_gap_is_not_over_merged()
    test_coalescing_applies_through_iter_events()
    print("\nALL PERF-01 TESTS PASSED")


if __name__ == "__main__":
    main()
