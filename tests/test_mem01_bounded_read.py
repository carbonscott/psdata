#!/usr/bin/env python3
"""MEM-01 -- the batch-read API must offer a BOUNDED / streaming read.

Bug (from the project bug matrix):

    MEM-01 -- ``read_events`` / ``read_stack`` materialize K x frame -- 335 GB
    at K=10000 jungfrau.  The OOM actually happened: reading a whole worker
    slice at once produced a 10 GB OOM, and ``read_batch=32`` was hand-rolled in
    the cube example to work around it.  The API is unchanged and still says
    nothing.

``RunIndex.read_events(ks)`` builds ONE ``list`` holding all K events, and
``RunIndex.read_stack(ks, det)`` allocates ONE ``(len(ks), *frame)`` buffer.  At
K=10000 jungfrau (32,512,1024) uint16 that buffer is ~335 GB: there is no bound,
no chunking, no warning -- a caller asking for a large slice OOMs and the API
gives no hint.

The fix adds a bounded, streaming read whose peak memory is O(batch_size) --
``iter_events`` (yields ``list[Event]`` batches) and ``iter_stack`` (yields
``(<=batch, *frame)`` ndarrays) -- and makes the whole-slice ``read_events`` /
``read_stack`` REFUSE (raise ``MemoryError`` naming the streaming API) a request
whose materialized size exceeds a memory budget, instead of silently attempting
the allocation.

This suite is the pre-fix/post-fix discriminator and is **fully self-contained**:
stdlib + numpy only, NO psana, NO SLAC data.  It builds the index's data
structures directly (as tests/test_index_format_idx02.py does) with a large
event count but a tiny per-event ``size``, so ``K x frame`` is "big" (drives the
guard) while the test itself stays small.  The per-event reader is monkeypatched
so no xtc2 file is touched -- the streaming assertions observe how many frames
are ever handed to the underlying reader at once (== peak live frames).  It
contains NO part of the fix.

Discriminator (parent SHA 34a5c89b48cdf15355ea8dcdea992dbe699f611d):

  * PARENT has no ``iter_events`` / ``iter_stack`` (``AttributeError``), and its
    ``read_events`` / ``read_stack`` have no guard -- an oversized request is NOT
    refused (no ``MemoryError``).  So the streaming tests and the refusal tests
    FAIL on the parent.
  * FIX adds both -> all pass.

Run (numpy only; no psana needed):
    python3 tests/test_mem01_bounded_read.py
"""

import os
import sys
import tempfile

import numpy as np  # noqa: F401  (index constructs numpy dtypes on the read path)

# --- locate the package (parent of this tests dir), robust to cwd -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")   # .../<repo>/src (holds psdata/)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import psdata.index as IX

# The default whole-slice memory budget the fix ships (kept as a literal here so
# the setup does not depend on the fix's own constant -- the parent must fail for
# the RIGHT reason, "no refusal", not "missing attribute").
LIMIT_BYTES = 2 * 1024 ** 3   # 2 GiB


# ==========================================================================
# Synthetic reader -- built directly, NO xtc2 file is read.
# ==========================================================================
def _synthetic_reader(n_events, per_event=64, path=None):
    """A :class:`RunIndex` with ``n_events`` positions whose per-event indexed
    ``size`` is ``per_event`` bytes -- enough state for ``read_events`` /
    ``read_stack`` / ``iter_events`` / ``iter_stack`` to run, with NO xtc2 file
    behind it.  ``path`` defaults to a guaranteed-absent file, so a whole-slice
    read that is NOT refused up front fails when it reaches disk (never a
    MemoryError) -- which is exactly the parent's behaviour."""
    class _StubRunConfig:
        stream_files = {}

    if path is None:
        path = os.path.join(
            tempfile.gettempdir(),
            "psdata-mem01-absent-%d.xtc2" % os.getpid())

    ridx = IX.RunIndex(_StubRunConfig())
    ridx.timestamps = list(range(n_events))
    # one contributing stream per event; the indexed 'size' (3rd tuple field) is
    # what the memory estimate sums, so it sets K x frame.
    ridx.entries = [
        {0: (path, 4096 + k * per_event, per_event)}
        for k in range(n_events)
    ]
    return ridx


# ==========================================================================
# 1. iter_events streams in fixed-size batches (peak memory O(batch)).
# ==========================================================================
def test_iter_events_streams_in_bounded_batches():
    K = 10000
    B = 32
    ridx = _synthetic_reader(K)

    # The bounded streaming API must EXIST -- the parent has none, so
    # read_events is the only path and it materializes all K frames at once.
    assert hasattr(ridx, "iter_events"), (
        "RunIndex has no iter_events -- MEM-01's bounded streaming read is "
        "absent (parent). read_events builds one list of all K events "
        "(335 GB at K=10000 jungfrau), with no way to read in bounded memory.")

    # Instrument the per-batch reader: no disk, and record how many positions
    # are handed to it at once (== peak live frames for that batch).
    call_sizes = []

    def fake_read_events(ks, max_bytes=None):
        ks = list(ks)
        call_sizes.append(len(ks))
        return [np.zeros(4, dtype=np.uint8) for _ in ks]   # one tiny 'frame' each

    ridx.read_events = fake_read_events

    gen = ridx.iter_events(range(K), batch_size=B)
    assert hasattr(gen, "__next__"), \
        "iter_events must return a lazy iterator/generator, not a materialized list"
    assert call_sizes == [], \
        "iter_events must be lazy -- nothing may be read before iteration starts"

    first = next(gen)
    assert len(call_sizes) == 1, (
        "after pulling ONE batch exactly ONE underlying read must have run "
        "(lazy, O(batch) peak) -- saw %d" % len(call_sizes))
    assert isinstance(first, list) and len(first) <= B, \
        "first batch is not a <= batch_size list of events: %r" % (type(first),)

    batches = [first] + list(gen)
    total = sum(len(b) for b in batches)
    assert total == K, "streamed %d events, expected %d" % (total, K)
    assert max(len(b) for b in batches) <= B, \
        "a yielded batch exceeded batch_size (%d)" % B
    assert max(call_sizes) <= B, (
        "iter_events handed %d positions to a single read (> %d): peak memory "
        "is not O(batch_size)" % (max(call_sizes), B))
    # decisive: the WHOLE K-slice was never materialized in one call.
    assert max(call_sizes) < K and K not in call_sizes, \
        "iter_events materialized the whole slice at once (not streamed)"
    print("[ok] iter_events streams %d events in <=%d-event batches "
          "(peak <=%d frames, %d underlying reads)"
          % (K, B, max(call_sizes), len(call_sizes)))


# ==========================================================================
# 2. iter_stack streams in fixed-size batches (never one (K, *frame) buffer).
# ==========================================================================
def test_iter_stack_streams_in_bounded_batches():
    K = 10000
    B = 32
    seg_shape = (2, 3)
    ridx = _synthetic_reader(K)

    assert hasattr(ridx, "iter_stack"), (
        "RunIndex has no iter_stack -- MEM-01's bounded streaming stack read is "
        "absent (parent). read_stack allocates one (K, *frame) buffer "
        "(335 GB at K=10000 jungfrau).")

    call_sizes = []

    def fake_read_stack(ks, det, field="raw", alg="raw", max_bytes=None):
        ks = list(ks)
        call_sizes.append(len(ks))
        return np.zeros((len(ks),) + seg_shape, dtype=np.uint16)

    ridx.read_stack = fake_read_stack

    gen = ridx.iter_stack(range(K), "jungfrau", batch_size=B)
    assert hasattr(gen, "__next__"), \
        "iter_stack must return a lazy iterator/generator"
    assert call_sizes == [], "iter_stack must be lazy"

    first = next(gen)
    assert len(call_sizes) == 1, \
        "after one batch exactly one read must have run -- saw %d" % len(call_sizes)
    assert isinstance(first, np.ndarray) and first.shape[0] <= B, \
        "first batch is not a <= batch_size ndarray: %r" % (getattr(first, "shape", first),)
    assert first.shape[1:] == seg_shape, "per-segment shape not preserved"

    batches = [first] + list(gen)
    rows = sum(b.shape[0] for b in batches)
    assert rows == K, "streamed %d rows, expected %d" % (rows, K)
    assert max(b.shape[0] for b in batches) <= B, \
        "a yielded stack batch exceeded batch_size (%d)" % B
    assert max(call_sizes) <= B and K not in call_sizes, (
        "iter_stack built a single (K, *frame) array (peak not O(batch)): "
        "max rows read at once = %d" % max(call_sizes))
    print("[ok] iter_stack streams %d rows in <=%d-row (b, *frame) arrays "
          "(peak <=%d rows, %d reads)" % (K, B, max(call_sizes), len(call_sizes)))


# ==========================================================================
# 3. Whole-slice read_events REFUSES an oversized request (never silent OOM).
# ==========================================================================
def test_read_events_refuses_oversized_request():
    K = 20000
    per_event = 200_000                        # 20000 * 200000 = ~3.7 GiB > 2 GiB
    ridx = _synthetic_reader(K, per_event=per_event)
    est = K * per_event
    assert est > LIMIT_BYTES, \
        "test setup error: request (%d B) must exceed the limit" % est

    refused = None
    other = None
    try:
        ridx.read_events(list(range(K)))       # default guard active
    except MemoryError as e:
        refused = e
    except Exception as e:                      # parent: absent file, etc.
        other = e

    assert refused is not None, (
        "read_events did NOT refuse an oversized request (MEM-01): it silently "
        "proceeds to materialize ~%.1f GiB in one list (335 GB at K=10000 "
        "jungfrau)%s"
        % (est / (1024 ** 3),
           "." if other is None else " -- it raised %r instead of refusing."
           % (other,)))
    msg = str(refused).lower()
    assert "iter_events" in msg or "stream" in msg, \
        "the refusal must point the caller at the streaming API: %r" % (refused,)
    print("[ok] read_events refuses a ~%.1f GiB request with MemoryError "
          "-> iter_events" % (est / (1024 ** 3)))


# ==========================================================================
# 4. Whole-slice read_stack REFUSES an oversized request (points at iter_stack).
# ==========================================================================
def test_read_stack_refuses_oversized_request():
    K = 20000
    per_event = 200_000
    ridx = _synthetic_reader(K, per_event=per_event)
    est = K * per_event
    assert est > LIMIT_BYTES

    refused = None
    other = None
    try:
        ridx.read_stack(list(range(K)), "jungfrau")
    except MemoryError as e:
        refused = e
    except Exception as e:
        other = e

    assert refused is not None, (
        "read_stack did NOT refuse an oversized request (MEM-01): it silently "
        "allocates a ~%.1f GiB (K, *frame) buffer%s"
        % (est / (1024 ** 3),
           "." if other is None else " -- it raised %r instead." % (other,)))
    msg = str(refused).lower()
    assert "iter_stack" in msg or "stream" in msg, \
        "the refusal must point the caller at iter_stack: %r" % (refused,)
    print("[ok] read_stack refuses a ~%.1f GiB request with MemoryError "
          "-> iter_stack" % (est / (1024 ** 3)))


# ==========================================================================
# 4b. read_stack sizes its guard by len(ks) ROWS, not distinct positions -- a
#     single position repeated K times still allocates a (K, *frame) buffer, so
#     it must be refused even though only ONE distinct event is read.
# ==========================================================================
def test_read_stack_duplicate_positions_refused():
    n_rep = 100_000
    per_event = 50_000                          # one frame: 50 KB << 2 GiB
    ridx = _synthetic_reader(1, per_event=per_event)   # only position 0 exists
    ks = [0] * n_rep
    distinct_est = per_event                    # what a distinct-only estimate sees
    buffer_bytes = n_rep * per_event            # the real (K, *frame) buffer
    assert distinct_est < LIMIT_BYTES < buffer_bytes, (
        "test setup: distinct estimate must slip under the limit while the true "
        "buffer blows past it")

    refused = None
    other = None
    try:
        ridx.read_stack(ks, "jungfrau")
    except MemoryError as e:
        refused = e
    except Exception as e:
        other = e

    assert refused is not None, (
        "read_stack did NOT refuse %d DUPLICATE positions (MEM-01): the "
        "(len(ks), *frame) output buffer is ~%.1f GiB, but a distinct-only "
        "estimate (~%d B, one frame) slipped past the guard -- the buffer has "
        "one row per position, so the budget must size by len(ks)%s"
        % (n_rep, buffer_bytes / (1024 ** 3), distinct_est,
           "." if other is None else " -- it raised %r instead." % (other,)))
    msg = str(refused).lower()
    assert "iter_stack" in msg or "stream" in msg, \
        "the refusal must point the caller at iter_stack: %r" % (refused,)
    print("[ok] read_stack refuses %d duplicate positions (~%.1f GiB buffer, "
          "1 distinct event) -> iter_stack" % (n_rep, buffer_bytes / (1024 ** 3)))


# ==========================================================================
# 5. The guard is SIZE-GATED, not a blanket refusal (existing small-K callers
#    must be byte-unchanged: a small in-limit request is never refused).
# ==========================================================================
def test_small_request_not_refused():
    ridx = _synthetic_reader(50, per_event=1000)   # 50 * 1000 = 50 KB << 2 GiB
    refused_mem = False
    try:
        ridx.read_events([0, 1, 2, 3, 4])
    except MemoryError:
        refused_mem = True
    except Exception:
        pass    # expected on both parent and fix: reaches the absent file
    assert not refused_mem, (
        "read_events refused a SMALL in-limit request -- the MEM-01 guard must "
        "be size-gated, not a blanket refusal; reasonable-K reads stay "
        "byte-unchanged.")
    print("[ok] a small in-limit request passes the guard (size-gated, not blanket)")


def main():
    test_iter_events_streams_in_bounded_batches()
    test_iter_stack_streams_in_bounded_batches()
    test_read_events_refuses_oversized_request()
    test_read_stack_refuses_oversized_request()
    test_read_stack_duplicate_positions_refused()
    test_small_request_not_refused()
    print("\nALL MEM-01 TESTS PASSED")


if __name__ == "__main__":
    main()
