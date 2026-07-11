#!/usr/bin/env python3
"""STR-04 regression: ``_enumerate_bd_chunks`` must NOT silently truncate a
non-contiguous bigdata chunk set.

The DAQ writes per-stream bigdata chunk files ``...-s0NN-c000.xtc2``,
``-c001.xtc2``, ``-c002.xtc2``, ...  The bug (STR-04): the old enumeration
walked the chunk ids by INCREMENTING from 0 until the first missing file, then
stopped.  If ``c001`` is absent but ``c002`` exists -- a non-contiguous set from
a partial copy, a filesystem hiccup, or a stray deletion -- it stopped at the
gap and silently indexed ONLY ``c000``, dropping every event in ``c002+`` with
no error.  The index was silently short and the caller never knew.

The fix ENUMERATES the ``-c00N`` siblings that actually exist and verifies
contiguity (``0..N`` with no hole).  A true interior hole is a HARD ERROR (fail
closed, naming the stream, the missing chunk(s), and the higher chunk(s) that
exist).  Reaching the last chunk is the legitimate end condition, not a hole, so
a contiguous set enumerates in order exactly as before.

This probe is SELF-CONTAINED: stdlib only (no numpy of its own, no psana, no
SLAC data).  It fabricates empty chunk files -- the enumeration keys off the
FILENAME convention, not file contents, so empty files exercise it fully.  It is
cwd-robust (imports psdata from the sibling ``src/`` regardless of where it is
run from) and has a ``main()`` / ``__main__`` entry point.

Parent vs fix (the probe's discriminating power):
  * On the PARENT, the gap case (``c000`` present, ``c001`` missing, ``c002``
    present) silently returns ``[c000]`` (no raise) -> the "must raise"
    assertion below fails -> the test exits NONZERO.
  * On the FIX, the same gap case RAISES a clear error -> the test PASSES.
  * The contiguous case (``c000``, ``c001``, ``c002``) must enumerate all three
    in order on BOTH parent and fix (no false alarm; the normal case is
    byte-unchanged).
"""

import os
import sys
import tempfile

# -- cwd-robust import of the psdata package from the sibling src/ tree --------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from psdata import index as psindex  # noqa: E402

_enumerate_bd_chunks = psindex._enumerate_bd_chunks

# A realistic per-stream naming: <exp>-r<run>-s<stream>-c<chunk>.xtc2
_EXP = "mfx101343025"
_RUN = 35
_STREAM = 7


def _chunk_name(chunk_id, stream=_STREAM):
    return f"{_EXP}-r{_RUN:04d}-s{stream:03d}-c{chunk_id:03d}.xtc2"


def _touch_chunks(d, chunk_ids, stream=_STREAM):
    """Create empty fake chunk files for ``chunk_ids`` in dir ``d``; return the
    absolute ``c000`` path (the entry point ``_enumerate_bd_chunks`` takes)."""
    for cid in chunk_ids:
        with open(os.path.join(d, _chunk_name(cid, stream)), "wb"):
            pass
    return os.path.join(d, _chunk_name(0, stream))


# ==========================================================================
# 1. Contiguous set (the normal case) -> enumerate all chunks IN ORDER
# ==========================================================================
def test_contiguous_set_enumerates_all_in_order():
    with tempfile.TemporaryDirectory() as d:
        c000 = _touch_chunks(d, [0, 1, 2])
        got = _enumerate_bd_chunks(c000)
        expected = [os.path.join(d, _chunk_name(cid)) for cid in (0, 1, 2)]
        assert got == expected, (
            f"contiguous chunk set must enumerate all three in order; "
            f"expected {expected!r}, got {got!r}")
    print("[ok] contiguous c000,c001,c002 -> all three, in order (no false alarm)")


# ==========================================================================
# 2. Single-chunk set (normal case) -> exactly [c000], unchanged
# ==========================================================================
def test_single_chunk_unchanged():
    with tempfile.TemporaryDirectory() as d:
        c000 = _touch_chunks(d, [0])
        got = _enumerate_bd_chunks(c000)
        assert got == [c000], (
            f"a single-chunk stream must return exactly its c000; got {got!r}")
    print("[ok] single-chunk c000 -> [c000] (unchanged)")


# ==========================================================================
# 3. THE STR-04 probe: interior gap -> MUST RAISE (not silent truncation)
# ==========================================================================
def test_interior_gap_raises():
    """c000 present, c001 MISSING, c002 present.  The parent silently returns
    [c000] (dropping c002); the fix must RAISE a clear error."""
    with tempfile.TemporaryDirectory() as d:
        c000 = _touch_chunks(d, [0, 2])   # gap at c001
        try:
            result = _enumerate_bd_chunks(c000)
        except ValueError as e:
            msg = str(e)
            # The error must be LOUD and specific: name the stream, the missing
            # chunk, and the higher chunk that exists.
            assert "s007" in msg, (
                f"error should name the stream (s007); got: {msg!r}")
            assert "c001" in msg, (
                f"error should name the missing chunk (c001); got: {msg!r}")
            assert "c002" in msg, (
                f"error should name the higher chunk that exists (c002); "
                f"got: {msg!r}")
            print("[ok] interior gap (c001 missing, c002 present) RAISES: "
                  f"{msg}")
        else:
            raise AssertionError(
                f"STR-04: _enumerate_bd_chunks did NOT raise on a "
                f"non-contiguous chunk set (c000 present, c001 MISSING, c002 "
                f"present); it silently returned {result!r}, dropping every "
                f"event in c002+.  A gap must fail closed, not truncate "
                f"silently.")


# ==========================================================================
# 4. Gap at c000 with higher chunks present -> also a hole -> MUST RAISE
# ==========================================================================
def test_missing_c000_with_higher_raises():
    """c000 MISSING but c001/c002 present is likewise a non-contiguous set.
    Passing the (absent) c000 path must fail closed, not index a short set."""
    with tempfile.TemporaryDirectory() as d:
        _touch_chunks(d, [1, 2])          # note: no c000 on disk
        c000 = os.path.join(d, _chunk_name(0))  # the (absent) c000 path
        try:
            result = _enumerate_bd_chunks(c000)
        except ValueError as e:
            msg = str(e)
            assert "c000" in msg, (
                f"error should name the missing chunk (c000); got: {msg!r}")
            print(f"[ok] missing c000 with c001/c002 present RAISES: {msg}")
        else:
            raise AssertionError(
                f"non-contiguous set (c000 missing, c001/c002 present) must "
                f"raise; got {result!r}")


def main():
    print("=" * 72)
    print("STR-04: non-contiguous bigdata chunk set must not silently truncate")
    print("=" * 72)
    test_contiguous_set_enumerates_all_in_order()
    test_single_chunk_unchanged()
    test_interior_gap_raises()
    test_missing_c000_with_higher_raises()
    print()
    print("ALL STR-04 CHECKS PASSED")


if __name__ == "__main__":
    main()
