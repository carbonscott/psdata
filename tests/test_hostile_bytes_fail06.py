#!/usr/bin/env python3
"""FAIL-06 regression: hostile xtc2 bytes must be rejected, always.

Self-contained: standard library + numpy only.  It imports **no** psana and
needs **no** SLAC data -- it hand-builds minimal xtc2 dgrams in bytes, corrupts
one field, and drives ``psdata.format`` directly.  It therefore runs anywhere,
and it is the pre-fix / post-fix discriminator for FAIL-06.

FAIL-06 names two distinct defects in ``psdata/format.py``:

  1. **`assert` as the validation layer.**  The parser validated untrusted
     on-disk bytes with bare ``assert`` statements.  ``python -O`` /
     ``PYTHONOPTIMIZE`` strips every ``assert``, so an optimized run -- which a
     performance-minded scientist WILL do, and which the demo-attack list names
     explicitly -- left the reader with no validation at all: it happily parsed
     a front dgram that is not a Configure.

  2. **Unbounded ``np.frombuffer``.**  A shape/count taken from the file drove
     an array view over the buffer with no bound against the *declared extent*
     of that dgram's Data payload.  A corrupt length field therefore let the
     returned array read past its own dgram into the following bytes -- silently
     returning data that belongs to a different event.

The two discriminating checks below:

  * **The `-O` hole** (the money assertion): re-execute the validation path in a
    subprocess launched with ``python3 -O`` feeding it a non-Configure front
    dgram, and require the parser to still reject it.  On the parent commit
    ``-O`` strips the ``assert`` and the malformed input is ACCEPTED -> FAIL.

  * **The over-read**: hand a ShapesData whose declared shape implies 16 bytes
    while its Data child declares only 4, with a recognizable sentinel pattern
    in the following bytes, and require the parser to raise.  On the parent the
    unbounded ``np.frombuffer`` returns an array that literally CONTAINS the
    sentinel -- the over-read is demonstrated concretely, then the leg FAILs.

This file contains no part of the fix and imports none of the fix's symbols
(so it also imports cleanly against the parent, where ``XtcFormatError`` does
not exist); it discriminates purely on observed parser behaviour.
"""

import os
import struct
import subprocess
import sys

import numpy as np

# -- robust import of the psdata under test (cwd-independent) ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(os.path.join(_SRC, "psdata")) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from psdata import format as psformat  # noqa: E402


# ==========================================================================
# Minimal, valid xtc2 byte builders (then we corrupt exactly one field)
# ==========================================================================
# Dgram on disk = Transition(12B) + a top-level Xtc.
#   Transition = "<III" (nsec, sec, env);   service = (env >> 24) & 0xf
#   Xtc header = "<IHHI" (src, damage, typeid, extent);  extent INCLUDES the
#   12-byte Xtc header, and the top Xtc's payload begins at byte 24 (DGRAM_HDR).
SENTINEL = 0xEEEEEEEE            # a recognizable uint32 for the over-read proof


def _xtc(src, typeid, payload, damage=0):
    """One Xtc = 12B header + payload.  ``extent`` counts the header + payload."""
    extent = psformat.XTC_HDR + len(payload)
    return struct.pack("<IHHI", src, damage, typeid, extent) + payload


def _dgram(service, top_xtc, nsec=0, sec=0):
    """One dgram = Transition(12B) + a top-level Xtc.  ``service`` is encoded in
    bits 24-27 of ``env`` exactly as ``parse_dgram_header`` decodes it."""
    env = (service & 0xf) << 24
    return struct.pack("<III", nsec, sec, env) + top_xtc


def _non_configure_dgram():
    """A structurally-valid 24-byte dgram whose ONLY defect is that its service
    is L1Accept, not Configure -- exactly what the front-dgram check must catch."""
    top = _xtc(src=0, typeid=psformat.TID_PARENT, payload=b"")  # no children
    return _dgram(service=psformat.SERVICE_L1ACCEPT, top_xtc=top)


def _overread_shapesdata():
    """Build a ShapesData whose declared shape overruns its Data child.

    Layout (offsets within the returned buffer)::

        [ 0,12)  ShapesData Xtc header (extent = 60)
        [12,44)  Shapes child  (extent 32): dims = (4,0,0,0,0)  -> count 4
        [44,60)  Data   child  (extent 16): payload = ONE uint32 (4 bytes)
        [60,76)  SENTINEL x4   (the "following bytes" -- next Xtc / next dgram)

    A rank-1 uint32 field therefore claims 4*4 = 16 bytes, but its Data child
    declares only 4.  An unbounded view reads 12 bytes past the Data payload,
    straight into the sentinel.
    """
    real_value = 0x11111111
    shapes_payload = struct.pack("<5I", 4, 0, 0, 0, 0)      # shape (4,) -> count 4
    shapes_xtc = _xtc(src=0, typeid=psformat.TID_SHAPES, payload=shapes_payload)
    data_payload = struct.pack("<I", real_value)            # only 1 element present
    data_xtc = _xtc(src=0, typeid=psformat.TID_DATA, payload=data_payload)
    sd_payload = shapes_xtc + data_xtc
    sd_xtc = _xtc(src=0, typeid=psformat.TID_SHAPESDATA, payload=sd_payload)
    trailer = struct.pack("<4I", SENTINEL, SENTINEL, SENTINEL, SENTINEL)
    buf = bytearray(sd_xtc + trailer)

    _src, _dmg, _tid, extent = struct.unpack_from("<IHHI", buf, 0)
    sd_hdr = dict(src=_src, damage=_dmg, typeid=_tid, extent=extent)
    table = {"names": [dict(name="x", type=2, rank=1)]}     # type 2 == uint32
    return buf, sd_hdr, table, real_value


# ==========================================================================
# Leg 1 -- the `-O` hole (the money assertion)
# ==========================================================================
# Child program: build the non-Configure dgram and run the Configure validation
# path.  Prints ACCEPTED if parse_configure returns, REJECTED:<type> if it
# raises.  Under `python -O`, a bare `assert` is stripped -- so on the parent
# this prints ACCEPTED; a real always-on check prints REJECTED.
_CHILD = r'''
import sys, struct
import psdata.format as f
top = struct.pack("<IHHI", 0, 0, f.TID_PARENT, f.XTC_HDR)   # Parent, no children
env = (f.SERVICE_L1ACCEPT & 0xf) << 24                       # service = L1Accept
buf = bytearray(struct.pack("<III", 0, 0, env) + top)        # 24-byte non-Configure
try:
    f.parse_configure(buf)
except BaseException as e:
    sys.stdout.write("REJECTED:" + type(e).__name__)
else:
    sys.stdout.write("ACCEPTED")
'''


def test_dash_O_still_rejects_non_configure():
    """Under `python -O`, the parser must STILL reject a non-Configure front
    dgram.  On the parent the `assert` is stripped and the input is accepted."""
    pkg_parent = os.path.dirname(
        os.path.dirname(os.path.abspath(psformat.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_parent + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-O", "-c", _CHILD],
        capture_output=True, text=True, env=env, cwd=_HERE)
    out = (proc.stdout or "").strip()
    assert out.startswith("REJECTED"), (
        "FAIL-06 (-O hole): under `python -O` the non-Configure front dgram was "
        f"NOT rejected -- parse_configure returned {out!r}. A bare `assert` was "
        "stripped by -O, so validation of untrusted bytes vanished. "
        f"[child stdout={proc.stdout!r} stderr={proc.stderr!r}]")


# ==========================================================================
# Leg 2 -- the over-read (unbounded np.frombuffer)
# ==========================================================================
def test_overread_is_bounded_not_returned():
    """A field whose declared shape overruns its Data child's declared extent
    must raise, NOT return an array that has read into the following bytes."""
    buf, sd_hdr, table, real_value = _overread_shapesdata()

    raised = None
    result = None
    try:
        result = psformat.extract_field(buf, 0, sd_hdr, table, "x")
    except Exception as exc:          # noqa: BLE001 -- any real rejection is fine
        raised = exc

    if raised is None:
        # Parent behaviour: the view over-read.  Prove it concretely -- the
        # returned array literally contains bytes from beyond the Data payload.
        arr = np.asarray(result[0]).reshape(-1)
        assert SENTINEL in set(int(v) for v in arr), (
            "expected the over-read to pull sentinel bytes into the array, but "
            f"got {arr!r}; the harness may be mis-built")
        raise AssertionError(
            "FAIL-06 (over-read): extract_field returned an array that read PAST "
            f"its Data child's 4-byte extent into the following bytes -- "
            f"array={list(int(v) for v in arr)!r} contains the sentinel "
            f"0x{SENTINEL:08X}. A hostile shape/count was not bounded by the "
            "declared payload extent; the reader silently returned another "
            "event's bytes.")

    # Fixed behaviour: rejected before constructing the view.  Make sure it was
    # rejected for the RIGHT reason (a bounds/format error), not because the
    # field was somehow not found.
    assert not isinstance(raised, KeyError), (
        f"extract_field raised KeyError ({raised!r}); expected a format/bounds "
        "rejection of the over-long shape, not a missing-field error")
    # And confirm the well-formed value was never handed back.
    assert result is None


# ==========================================================================
# Leg 3 -- sanity: even in-process, the front-dgram check must reject
# (non-discriminating: parent raises AssertionError here, the fix raises its
#  own error -- both reject; the discriminator for this defect is the -O leg.)
# ==========================================================================
def test_non_configure_rejected_in_process():
    buf = bytearray(_non_configure_dgram())
    raised = None
    try:
        psformat.parse_configure(buf)
    except Exception as exc:          # noqa: BLE001
        raised = exc
    assert raised is not None, (
        "parse_configure accepted a non-Configure front dgram in-process")


# ==========================================================================
def main():
    legs = [
        ("dash_O_still_rejects_non_configure", test_dash_O_still_rejects_non_configure),
        ("overread_is_bounded_not_returned", test_overread_is_bounded_not_returned),
        ("non_configure_rejected_in_process", test_non_configure_rejected_in_process),
    ]
    print("=" * 72)
    print("FAIL-06 regression: hostile xtc2 bytes must always be rejected")
    print("=" * 72)
    failures = []
    for name, fn in legs:
        try:
            fn()
            print(f"[ok] {name}")
        except Exception as exc:      # noqa: BLE001
            failures.append((name, exc))
            print(f"[FAIL] {name}: {exc}")
    if failures:
        print(f"\n{len(failures)} of {len(legs)} FAIL-06 checks FAILED")
        # Re-raise the first so the process exits nonzero with a traceback.
        raise failures[0][1]
    print("\nALL FAIL-06 CHECKS PASSED")


if __name__ == "__main__":
    main()
