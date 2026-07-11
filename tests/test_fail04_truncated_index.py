#!/usr/bin/env python3
"""FAIL-04 regression: a TRUNCATED xtc file must not yield a silently short
random-access index -- the index-build scan must fail closed, like streaming.

The bug (FAIL-04): when an xtc2 stream file is cut off mid-stream (a partial
transfer, an aborted DAQ), the two read paths of ``psdata`` disagree.  The
forward STREAMING path raises ``RuntimeError`` the moment it meets the
truncation (``stream.py`` ``StreamCursor.peek`` -- "truncated dgram at file
offset ..."), but the INDEX build silently stopped at the last COMPLETE dgram
and returned a short index with NO diagnostic.  A user reading by random access
then just... doesn't get the missing events, and never learns the file was
short.  The reader contradicts itself and the index path loses data silently.

The fix makes both scan loops that walk dgrams to EOF --
:func:`psdata.index._scan_bigdata_stream` (the bigdata-header walk) and
:func:`psdata.index._scan_smd_stream` (the SMD-sidecar walk) -- FAIL CLOSED on a
truncated file: a header declaring an extent past EOF, or trailing bytes too
few for even a 24-byte dgram header, raises ``RuntimeError`` naming the file,
the byte offset of the truncation, and the last complete event index (mirroring
the streaming path).  A CLEAN end (the last dgram finishes exactly at the file
boundary) is the normal case and must NOT raise.  An explicit opt-in
``tolerate_truncation=True`` restores the old behavior deliberately: index the
intact prefix and stop.

This probe is SELF-CONTAINED: stdlib + numpy only (NO psana, NO SLAC data, no
network).  It fabricates minimal but VALID xtc2 bytes -- a Configure with a real
``smdinfo`` Names table plus L1Accept dgrams for the SMD path, and bare L1Accept
dgram headers for the bigdata path -- then cuts them mid-dgram.  The dgram /
Xtc / Names byte layout is taken from ``psdata.format`` (read-only).  It is
cwd-robust and has a ``main()`` / ``__main__`` entry point.

Parent vs fix (the probe's discriminating power):
  * On the PARENT, a truncated file scans WITHOUT raising and returns a short
    record list (silent data loss) -> the "must raise" assertions below fail
    -> the test exits NONZERO.
  * On the FIX, the same truncated files RAISE a clear, offset-naming error
    -> the test PASSES.
  * The CLEANLY-ended files must scan WITHOUT error on BOTH parent and fix,
    returning every fabricated event (no false alarm; the normal case is
    byte-unchanged).
"""

import os
import struct
import sys
import tempfile

# -- cwd-robust import of the psdata package from the sibling src/ tree --------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from psdata import format as _f      # noqa: E402  (byte-layout constants only)
from psdata import index as _ix      # noqa: E402

# ==========================================================================
# Minimal xtc2 byte fabrication (layout per psdata.format, read-only)
# ==========================================================================
_XTC_HDR = _f.XTC_HDR                 # 12: Src+Damage+TypeId+extent
_DGRAM_HDR = _f.DGRAM_HDR             # 24: Transition(12) + Xtc(12)
_UINT64 = 3                           # DataType.UINT64 (rank-0 smdinfo fields)

# namesid_of(src) = ((src>>8)&0xfff, src&0xff); src=1 -> (nodeId=0, namesId=1).
_SMDINFO_SRC = 1


def _xtc(src, typeid, payload, damage=0):
    """One Xtc: 12-byte header (extent INCLUDES the header) + payload."""
    extent = _XTC_HDR + len(payload)
    return struct.pack("<IHHI", src, damage, typeid, extent) + payload


def _dgram(service, ts, top_payload, top_typeid=_f.TID_PARENT, src=0):
    """One on-disk dgram: Transition(12) + top Xtc(12) header + top_payload.
    ``service`` rides in the top 4 bits of Transition.env, matching
    ``parse_dgram_header`` (``service = (env >> 24) & 0xf``)."""
    nsec = ts & 0xffffffff
    sec = (ts >> 32) & 0xffffffff
    env = (service & 0xf) << 24
    extent = _XTC_HDR + len(top_payload)
    transition = struct.pack("<III", nsec, sec, env)
    xtc_hdr = struct.pack("<IHHI", src, 0, top_typeid, extent)
    return transition + xtc_hdr + top_payload


def _cstr(s, n=256):
    b = s.encode("latin-1")[:n]
    return b + b"\x00" * (n - len(b))


def _names_payload(det_type, det_name, alg_name, segment, fields):
    """One Names table payload: a NAMEINFO block (1036B) then one Name (524B)
    per field, exactly as ``parse_names_block`` reads it."""
    p = struct.pack("<I", len(fields))                    # num_arrays
    p += _cstr(det_type) + _cstr(det_name) + _cstr("detid-serial")
    p += _cstr(alg_name) + struct.pack("<I", 1) + struct.pack("<I", segment)
    assert len(p) == _f.NAMEINFO_SZ, (len(p), _f.NAMEINFO_SZ)
    for fname, ftype, frank in fields:
        nm = b"\x00" * _f.ALG_SZ                          # field's own Alg
        nm += _cstr(fname) + struct.pack("<I", ftype) + struct.pack("<I", frank)
        assert len(nm) == _f.NAME_SZ, (len(nm), _f.NAME_SZ)
        p += nm
    return p


def _smd_configure(ts=1):
    """A Configure dgram declaring the ``smdinfo``/``offsetAlg`` Names table
    (fields ``intOffset``, ``intDgramSize``) the SMD scan requires."""
    npay = _names_payload("offset", "smdinfo", "offsetAlg", 0,
                          [("intOffset", _UINT64, 0),
                           ("intDgramSize", _UINT64, 0)])
    return _dgram(_f.SERVICE_CONFIGURE, ts, _xtc(_SMDINFO_SRC, _f.TID_NAMES, npay))


def _smd_l1accept(ts, intoff, intsize):
    """An L1Accept carrying an ``smdinfo`` ShapesData whose Data child holds the
    two rank-0 uint64 fields (``intOffset``, ``intDgramSize``)."""
    data_xtc = _xtc(0, _f.TID_DATA, struct.pack("<QQ", intoff, intsize))
    sd_xtc = _xtc(_SMDINFO_SRC, _f.TID_SHAPESDATA, data_xtc)
    return _dgram(_f.SERVICE_L1ACCEPT, ts, sd_xtc)


def _bd_l1accept(ts, payload_len=48):
    """A bare L1Accept dgram for the bigdata scan (which reads only the 24-byte
    header, so the payload bytes are irrelevant -- any complete dgram works)."""
    return _dgram(_f.SERVICE_L1ACCEPT, ts, b"\x00" * payload_len)


def _write(path, data):
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _assert_truncation_error(msg, path):
    """The truncation error must be LOUD and actionable: name the file, name a
    byte offset, and report the last complete event index."""
    assert "truncat" in msg.lower(), f"error should say 'truncated'; got: {msg!r}"
    assert os.path.basename(path) in msg, \
        f"error should name the file {os.path.basename(path)!r}; got: {msg!r}"
    assert "byte offset" in msg, \
        f"error should name the truncation byte offset; got: {msg!r}"
    assert "last complete event index" in msg, \
        f"error should name the last complete event index; got: {msg!r}"


# ==========================================================================
# 1. BIGDATA scan -- clean file scans all events (no false alarm)
# ==========================================================================
def test_bigdata_clean_scans_all():
    with tempfile.TemporaryDirectory() as d:
        # non-chunk name -> _enumerate_bd_chunks returns [path] unchanged.
        path = os.path.join(d, "bigdata_clean.bin")
        n = 5
        _write(path, b"".join(_bd_l1accept(1000 + i, 40 + i) for i in range(n)))
        recs, _nbytes, chunks = _ix._scan_bigdata_stream(path)
        assert chunks == [path]
        assert len(recs) == n, \
            f"clean bigdata file must yield all {n} events; got {len(recs)}"
        assert [r[0] for r in recs] == [1000 + i for i in range(n)]
    print(f"[ok] bigdata clean file (last dgram ends at EOF) -> all {n} "
          f"events, no false alarm")


# ==========================================================================
# 2. BIGDATA scan -- header declaring an extent past EOF -> MUST RAISE
# ==========================================================================
def test_bigdata_truncated_header_past_eof_raises():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bigdata_trunc_hdr.bin")
        good = b"".join(_bd_l1accept(2000 + i) for i in range(4))
        last = _bd_l1accept(2999, payload_len=400)   # full dgram = 24 + 400
        cut = last[:_DGRAM_HDR + 100]                # header intact, payload cut
        _write(path, good + cut)
        try:
            recs, _nb, _c = _ix._scan_bigdata_stream(path)
        except RuntimeError as e:
            _assert_truncation_error(str(e), path)
            print(f"[ok] bigdata header-past-EOF RAISES: {str(e)[:90]}...")
        else:
            raise AssertionError(
                "FAIL-04: _scan_bigdata_stream did NOT raise on a file whose "
                "final dgram header declares an extent past EOF; it silently "
                f"returned {len(recs)} records (dropping the truncated event "
                "and any following it). A truncated file must fail closed.")


# ==========================================================================
# 3. BIGDATA scan -- trailing bytes too short for a header -> MUST RAISE
# ==========================================================================
def test_bigdata_truncated_trailing_bytes_raises():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bigdata_trunc_tail.bin")
        good = b"".join(_bd_l1accept(3000 + i) for i in range(4))
        _write(path, good + b"\x01\x02\x03\x04\x05")   # 5 trailing bytes (<24)
        try:
            recs, _nb, _c = _ix._scan_bigdata_stream(path)
        except RuntimeError as e:
            _assert_truncation_error(str(e), path)
            print(f"[ok] bigdata trailing-bytes RAISES: {str(e)[:90]}...")
        else:
            raise AssertionError(
                "FAIL-04: _scan_bigdata_stream did NOT raise on a file ending "
                "in bytes too few for even a dgram header; it silently returned "
                f"{len(recs)} records. Trailing partial bytes are a truncation.")


# ==========================================================================
# 4. BIGDATA scan -- opt-in tolerance returns the intact prefix (no raise)
# ==========================================================================
def test_bigdata_tolerate_returns_intact_prefix():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bigdata_tolerate.bin")
        good = b"".join(_bd_l1accept(4000 + i) for i in range(4))
        last = _bd_l1accept(4999, payload_len=400)
        _write(path, good + last[:_DGRAM_HDR + 100])
        recs, _nb, _c = _ix._scan_bigdata_stream(path, tolerate_truncation=True)
        assert len(recs) == 4, \
            f"tolerate_truncation must index the 4 intact events; got {len(recs)}"
    print("[ok] bigdata tolerate_truncation=True -> intact prefix (4), no raise")


# ==========================================================================
# 5. SMD scan -- clean file scans all events (no false alarm)
# ==========================================================================
def test_smd_clean_scans_all():
    with tempfile.TemporaryDirectory() as d:
        bd_c000 = os.path.join(d, "exp-r0001-s000-c000.xtc2")   # referenced path
        path = os.path.join(d, "exp-r0001-s000-c000.smd.xtc2")
        n = 6
        body = b"".join(_smd_l1accept(5000 + i, 100 * i, 96) for i in range(n))
        _write(path, _smd_configure() + body)
        recs, _nbytes, _chunks = _ix._scan_smd_stream(path, bd_c000, 1 << 22)
        assert len(recs) == n, \
            f"clean SMD file must yield all {n} events; got {len(recs)}"
        assert [r[0] for r in recs] == [5000 + i for i in range(n)]
    print(f"[ok] SMD clean file (last dgram ends at EOF) -> all {n} events, "
          f"no false alarm")


# ==========================================================================
# 6. SMD scan -- header declaring an extent past EOF -> MUST RAISE
# ==========================================================================
def test_smd_truncated_header_past_eof_raises():
    with tempfile.TemporaryDirectory() as d:
        bd_c000 = os.path.join(d, "exp-r0002-s000-c000.xtc2")
        path = os.path.join(d, "exp-r0002-s000-c000.smd.xtc2")
        good = _smd_configure() + b"".join(
            _smd_l1accept(6000 + i, 100 * i, 96) for i in range(4))
        last = _smd_l1accept(6999, 500, 96)
        _write(path, good + last[:_DGRAM_HDR + 10])   # header intact, payload cut
        try:
            recs, _nb, _c = _ix._scan_smd_stream(path, bd_c000, 1 << 22)
        except RuntimeError as e:
            _assert_truncation_error(str(e), path)
            print(f"[ok] SMD header-past-EOF RAISES: {str(e)[:90]}...")
        else:
            raise AssertionError(
                "FAIL-04: _scan_smd_stream did NOT raise on an SMD file whose "
                "final dgram header declares an extent past EOF; it silently "
                f"returned {len(recs)} records. A truncated SMD sidecar must "
                "fail closed, mirroring the streaming path.")


# ==========================================================================
# 7. SMD scan -- trailing bytes too short for a header -> MUST RAISE
# ==========================================================================
def test_smd_truncated_trailing_bytes_raises():
    with tempfile.TemporaryDirectory() as d:
        bd_c000 = os.path.join(d, "exp-r0003-s000-c000.xtc2")
        path = os.path.join(d, "exp-r0003-s000-c000.smd.xtc2")
        good = _smd_configure() + b"".join(
            _smd_l1accept(7000 + i, 100 * i, 96) for i in range(4))
        _write(path, good + b"\x09" * 7)              # 7 trailing bytes (<24)
        try:
            recs, _nb, _c = _ix._scan_smd_stream(path, bd_c000, 1 << 22)
        except RuntimeError as e:
            _assert_truncation_error(str(e), path)
            print(f"[ok] SMD trailing-bytes RAISES: {str(e)[:90]}...")
        else:
            raise AssertionError(
                "FAIL-04: _scan_smd_stream did NOT raise on an SMD file ending "
                "in bytes too few for a dgram header; it silently returned "
                f"{len(recs)} records.")


# ==========================================================================
# 8. SMD scan -- opt-in tolerance returns the intact prefix (no raise)
# ==========================================================================
def test_smd_tolerate_returns_intact_prefix():
    with tempfile.TemporaryDirectory() as d:
        bd_c000 = os.path.join(d, "exp-r0004-s000-c000.xtc2")
        path = os.path.join(d, "exp-r0004-s000-c000.smd.xtc2")
        good = _smd_configure() + b"".join(
            _smd_l1accept(8000 + i, 100 * i, 96) for i in range(4))
        last = _smd_l1accept(8999, 500, 96)
        _write(path, good + last[:_DGRAM_HDR + 10])
        recs, _nb, _c = _ix._scan_smd_stream(path, bd_c000, 1 << 22,
                                             tolerate_truncation=True)
        assert len(recs) == 4, \
            f"tolerate_truncation must index the 4 intact events; got {len(recs)}"
    print("[ok] SMD tolerate_truncation=True -> intact prefix (4), no raise")


def main():
    print("=" * 72)
    print("FAIL-04: a truncated xtc file must not yield a silently short index")
    print("=" * 72)
    test_bigdata_clean_scans_all()
    test_bigdata_truncated_header_past_eof_raises()
    test_bigdata_truncated_trailing_bytes_raises()
    test_bigdata_tolerate_returns_intact_prefix()
    test_smd_clean_scans_all()
    test_smd_truncated_header_past_eof_raises()
    test_smd_truncated_trailing_bytes_raises()
    test_smd_tolerate_returns_intact_prefix()
    # the whole probe stays framework-pure (no psana / mpi4py / h5py leaked in).
    _ix.assert_no_framework_imports()
    print()
    print("ALL FAIL-04 CHECKS PASSED")


if __name__ == "__main__":
    main()
