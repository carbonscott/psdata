#!/usr/bin/env python3
"""DET-10 regression: psdata must SIGNAL an unvalidated detector raw version.

Bug (DET-10).  ``psdata.format.parse_names_block`` parses each Names table's
algorithm name and its packed ``major.minor.micro`` version, then throws the
version away.  psana instead dispatches its detector data on the
``(det_type, alg, version)`` triple -- its ``DetectorImpl`` class is literally
``f"{det_type}_{alg}_{major}_{minor}_{micro}"`` (e.g. ``epix10ka_raw_2_0_1``) --
and raises ``KeyError`` on a triple it has no class for.  The deployed release
ships newer raw classes an old class table never had (``epix10ka_raw_3_0_1``,
``epixuhr3x2_raw_0_1_0``), so a reader pinned to an old table hits that path on
real, current data.  psdata's raw decode is generic and self-describing, so it
cannot *mis-decode* a bumped, additive version -- but on the PARENT commit it has
**no "I don't know this version" signal at all**: it parses an unknown raw
version silently.

The contract this probe pins:

  * An unvalidated ``(det_type, 'raw', version)`` (e.g. ``9.9.9``) SIGNALS --
    it emits an ``UnvalidatedVersionWarning`` naming the detector and version
    (or, in strict mode, raises).
  * A validated version (``jungfrau`` raw ``0.1.0``) parses with NO signal and
    byte-identical output -- no false alarm, decode unchanged.
  * The signal is suppressible once a new version is acknowledged
    (``PSDATA_ACK_VERSIONS``), so it is not perpetual noise.

Discriminator.  On the PARENT commit ``parse_names_block`` never signals, so the
unknown version is parsed with zero warnings -- ``len(recorded) == 0`` -- and the
"must signal" assertion below fails, so this probe exits non-zero.  On the fixed
commit it warns (and strict mode raises), so the probe passes.

Self-contained: standard library + numpy only.  No psana, no SLAC data, no real
xtc2 file -- the Names payload is built by hand, in bytes, to the exact layout
``parse_names_block`` reads (a minimal valid one-field table), with the alg
version set to a known / unknown triple.  ``main()`` + ``__main__``; cwd-robust.
"""

import logging
import os
import struct
import sys
import warnings

import numpy as np  # noqa: F401  (psdata's only declared dependency; asserts env)

# --- locate the package under test (parent-of-tests/src), cwd-robust ---------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../<repo>/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import psdata.format as fmt  # noqa: E402  (after sys.path shim)


# ---------------------------------------------------------------------------
# Hand-build one Names Xtc payload, byte-for-byte to the layout parse_names_block
# reads (see src/psdata/format.py).  Offsets within the payload:
#     0     uint32   num_arrays
#     4     char[256] det_type
#     260   char[256] det_name
#     516   char[256] det_id
#     772   char[256] alg_name
#     1028  uint32   alg_version      (packed major.minor.micro)
#     1032  uint32   segment
#     1036  Name[]   entries, each NAME_SZ=524 bytes:
#              +260  char[256] field name
#              +516  uint32   type code
#              +520  uint32   rank
# ---------------------------------------------------------------------------
_NAMEINFO_SZ = 1036
_NAME_SZ = 524


def _pack_version(major, minor, micro):
    """psana's Alg packing: (major<<16) | (minor<<8) | micro."""
    return (major << 16) | (minor << 8) | micro


def _put_str(buf, off, s):
    b = s.encode("latin-1")[:255]           # leave room for the NUL terminator
    buf[off:off + len(b)] = b                # remaining bytes stay 0x00


def build_names_payload(det_type, det_name, det_id, alg_name, version,
                        segment, fields, num_arrays):
    """Return the bytes of one Names Xtc payload.  ``version`` is a
    ``(major, minor, micro)`` triple; ``fields`` is a list of
    ``(name, type_code, rank)``."""
    n = len(fields)
    buf = bytearray(_NAMEINFO_SZ + n * _NAME_SZ)
    struct.pack_into("<I", buf, 0, num_arrays)
    _put_str(buf, 4, det_type)
    _put_str(buf, 260, det_name)
    _put_str(buf, 516, det_id)
    _put_str(buf, 772, alg_name)
    struct.pack_into("<I", buf, 1028, _pack_version(*version))
    struct.pack_into("<I", buf, 1032, segment)
    for k, (fname, ftype, frank) in enumerate(fields):
        no = _NAMEINFO_SZ + k * _NAME_SZ
        _put_str(buf, no + 260, fname)
        struct.pack_into("<I", buf, no + 516, ftype)
        struct.pack_into("<I", buf, no + 520, frank)
    return buf


def _parse(buf):
    return fmt.parse_names_block(buf, 0, len(buf))


def _reset_signal_dedup():
    """Clear the fix's once-per-process dedup so a fresh signal can be observed.
    Guarded: the attribute does not exist on the PARENT (no fix), where this is a
    no-op and the probe still fails on the 'must signal' assertion below."""
    dedup = getattr(fmt, "_signaled_versions", None)
    if dedup is not None:
        dedup.clear()


def _unval_warnings(recorded):
    """The DET-10 warnings among ``recorded`` -- matched by category NAME so the
    probe never imports a symbol the PARENT lacks."""
    return [w for w in recorded
            if w.category.__name__ == "UnvalidatedVersionWarning"]


class _ListHandler(logging.Handler):
    """Capture ``logging`` records so the durable (filter-proof) DET-10 channel
    can be asserted."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


# The expected decode of the one-field table below -- what parse_names_block must
# return regardless of whether the version is known or unknown (the signal must
# never alter the decoded output).
def _expected_table(version, det_type="jungfrau", det_name="jungfrau",
                    det_id="serialX", segment=0):
    return dict(det_type=det_type, det_name=det_name, det_id=det_id,
                alg_name="raw", alg_version=_pack_version(*version),
                segment=segment, num_arrays=0,
                names=[dict(name="raw", type=1, rank=0)])  # rank-0 uint16 field


# ===========================================================================
# 1. THE DISCRIMINATOR: an unknown raw version SIGNALS (parent is silent -> fail)
# ===========================================================================
def check_unknown_version_signals():
    version = (9, 9, 9)               # a version no reader was validated against
    buf = build_names_payload("jungfrau", "jungfrau", "serialX", "raw",
                              version, 0, [("raw", 1, 0)], num_arrays=0)

    _reset_signal_dedup()
    os.environ.pop("PSDATA_STRICT_VERSIONS", None)
    os.environ.pop("PSDATA_ACK_VERSIONS", None)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        table = _parse(buf)

    unval = _unval_warnings(recorded)
    # Core "must signal" assertion.  On the PARENT no warning is emitted at all,
    # so len(recorded) == 0 and this fails -> the probe exits non-zero.
    if len(unval) < 1:
        raise AssertionError(
            "DET-10: parse_names_block parsed raw version 9.9.9 with NO signal "
            f"({len(recorded)} warning(s) recorded, {len(unval)} of the DET-10 "
            "category). An unvalidated detector raw version must be signalled "
            "(warned), not accepted silently.")

    msg = str(unval[0].message)
    assert "9.9.9" in msg, \
        f"the signal must name the unvalidated version 9.9.9; got: {msg!r}"
    assert "jungfrau" in msg, \
        f"the signal must name the detector; got: {msg!r}"

    # The signal must NOT change the decoded output: the table is byte-identical
    # to the reference decode of these bytes.
    assert table == _expected_table(version), \
        f"unknown-version decode changed: {table!r} != {_expected_table(version)!r}"

    print("OK: unknown raw version 9.9.9 SIGNALS "
          f"(UnvalidatedVersionWarning) and decode is unchanged; msg={msg!r}")


# ===========================================================================
# 2. NO FALSE ALARM: a validated version parses silently, byte-unchanged
# ===========================================================================
def check_known_version_no_signal_and_byte_unchanged():
    version = (0, 1, 0)               # jungfrau_raw_0_1_0 -- a validated class
    buf = build_names_payload("jungfrau", "jungfrau", "serialX", "raw",
                              version, 0, [("raw", 1, 0)], num_arrays=0)

    _reset_signal_dedup()
    os.environ.pop("PSDATA_STRICT_VERSIONS", None)
    os.environ.pop("PSDATA_ACK_VERSIONS", None)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        table = _parse(buf)

    unval = _unval_warnings(recorded)
    assert len(unval) == 0, \
        f"a validated raw version must NOT signal; got {len(unval)}: " \
        f"{[str(w.message) for w in unval]!r}"

    # Byte-exactness: the decoded table equals the reference decode exactly.
    assert table == _expected_table(version), \
        f"known-version decode changed: {table!r} != {_expected_table(version)!r}"
    print("OK: validated raw version 0.1.0 parses with NO signal, "
          "decode byte-unchanged")


# ===========================================================================
# 3. SUPPRESS: an acknowledged (accepted) new version stops signalling
# ===========================================================================
def check_ack_suppresses_signal():
    version = (9, 9, 9)
    buf = build_names_payload("jungfrau", "jungfrau", "serialX", "raw",
                              version, 0, [("raw", 1, 0)], num_arrays=0)

    _reset_signal_dedup()
    os.environ.pop("PSDATA_STRICT_VERSIONS", None)
    os.environ["PSDATA_ACK_VERSIONS"] = "jungfrau:9.9.9"
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            _parse(buf)
    finally:
        os.environ.pop("PSDATA_ACK_VERSIONS", None)

    unval = _unval_warnings(recorded)
    assert len(unval) == 0, \
        "an acknowledged version (PSDATA_ACK_VERSIONS) must not signal; got " \
        f"{len(unval)}: {[str(w.message) for w in unval]!r}"
    print("OK: PSDATA_ACK_VERSIONS suppresses the signal for an accepted version")


# ===========================================================================
# 4. STRICT: opt-in fail-closed mode raises instead of warning (2nd discriminator)
# ===========================================================================
def check_strict_mode_raises():
    version = (9, 9, 9)
    buf = build_names_payload("jungfrau", "jungfrau", "serialX", "raw",
                              version, 0, [("raw", 1, 0)], num_arrays=0)

    _reset_signal_dedup()
    os.environ.pop("PSDATA_ACK_VERSIONS", None)
    os.environ["PSDATA_STRICT_VERSIONS"] = "1"
    raised = None
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _parse(buf)
    except Exception as exc:          # noqa: BLE001 -- probe records anything
        raised = exc
    finally:
        os.environ.pop("PSDATA_STRICT_VERSIONS", None)

    # On the PARENT nothing is raised (no strict mode), so raised is None and
    # this fails -- the probe exits non-zero.
    assert raised is not None, \
        "PSDATA_STRICT_VERSIONS=1 must make an unvalidated raw version RAISE; " \
        "parse_names_block returned normally"
    assert type(raised).__name__ == "UnvalidatedVersionError", \
        f"strict mode must raise UnvalidatedVersionError; got {type(raised).__name__}"
    assert "9.9.9" in str(raised), \
        f"the strict error must name the version; got: {str(raised)!r}"
    print(f"OK: PSDATA_STRICT_VERSIONS=1 raises {type(raised).__name__} naming 9.9.9")


# ===========================================================================
# 5. DURABILITY (NIT 1): even when the warnings channel is fully suppressed, the
#    signal must survive via logging -- else the once-per-process dedup would
#    make the unvalidated version permanently silent.
# ===========================================================================
def check_suppressed_warnings_still_logs():
    version = (9, 9, 9)
    buf = build_names_payload("jungfrau", "jungfrau", "serialX", "raw",
                              version, 0, [("raw", 1, 0)], num_arrays=0)

    _reset_signal_dedup()
    os.environ.pop("PSDATA_STRICT_VERSIONS", None)
    os.environ.pop("PSDATA_ACK_VERSIONS", None)

    logger = logging.getLogger("psdata")
    handler = _ListHandler()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        # A suppressing context: warnings are IGNORED (as `-W ignore` /
        # PYTHONWARNINGS=ignore / an outer catch_warnings would do), so the
        # warnings channel records nothing.
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("ignore")
            _parse(buf)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    # The warnings channel is genuinely silenced here...
    assert len(_unval_warnings(recorded)) == 0, \
        "precondition: warnings must be suppressed in this context"

    # ...but the DURABLE logging channel must still carry the signal (WARNING,
    # on the 'psdata' logger, naming the unvalidated version).  On the PARENT
    # parse_names_block emits nothing to logging, so this fails -> exit non-zero.
    logged = [r for r in handler.records
              if r.levelno >= logging.WARNING and "9.9.9" in r.getMessage()]
    if not logged:
        raise AssertionError(
            "DET-10 durability: with Python warnings suppressed, an unvalidated "
            "raw version emitted NO logging record -- the dedup would then keep "
            "it permanently silent. It must also log (WARNING) so a warnings-"
            "silencing pipeline still sees the signal.")
    assert "jungfrau" in logged[0].getMessage(), \
        f"the logged signal must name the detector; got: {logged[0].getMessage()!r}"
    print("OK: warnings suppressed -> signal still emitted on the 'psdata' logger "
          "(durable channel survives warnings filters)")


# ===========================================================================
# 6. ACK NORMALISATION (NIT 2): a short 'major.minor' acknowledgement is padded
#    to the canonical triple and still matches.
# ===========================================================================
def check_ack_short_version_normalizes():
    # The env token uses the two-component form 'jungfrau:9.9', which must
    # normalise to (9, 9, 0)...
    v_env = (9, 9, 0)
    buf_env = build_names_payload("jungfrau", "jungfrau", "serialX", "raw",
                                  v_env, 0, [("raw", 1, 0)], num_arrays=0)
    _reset_signal_dedup()
    os.environ.pop("PSDATA_STRICT_VERSIONS", None)
    os.environ["PSDATA_ACK_VERSIONS"] = "jungfrau:9.9"
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            _parse(buf_env)
    finally:
        os.environ.pop("PSDATA_ACK_VERSIONS", None)
    assert len(_unval_warnings(recorded)) == 0, \
        "env token 'jungfrau:9.9' must normalise to (9,9,0) and suppress it; " \
        f"got {[str(w.message) for w in recorded]!r}"

    # ...and the programmatic acknowledge_version accepts a short string too,
    # padding '2.5' -> (2,5,0).  Guarded: acknowledge_version does not exist on
    # the PARENT, so skip that leg there (the durability/signal checks above are
    # the discriminators).
    ack = getattr(fmt, "acknowledge_version", None)
    if ack is not None:
        v_api = (2, 5, 0)
        buf_api = build_names_payload("epix10ka", "epixquad", "serialY", "raw",
                                      v_api, 0, [("raw", 1, 0)], num_arrays=0)
        _reset_signal_dedup()
        ack("epix10ka", "2.5")            # short form, padded to (2,5,0)
        try:
            with warnings.catch_warnings(record=True) as recorded2:
                warnings.simplefilter("always")
                _parse(buf_api)
        finally:
            fmt._acknowledged_versions.discard(("epix10ka", (2, 5, 0)))
        assert len(_unval_warnings(recorded2)) == 0, \
            "acknowledge_version('epix10ka', '2.5') must pad to (2,5,0) and " \
            f"suppress; got {[str(w.message) for w in recorded2]!r}"
    print("OK: short 'major.minor' acknowledgements normalise to the canonical "
          "triple (env + programmatic)")


def main():
    check_unknown_version_signals()
    check_known_version_no_signal_and_byte_unchanged()
    check_ack_suppresses_signal()
    check_strict_mode_raises()
    check_suppressed_warnings_still_logs()
    check_ack_short_version_normalizes()
    print("\nALL DET-10 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
