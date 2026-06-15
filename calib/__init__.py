"""psdata.calib -- one-time calibration-constant snapshot + pinning (US-006).

This is a **separate layer** from the :mod:`psdata` raw reader.  The reader
(``psdata.format`` / ``stream`` / ``index`` / ``run``) is detector-universal and
imports only numpy; calibration is detector-specific and -- for the one-time
snapshot step only -- needs a working ``psana`` to reach the calib DB.  Keeping
it here means importing ``psdata`` never drags in psana, while the calibrated
HDR-image render (US-007) can build on the pinned snapshot offline.

Two halves, deliberately split by their dependencies:

  * :func:`snapshot_calib` -- the **only** psana-using entry point.  It opens a
    run, pulls ``det.raw._calibconst`` (a dict ``{ctype: (ndarray|str, meta)}``)
    and ``det.raw._mask(status=True)``, and writes a self-describing on-disk
    snapshot pinned by ``(detector_uniqueid, run)``.  Run it once, in the
    ``psconda.sh`` psana env, on a host that can reach the calib DB.

  * :class:`CalibSnapshot` / :func:`load_snapshot` -- **pure numpy, no psana**.
    Reloads a snapshot's arrays and validity metadata for fully-offline use.
    ``snapshot.calibconst`` reconstructs psana's ``{ctype: (array, meta)}`` dict
    byte-for-byte, so a reload reproduces the exact arrays psana returned.

Pinning / staleness
-------------------
A snapshot records the run it was taken for (the *pin*) and, per ctype, the
validity metadata psana attached (``run`` = first run the constant is valid
for, ``run_end``, ``version``).  Constants are keyed by ``(detector_uniqueid,
run)`` with validity *ranges*; reusing a snapshot outside its range gives
**wrong-but-silent** results.  This package retains the metadata so a caller can
check, but does not itself refuse a stale reload.

Importing :mod:`psdata.calib` (and reloading a snapshot) pulls in only the
standard library + numpy; see :func:`assert_no_framework_imports`.
"""

from .snapshot import (
    CalibSnapshot,
    load_snapshot,
    snapshot_calib,
    assert_no_framework_imports,
    SNAPSHOT_CTYPES,
    MANIFEST_NAME,
)

__all__ = [
    "snapshot_calib",
    "load_snapshot",
    "CalibSnapshot",
    "assert_no_framework_imports",
    "SNAPSHOT_CTYPES",
    "MANIFEST_NAME",
]
