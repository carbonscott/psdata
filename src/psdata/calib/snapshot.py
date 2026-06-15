#!/usr/bin/env python3
"""psdata.calib.snapshot -- RE-EXPORT SHIM of pscalib.providers.snapshot.

The calibration engine's canonical home moved to the standalone ``pscalib``
package (``pscalib.providers.snapshot``); psdata's calibration layer now
RE-EXPORTS it so there is exactly one implementation and no drift.  See
pscalib's README "psdata relationship" section and US-000.

psdata is retained as the framework-free *reader*; this module is kept so
``import psdata.calib`` and the existing US-006 API keep working, but the
``CalibSnapshot`` / ``load_snapshot`` / ``snapshot_calib`` objects it exposes
ARE the pscalib objects (proven by identity in pscalib's
``tests/test_no_drift_us000.py``).

pscalib is numpy-only at import time, so re-exporting it keeps psdata's
reload/snapshot path framework-free; psdata's own ``assert_no_framework_imports``
(its original ``('psana','mpi4py','h5py')`` contract) is kept here.
"""

# Canonical implementation lives in pscalib (numpy-only import).
from pscalib.providers.snapshot import (  # noqa: F401
    CalibSnapshot,
    load_snapshot,
    snapshot_calib,
    SNAPSHOT_CTYPES,
    MANIFEST_NAME,
)

#: psdata's original forbidden set (the reader's contract -- the three psdata
#: shipped with).  pscalib EXTENDS this with ``dgram`` + ``pymongo``; psdata
#: keeps its narrower historical contract here.
_FORBIDDEN_MODULES = ("psana", "mpi4py", "h5py")


def assert_no_framework_imports():
    """Raise AssertionError if psana / mpi4py / h5py leaked into ``sys.modules``.

    Re-export-shim flavour of the check, preserving psdata's original
    ``('psana','mpi4py','h5py')`` contract.  The canonical engine is in
    pscalib (which uses a stricter 5-module set); see
    :func:`pscalib.assert_no_framework_imports`.
    """
    import sys
    leaked = [m for m in _FORBIDDEN_MODULES if m in sys.modules]
    assert not leaked, (
        f"psdata.calib (reload path) must not import {_FORBIDDEN_MODULES}; "
        f"found {leaked} in sys.modules")
