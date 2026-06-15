"""psdata.hdr.render -- RE-EXPORT SHIM of pscalib.render.

The offline HDR render engine's canonical home moved to ``pscalib.render``;
this module re-exports it (one implementation, no drift -- see US-000).  The
re-exported ``HDRImager`` / ``from_snapshot_dir`` ARE the pscalib objects
(proven by identity in pscalib's ``tests/test_no_drift_us000.py``).

NOTE: ``pscalib.render.from_snapshot_dir`` loads via
``pscalib.providers.snapshot.load_snapshot`` -- which is the same object
psdata.calib re-exports -- so a snapshot dir taken with psdata round-trips
identically.
"""

from pscalib.render import (  # noqa: F401
    HDRImager,
    from_snapshot_dir,
)
