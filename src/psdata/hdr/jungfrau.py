"""psdata.hdr.jungfrau -- RE-EXPORT SHIM of pscalib.apply.jungfrau.

The Jungfrau gain decode's canonical home moved to ``pscalib.apply.jungfrau``;
this module re-exports it (one implementation, no drift -- see US-000).  The
re-exported ``calib_jungfrau`` IS the pscalib object (proven by identity in
pscalib's ``tests/test_no_drift_us000.py``).
"""

from pscalib.apply.jungfrau import (  # noqa: F401
    calib_jungfrau,
    gain_stage_map,
    MSK,
    BSH,
    N_GAIN_STAGES,
)
