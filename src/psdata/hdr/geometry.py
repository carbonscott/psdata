"""psdata.hdr.geometry -- RE-EXPORT SHIM of pscalib.geometry.

The geometry-text -> pixel-index-map derivation's canonical home moved to
``pscalib.geometry``; this module re-exports it (one implementation, no drift --
see US-000).  The lazy psana import (the one prep-time touch) lives in pscalib.
"""

from pscalib.geometry import (  # noqa: F401
    pixel_coord_indexes_from_text,
    cache_pixel_indexes_for_snapshot,
    load_pixel_indexes,
    IX_FILE,
    IY_FILE,
)
