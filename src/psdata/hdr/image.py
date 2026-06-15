"""psdata.hdr.image -- RE-EXPORT SHIM of pscalib.image.

The pixel-array -> 2-D image remap's canonical home moved to ``pscalib.image``;
this module re-exports it (one implementation, no drift -- see US-000).  The
re-exported ``assemble_image`` IS the pscalib object (proven by identity in
pscalib's ``tests/test_no_drift_us000.py``).
"""

from pscalib.image import (  # noqa: F401
    assemble_image,
    image_shape,
    statistics_of_pixel_arrays,
    statistics_of_holes,
    img_from_pixel_arrays,
    img_multipixel_max,
    fill_holes,
)
