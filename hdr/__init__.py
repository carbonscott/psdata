"""psdata.hdr -- standalone offline calibrated 2-D HDR image render (US-007).

A **separate, optional** layer from the :mod:`psdata` raw reader, built on a
US-006 calibration snapshot (:mod:`psdata.calib`).  It turns a raw detector
stack into the calibrated 2-D HDR image *fully offline* -- no web calib DB, no
MPI, no psana framework at render time (only numpy).

Pipeline (Jungfrau 8M reference)::

    raw (32,512,1024) uint16
        -> gain decode (cached pedestals/pixel_gain/pixel_offset/mask)
        -> calib (32,512,1024) f32      == det.raw.calib(evt)   (max|diff|==0)
        -> geometry remap (cached pixel index maps ix/iy)
        -> image (4216,4432) f32        == det.raw.image(evt)   (max|diff|==0)

Usage (after a one-time US-006 snapshot has been taken)::

    from psdata.calib import load_snapshot
    from psdata.hdr import HDRImager

    snap = load_snapshot("snapshots/jungfrau_r0051")   # pure numpy
    imager = HDRImager(snap)                            # offline render engine
    calib, image = imager.render(raw_stack)            # numpy only

Module map
----------
* :mod:`psdata.hdr.jungfrau`  -- Jungfrau gain decode (vendored numpy,
  ``== det.raw.calib``).
* :mod:`psdata.hdr.image`     -- generic pixel-array -> 2-D image remap
  (vendored numpy, ``mapmode=2``/``fillholes`` == ``det.raw.image``).
* :mod:`psdata.hdr.geometry`  -- derive/cache the per-pixel image index maps
  from geometry text (the one psana touch; a one-time, snapshot-time prep
  step, imported lazily so importing :mod:`psdata.hdr` stays framework-free).
* :mod:`psdata.hdr.render`    -- :class:`HDRImager`, the public render engine.

This package is **not** imported by :mod:`psdata` (the reader stays numpy-only
and detector-universal); ``import psdata.hdr`` explicitly when you want the
detector-specific calibrated render.  The render is per-detector-type; today
only Jungfrau is wired in.
"""

from . import jungfrau   # noqa: F401
from . import image      # noqa: F401
from . import geometry   # noqa: F401
from . import render      # noqa: F401
from .jungfrau import calib_jungfrau
from .image import assemble_image
from .geometry import (
    pixel_coord_indexes_from_text,
    cache_pixel_indexes_for_snapshot,
    load_pixel_indexes,
)
from .render import HDRImager, from_snapshot_dir

__all__ = [
    "jungfrau", "image", "geometry", "render",
    "calib_jungfrau", "assemble_image",
    "pixel_coord_indexes_from_text", "cache_pixel_indexes_for_snapshot",
    "load_pixel_indexes",
    "HDRImager", "from_snapshot_dir",
]
