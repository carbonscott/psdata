"""psdata.hdr.render -- standalone offline calibrated 2-D HDR image (US-007).

Ties together the vendored gain decode (:mod:`psdata.hdr.jungfrau`), the
vendored image remap (:mod:`psdata.hdr.image`), and a US-006 calibration
snapshot (:mod:`psdata.calib`) into a single offline pipeline:

    raw (N,512,1024) uint16  -- e.g. from psdata.run.Event.stack('jungfrau')
        |  gain decode (cached constants)
        v
    calib (N,512,1024) f32   == det.raw.calib(evt)   (max|diff| == 0)
        |  geometry remap (cached pixel index maps)
        v
    image (4216,4432) f32    == det.raw.image(evt)   (max|diff| == 0)

This is a **separate** module from the psdata reader: calibration is detector-
specific and stays its own layer (the reader is detector-universal and numpy-
only).  At *render time* this pulls in only numpy -- no psana, no DB, no MPI.
The one psana-touching step is the one-time snapshot + index-map prep
(:func:`psdata.calib.snapshot_calib` + :func:`psdata.hdr.geometry`).

The render is per-detector-type (the gain decode and geometry differ by
detector).  Today only Jungfrau is wired in; :class:`HDRImager` dispatches on
``snapshot.detname`` so other detector types can be added without changing the
public surface.
"""

import numpy as np

from . import image as _image
from . import jungfrau as _jungfrau
from . import geometry as _geometry

#: Detectors with a wired-in gain decode.  Maps the snapshot detname (matched
#: case-insensitively as a prefix) to its calibrate function.
_GAIN_DECODERS = {
    "jungfrau": _jungfrau.calib_jungfrau,
}


def _decoder_for(detname):
    key = (detname or "").lower()
    for name, fn in _GAIN_DECODERS.items():
        if key.startswith(name):
            return fn
    raise NotImplementedError(
        f"no HDR gain decode wired in for detector {detname!r}; "
        f"supported: {sorted(_GAIN_DECODERS)}")


class HDRImager:
    """Offline calibrated 2-D HDR image renderer pinned to one calib snapshot.

    Construct from a :class:`psdata.calib.CalibSnapshot` (US-006).  All inputs
    -- constants, mask, geometry index maps -- come from the snapshot, so once
    built the renderer touches no psana, DB, or network.

    The geometry index maps (``ix``/``iy``) are read from the snapshot if they
    were cached (:func:`psdata.hdr.geometry.cache_pixel_indexes_for_snapshot`),
    else derived once from the snapshot's geometry text via psana's
    ``GeometryAccess`` (the single lazy psana touch) and cached for next time.

    Parameters
    ----------
    snapshot : psdata.calib.CalibSnapshot
        A pinned calibration snapshot for the detector + run to render.
    derive_geometry_if_missing : bool
        If the snapshot has no cached ``ix``/``iy`` but does carry geometry
        text, derive them (one lazy psana ``GeometryAccess`` call) and cache
        them into the snapshot dir.  Default True.  Set False to force a
        fully-offline construction that fails fast if the maps were never
        cached.
    """

    def __init__(self, snapshot, derive_geometry_if_missing=True):
        self.snapshot = snapshot
        self.detname = snapshot.detname
        self.run = snapshot.run
        self._calibrate = _decoder_for(self.detname)

        self.pedestals = snapshot.pedestals
        self.pixel_gain = snapshot.pixel_gain
        self.pixel_offset = snapshot.pixel_offset          # may be None
        self.mask = snapshot.mask
        if self.pedestals is None or self.pixel_gain is None:
            raise ValueError(
                f"snapshot {snapshot!r} is missing pedestals/pixel_gain -- "
                f"cannot calibrate")

        idx = _geometry.load_pixel_indexes(snapshot.path)
        if idx is None:
            if not (derive_geometry_if_missing and snapshot.geometry):
                raise ValueError(
                    f"snapshot {snapshot.path!r} has no cached pixel index "
                    f"maps and {'geometry text is absent' if not snapshot.geometry else 'derive_geometry_if_missing=False'}; "
                    f"run psdata.hdr.geometry.cache_pixel_indexes_for_snapshot "
                    f"once (with psana) to create them")
            _geometry.cache_pixel_indexes_for_snapshot(snapshot.path)
            idx = _geometry.load_pixel_indexes(snapshot.path)
        self.ix, self.iy = idx
        # full-detector image-grid extent, pinned once.
        self._rc_tot_max = [int(np.max(self.ix.ravel())),
                            int(np.max(self.iy.ravel()))]

    # -- the two products -------------------------------------------------
    def calib(self, raw):
        """Calibrate a raw stack into ADU (``== det.raw.calib(evt)``).

        Parameters
        ----------
        raw : ndarray ``(N, 512, 1024)`` uint16

        Returns
        -------
        ndarray ``(N, 512, 1024)`` float32
        """
        return self._calibrate(raw, self.pedestals, self.pixel_gain,
                               self.pixel_offset, self.mask)

    def image(self, calib_or_raw, is_raw=False):
        """Assemble the calibrated 2-D image (``== det.raw.image(evt)``).

        Parameters
        ----------
        calib_or_raw : ndarray
            Either a calibrated stack (default) or, if ``is_raw=True``, a raw
            stack to calibrate first.
        is_raw : bool
            Treat the input as raw uint16 and calibrate it before assembly.

        Returns
        -------
        ndarray, 2-D, float32  (e.g. ``(4216, 4432)`` for Jungfrau 8M)
        """
        calib = self.calib(calib_or_raw) if is_raw else calib_or_raw
        return _image.assemble_image(calib, self.ix, self.iy,
                                     rc_tot_max=self._rc_tot_max)

    def render(self, raw):
        """Full pipeline: raw -> (calib, image).  Convenience wrapper.

        Returns
        -------
        (calib, image) : (ndarray (N,512,1024) f32, ndarray 2-D f32)
        """
        calib = self.calib(raw)
        image = self.image(calib)
        return calib, image

    def __repr__(self):
        return (f"HDRImager(detname={self.detname!r}, run={self.run}, "
                f"image_shape=({self._rc_tot_max[0] + 1},"
                f"{self._rc_tot_max[1] + 1}))")


def from_snapshot_dir(snap_dir, **kwa):
    """Build an :class:`HDRImager` from a calib snapshot *directory* path.

    Loads the snapshot with :func:`psdata.calib.load_snapshot` (pure numpy)
    then constructs the imager.  Convenience for the common "I have a snapshot
    on disk" case.
    """
    from psdata.calib import load_snapshot
    return HDRImager(load_snapshot(snap_dir), **kwa)
