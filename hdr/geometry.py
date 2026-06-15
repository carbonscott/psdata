"""psdata.hdr.geometry -- geometry text -> per-pixel image index maps.

The pixel-coordinate index maps ``ix, iy`` (mapping each ``(seg, row, col)``
data pixel to its ``(image_row, image_col)``) are derived from the detector's
geometry text via psana's pure-numpy ``GeometryAccess`` --
``GeometryAccess.load_pars_from_str/file`` (~195/223) +
``get_pixel_coord_indexes(do_tilt=True, cframe=0)`` (~592).  Verified
byte-identical (``np.array_equal``) to ``det.raw._pixel_coord_indexes()``.

This is the **one** psana-touching step of the HDR render, and it is a
*prep/snapshot-time* step, not an apply-time one: the resulting ``ix.npy`` /
``iy.npy`` are run-pinned constants (the geometry is fixed for a run), cached
once alongside the US-006 calibration snapshot.  At render time
:mod:`psdata.hdr.image` consumes the cached ``ix``/``iy`` with numpy only -- no
``GeometryAccess``, no psana (see the handoff appendix "Shortcut": cache the
precomputed index maps so even ``GeometryAccess`` isn't needed at apply time).

``GeometryAccess`` itself is pure numpy, but it lives in the psana tree, so to
keep ``import psdata.hdr`` framework-free the psana import here is **lazy** --
it only happens when you actually derive indexes (a one-time op), exactly
mirroring ``psdata.calib.snapshot_calib``.
"""

import os

import numpy as np

#: Filenames the index maps are cached under inside a snapshot directory.
IX_FILE = "pixel_index_ix.npy"
IY_FILE = "pixel_index_iy.npy"


def pixel_coord_indexes_from_text(geometry_text, do_tilt=True, cframe=0):
    """Derive ``(ix, iy)`` per-pixel image index maps from geometry text.

    Uses psana's pure-numpy ``GeometryAccess`` (imported lazily).  Run once
    (in the ``psconda.sh`` psana env) to produce the run-pinned index maps;
    the result is byte-identical to ``det.raw._pixel_coord_indexes()``.

    Parameters
    ----------
    geometry_text : str
        The geometry definition text (e.g.
        ``CalibSnapshot.geometry`` / ``det.raw._calibconst['geometry'][0]``).
    do_tilt : bool
        Apply per-segment tilt (default True -- the ``det.raw.image`` default).
    cframe : int
        Coordinate frame (default 0 -- psana's default).

    Returns
    -------
    (ix, iy) : (ndarray, ndarray)
        Per-pixel image row / column index maps, shaped as the data
        (``(nsegs, 512, 1024)`` for Jungfrau), dtype as psana returns
        (``uint64``).
    """
    # Lazy psana import: keeps importing psdata.hdr framework-free.
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess

    geo = GeometryAccess()
    geo.load_pars_from_str(geometry_text)
    ix, iy = geo.get_pixel_coord_indexes(do_tilt=do_tilt, cframe=cframe)
    return np.asarray(ix), np.asarray(iy)


def cache_pixel_indexes_for_snapshot(snap_dir, do_tilt=True, cframe=0,
                                     overwrite=False):
    """Derive and cache ``ix.npy``/``iy.npy`` into a US-006 calib snapshot dir.

    Reads ``geometry.txt`` from the snapshot (written by
    :func:`psdata.calib.snapshot_calib`), derives the index maps with psana's
    ``GeometryAccess``, and writes :data:`IX_FILE` / :data:`IY_FILE` next to
    the constants.  This is the one-time, psana-using augmentation that makes
    the snapshot self-sufficient for a fully-offline render.

    Parameters
    ----------
    snap_dir : str
        A snapshot directory (``{detname}_r{run:04d}/``) containing
        ``geometry.txt``.
    do_tilt, cframe :
        Passed to :func:`pixel_coord_indexes_from_text`.
    overwrite : bool
        If False (default) and the index files already exist, return their
        paths without recomputing.

    Returns
    -------
    (ix_path, iy_path) : (str, str)
        Absolute paths of the written (or pre-existing) index files.
    """
    ix_path = os.path.join(snap_dir, IX_FILE)
    iy_path = os.path.join(snap_dir, IY_FILE)
    if (not overwrite and os.path.isfile(ix_path) and os.path.isfile(iy_path)):
        return os.path.abspath(ix_path), os.path.abspath(iy_path)

    geo_path = os.path.join(snap_dir, "geometry.txt")
    if not os.path.isfile(geo_path):
        raise FileNotFoundError(
            f"no geometry.txt in {snap_dir!r} -- snapshot has no geometry to "
            f"derive pixel indexes from")
    with open(geo_path, encoding="utf-8") as fh:
        geometry_text = fh.read()

    ix, iy = pixel_coord_indexes_from_text(geometry_text, do_tilt=do_tilt,
                                           cframe=cframe)
    np.save(ix_path, ix, allow_pickle=False)
    np.save(iy_path, iy, allow_pickle=False)
    return os.path.abspath(ix_path), os.path.abspath(iy_path)


def load_pixel_indexes(snap_dir):
    """Load cached ``(ix, iy)`` index maps from a snapshot dir (pure numpy).

    Returns ``None`` if they have not been cached (call
    :func:`cache_pixel_indexes_for_snapshot` once, with psana, to create them).
    """
    ix_path = os.path.join(snap_dir, IX_FILE)
    iy_path = os.path.join(snap_dir, IY_FILE)
    if not (os.path.isfile(ix_path) and os.path.isfile(iy_path)):
        return None
    return (np.load(ix_path, allow_pickle=False),
            np.load(iy_path, allow_pickle=False))
