"""psdata.hdr.image -- raw-pixel-array -> 2-D image remap (vendored numpy).

Framework-free numpy re-implementation of psana's image assembly --
``psana.detector.UtilsAreaDetector`` (``statistics_of_pixel_arrays`` ~76,
``img_from_pixel_arrays`` ~124, ``img_multipixel_max`` ~155, ``fill_holes``
~238, plus the hole-finding helpers) composed exactly as
``psana.detector.calibconstants.CalibConstants.image`` does for ``mapmode=2,
fillholes=True`` (the path ``det.raw.image`` takes).

Given a calibrated stack and the per-pixel image-index maps ``ix, iy`` (the
``(seg,row,col) -> (image_row, image_col)`` scatter, from
:mod:`psdata.hdr.geometry`), :func:`assemble_image` scatters each pixel's
value into the 2-D image grid, taking the *max* where multiple pixels land on
one image bin (``mapmode=2``) and filling single-bin holes with the min of
their four neighbours (``fillholes=True``).

Verified ``np.array_equal`` to ``det.raw.image(evt)`` for the reference
Jungfrau dataset (image shape ``(4216, 4432)`` f32).
"""

import numpy as np


# --------------------------------------------------------------------------
# vendored helpers (UtilsAreaDetector)
# --------------------------------------------------------------------------
def image_shape(rows, cols, rc_tot_max=None):
    """Image shape ``(nrows, ncols)`` -- psana ``image_shape`` (~194).

    ``rc_tot_max`` (the *total*-detector max index, computed before any
    segment sub-selection) pins the grid size so a partial-segment render
    still lands in the full-detector frame.
    """
    rmax, cmax = (rows.max(), cols.max()) if rc_tot_max is None else rc_tot_max
    return int(rmax) + 1, int(cmax) + 1


def statistics_of_pixel_arrays(rows, cols, rc_tot_max=None):
    """Map of overlapping pixels -- psana ``statistics_of_pixel_arrays`` (~76).

    Returns ``multinds = {pixel_ravel_index: image_ravel_index}`` for every
    pixel that lands on an image bin shared by more than one pixel.  Only the
    overlaps matter for ``img_multipixel_max``; the histogram itself is
    transient.
    """
    img_shape = _nrows, ncols = image_shape(rows, cols, rc_tot_max)
    rr = rows.ravel()
    cc = cols.ravel()
    img_sta = np.zeros(img_shape, dtype=np.uint16)
    # explicit loop: fancy-indexed += does NOT accumulate repeated indices.
    for r, c in zip(rr, cc):
        img_sta[r, c] += 1
    cond = img_sta > 1
    multinds = {i: int(r * ncols + c)
                for i, (r, c) in enumerate(zip(rr, cc)) if cond[r, c]}
    return multinds


def _image_of_holes(busy_img_bins):
    """psana ``image_of_holes`` (~218): True where an empty bin is surrounded
    (in all 4 cardinal directions) by occupied bins."""
    nonem = busy_img_bins
    nrows, ncols = nonem.shape
    empty = np.logical_not(nonem)
    empty[0:nrows - 1, :] = np.logical_and(empty[0:nrows - 1, :], nonem[1:nrows, :])
    empty[1:nrows, :]     = np.logical_and(empty[1:nrows, :],     nonem[0:nrows - 1, :])
    empty[:, 0:ncols - 1] = np.logical_and(empty[:, 0:ncols - 1], nonem[:, 1:ncols])
    empty[:, 1:ncols]     = np.logical_and(empty[:, 1:ncols],     nonem[:, 0:ncols - 1])
    return empty


def statistics_of_holes(rows, cols, rc_tot_max=None):
    """Hole rows/cols -- psana ``statistics_of_holes`` (~244, the pieces used
    by ``CalibConstants.image``).

    Returns ``(hole_rows, hole_cols)`` index arrays for image bins that have no
    pixel of their own but whose four neighbours are all occupied.
    """
    img_shape = image_shape(rows, cols, rc_tot_max)
    img_pix_ascend_ind = -np.ones(img_shape, dtype=np.int32)
    img_pix_ascend_ind[rows.ravel(), cols.ravel()] = np.arange(
        rows.size, dtype=np.int32)
    busy_img_bins = img_pix_ascend_ind > -1
    img_holes = _image_of_holes(busy_img_bins)
    hole_rows, hole_cols = np.where(img_holes)
    return hole_rows, hole_cols


def img_from_pixel_arrays(rows, cols, weight, vbase=0, rc_tot_max=None):
    """psana ``img_from_pixel_arrays`` (~124): scatter ``weight`` into the grid.

    ``img[rows.ravel(), cols.ravel()] = weight.ravel()`` -- last write wins for
    overlapping bins (``img_multipixel_max`` then fixes those up).
    """
    img_shape = image_shape(rows, cols, rc_tot_max)
    if vbase:
        img = np.ones(img_shape, dtype=np.float32) * vbase
    else:
        img = np.zeros(img_shape, dtype=np.float32)
    img[rows.ravel(), cols.ravel()] = np.asarray(weight, dtype=np.float32).ravel()
    return img


def img_multipixel_max(img, weight, multinds):
    """psana ``img_multipixel_max`` (~155): for each overlapping image bin keep
    the max over the pixels that land on it.  Modifies ``img`` in place."""
    imgrav = img.ravel()
    wrav = np.asarray(weight, dtype=np.float32).ravel()
    for ia, i in multinds.items():
        imgrav[i] = max(imgrav[i], wrav[ia])


def fill_holes(img, hole_rows, hole_cols):
    """psana ``fill_holes`` (~238): set each hole to the min of its four
    cardinal neighbours.  Modifies ``img`` in place."""
    if hole_rows.size == 0:
        return
    img[hole_rows, hole_cols] = np.minimum(
        np.minimum(img[hole_rows - 1, hole_cols], img[hole_rows + 1, hole_cols]),
        np.minimum(img[hole_rows, hole_cols - 1], img[hole_rows, hole_cols + 1]))


# --------------------------------------------------------------------------
# top-level assembly (mapmode=2, fillholes=True -- the det.raw.image path)
# --------------------------------------------------------------------------
def assemble_image(values, ix, iy, vbase=0, fillholes=True, rc_tot_max=None):
    """Assemble a 2-D image from a per-pixel value stack and its index maps.

    Replicates ``CalibConstants.image(nda, mapmode=2, fillholes=True)`` exactly
    in pure numpy -- no psana, no DB, no MPI.

    Parameters
    ----------
    values : ndarray
        Per-pixel values shaped like the data (e.g. the calibrated
        ``(N, 512, 1024)`` stack).  Flattened in C order to match ``ix``/``iy``.
    ix, iy : ndarray, integer
        Per-pixel image row / column indexes (same shape as ``values``), from
        :mod:`psdata.hdr.geometry` (i.e. psana
        ``GeometryAccess.get_pixel_coord_indexes`` /
        ``det.raw._pixel_coord_indexes``).
    vbase : float
        Value for image bins with no contributing pixel (default 0).
    fillholes : bool
        Fill single-bin holes with the min of four neighbours (default True,
        matching ``det.raw.image``).
    rc_tot_max : (int, int) or None
        ``(max_image_row, max_image_col)`` over the *full* detector.  Defaults
        to the max of the passed ``ix``/``iy`` -- correct when ``ix``/``iy``
        cover every segment (as in the reference run).  Pass the full-detector
        max explicitly when rendering a subset of segments so the partial
        render lands in the full frame.

    Returns
    -------
    ndarray, 2-D, float32
        The assembled image, shape ``(max_image_row+1, max_image_col+1)``.
    """
    ix = np.asarray(ix)
    iy = np.asarray(iy)
    values = np.asarray(values, dtype=np.float32)
    if ix.shape != iy.shape:
        raise ValueError(f"ix/iy shape mismatch: {ix.shape} vs {iy.shape}")
    if values.size != ix.size:
        raise ValueError(
            f"values ({values.size}) and ix/iy ({ix.size}) sizes differ")

    if rc_tot_max is None:
        rc_tot_max = [int(np.max(ix.ravel())), int(np.max(iy.ravel()))]

    rows = ix.reshape(values.shape)
    cols = iy.reshape(values.shape)

    multinds = statistics_of_pixel_arrays(rows, cols, rc_tot_max)
    img = img_from_pixel_arrays(rows, cols, weight=values, vbase=vbase,
                                rc_tot_max=rc_tot_max)
    img_multipixel_max(img, values, multinds)             # mapmode == 2
    if fillholes:
        hole_rows, hole_cols = statistics_of_holes(rows, cols, rc_tot_max)
        fill_holes(img, hole_rows, hole_cols)
    return img
