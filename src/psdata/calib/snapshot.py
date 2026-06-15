#!/usr/bin/env python3
"""psdata.calib.snapshot -- one-time calibration-constant snapshot + pinning.

See :mod:`psdata.calib` for the design.  This module has two halves split by
their dependencies:

  * :func:`snapshot_calib` -- the ONLY psana-using function.  It is imported
    lazily (``import psana`` happens *inside* the function) so that importing
    this module, reloading a snapshot, and the rest of psdata stay psana-free.

  * :class:`CalibSnapshot` / :func:`load_snapshot` -- pure numpy.  Reload a
    snapshot's arrays + validity metadata for fully-offline use.

On-disk layout (one directory per ``(detector, run)`` pin)::

    <detname>_r<run:04d>/
        manifest.json        # pin + per-ctype validity metadata + index
        pedestals.npy        # (3,32,512,1024) f32   ] the HDR calibration
        pixel_gain.npy       # (3,32,512,1024) f32   ] constants -- leading
        pixel_offset.npy     # (3,32,512,1024) f32   ] axis = 3 gain stages
        pixel_status.npy     # (3,32,512,1024) u64   ] (+ pixel_rms/max/min ...)
        ...                  # every other ndarray ctype psana returned
        mask.npy             # (32,512,1024) u8      -- det.raw._mask(status=True)
        geometry.txt         # ~5-8 KB geometry text -- det.raw._calibconst['geometry']

The manifest is JSON (human-auditable); arrays are ``.npy`` (byte-exact, no
re-encoding).  ``CalibSnapshot.calibconst`` rebuilds psana's
``{ctype: (array, meta)}`` dict so a reload reproduces the exact arrays.
"""

import json
import os

import numpy as np

#: Calibration ctypes the snapshot is *expected* to carry for an area detector.
#: ``snapshot_calib`` actually persists *every* ndarray ctype psana returns
#: (so the snapshot is complete), but these are the ones US-006 names and that
#: downstream calibration (US-007) consumes.  ``pixel_offset`` may be absent for
#: some detectors/runs -- callers treat a missing offset as 0.
SNAPSHOT_CTYPES = ("pedestals", "pixel_gain", "pixel_offset")

#: Name of the JSON manifest inside a snapshot directory.
MANIFEST_NAME = "manifest.json"

#: ctype name under which the geometry text lives in psana's _calibconst.
_GEOMETRY_CTYPE = "geometry"

#: Filenames (inside the snapshot dir) for the non-ndarray / derived artifacts.
_MASK_FILE = "mask.npy"
_GEOMETRY_FILE = "geometry.txt"

#: Validity-metadata fields kept per ctype (psana attaches many more; these are
#: the ones that define the validity range + provenance for pinning).
_META_KEEP = (
    "run", "run_end", "version", "ctype", "detector", "detname", "dettype",
    "longname", "time_sec", "time_stamp", "experiment",
)

_FORBIDDEN_MODULES = ("psana", "mpi4py", "h5py")


# ==========================================================================
# Reload side -- pure numpy, no psana
# ==========================================================================
class CalibSnapshot:
    """A pinned, on-disk calibration snapshot -- reloaded with numpy only.

    Construct via :func:`load_snapshot` (or :meth:`load`).  Exposes the cached
    arrays, the geometry text, and the per-ctype validity metadata, plus the
    ``(detector_uniqueid, run)`` pin the snapshot was taken for.

    Attributes
    ----------
    path : str
        Directory the snapshot was loaded from.
    detname : str
        Detector short name (e.g. ``"jungfrau"``).
    detector_uniqueid : str
        The detector's unique id -- ``det.raw._uniqueid`` at snapshot time.  Half
        of the pin; pairs with :attr:`run`.
    run : int
        The run the snapshot was *taken for* (the pin).  NOTE: each constant's
        own validity range (see :meth:`validity`) may begin at an *earlier* run
        -- ``run`` is "the run you asked psana to calibrate", not the constant's
        first valid run.
    exp : str | None
        Experiment id the snapshot was taken from.
    """

    def __init__(self, path, manifest, arrays, geometry):
        self.path = path
        self._manifest = manifest
        self._arrays = arrays                 # {ctype: ndarray}, includes 'mask'
        self.geometry = geometry              # str | None
        pin = manifest["pin"]
        self.detname = pin["detname"]
        self.detector_uniqueid = pin["detector_uniqueid"]
        self.run = pin["run"]
        self.exp = pin.get("exp")

    # -- the named HDR constants (US-006) ---------------------------------
    @property
    def pedestals(self):
        """``(3,32,512,1024) f32`` pedestals (leading axis = 3 gain stages)."""
        return self._arrays.get("pedestals")

    @property
    def pixel_gain(self):
        """``(3,32,512,1024) f32`` per-pixel gain (leading axis = gain stages)."""
        return self._arrays.get("pixel_gain")

    @property
    def pixel_offset(self):
        """``(3,32,512,1024) f32`` per-pixel offset, or ``None`` if not cached
        (callers treat a missing offset as 0)."""
        return self._arrays.get("pixel_offset")

    @property
    def mask(self):
        """``(32,512,1024) u8`` status mask -- ``det.raw._mask(status=True)``."""
        return self._arrays.get("mask")

    # -- generic access ----------------------------------------------------
    def array(self, ctype):
        """Return the cached ndarray for ``ctype`` (``None`` if not present)."""
        return self._arrays.get(ctype)

    def ctypes(self):
        """Sorted list of array ctypes in the snapshot (excludes the derived
        ``mask`` and the text ``geometry``)."""
        return sorted(k for k in self._arrays if k != "mask")

    def validity(self, ctype):
        """Per-ctype validity metadata dict (``run`` / ``run_end`` / ``version``
        / provenance) -- the doc psana attached to this constant.  ``run`` is
        the *first* run the constant is valid for; ``run_end`` the last (or the
        sentinel ``'end'``)."""
        return dict(self._manifest["validity"].get(ctype, {}))

    @property
    def pin(self):
        """The ``(detector_uniqueid, run)`` pin (plus detname/exp) as a dict."""
        return dict(self._manifest["pin"])

    def calibconst(self):
        """Reconstruct psana's ``det.raw._calibconst`` dict:
        ``{ctype: (array_or_text, metadata)}``.

        This is the byte-for-byte inverse of what :func:`snapshot_calib`
        captured, so feeding it where psana's ``_calibconst`` is expected (e.g.
        the US-007 HDR render) reproduces psana's exact arrays.
        """
        out = {}
        for ctype in self.ctypes():
            out[ctype] = (self._arrays[ctype], self.validity(ctype))
        if self.geometry is not None:
            out[_GEOMETRY_CTYPE] = (self.geometry, self.validity(_GEOMETRY_CTYPE))
        return out

    def is_valid_for_run(self, run):
        """Best-effort check that *every* cached constant's validity range
        covers ``run`` (``run <= run <= run_end``; ``'end'`` means open-ended).

        Returns ``True`` if all ranges cover ``run``.  This is advisory only --
        :func:`load_snapshot` never refuses a stale reload (staleness is silent
        by design); use this to opt into a check.
        """
        run = int(run)
        for ctype in list(self.ctypes()) + ([_GEOMETRY_CTYPE]
                                            if self.geometry is not None else []):
            meta = self._manifest["validity"].get(ctype, {})
            lo = meta.get("run")
            hi = meta.get("run_end")
            if lo is not None and run < int(lo):
                return False
            if hi not in (None, "end") and run > int(hi):
                return False
        return True

    @classmethod
    def load(cls, path):
        """Load the snapshot directory at ``path`` (pure numpy; no psana)."""
        return load_snapshot(path)

    def __repr__(self):
        return (f"CalibSnapshot(detname={self.detname!r}, run={self.run}, "
                f"ctypes={self.ctypes()}, "
                f"has_mask={self.mask is not None}, "
                f"has_geometry={self.geometry is not None})")


def load_snapshot(path):
    """Load a calibration snapshot from directory ``path`` (pure numpy).

    Reads the manifest, every ``.npy`` array it indexes, the ``mask.npy``, and
    the ``geometry.txt`` text.  Imports no psana / mpi4py / h5py.

    Returns
    -------
    CalibSnapshot
    """
    manifest_path = os.path.join(path, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"no {MANIFEST_NAME} in {path!r} -- not a psdata calib snapshot")
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    arrays = {}
    for ctype, fname in manifest["files"].items():
        arr_path = os.path.join(path, fname)
        # allow_pickle=False: arrays are plain numeric ndarrays, never objects.
        arrays[ctype] = np.load(arr_path, allow_pickle=False)

    geometry = None
    geo_file = manifest.get("geometry_file")
    if geo_file is not None:
        with open(os.path.join(path, geo_file), encoding="utf-8") as fh:
            geometry = fh.read()

    return CalibSnapshot(os.path.abspath(path), manifest, arrays, geometry)


# ==========================================================================
# Snapshot side -- the ONLY psana-using function (lazy import)
# ==========================================================================
def _slim_meta(meta):
    """Keep the validity + provenance fields from a psana ctype metadata dict.

    psana attaches a large metadata doc per ctype; we retain the fields that
    define the validity range (``run`` / ``run_end`` / ``version``) and identify
    the constant, dropping DB-internal bookkeeping (``_id``, ``cwd``, ``host``,
    ...) that is not load-bearing for offline reuse.
    """
    if not isinstance(meta, dict):
        return {"_raw": str(meta)}
    return {k: meta[k] for k in _META_KEEP if k in meta}


def snapshot_calib(exp, run, dir, detname, out_dir,
                   overwrite=False, slim_metadata=True):
    """One-time snapshot of a detector's calibration constants for ``run``.

    **This is the only psana-using entry point** -- ``import psana`` happens
    inside.  Run it once, in the ``psconda.sh`` psana env, on a host that can
    reach the calib DB.  It opens the run, reads ``det.raw._calibconst`` and
    ``det.raw._mask(status=True)``, and writes a self-describing on-disk
    snapshot (see :mod:`psdata.calib.snapshot` for the layout) pinned by
    ``(det.raw._uniqueid, run)``.

    Parameters
    ----------
    exp, run, dir : str, int, str
        The reference run (e.g. ``"mfx100848724"``, ``51``,
        ``"/sdf/data/lcls/ds/prj/public01/xtc"``).
    detname : str
        Detector short name (e.g. ``"jungfrau"``).
    out_dir : str
        Parent directory; the snapshot is written to
        ``{out_dir}/{detname}_r{run:04d}/``.
    overwrite : bool
        If False (default), refuse to overwrite an existing snapshot dir.
    slim_metadata : bool
        If True (default) retain only the validity + provenance metadata fields
        (:data:`_META_KEEP`); if False keep psana's full per-ctype metadata doc
        (JSON-coerced).

    Returns
    -------
    str
        The path of the written snapshot directory.
    """
    # Lazy psana import: keeps this module psana-free unless you actually snapshot.
    from psana import DataSource

    ds = DataSource(exp=exp, run=int(run), dir=dir)
    myrun = next(ds.runs())
    det = myrun.Detector(detname)
    raw = det.raw

    calibconst = raw._calibconst          # {ctype: (ndarray|str, meta_dict)}
    if calibconst is None:
        raise RuntimeError(
            f"det.raw._calibconst is None for {detname!r} run {run} -- the "
            f"calib DB returned nothing (wrong detname/run, or no network?)")
    mask = raw._mask(status=True)         # (segs, ...) u8 status mask
    uniqueid = raw._uniqueid              # the detector_uniqueid -- half the pin

    snap_dir = os.path.join(out_dir, f"{detname}_r{int(run):04d}")
    if os.path.isdir(snap_dir) and os.listdir(snap_dir) and not overwrite:
        raise FileExistsError(
            f"snapshot dir {snap_dir!r} already exists and is non-empty; "
            f"pass overwrite=True to replace it")
    os.makedirs(snap_dir, exist_ok=True)

    files = {}                  # ctype -> npy filename (ndarray ctypes only)
    validity = {}               # ctype -> kept metadata dict
    shapes = {}                 # ctype -> recorded shape/dtype (audit)
    geometry_file = None

    meta_fn = _slim_meta if slim_metadata else _full_meta

    for ctype, value in calibconst.items():
        arr, meta = value[0], value[1]
        validity[ctype] = meta_fn(meta)
        if ctype == _GEOMETRY_CTYPE and isinstance(arr, str):
            geometry_file = _GEOMETRY_FILE
            with open(os.path.join(snap_dir, _GEOMETRY_FILE), "w",
                      encoding="utf-8") as fh:
                fh.write(arr)
            shapes[ctype] = {"kind": "text", "len": len(arr)}
            continue
        if not isinstance(arr, np.ndarray):
            # Non-array, non-geometry ctype: record metadata only, skip the blob.
            shapes[ctype] = {"kind": type(arr).__name__}
            continue
        fname = f"{ctype}.npy"
        np.save(os.path.join(snap_dir, fname), arr, allow_pickle=False)
        files[ctype] = fname
        shapes[ctype] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}

    # The status mask is a *derived* product (not a raw _calibconst entry), so
    # store it under its own filename and key.
    if mask is not None:
        np.save(os.path.join(snap_dir, _MASK_FILE), np.asarray(mask),
                allow_pickle=False)
        files["mask"] = _MASK_FILE
        shapes["mask"] = {"shape": list(np.asarray(mask).shape),
                          "dtype": str(np.asarray(mask).dtype),
                          "kind": "mask(status=True)"}

    manifest = {
        "schema": "psdata.calib.snapshot/v1",
        "pin": {
            "detname": detname,
            "detector_uniqueid": uniqueid,
            "run": int(run),
            "exp": exp,
            "dir": dir,
        },
        "files": files,
        "geometry_file": geometry_file,
        "validity": validity,
        "shapes": shapes,
    }
    with open(os.path.join(snap_dir, MANIFEST_NAME), "w") as fh:
        json.dump(manifest, fh, indent=2, default=_json_default)

    return os.path.abspath(snap_dir)


def _full_meta(meta):
    """JSON-coerce a full psana metadata dict (drop nothing but un-serializable
    bits, which are stringified)."""
    if not isinstance(meta, dict):
        return {"_raw": str(meta)}
    return {k: _json_default(v) if not _json_safe(v) else v
            for k, v in meta.items()}


def _json_safe(v):
    return isinstance(v, (str, int, float, bool, type(None), list, dict))


def _json_default(v):
    """Fallback JSON encoder for numpy scalars / ObjectId / etc."""
    if isinstance(v, np.generic):
        return v.item()
    return str(v)


# ==========================================================================
# Import-purity self check (the reload side must stay psana-free)
# ==========================================================================
def assert_no_framework_imports():
    """Raise AssertionError if psana / mpi4py / h5py leaked into ``sys.modules``.

    ``snapshot_calib`` imports psana lazily *on call*; merely importing
    :mod:`psdata.calib` and reloading a snapshot must not.  Call this after a
    reload to assert the offline path stayed clean.
    """
    import sys
    leaked = [m for m in _FORBIDDEN_MODULES if m in sys.modules]
    assert not leaked, (
        f"psdata.calib (reload path) must not import {_FORBIDDEN_MODULES}; "
        f"found {leaked} in sys.modules")
