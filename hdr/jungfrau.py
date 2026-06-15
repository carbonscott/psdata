"""psdata.hdr.jungfrau -- Jungfrau 3-gain HDR gain decode (vendored numpy).

This is the per-detector-type gain-decode half of the standalone calibrated
2-D HDR render (US-007).  It is a faithful, framework-free numpy
re-implementation of psana's
``psana.detector.UtilsJungfrau.calib_jungfrau_single_panel`` (looped over
segments, exactly as ``calib_jungfrau`` does), verified byte-identical
(``np.array_equal``) to ``det.raw.calib(evt)`` for the reference Jungfrau 8M
dataset (exp=mfx100848724, run=51, det='jungfrau').

The Jungfrau is an *auto-ranging* detector: every 16-bit raw word carries its
own gain stage in the top 2 bits and a 14-bit ADC code in the low 14 bits.
The three gain stages (the "HDR signature") are the leading ``3`` axis of the
``pedestals`` / ``pixel_gain`` / ``pixel_offset`` constants.

Gain-bit decode (psana ``calib_jungfrau_single_panel`` + constants ``MSK``/``BSH``
at ``UtilsJungfrau.py`` ~lines 44-45, 652)::

    gbits = raw >> 14          # 0, 1, 2, or 3
    stage 0  <- gbits == 0
    stage 1  <- gbits == 1
    stage 2  <- gbits == 3     # NOTE: binary 11 (==3), NOT 2
    "bad"    <- gbits == 2     # binary 10 -> no gain stage; contributes 0
    adc = raw & 0x3fff
    calib[stage] = (adc - (pedestals + pixel_offset)[stage]) / pixel_gain[stage] * mask

Common-mode correction is OFF by default (matching psana's ``cmpars`` default
in this path being unused); this module does not apply it.  ``pixel_offset`` may
be absent for some runs/detectors -- callers pass ``None`` and it is treated as
0 (matching ``psdata.calib`` snapshot semantics).
"""

import numpy as np

#: 14-bit ADC mask -- psana ``UtilsJungfrau.MSK`` (``0x3fff``, ``(1<<14)-1``).
MSK = 0x3fff
#: Gain-bit shift -- psana ``UtilsJungfrau.BSH`` (the gain code is ``raw >> 14``).
BSH = 14

#: Expected number of gain stages on the leading constant axis (HDR signature).
N_GAIN_STAGES = 3


def calib_jungfrau(raw, pedestals, pixel_gain, pixel_offset=None, mask=None):
    """Calibrate a raw Jungfrau stack into ADU, fully offline (numpy only).

    Faithful re-implementation of psana
    ``UtilsJungfrau.calib_jungfrau`` / ``calib_jungfrau_single_panel`` (looped
    over segments).  Verified ``np.array_equal`` to ``det.raw.calib(evt)``.

    Parameters
    ----------
    raw : ndarray, shape ``(N, 512, 1024)``, uint16
        The raw detector stack (e.g. from :meth:`psdata.run.Event.stack`).
        ``N`` is the number of segments present this event.
    pedestals : ndarray, shape ``(3, S, 512, 1024)``, float32
        Per-(stage, segment) pedestals.  Leading axis = 3 gain stages.
    pixel_gain : ndarray, shape ``(3, S, 512, 1024)``, float32
        Per-(stage, segment) gain (ADU per keV-equivalent).  Calibration
        divides by this (protected: a 0 gain yields a 0 factor).
    pixel_offset : ndarray or None, shape ``(3, S, 512, 1024)``, float32
        Per-(stage, segment) pedestal offset, added to ``pedestals``.  ``None``
        is treated as 0 (matching psana / the snapshot, where it may be absent).
    mask : ndarray or None, shape ``(S, 512, 1024)``
        Per-segment status mask (0 = bad pixel).  ``None`` => no masking.

    Returns
    -------
    ndarray, shape ``(N, 512, 1024)``, float32
        Calibrated stack in ADU.  Bad-gain-code pixels (``gbits == 2``) and
        masked pixels are 0.

    Notes
    -----
    ``raw`` carries ``N`` segments; the constants carry ``S`` segments
    (``S`` may exceed ``N`` if not every segment is present this event).  Each
    raw segment ``s`` indexes into constant segment ``s`` -- the caller must
    pass a ``raw`` stack ordered by ascending segment id (as
    :meth:`psdata.run.Event.stack` produces) and constants for those same
    segments.  For the reference run all 32 segments are always present.
    """
    raw = np.asarray(raw)
    if raw.ndim != 3:
        raise ValueError(f"raw must be 3-D (N,512,1024); got shape {raw.shape}")

    pedestals = np.asarray(pedestals, dtype=np.float32)
    pixel_gain = np.asarray(pixel_gain, dtype=np.float32)
    if pedestals.shape[0] != N_GAIN_STAGES:
        raise ValueError(
            f"pedestals leading axis must be {N_GAIN_STAGES} gain stages; "
            f"got shape {pedestals.shape}")

    # poff = pedestals + pixel_offset  (offset absent -> 0)
    if pixel_offset is None:
        poff = pedestals.copy()
    else:
        poff = (pedestals + np.asarray(pixel_offset, dtype=np.float32)
                ).astype(np.float32)

    # gfac = 1 / pixel_gain, protected (gain==0 -> factor 0, matches psana)
    gfac = np.divide(1.0, pixel_gain,
                     out=np.zeros_like(pixel_gain, dtype=np.float32),
                     where=pixel_gain != 0).astype(np.float32)

    nseg = raw.shape[0]
    out = np.zeros(raw.shape, dtype=np.float32)
    for s in range(nseg):
        arr = raw[s]                                    # (512,1024) uint16
        # gain bits: 00/01/11 select stages 0/1/2; 10 (==2) is the bad code.
        gbits = (arr >> BSH).astype(np.uint8)
        gr0, gr1, gr2 = gbits == 0, gbits == 1, gbits == 3
        factor = np.select((gr0, gr1, gr2),
                           (gfac[0, s], gfac[1, s], gfac[2, s]), default=0)
        pedoff = np.select((gr0, gr1, gr2),
                           (poff[0, s], poff[1, s], poff[2, s]), default=0)
        arrf = (arr & MSK).astype(np.float32)
        arrf -= pedoff
        arrf *= factor                                  # bad code -> factor 0
        if mask is not None:
            arrf *= np.asarray(mask)[s].astype(np.float32)
        out[s] = arrf
    return out


def gain_stage_map(raw):
    """Return the per-pixel gain *stage* (0/1/2) and a ``bad`` mask for ``raw``.

    Convenience/introspection helper (not used by :func:`calib_jungfrau`,
    which decodes inline).  ``stage`` is the index into the leading 3-axis of
    the HDR constants; ``bad`` marks the ``gbits == 2`` code that has no stage.

    Returns
    -------
    (stage, bad) : (ndarray int8, ndarray bool), same shape as ``raw``
    """
    raw = np.asarray(raw)
    gbits = (raw >> BSH).astype(np.uint8)
    stage = np.full(raw.shape, -1, dtype=np.int8)
    stage[gbits == 0] = 0
    stage[gbits == 1] = 1
    stage[gbits == 3] = 2
    bad = gbits == 2
    return stage, bad
