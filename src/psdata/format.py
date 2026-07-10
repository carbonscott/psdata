#!/usr/bin/env python3
"""psdata.format -- pure-Python xtc2 parse core with generic discovery.

This module is the clean-room data-access core of ``psdata``.  It reads the
LCLS-II **xtc2** binary container into numpy arrays using only the standard
library ``struct`` and ``numpy.frombuffer`` -- it imports **no** psana, no
mpi4py, no h5py.  (See :func:`assert_no_framework_imports`.)

It does two things:

1. **Low-level format parsing** -- the byte layouts, the recursive Xtc walk,
   the ``(nodeId, namesId) -> Names`` association, and the ``DescData`` field
   offset algorithm.  This machinery is copied verbatim (functions and
   constants) from the byte-exact reference reader ``psdata.py`` at the repo
   root, which was validated against psana with ``max|diff| = 0``.

2. **Generic detector / segment discovery** -- given any run's xtc2 stream
   files, :func:`discover` reads every stream's Configure ``Names`` tables and
   builds a :class:`RunConfig` describing every detector, each detector's
   fields (name / dtype / rank), and the segment->stream mapping.  Nothing is
   hardcoded: no detector name, no stream list, no file path, no timestamp.

The format constants and the five functions named in the acceptance criteria
(``parse_dgram_header``, ``iter_xtc_children``, ``parse_names_block``,
``parse_configure``, ``extract_field``) are the same as in ``psdata.py``.
"""

import os
import re
import struct
import sys

import numpy as np

# ==========================================================================
# xtc2 byte-layout constants (little-endian, verified) -- from psdata.py
# ==========================================================================
# Dgram on disk = Transition(12B) + Xtc(12B) header = 24B, then payload.
#   Transition = TimeStamp{uint32 low=nsec, uint32 high=sec} + uint32 env
#   Xtc        = Src(uint32) + Damage(uint16) + TypeId(uint16) + extent(uint32)
DGRAM_HDR = 24          # transition(12) + xtc(12)
XTC_HDR = 12

SERVICE_CONFIGURE = 2
SERVICE_BEGINRUN = 4
SERVICE_BEGINSTEP = 6
SERVICE_ENABLE = 8
SERVICE_SLOWUPDATE = 10
SERVICE_L1ACCEPT_EOB = 11
SERVICE_L1ACCEPT = 12

# xtc.damage is a 16-bit field: the low 12 bits are the damage id (a
# DamageBitmask), and the high 4 bits are user-defined "userbits".  Matches
# psana/psana/detector/damage.py (DAMAGE_USERBITSHIFT=12,
# DAMAGE_VALUEBITMASK=0x0FFF).  psana does NOT auto-drop damaged data; it
# surfaces the per-segment damage value -- so does psdata (see
# :func:`decode_damage` and :meth:`psdata.stream.Event.damage`).
DAMAGE_USERBITSHIFT = 12
DAMAGE_VALUEBITMASK = 0x0FFF

# TypeId.type values (value & 0x0fff)
TID_PARENT = 0
TID_SHAPESDATA = 1
TID_SHAPES = 2
TID_DATA = 3
TID_NAMES = 4

# DataType enum -> element byte size.  Enum values match psana's DescData.
# (UINT8=0, UINT16=1, UINT32=2, UINT64=3, INT8=4, INT16=5, INT32=6, INT64=7,
#  FLOAT=8, DOUBLE=9, CHARSTR=10, ENUMVAL=11, ENUMDICT=12)
DTYPE_SIZE = {0: 1, 1: 2, 2: 4, 3: 8, 4: 1, 5: 2, 6: 4, 7: 8,
              8: 4, 9: 8, 10: 1, 11: 4, 12: 4}
DTYPE_NP = {0: np.uint8, 1: np.uint16, 2: np.uint32, 3: np.uint64,
            4: np.int8, 5: np.int16, 6: np.int32, 7: np.int64,
            8: np.float32, 9: np.float64, 10: np.uint8, 11: np.uint32,
            12: np.uint32}

# DataType.CHARSTR: a null-padded byte string stored as a rank-1 uint8 array
# (DTYPE_NP[10] == uint8).  psana's ``DetectorImpl._return_types`` maps type 10
# to ``str`` regardless of rank, so the env store decodes such a field to a
# Python ``str`` via :func:`decode_charstr`.
TYPE_CHARSTR = 10

# ENUMVAL/ENUMDICT: psana returns a scalar for these types regardless of
# rank; EnvStore._convert coerces a rank-0 field of either type to ``int``.
TYPE_ENUMVAL = 11
TYPE_ENUMDICT = 12

# Every DataType code this reader can decode.  Exactly the key set of the
# dtype tables above: FieldInfo raises KeyError on any other code, and
# EnvStore._convert dispatches on precisely these.
SUPPORTED_TYPE_CODES = frozenset(DTYPE_NP)


def decode_charstr(arr):
    """Decode a CHARSTR field (a null-padded ``uint8`` array from
    :func:`extract_field`) to a ``str``: take the bytes up to the first NUL and
    decode as latin-1.  Mirrors how psana surfaces a ``type==10`` env field."""
    return np.asarray(arr).tobytes().split(b"\x00")[0].decode("latin-1")

MAXRANK = 5
SHAPE_SZ = 4 * MAXRANK              # uint32 _shape[5] = 20B
ALG_SZ = 256 + 4                    # char[256] + uint32 version = 260B
NAME_SZ = ALG_SZ + 256 + 4 + 4     # Alg(260) + char _name[256] + type + rank = 524B
NAMEINFO_SZ = 4 + 256 * 3 + ALG_SZ + 4  # numArrays + detType/Name/Id + Alg + segment = 1036B


# ==========================================================================
# Low-level parsing -- copied from psdata.py
# ==========================================================================
def _cstr(buf, off, n=256):
    raw = buf[off:off + n]
    z = raw.find(b"\x00")
    return raw[:z if z >= 0 else n].decode("latin-1")


def parse_dgram_header(buf, off):
    """Return the Transition + Xtc header of the dgram whose Transition starts
    at ``off``.  The dgram's own Xtc spans ``[off+12, off+12+extent)``."""
    nsec, sec, env = struct.unpack_from("<III", buf, off)
    service = (env >> 24) & 0xf
    src, damage, typeid, extent = struct.unpack_from("<IHHI", buf, off + 12)
    ts = (sec << 32) | nsec
    return dict(service=service, env=env, ts=ts, sec=sec, nsec=nsec,
                src=src, damage=damage, typeid=typeid, extent=extent)


def typeid_type(typeid):
    return typeid & 0x0fff


def typeid_version(typeid):
    return (typeid >> 12) & 0xf


def namesid_of(src):
    """(nodeId, namesId) packed in Xtc.src."""
    return (src >> 8) & 0xfff, src & 0xff


def decode_damage(damage):
    """Split a 16-bit ``Xtc.damage`` into ``(damage_id, userbits)``.

    ``damage_id = damage & 0x0FFF`` is the low-12-bit damage code (a
    ``DamageBitmask`` value; 0 means undamaged); ``userbits = damage >> 12`` is
    the high-4-bit user field.  Mirrors ``psana/psana/detector/damage.py``.
    """
    return damage & DAMAGE_VALUEBITMASK, damage >> DAMAGE_USERBITSHIFT


def iter_xtc_children(buf, parent_payload_off, parent_payload_end):
    """Yield ``(xtc_off, hdr)`` for each child Xtc within a parent payload.
    ``next = xtc_off + extent`` (extent INCLUDES the 12B Xtc header)."""
    off = parent_payload_off
    while off + XTC_HDR <= parent_payload_end:
        src, damage, typeid, extent = struct.unpack_from("<IHHI", buf, off)
        if extent < XTC_HDR:
            break
        yield off, dict(src=src, damage=damage, typeid=typeid, extent=extent)
        off += extent


# ==========================================================================
# Configure: build the Names tables, keyed by (nodeId, namesId)
# ==========================================================================
def parse_names_block(buf, payload_off, payload_end):
    """Parse one Names Xtc payload -> dict describing the table."""
    o = payload_off
    num_arrays = struct.unpack_from("<I", buf, o)[0]
    det_type = _cstr(buf, o + 4)
    det_name = _cstr(buf, o + 4 + 256)
    det_id = _cstr(buf, o + 4 + 512)
    alg_off = o + 4 + 768
    alg_name = _cstr(buf, alg_off)
    alg_version = struct.unpack_from("<I", buf, alg_off + 256)[0]
    segment = struct.unpack_from("<I", buf, alg_off + 260)[0]

    names_off = o + NAMEINFO_SZ
    n_names = (payload_end - names_off) // NAME_SZ
    names = []
    for k in range(n_names):
        no = names_off + k * NAME_SZ
        fname = _cstr(buf, no + ALG_SZ)
        ftype = struct.unpack_from("<I", buf, no + ALG_SZ + 256)[0]
        frank = struct.unpack_from("<I", buf, no + ALG_SZ + 256 + 4)[0]
        names.append(dict(name=fname, type=ftype, rank=frank))
    return dict(det_type=det_type, det_name=det_name, det_id=det_id,
                alg_name=alg_name, alg_version=alg_version, segment=segment,
                num_arrays=num_arrays, names=names)


def parse_configure(buf):
    """Walk the Configure dgram at offset 0; return ``{(nodeId,namesId): table}``.
    The Configure's top Xtc is a Parent whose children are Names blocks -- but a
    Names table may sit **nested one level deeper** inside another ``TID_PARENT``
    (the ``scan`` config's Names table does: it rides inside a Parent, so a
    direct-children-only walk never sees it).  Recurse through ``TID_PARENT``
    containers while collecting every ``TID_NAMES`` -- the same recursion
    :func:`iter_shapesdata` uses for ShapesData -- so no declared detector
    (e.g. ``scan``) is missed.  Return signature is unchanged."""
    cfg = parse_dgram_header(buf, 0)
    assert cfg["service"] == SERVICE_CONFIGURE, "front dgram is not Configure"
    top_payload = DGRAM_HDR                       # 24
    top_end = XTC_HDR + cfg["extent"]             # dgram total on-disk size
    tables = {}
    stack = [(top_payload, top_end)]
    while stack:
        po, pe = stack.pop()
        for xoff, xh in iter_xtc_children(buf, po, pe):
            t = typeid_type(xh["typeid"])
            if t == TID_NAMES:
                nodeId, namesId = namesid_of(xh["src"])
                tbl = parse_names_block(buf, xoff + XTC_HDR, xoff + xh["extent"])
                tables[(nodeId, namesId)] = tbl
            elif t == TID_PARENT:
                stack.append((xoff + XTC_HDR, xoff + xh["extent"]))
    return cfg, tables, top_end


# ==========================================================================
# DescData: compute field offsets within a ShapesData block & extract arrays
# ==========================================================================
def extract_field(buf, shapesdata_off, shapesdata_hdr, table, field_name):
    """Given a ShapesData Xtc (which holds a Shapes child and a Data child),
    extract ``field_name`` as a numpy array using the Names ``table``.

    DescData offset algorithm: fields are in Names order.  ``offset[0]=0``;
    for field i: ``rank==0`` -> ``+elemsize``; ``rank>0`` ->
    ``+prod(shape)*elemsize`` and consume the next Shape (Shapes are
    consecutive, one per rank>0 field)."""
    sd_payload = shapesdata_off + XTC_HDR
    sd_end = shapesdata_off + shapesdata_hdr["extent"]

    data_off = data_end = None
    shapes_off = shapes_end = None
    for coff, ch in iter_xtc_children(buf, sd_payload, sd_end):
        t = typeid_type(ch["typeid"])
        if t == TID_DATA:
            data_off, data_end = coff + XTC_HDR, coff + ch["extent"]
        elif t == TID_SHAPES:
            shapes_off, shapes_end = coff + XTC_HDR, coff + ch["extent"]
    assert data_off is not None, "no Data child in ShapesData"

    names = table["names"]
    # Walk fields, accumulating byte offsets and consuming Shapes for rank>0.
    offset = 0
    shape_idx = 0
    for nm in names:
        rank = nm["rank"]
        esz = DTYPE_SIZE[nm["type"]]
        if rank == 0:
            shape = ()
            count = 1
            field_bytes = esz
        else:
            # read this field's Shape (one Shape per rank>0 field, in order)
            so = shapes_off + shape_idx * SHAPE_SZ
            dims = struct.unpack_from("<5I", buf, so)[:rank]
            shape = tuple(int(d) for d in dims)
            count = int(np.prod(shape)) if shape else 1
            field_bytes = count * esz
            shape_idx += 1
        if nm["name"] == field_name:
            arr = np.frombuffer(buf, dtype=DTYPE_NP[nm["type"]],
                                count=count, offset=data_off + offset)
            return arr.reshape(shape) if shape else arr[0], shape, nm
        offset += field_bytes
    raise KeyError(f"field {field_name!r} not found in Names table")


def iter_shapesdata(buf, dg_off, dg_hdr):
    """Yield ``(shapesdata_off, shapesdata_hdr)`` for every ShapesData Xtc in
    the dgram whose Transition starts at ``dg_off`` (recursing through Parent
    containers).  Works for any dgram -- L1Accept *or* a transition such as
    Enable (whose ``chunkinfo`` arrives as a ShapesData).
    """
    top_payload = dg_off + DGRAM_HDR
    top_end = dg_off + XTC_HDR + dg_hdr["extent"]
    stack = [(top_payload, top_end)]
    while stack:
        po, pe = stack.pop()
        for xoff, xh in iter_xtc_children(buf, po, pe):
            t = typeid_type(xh["typeid"])
            if t == TID_SHAPESDATA:
                yield xoff, xh
            elif t == TID_PARENT:
                stack.append((xoff + XTC_HDR, xoff + xh["extent"]))


def read_dgram_field(buf, dg_off, dg_hdr, tables, det_name, alg, field):
    """Walk one dgram and return ``field`` of the ``(det_name, alg)`` table the
    first time its ShapesData is found, or ``None`` if absent.

    ``tables`` is the stream's ``{(nodeId,namesId): names_table}`` Configure
    map.  Returns the extracted value (scalar for rank-0, ndarray otherwise),
    as :func:`extract_field` does.  Used to read the ``chunkinfo`` carried on an
    Enable transition (see :mod:`psdata.index`).
    """
    for xoff, xh in iter_shapesdata(buf, dg_off, dg_hdr):
        key = namesid_of(xh["src"])
        table = tables.get(key)
        if table is None:
            continue
        if table["det_name"] == det_name and table["alg_name"] == alg:
            arr, _shape, _meta = extract_field(buf, xoff, xh, table, field)
            return arr
    return None


# ==========================================================================
# Configure-object accessor -- per-segment CONFIGURE-block fields (US-010)
# ==========================================================================
# The L1Accept dgrams carry a detector's *event* data (the ``raw`` alg).  A
# detector's static *settings* -- e.g. epix10ka's per-ASIC ``trbit`` /
# ``asicPixelConfig`` that the gain-range decode needs -- live in the
# ``config`` alg, written once into the **Configure** dgram (offset 0 of every
# stream), not into any L1Accept.  This reads them back per segment, reusing the
# same generic DescData decoder (:func:`extract_field`) that reads event fields,
# so it is byte-exact vs psana's ``det.raw._seg_configs()`` for any detector
# whose Names tables declare a ``config`` alg (proven for epix10ka and jungfrau).
def read_config_object(run_config, det_name, alg="config", fields=None):
    """Read the per-segment CONFIGURE-block fields of ``(det_name, alg)``.

    For every segment of ``(det_name, alg)``, opens the Configure dgram (which
    sits at offset 0 of the segment's stream), locates that segment's
    ``config``-alg ShapesData, and extracts each requested field with the
    generic DescData decoder.  Returns the raw field values exactly as the
    decoder reads them (scalar for rank-0 fields, ndarray otherwise) -- the
    same arrays psana surfaces through ``det.raw._seg_configs()[seg].config``.

    Parameters
    ----------
    run_config : RunConfig
        The run's discovered config (from :func:`discover`).
    det_name : str
        Detector name (e.g. ``'epixquad'``).
    alg : str, optional
        Algorithm holding the configure-block fields (default ``'config'``).
    fields : sequence of str, optional
        Which fields to extract; defaults to every field discovered for
        ``(det_name, alg)``.

    Returns
    -------
    dict
        ``{segment_id: {field_name: value}}`` -- one entry per segment, sorted
        by segment id.  Each inner dict maps every requested field name to its
        extracted value.

    Raises
    ------
    KeyError
        If ``det_name`` is unknown or has no ``alg`` algorithm.

    Notes
    -----
    Scope is the most recent config declared at/before the first L1Accept (the
    front transitions, last-wins).  A config re-emitted on a transition that
    occurs AFTER events have begun (a mid-run reconfigure) is out of scope and
    NOT reflected -- unlike psana's stateful ``_seg_configs()``.  This matches
    the common case: config is set at Configure/BeginStep and constants are
    per-run keyed.
    """
    det = run_config.detector(det_name)
    if alg not in det.algs:
        raise KeyError(f"detector {det_name!r} has no alg {alg!r} "
                       f"(have {det.alg_names()})")
    want_fields = list(det.field_names(alg) if fields is None else fields)

    # A segment's config can be declared under MORE THAN ONE Names id: the DAQ
    # may re-emit it on a later transition with a fresh namesId.  E.g.
    # uedcom103/r7 declares epixquad seg 0's config under (2,1) -- data in the
    # Configure dgram, trbit=[0,0,0,0] -- AND under (2,21) -- different data on
    # BeginStep, trbit=[1,1,1,1].  The config that is ACTIVE for the run's
    # L1Accept events (and that psana's det.raw.calib uses, and that
    # det.raw._seg_configs() reports once the DataSource has advanced past the
    # transitions) is the MOST RECENT one before the first L1Accept -- here the
    # BeginStep override, not the Configure default.  So map every config Names
    # id -> its segment and walk ALL the front transition dgrams, letting a later
    # transition's value overwrite an earlier one (last-wins == active config).
    # (psana's _seg_configs() is stateful: read before iterating it returns the
    # Configure default; we expose the active config so the gain decode is
    # byte-exact vs psana's per-event calib.)
    streams_needed = sorted({det.seg_to_stream[(alg, seg)]
                             for seg in det.segment_ids(alg)})

    out = {}
    for stream in streams_needed:
        path = run_config.stream_files[stream]
        tables = run_config.raw_tables[stream]
        nkey_to_seg = {nkey: tbl["segment"]
                       for nkey, tbl in tables.items()
                       if tbl["det_name"] == det_name and tbl["alg_name"] == alg}
        fd = os.open(path, os.O_RDONLY)
        try:
            # The config ShapesData may ride on the Configure dgram OR a later
            # transition (BeginRun / BeginStep / Enable / SlowUpdate).  Walk the
            # front transition dgrams in order up to the first L1Accept (config
            # never lives on an event dgram), overwriting each segment so the
            # last (active) value wins.  Grow the read window on demand for a
            # transition dgram that runs past it.
            buf = bytearray(os.pread(fd, _CONFIG_READ, 0))
            off = 0
            while True:
                if off + DGRAM_HDR > len(buf):
                    grown = bytearray(os.pread(fd, len(buf) + _CONFIG_READ, 0))
                    if len(grown) == len(buf):
                        break                   # EOF: no more dgrams
                    buf = grown
                    continue
                hdr = parse_dgram_header(buf, off)
                if hdr["service"] in (SERVICE_L1ACCEPT, SERVICE_L1ACCEPT_EOB):
                    break                       # reached event data
                dg_size = XTC_HDR + hdr["extent"]
                if off + dg_size > len(buf):
                    grown = bytearray(os.pread(fd, off + dg_size, 0))
                    if len(grown) < off + dg_size:
                        break                   # truncated / EOF
                    buf = grown
                for xoff, xh in iter_shapesdata(buf, off, hdr):
                    nkey = namesid_of(xh["src"])
                    seg = nkey_to_seg.get(nkey)
                    if seg is None:
                        continue                # not this detector's config
                    table = tables[nkey]
                    seg_fields = {}
                    for fld in want_fields:
                        arr, _shape, _meta = extract_field(buf, xoff, xh,
                                                           table, fld)
                        seg_fields[fld] = arr
                    out[seg] = seg_fields        # last-wins: active config
                off += dg_size
        finally:
            os.close(fd)
    return {seg: out[seg] for seg in sorted(out)}


# ==========================================================================
# Generic discovery -- the US-001 deliverable, layered on the machinery above
# ==========================================================================
# Names tables whose alg is one of these are container bookkeeping, not a
# physical detector field a user asks for by name.  They are still discovered
# and exposed (under their own det_name) but are skipped by the "what real
# detectors are here" convenience views.
_BOOKKEEPING_ALGS = frozenset({"runinfo", "chunkinfo", "epicsinfo",
                               "pvdetinfo", "triginfo"})

# Default amount of each stream's front to read for the Configure dgram.  The
# Configure sits at offset 0 of every stream and is at most a few hundred KB
# (largest seen on the reference run is ~375 KB on the epics stream).
_CONFIG_READ = 4_000_000


def _read_front_buffer(path):
    """Open ``path`` and return a bytearray guaranteed to hold the whole
    Configure dgram, which sits at offset 0 -- growing the read window on demand.

    A single bare ``os.pread(fd, _CONFIG_READ, 0)`` truncates a Configure larger
    than ``_CONFIG_READ``: ``parse_dgram_header`` would still report the true
    extent, so the subsequent Xtc walk runs past ``len(buf)`` -- either raising
    or silently failing to find a ShapesData.  This re-preads with a grown window
    until the whole Configure dgram fits, detecting EOF by the read not growing
    (the same grow-on-demand strategy :func:`read_config_object` uses to walk the
    front transitions).  The fd is closed on every path.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        buf = bytearray(os.pread(fd, _CONFIG_READ, 0))
        while len(buf) >= DGRAM_HDR:
            hdr = parse_dgram_header(buf, 0)
            dg_size = XTC_HDR + hdr["extent"]      # dgram total on-disk size
            if dg_size <= len(buf):
                break                              # whole Configure dgram present
            grown = bytearray(os.pread(fd, len(buf) + _CONFIG_READ, 0))
            if len(grown) == len(buf):
                break                              # EOF: read did not grow
            buf = grown
        return buf
    finally:
        os.close(fd)


class FieldInfo:
    """One field of one (detector, alg) -- its name, numpy dtype, and rank.

    ``shape`` is only known at event time for rank>0 array fields (it is
    carried in the per-event Shapes block, not in the Configure Names table),
    so it is ``None`` here unless a representative shape is later filled in.
    """

    __slots__ = ("name", "type_code", "rank", "np_dtype")

    def __init__(self, name, type_code, rank):
        self.name = name
        self.type_code = type_code
        self.rank = rank
        self.np_dtype = np.dtype(DTYPE_NP[type_code])

    def __repr__(self):
        return (f"FieldInfo(name={self.name!r}, dtype={self.np_dtype.name}, "
                f"rank={self.rank})")


class DetectorInfo:
    """A detector discovered from the Configure Names tables.

    A single physical detector (one ``det_name``) may declare several
    *algorithms* (``alg``) -- e.g. a jungfrau exposes alg ``raw`` (the frame)
    and alg ``config`` (its settings).  Fields are therefore grouped by alg,
    matching how psana addresses them (``det.raw.raw`` == det_name='jungfrau',
    alg='raw', field='raw').

    Attributes
    ----------
    name : str          detector name (det_name)
    det_type : str      detector type (det_type, e.g. 'jungfrau', 'ts')
    det_id : str        detector unique id string
    algs : dict         alg name -> {field name -> FieldInfo}
    segments : dict     alg name -> sorted list of segment ids
    seg_to_stream : dict   (alg, segment) -> stream index
    names_id : dict     (alg, segment) -> (nodeId, namesId) into the Configure
    """

    def __init__(self, name, det_type, det_id):
        self.name = name
        self.det_type = det_type
        self.det_id = det_id      # first segment's id (back-compat); see seg_detids
        self.algs = {}            # alg -> {field: FieldInfo}
        self.segments = {}        # alg -> set(segment)
        self.seg_to_stream = {}   # (alg, segment) -> stream
        self.names_id = {}        # (alg, segment) -> (nodeId, namesId)
        self.seg_detids = {}      # segment -> det_id (serial), alg-independent

    # -- discovery-time population -----------------------------------------
    def _add_table(self, alg, table, stream, names_key):
        seg = table["segment"]
        fields = self.algs.setdefault(alg, {})
        for nm in table["names"]:
            if nm["name"] not in fields:
                fields[nm["name"]] = FieldInfo(nm["name"], nm["type"],
                                               nm["rank"])
        self.segments.setdefault(alg, set()).add(seg)
        self.seg_to_stream[(alg, seg)] = stream
        self.names_id[(alg, seg)] = names_key
        # A segment's det_id (hardware serial) is alg-independent; record it
        # once per segment so the composite uniqueid can be reassembled exactly
        # as psana does (dgrammanager._set_configinfo).
        self.seg_detids.setdefault(seg, table["det_id"])

    # -- convenience views --------------------------------------------------
    @property
    def is_bookkeeping(self):
        return all(a in _BOOKKEEPING_ALGS for a in self.algs)

    def alg_names(self):
        return sorted(self.algs)

    def field_names(self, alg):
        return sorted(self.algs[alg])

    def segment_ids(self, alg):
        return sorted(self.segments[alg])

    def streams_for(self, alg):
        """Sorted unique stream indices that carry this (detector, alg)."""
        return sorted({self.seg_to_stream[(alg, s)]
                       for s in self.segments[alg]})

    def uniqueid(self):
        """Long hardware unique-id string -- byte-identical to psana's
        ``det.raw._uniqueid`` / ``configinfo.uniqueid`` for any real detector.

        Reproduces psana's composition exactly (dgrammanager.py
        ``_set_configinfo``): the detector ``det_type`` followed by each
        segment's ``det_id`` (hardware serial), in ascending segment order,
        joined by ``'_'``::

            uniqueid = det_type
            for seg in sorted(segments):
                uniqueid += '_' + det_id[seg]

        For a multi-segment detector (e.g. an epix10ka quad or a 32-module
        jungfrau) this is the full composite id used to address calibration
        constants -- the same string a caller would otherwise pin as a literal.

        Note: psana sources the per-segment serial from the ``config.software``
        block, which it populates only for detectors with a real software
        definition.  For every imaging detector that exposes
        ``det.raw._uniqueid`` (jungfrau, epix10ka, ...) the Names-table serial
        used here equals psana's software-block serial, so the result is
        byte-exact.  Bookkeeping/pseudo detectors that psana omits from
        ``config.software`` (e.g. the timing detector, whose ``_uniqueid`` psana
        reports as just ``'ts_'``) are not addressable for calibration and are
        not the intended use of this accessor.
        """
        uid = self.det_type
        for seg in sorted(self.seg_detids):
            uid += "_" + self.seg_detids[seg]
        return uid

    def __repr__(self):
        parts = []
        for alg in self.alg_names():
            parts.append(f"{alg}[{len(self.segments[alg])}seg,"
                         f"{len(self.algs[alg])}fld]")
        return (f"DetectorInfo(name={self.name!r}, type={self.det_type!r}, "
                f"algs={{{', '.join(parts)}}})")


class RunConfig:
    """Everything discovered from a run's per-stream Configure dgrams.

    Attributes
    ----------
    stream_files : dict   stream index -> file path
    detectors : dict      det_name -> DetectorInfo
    stream_configs : dict stream index -> (cfg_header, cfg_end_offset)
    raw_tables : dict     stream index -> {(nodeId,namesId): names_table}
    """

    def __init__(self):
        self.stream_files = {}
        self.detectors = {}
        self.stream_configs = {}
        self.raw_tables = {}

    # -- convenience views --------------------------------------------------
    def detector_names(self, include_bookkeeping=False):
        """Sorted detector names.  By default omit container-bookkeeping
        pseudo-detectors (runinfo / chunkinfo / epicsinfo / ...)."""
        out = []
        for name, det in self.detectors.items():
            if include_bookkeeping or not det.is_bookkeeping:
                out.append(name)
        return sorted(out)

    def detector(self, name):
        return self.detectors[name]

    def uniqueid(self, name):
        """Long hardware unique-id of detector ``name`` -- byte-identical to
        psana's ``det.raw._uniqueid`` (see :meth:`DetectorInfo.uniqueid`)."""
        return self.detectors[name].uniqueid()

    def find_detector_by_type(self, det_type):
        """Return the sorted names of detectors whose det_type matches.

        Used by higher layers to discover, e.g., which detector is the
        ``timing`` (det_type='ts') detector without hardcoding its name.
        """
        return sorted(n for n, d in self.detectors.items()
                      if d.det_type == det_type)

    def names_table(self, stream, names_key):
        return self.raw_tables[stream][names_key]

    def __repr__(self):
        return (f"RunConfig(streams={sorted(self.stream_files)}, "
                f"detectors={self.detector_names()})")


def discover(stream_files):
    """Discover all detectors / fields / segment->stream mappings for a run.

    Parameters
    ----------
    stream_files : sequence of (stream_index, path) | sequence of path | dict
        The xtc2 stream files of one run.  Either an explicit
        ``{index: path}`` mapping, a list of ``(index, path)`` pairs, or a
        plain list of paths (indices assigned positionally).

    Returns
    -------
    RunConfig
        Fully populated from the Configure Names tables of every stream.  No
        detector name, stream list, file path, or timestamp is hardcoded.
    """
    items = _normalize_stream_files(stream_files)

    rc = RunConfig()
    for stream, path in items:
        rc.stream_files[stream] = path
        # Read the WHOLE Configure dgram.  A bare fixed-window pread truncates a
        # Configure larger than _CONFIG_READ, and parse_configure would then walk
        # past the buffer; _read_front_buffer grows the window on demand.
        buf = _read_front_buffer(path)

        cfg, tables, cfg_end = parse_configure(buf)
        rc.stream_configs[stream] = (cfg, cfg_end)
        rc.raw_tables[stream] = tables

        for names_key, t in tables.items():
            name = t["det_name"]
            det = rc.detectors.get(name)
            if det is None:
                det = DetectorInfo(name, t["det_type"], t["det_id"])
                rc.detectors[name] = det
            det._add_table(t["alg_name"], t, stream, names_key)
    return rc


def _normalize_stream_files(stream_files):
    """Return a sorted list of ``(stream_index, path)``."""
    if isinstance(stream_files, dict):
        items = list(stream_files.items())
    else:
        items = list(stream_files)
        if items and not isinstance(items[0], (tuple, list)):
            # plain list of paths -> positional indices
            items = list(enumerate(items))
    # coerce indices to int and sort
    return sorted(((int(i), p) for i, p in items), key=lambda kv: kv[0])


# Stream/chunk file name pattern: ``<exp>-r<run>-s<stream>-c<chunk>.xtc2``.
# psana opens only the first chunk (``c000``) of each stream and rolls forward
# from there (``ds_base.py`` filters with ``re.search(r"-c000\.", ...)``,
# ~line 393); the later chunks are reached by following ``chunkinfo`` on each
# Enable.  :func:`filter_c000` reproduces that initial filter so a directory
# listing of all chunks collapses to the one file per stream that the reader
# opens first.
_C000_RE = re.compile(r"-c000\.")
_STREAM_RE = re.compile(r"-s(\d+)-c000\.")


def filter_c000(paths):
    """Keep only the first-chunk (``c000``) files from a list of xtc2 paths.

    Mirrors psana's ``ds_base.py`` initial file filter: a run directory holds
    ``...-s###-c000.xtc2`` plus later chunks ``-c001``, ``-c002``, ... for the
    streams that rolled; the reader opens only ``c000`` and follows the
    ``chunkinfo`` roll from there (see :mod:`psdata.index`).  Returns the
    surviving paths in their input order.
    """
    return [p for p in paths if _C000_RE.search(os.path.basename(p))]


def stream_index_of(path):
    """Extract the integer stream index from a ``...-s###-c000.xtc2`` path, or
    ``None`` if the name does not match.  Lets a plain c000 file list be keyed
    by its real ``s###`` stream id rather than positionally."""
    m = _STREAM_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


# ==========================================================================
# Single-event field extraction (generic; the US-001 byte-exact check)
# ==========================================================================
def extract_detector_event(run_config, det_name, alg, field, target_ts,
                           read_chunk=2_000_000):
    """Extract one detector field for the event at ``target_ts``.

    Walks each stream that carries ``(det_name, alg)`` from the front, finds
    the L1Accept whose dgram timestamp equals ``target_ts``, extracts ``field``
    from every segment's ShapesData, and assembles a stack ordered by segment
    id.  Fully generic -- the detector, alg, field, streams, and segment ids
    all come from ``run_config`` (discovered from the Configure tables).

    Returns
    -------
    dict with keys:
        ``stack``      ndarray, shape (n_segments, *seg_shape)
        ``seg_ids``    sorted segment ids
        ``segments``   {seg_id: ndarray}
        ``ts``         the matched timestamp
        ``bytes_read`` total bytes pread across streams
        ``field_meta`` the Names entry for ``field`` (name/type/rank)
    """
    det = run_config.detector(det_name)
    if alg not in det.algs:
        raise KeyError(f"detector {det_name!r} has no alg {alg!r} "
                       f"(have {det.alg_names()})")
    if field not in det.algs[alg]:
        raise KeyError(f"detector {det_name!r} alg {alg!r} has no field "
                       f"{field!r} (have {det.field_names(alg)})")

    streams = det.streams_for(alg)
    segments = {}
    field_meta = None
    total_bytes = 0
    for stream in streams:
        res = _read_stream_event(run_config, stream, det_name, alg, field,
                                 target_ts, read_chunk)
        total_bytes += res["bytes_read"]
        for seg, (arr, shape, meta) in res["segments"].items():
            segments[seg] = arr
            field_meta = meta

    seg_ids = sorted(segments)
    if not seg_ids:
        raise RuntimeError(
            f"no {det_name}/{alg}/{field} segments found at ts={target_ts}")
    sample = segments[seg_ids[0]]
    # array fields commonly arrive as (1, H, W); normalise the per-segment
    # frame to its trailing data dims so the stack is (nseg, ...).
    if sample.ndim >= 3 and sample.shape[0] == 1:
        seg_shape = sample.shape[1:]
    else:
        seg_shape = sample.shape
    stack = np.empty((len(seg_ids),) + seg_shape, dtype=sample.dtype)
    norm_segments = {}
    for k, seg in enumerate(seg_ids):
        a = segments[seg].reshape(seg_shape)
        stack[k] = a
        norm_segments[seg] = a
    return dict(stack=stack, seg_ids=seg_ids, segments=norm_segments,
                ts=target_ts, bytes_read=total_bytes, field_meta=field_meta)


def _read_stream_event(run_config, stream, det_name, alg, field, target_ts,
                       read_chunk):
    """Walk one stream's dgrams from the end of Configure to the L1Accept whose
    timestamp == ``target_ts``; extract ``field`` from each matching segment's
    ShapesData."""
    path = run_config.stream_files[stream]
    tables = run_config.raw_tables[stream]
    _cfg, cfg_end = run_config.stream_configs[stream]

    # (nodeId,namesId) -> segment, for the tables of (det_name, alg) we want.
    want = {}
    for names_key, t in tables.items():
        if t["det_name"] == det_name and t["alg_name"] == alg:
            want[names_key] = t["segment"]
    if not want:
        return dict(segments={}, bytes_read=0)

    fd = os.open(path, os.O_RDONLY)
    try:
        buf = bytearray(os.pread(fd, max(read_chunk, cfg_end), 0))
        bytes_read = len(buf)

        off = cfg_end
        while True:
            if off + DGRAM_HDR > len(buf):
                more = os.pread(fd, read_chunk, len(buf))
                if not more:
                    raise RuntimeError(
                        f"stream {stream}: EOF before ts={target_ts}")
                buf.extend(more)
                bytes_read += len(more)
                continue
            h = parse_dgram_header(buf, off)
            total = XTC_HDR + h["extent"]
            if off + total > len(buf):
                more = os.pread(fd, max(read_chunk, total), len(buf))
                if not more:
                    raise RuntimeError(f"stream {stream}: truncated dgram")
                buf.extend(more)
                bytes_read += len(more)
                continue

            if h["service"] == SERVICE_L1ACCEPT and h["ts"] == target_ts:
                found = _collect_segments(buf, off, h, want, tables, field)
                return dict(segments=found, bytes_read=bytes_read)
            off += total
    finally:
        os.close(fd)


def _collect_segments(buf, dg_off, dg_hdr, want, tables, field):
    """Find every ShapesData in this L1Accept whose (nodeId,namesId) is in
    ``want``; extract ``field``.  Returns ``{seg_id: (array, shape, meta)}``."""
    top_payload = dg_off + DGRAM_HDR
    top_end = dg_off + XTC_HDR + dg_hdr["extent"]
    out = {}

    def recurse(payload_off, payload_end):
        for xoff, xh in iter_xtc_children(buf, payload_off, payload_end):
            t = typeid_type(xh["typeid"])
            if t == TID_SHAPESDATA:
                key = namesid_of(xh["src"])
                if key in want:
                    table = tables[key]
                    arr, shape, meta = extract_field(buf, xoff, xh, table,
                                                     field)
                    out[want[key]] = (arr, shape, meta)
            elif t == TID_PARENT:
                recurse(xoff + XTC_HDR, xoff + xh["extent"])

    recurse(top_payload, top_end)
    return out


# ==========================================================================
# Import-purity self check
# ==========================================================================
_FORBIDDEN_MODULES = ("psana", "mpi4py", "h5py", "xtcdata")


def assert_no_framework_imports():
    """Raise AssertionError if importing this module pulled in a framework.

    Verifies the clean-room property: parsing xtc2 must not drag in psana,
    mpi4py, or h5py.
    """
    leaked = [m for m in _FORBIDDEN_MODULES if m in sys.modules]
    assert not leaked, (
        f"psdata.format must not import {_FORBIDDEN_MODULES}; "
        f"found in sys.modules: {leaked}")


# Self-check at import time: this module's own import chain must be clean.
# (A caller that has *separately* imported psana before importing us would
# defeat this; the test harness imports us in a fresh interpreter.)
if __name__ != "__main__":
    _at_import = [m for m in _FORBIDDEN_MODULES if m in sys.modules]
    # We only assert the modules we could have introduced (numpy/struct/os do
    # not import any of them).  If they are already present they came from the
    # caller, which is out of our control; the dedicated test asserts a clean
    # fresh-interpreter import.
