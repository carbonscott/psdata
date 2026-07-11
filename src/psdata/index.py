#!/usr/bin/env python3
"""psdata.index -- random-access-by-event index and read-by-timestamp.

This is the US-003 layer on top of the US-001 parse core
(:mod:`psdata.format`) and the US-002 event-assembly layer
(:mod:`psdata.stream`).  It builds a compact ``timestamp -> {stream:
(offset, size)}`` index by scanning only the **small SMD files**
(``smalldata/{exp}-r{run:04d}-s###-c000.smd.xtc2``), then serves an arbitrary
event by timestamp with a single ``os.pread`` per contributing stream into the
GB-scale bigdata files -- **without** sequentially scanning them.

How it works
------------
Every SMD file carries, on each ``L1Accept`` dgram, a ``smdinfo`` detector
(``det_type='offset'``, ``alg='offsetAlg'``) with two rank-0 ``uint64`` fields:

  * ``intOffset``    -- byte offset of the matching bigdata dgram, and
  * ``intDgramSize`` -- its full on-disk size (``XTC_HDR + extent``),

both pointing into the **bigdata file of the same stream index** (psana
indexes SMD file and bigdata file 1:1 -- ``event_manager.py`` reads
``d.smdinfo[0].offsetAlg.intOffset`` / ``.intDgramSize`` ~lines 106-107 and
opens ``self.dm.xtc_files[i_smd]`` for stream ``i_smd``).  So scanning the tiny
SMD files yields, per event timestamp, exactly where to ``pread`` each big
dgram.

An event is assembled from the ``pread``-ed bigdata dgrams using the same
machinery as the streaming path (:func:`psdata.stream._index_dgram` +
:class:`psdata.stream.Event`), so a randomly-read event is byte-identical to
the same event obtained by streaming.

Multi-chunk runs (US-004)
-------------------------
A long run's bigdata is split into chunk files ``...-s###-c000.xtc2``,
``-c001``, ``-c002``, ...; each chunk file's ``intOffset`` values restart at 0,
so an offset is meaningful only *relative to the chunk file it indexes*.  psana
opens ``c000`` and, on every ``Enable`` transition, reads a ``chunkinfo``
container (``det_name='chunkinfo'``, ``alg='chunkinfo'``; fields ``filename``
and ``chunkid``) and, when ``chunkid`` advances, reopens the bigdata fd on the
new file (``event_manager.py`` ~208-254, ``_open_new_bd_file``).

This module replicates that roll while scanning the SMD: each per-event index
entry records not just ``(offset, size)`` but the **bigdata chunk path** the
offset indexes, so :meth:`RunIndex.read_event` ``os.pread`` reads the correct
chunk file -- byte-identical to psana across the boundary.

Building the index WITHOUT SMD
------------------------------
The SMD sidecars are only an *index artifact*: every ``(intOffset,
intDgramSize)`` they record is just the running byte cursor and on-disk size of
the matching bigdata dgram (the DAQ/`smdwriter` write them in lockstep with the
bigdata).  So they are not *required* -- :meth:`RunIndex.build_from_bigdata`
rebuilds the identical index by walking the bigdata dgram **headers** directly
(reading each 24-byte header and seeking past the GB-scale payloads), the
``smdwriter`` algorithm in pure Python.  By default the index it produces is
byte-identical to one built from SMD -- both are clamped to the *canonical*
event set (events carrying the timing/master stream); the raw walk can also
recover the ragged DAQ end-of-run tail via ``include_shutdown_tail=True`` (see
:meth:`RunIndex.build_from_bigdata`).  So :func:`build_index` defaults to
``source="auto"``: it uses the SMD sidecars as a fast cache *when present*, and
falls back to the bigdata scan when they are absent -- the random-access
capability never *depends* on an artifact owned by the psana/xtcdata toolchain.

Like the rest of ``psdata``, this module imports **no** psana / mpi4py / h5py
-- only the standard library and numpy.
"""

import base64
import bisect
import hashlib
import json
import os
import re
import struct
import time

import numpy as np

from . import format as _f
from . import stream as _s

# The bookkeeping detector that carries the per-event bigdata offset/size.
_SMDINFO_DET = "smdinfo"
_SMDINFO_ALG = "offsetAlg"
_OFFSET_FIELD = "intOffset"
_SIZE_FIELD = "intDgramSize"

# The bookkeeping detector that carries the chunk-roll info on Enable dgrams.
_CHUNKINFO_DET = "chunkinfo"
_CHUNKINFO_ALG = "chunkinfo"
_CHUNK_FILENAME_FIELD = "filename"
_CHUNK_ID_FIELD = "chunkid"

# Env (slow-data) transition services -> the env store they feed.  Mirrors
# psana's ``EnvStoreManager.update_by_event``: SlowUpdate feeds the ``epics``
# store; BeginStep and BeginRun feed the ``scan`` store (BeginRun carries no
# scan container -- the env-store backward scan skips it at lookup time).  The
# per-dgram (ts, path, offset, size) is bucketed while the L1Accept walk already
# reads every dgram header, so this costs no extra I/O; the payload is decoded
# lazily at lookup (see :mod:`psdata.envstore`).
_ENV_SERVICE_STORE = {
    _f.SERVICE_SLOWUPDATE: "epics",   # service 10
    _f.SERVICE_BEGINSTEP: "scan",     # service 6
    _f.SERVICE_BEGINRUN: "scan",      # service 4
}


def _decode_chunk_filename(arr):
    """``chunkinfo.filename`` is a rank-1 CHARSTR (uint8) field, null-padded.
    Return it as a ``str`` (the bare basename of the next bigdata chunk)."""
    return _f.decode_charstr(arr)


# ==========================================================================
# Safe, versioned on-disk index format (IDX-02)
# ==========================================================================
# The persisted index is a shareable artifact re-read off a multi-user
# analysis filesystem, so its on-disk format MUST NOT be a pickle:
# ``pickle.load`` executes arbitrary code baked into the file (an RCE vector --
# any user who can write the directory a victim's job loads from would get code
# execution in that job), and a bare pickle also has no magic, no format
# version, and no integrity check (a truncated/corrupt/drifted file crashes
# inscrutably, or silently loads wrong data).
#
# Instead ``RunIndex.save``/``load`` write a self-describing container:
#   magic + version + a JSON header (checksum) + a JSON payload,
# where the payload is ``_persist_state()`` rendered by :func:`_index_encode`
# and read back by :func:`_index_decode`.  The decoder dispatches on a FIXED
# whitelist of type tags and can reconstruct ONLY plain data (dict/list/tuple/
# set/str/int/float/bool/None, plus numpy dtype/ndarray) and the three psdata
# config classes (RunConfig/DetectorInfo/FieldInfo) -- there is no tag that
# imports or calls an arbitrary object, so loading a file can never execute
# embedded code the way ``pickle.load`` does.  The encoder REFUSES any type it
# does not recognise (rather than falling back to pickle or to a numpy object
# array), which also forecloses the "ragged list-of-lists silently becomes an
# object/pickled array" trap: the ragged per-event ``entries`` are encoded as
# explicit tagged lists/dicts, never handed to ``np.array``/``np.savez``.
_INDEX_MAGIC = b"PSDATIDX"          # 8 bytes; identifies a psdata index file
_INDEX_FORMAT_VERSION = 2           # bump on ANY incompatible layout change
# v1 -> v2 (IDX-03): the payload gained a ``portability`` record and stores its
# file paths RELATIVE to a common root (relocatable index).  ``load`` still
# reads a v1 payload (no ``portability`` key -> absolute paths, as before), so
# older-but-magic indexes are not orphaned; only the on-disk layout changed.
_INDEX_SUPPORTED_VERSIONS = frozenset({1, 2})
_INDEX_CHECKSUM_ALGO = "sha256"     # over the payload bytes, checked on load
# The ONLY checksum algorithms load() will honor -- an explicit allowlist, not
# ``hashlib.algorithms_available``.  That set also contains variable-length XOFs
# (e.g. ``shake_128`` / ``shake_256``) whose ``hexdigest()`` REQUIRES a length
# argument, so honoring one would raise an uncaught ``TypeError`` instead of a
# clean integrity ``ValueError``.  Restrict to the fixed-length digest we
# actually write; add an entry here (never a blanket ``algorithms_available``)
# if a future format writes a different one.
_INDEX_ALLOWED_CHECKSUM_ALGOS = frozenset({"sha256"})


def _index_encode(obj):
    """Encode ``obj`` (a ``_persist_state`` dict) to a JSON-safe, type-tagged
    document.  Strings are pooled (repeated chunk paths cost one entry).  Raises
    ``TypeError`` on any type outside the supported set -- so nothing is ever
    silently dropped, coerced to a numpy object array, or pickled."""
    pool = []
    interned = {}

    def sref(s):
        i = interned.get(s)
        if i is None:
            i = len(pool)
            pool.append(s)
            interned[s] = i
        return {"t": "s", "i": i}

    def enc(o):
        # bool BEFORE int (bool is an int subclass); None/bool/int/float are
        # JSON natives and pass through unwrapped.
        if o is None or isinstance(o, bool):
            return o
        if isinstance(o, int):
            return o
        if isinstance(o, float):
            return o
        if isinstance(o, str):
            return sref(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, dict):
            return {"t": "dict", "d": [[enc(k), enc(v)] for k, v in o.items()]}
        if isinstance(o, list):
            return {"t": "list", "d": [enc(x) for x in o]}
        if isinstance(o, tuple):
            return {"t": "tuple", "d": [enc(x) for x in o]}
        if isinstance(o, frozenset):
            return {"t": "frozenset", "d": [enc(x) for x in o]}
        if isinstance(o, set):
            return {"t": "set", "d": [enc(x) for x in o]}
        if isinstance(o, np.dtype):
            return {"t": "dtype", "v": o.str}
        if isinstance(o, np.ndarray):
            return {"t": "ndarray", "dtype": o.dtype.str,
                    "shape": list(o.shape),
                    "b64": base64.b64encode(
                        np.ascontiguousarray(o).tobytes()).decode("ascii")}
        if isinstance(o, _f.RunConfig):
            return {"t": "RunConfig", "s": enc(o.__dict__)}
        if isinstance(o, _f.DetectorInfo):
            return {"t": "DetectorInfo", "s": enc(o.__dict__)}
        if isinstance(o, _f.FieldInfo):
            # reconstructed via its constructor (recomputes np_dtype), so only
            # the three defining scalars are stored.
            return {"t": "FieldInfo", "name": enc(o.name),
                    "type_code": int(o.type_code), "rank": int(o.rank)}
        raise TypeError(
            "psdata index save: refusing to serialize unsupported type %r -- a "
            "safe self-describing index stores only plain data and the psdata "
            "config classes (no arbitrary objects, no pickle)." % (type(o),))

    return {"strings": pool, "root": enc(obj)}


def _index_decode(doc):
    """Inverse of :func:`_index_encode`.  Dispatches on a FIXED whitelist of
    type tags; an unknown tag (or a malformed node) raises ``ValueError``.  The
    only classes it can ever instantiate are the three psdata config classes --
    there is NO tag that imports or calls an arbitrary object, so decoding a
    hostile file cannot execute embedded code."""
    if (not isinstance(doc, dict) or "strings" not in doc
            or "root" not in doc):
        raise ValueError("psdata index: malformed payload envelope")
    strings = doc["strings"]
    if not isinstance(strings, list) or not all(
            isinstance(s, str) for s in strings):
        raise ValueError("psdata index: malformed string pool")

    def dec(n):
        # JSON natives pass straight through.
        if n is None or isinstance(n, bool) or isinstance(n, (int, float)):
            return n
        if not isinstance(n, dict):
            raise ValueError(
                "psdata index: unexpected JSON node of type %r" % (type(n),))
        t = n.get("t")
        if t == "s":
            i = n["i"]
            if not isinstance(i, int) or not (0 <= i < len(strings)):
                raise ValueError("psdata index: bad string reference")
            return strings[i]
        if t == "dict":
            return {dec(k): dec(v) for k, v in n["d"]}
        if t == "list":
            return [dec(x) for x in n["d"]]
        if t == "tuple":
            return tuple(dec(x) for x in n["d"])
        if t == "set":
            return set(dec(x) for x in n["d"])
        if t == "frozenset":
            return frozenset(dec(x) for x in n["d"])
        if t == "dtype":
            return np.dtype(n["v"])
        if t == "ndarray":
            arr = np.frombuffer(base64.b64decode(n["b64"]),
                                dtype=np.dtype(n["dtype"]))
            return arr.reshape(n["shape"]).copy()
        if t == "RunConfig":
            obj = _f.RunConfig.__new__(_f.RunConfig)
            obj.__dict__.update(dec(n["s"]))
            return obj
        if t == "DetectorInfo":
            obj = _f.DetectorInfo.__new__(_f.DetectorInfo)
            obj.__dict__.update(dec(n["s"]))
            return obj
        if t == "FieldInfo":
            return _f.FieldInfo(dec(n["name"]), n["type_code"], n["rank"])
        raise ValueError(
            "psdata index: unknown/unsafe type tag %r in payload" % (t,))

    return dec(doc["root"])


# ==========================================================================
# Portable (relocatable) path serialization -- IDX-03
# ==========================================================================
# The build records every xtc/bigdata file by ABSOLUTE path (e.g.
# ``/sdf/data/lcls/ds/<exp>/xtc/...-c000.xtc2``).  psdata's headline feature is
# a persisted, SHAREABLE index artifact, so it must survive being copied to
# another mount, host, or container where the same data lives under a DIFFERENT
# prefix -- a hard-coded absolute path would then load a wrong/absent file.
#
# Only the SERIALIZED form is made relocatable; the in-memory index is never
# touched.  On :func:`_index_encode`, every string is pooled once, and in this
# domain every ABSOLUTE pool string is a data-file path (detector names, ids,
# alg/field names and dtype strings are never absolute).  We compute the common
# parent directory ``root`` of those paths, rewrite each to ``relpath(p, root)``
# in the pool, and record ``root`` + the rewritten pool indices under a
# ``portability`` key.  On load the indices are re-joined to a base directory
# (see :func:`_index_resolve_paths`) BEFORE decoding -- so a load from the
# ORIGINAL location reconstructs byte-identical absolute paths (downstream reads
# unchanged) and a relocated load resolves to the data's new home.

def _index_portablize_paths(doc):
    """In place: relativize the absolute path strings in ``doc['strings']`` and
    attach ``doc['portability'] = {"root": <common dir | None>, "rel": [i,...]}``
    naming the rewritten pool indices.  A no-op (``root=None, rel=[]``) when the
    pool holds no absolute path.  ``doc`` is the throwaway encoder output, so the
    live index's own strings are never mutated."""
    strings = doc["strings"]
    abs_idx = [i for i, s in enumerate(strings) if os.path.isabs(s)]
    if not abs_idx:
        doc["portability"] = {"root": None, "rel": []}
        return doc
    dirs = sorted({os.path.dirname(strings[i]) for i in abs_idx})
    root = dirs[0] if len(dirs) == 1 else os.path.commonpath(dirs)
    rel = []
    for i in abs_idx:
        # Defensive: only paths genuinely under ``root`` are relativized (an
        # unrelated absolute string, should one ever appear, is left absolute
        # and still loads verbatim -- ``_index_resolve_paths`` skips it).
        if os.path.commonpath([root, strings[i]]) == root:
            strings[i] = os.path.relpath(strings[i], root)
            rel.append(i)
    doc["portability"] = {"root": root, "rel": rel}
    return doc


def _index_resolve_paths(doc, index_path, dir=None):
    """In place inverse of :func:`_index_portablize_paths`: re-absolutize the
    relativized pool strings against a base directory, chosen by this
    precedence, then leave ``doc`` ready for :func:`_index_decode`.

      * explicit ``dir=`` override -- STRICT: every relativized file must exist
        under ``dir`` or a clear ``FileNotFoundError`` names the first miss.
        This is how an index built under ``/sdf/...`` on host A is loaded under
        ``/data/...`` on host B.
      * else the stored ``root`` if the files are found there -- the ORIGINAL
        location; reproduces the exact absolute paths byte-for-byte.
      * else the index file's OWN directory if the files are found beside it --
        an index shipped together with its data.
      * else fall back to the stored ``root`` (reproduce the original absolute
        paths).  This branch NEVER raises, so an index whose data is momentarily
        offline still loads and a genuine miss surfaces at read time exactly as
        it did before portable paths -- and no wrong file is ever opened.

    A v1 payload has no ``portability`` record (its paths are absolute): nothing
    to resolve, and an explicit ``dir=`` is refused with an actionable error."""
    port = doc.get("portability")
    if not isinstance(port, dict):
        if dir is not None:
            raise ValueError(
                "%r: this psdata index predates portable paths (it stores "
                "absolute file paths) and cannot be relocated with dir=%r -- "
                "rebuild it with the current psdata to get a relocatable index."
                % (index_path, dir))
        return doc
    strings = doc["strings"]
    rel = port.get("rel") or []
    if not rel:
        return doc
    root = port.get("root")

    def _first_missing(base):
        """Pool index of the first relativized file absent under ``base``, or
        ``None`` if they all exist there."""
        for i in rel:
            if not os.path.exists(os.path.normpath(os.path.join(base, strings[i]))):
                return i
        return None

    idx_dir = os.path.dirname(os.path.abspath(index_path))
    if dir is not None:
        base = os.path.abspath(dir)
        miss = _first_missing(base)
        if miss is not None:
            missing = os.path.normpath(os.path.join(base, strings[miss]))
            raise FileNotFoundError(
                "psdata index load: dir=%r does not hold this index's data -- "
                "expected file %r is missing.  Pass the directory that contains "
                "the run's xtc2 files (the index stores them relative to %r)."
                % (dir, missing, root))
        chosen = base
    else:
        chosen = None
        for cand in (root, idx_dir):
            if cand is not None and _first_missing(cand) is None:
                chosen = cand
                break
        if chosen is None:
            # Data not found at the original root nor beside the index: reproduce
            # the ORIGINAL absolute paths so the load still succeeds; any real
            # miss surfaces at read time (os.open), never as a wrong file.
            chosen = root if root is not None else idx_dir
    for i in rel:
        strings[i] = os.path.normpath(os.path.join(chosen, strings[i]))
    return doc


# ==========================================================================
# Locating the smdinfo table in an SMD stream's Configure
# ==========================================================================
def _find_smdinfo_key(tables):
    """Return the ``(nodeId, namesId)`` of the ``smdinfo`` Names table in one
    SMD stream's Configure ``tables`` (``{(nodeId,namesId): names_table}``), or
    ``None`` if this stream has no smdinfo (e.g. it is a bigdata, not an SMD,
    file)."""
    for key, t in tables.items():
        if t["det_name"] == _SMDINFO_DET and t["alg_name"] == _SMDINFO_ALG:
            if (_OFFSET_FIELD in {n["name"] for n in t["names"]}
                    and _SIZE_FIELD in {n["name"] for n in t["names"]}):
                return key
    return None


def _read_smdinfo(buf, dg_off, dg_hdr, smdinfo_key, table):
    """Find the ``smdinfo`` ShapesData in one SMD dgram and return
    ``(intOffset, intDgramSize)`` as Python ints, or ``None`` if absent."""
    top_payload = dg_off + _f.DGRAM_HDR
    top_end = dg_off + _f.XTC_HDR + dg_hdr["extent"]
    found = [None]

    def recurse(payload_off, payload_end):
        if found[0] is not None:
            return
        for xoff, xh in _f.iter_xtc_children(buf, payload_off, payload_end):
            t = _f.typeid_type(xh["typeid"])
            if t == _f.TID_SHAPESDATA:
                if _f.namesid_of(xh["src"]) == smdinfo_key:
                    off_arr, _s1, _m1 = _f.extract_field(
                        buf, xoff, xh, table, _OFFSET_FIELD)
                    sz_arr, _s2, _m2 = _f.extract_field(
                        buf, xoff, xh, table, _SIZE_FIELD)
                    found[0] = (int(off_arr), int(sz_arr))
                    return
            elif t == _f.TID_PARENT:
                recurse(xoff + _f.XTC_HDR, xoff + xh["extent"])
                if found[0] is not None:
                    return

    recurse(top_payload, top_end)
    return found[0]


# ==========================================================================
# The random-access index
# ==========================================================================
class RunIndex:
    """A timestamp -> ``{stream: (bigdata_offset, bigdata_size)}`` index for a
    run, built by scanning only the small SMD files.

    Attributes
    ----------
    timestamps : list[int]
        L1Accept event timestamps in ascending order (the index of an event in
        this list is its event number / position ``k``).
    entries : list[dict]
        ``entries[k]`` is ``{stream: (chunk_path, offset, size)}`` -- which
        bigdata streams carry event ``k``, the **chunk file** that event's
        dgram lives in (``c000``/``c001``/... after the roll), and the offset
        and size within it.  Only streams that contributed a segment to event
        ``k`` are present (a stream missing for an event is simply absent, so
        the US-002 missing-segment rule applies).
    bd_files : dict
        ``{stream: bigdata_c000_path}`` -- the first (``c000``) bigdata file of
        each stream (from the run config, same stream index as the SMD file).
        Later chunks are recorded per-event in ``entries`` and in
        ``chunk_files``.
    chunk_files : dict
        ``{stream: [chunk_path, ...]}`` -- the ordered bigdata chunk files each
        stream rolls through (``[c000]`` for a single-chunk stream, ``[c000,
        c001, ...]`` for a multi-chunk one).
    run_config : psdata.format.RunConfig
        The run's discovered detector / segment configuration (shared with the
        streaming path so assembled events are identical).
    build_seconds : float
        Wall-clock time spent building the index (SMD scan only).
    smd_bytes_read : int
        Total bytes read from the SMD files during the build.
    multichunk_streams : set[int]
        Streams that rolled through more than one chunk file (followed the
        ``chunkinfo`` roll).  Empty for a clean single-chunk run.
    """

    def __init__(self, run_config):
        self.run_config = run_config
        self.timestamps = []          # ascending L1Accept ts
        self.entries = []             # entries[k] = {stream:(path,offset,size)}
        # Env (slow-data) records, ADDITIVE to the canonical L1Accept index and
        # never merged into timestamps/entries: {store_name: {stream: [(ts,
        # path, offset, size), ...]}} ascending by ts, store_name in
        # {"epics","scan"}.  ``path`` is the file the env dgram bytes live in
        # (the SMD sidecar when scanned from SMD, the bigdata chunk from
        # bigdata).  Consumed by :mod:`psdata.envstore` for as-of lookups.
        self.env_records = {}
        self.bd_files = dict(run_config.stream_files)
        self.chunk_files = {s: [p] for s, p in run_config.stream_files.items()}
        self.build_seconds = 0.0
        self.smd_bytes_read = 0
        self.multichunk_streams = set()
        self.scan_source = "smd"      # "smd" (sidecar cache) | "bigdata" (no SMD)
        self.scan_bytes_read = 0      # bytes the index scan read (smd OR bigdata)
        self.include_shutdown_tail = False  # True iff the raw end-of-run tail
        #   (events lacking the timing/master stream; pulseId=None) is kept;
        #   default False clamps to the canonical (SMD-equivalent) event set.
        # private: open bigdata fds, lazily, cached per chunk path
        self._bd_fds = {}
        self._ts_to_k = None          # built lazily for read_event(ts)

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, smd_files, run_config, smd_read_chunk=1 << 22):
        """Build a :class:`RunIndex` by scanning the SMD files only.

        Parameters
        ----------
        smd_files : dict | sequence
            The run's SMD files, same forms as
            :func:`psdata.format.discover` accepts (``{stream: path}``,
            ``(stream, path)`` pairs, or a plain path list keyed positionally).
            The stream index of each SMD file **must** match the bigdata stream
            index in ``run_config`` (they share ``s###`` by construction).
        run_config : psdata.format.RunConfig
            The run config discovered from the *bigdata* stream files (its
            ``stream_files`` map gives the bigdata path per stream).
        smd_read_chunk : int
            Read granularity for the growing SMD window.

        Returns
        -------
        RunIndex
        """
        items = _f._normalize_stream_files(smd_files)
        idx = cls(run_config)

        t0 = time.monotonic()
        # per-stream ordered lists of (ts, chunk_path, offset, size)
        per_stream = {}
        for stream, smd_path in items:
            # the bigdata c000 path this SMD stream indexes into (chunkinfo
            # filenames on Enable name the later chunks, in the same directory).
            bd_c000 = run_config.stream_files.get(stream)
            env_recs = {}
            recs, nbytes, chunks = _scan_smd_stream(
                smd_path, bd_c000, smd_read_chunk, env_recs)
            per_stream[stream] = recs
            idx.smd_bytes_read += nbytes
            idx.chunk_files[stream] = chunks
            if len(chunks) > 1:
                idx.multichunk_streams.add(stream)
            for store, erecs in env_recs.items():
                idx.env_records.setdefault(store, {})[stream] = erecs

        # SMD is already the canonical reference set -- the offline
        # smdwriter pre-truncated the end-of-run shutdown tail, so the clamp
        # (which exists only to trim the bigdata walk's tail) has nothing to
        # remove here.  Pass include_shutdown_tail=True so the SMD path is
        # provably unchanged: it never re-derives the timing predicate to
        # second-guess SMD, and the (still-open) damaged-data policy is not
        # silently decided on this path.
        idx._merge_streams(per_stream, include_shutdown_tail=True)
        idx.build_seconds = time.monotonic() - t0
        idx.scan_source = "smd"
        idx.scan_bytes_read = idx.smd_bytes_read
        return idx

    @classmethod
    def build_from_bigdata(cls, run_config, include_shutdown_tail=False):
        """Build a :class:`RunIndex` WITHOUT any SMD file, by walking the
        bigdata chunk files' dgram headers directly.

        This is the offline ``smdwriter`` algorithm reimplemented in psdata's
        own pure-Python parser: for each stream, walk its bigdata chunk files
        (``c000``, ``c001``, ...) from offset 0, record the running byte cursor
        at every ``L1Accept`` (the same ``intOffset`` the SMD file would store)
        and its on-disk size ``XTC_HDR + extent`` (the same ``intDgramSize``),
        advancing the cursor across every dgram (transitions included).

        Event set: by default (``include_shutdown_tail=False``) the index is
        clamped to the *canonical* event set -- the events that carry the
        timing/master stream (``det_type='ts'``) and so have a ``pulseId`` --
        which is byte-for-byte identical to one :meth:`build` produces from the
        SMD sidecars (the offline ``smdwriter`` writes exactly that set), but
        needs no ``.smd.xtc2`` artifact.  Pass ``include_shutdown_tail=True`` to
        instead keep every physical L1Accept on disk, including the ragged DAQ
        end-of-run tail: events some detector streams kept writing after the
        timing/master streams closed.  Those tail events lack the timing stream,
        so ``Event.pulseId`` is ``None`` and -- once a detector stream also
        closes -- they go partial (``Event.stack`` then returns ``None`` by the
        missing-segment rule); they are a strict superset of the canonical set.

        Only each dgram's 24-byte header is read; the (GB-scale) payloads are
        seeked past, so the cost is one ``pread`` per dgram (~thousands per
        stream for large-dgram detectors), not the file size.  Persist the built
        index with :meth:`save` so this one-time scan is never repeated.

        The returned index sets ``scan_source = "bigdata"`` and
        ``smd_bytes_read = 0`` (no SMD I/O); ``scan_bytes_read`` is the total of
        the 24-byte headers walked.
        """
        idx = cls(run_config)
        idx.scan_source = "bigdata"
        idx.include_shutdown_tail = include_shutdown_tail
        t0 = time.monotonic()
        # per-stream ordered lists of (ts, chunk_path, offset, size)
        per_stream = {}
        for stream, bd_c000 in _f._normalize_stream_files(
                run_config.stream_files):
            env_recs = {}
            recs, nbytes, chunks = _scan_bigdata_stream(bd_c000, env_recs)
            per_stream[stream] = recs
            idx.scan_bytes_read += nbytes
            idx.chunk_files[stream] = chunks
            if len(chunks) > 1:
                idx.multichunk_streams.add(stream)
            for store, erecs in env_recs.items():
                idx.env_records.setdefault(store, {})[stream] = erecs

        idx._merge_streams(per_stream,
                           include_shutdown_tail=include_shutdown_tail)
        idx.build_seconds = time.monotonic() - t0
        return idx

    def _timing_streams(self):
        """Stream indices that carry the timing/master detector (``det_type
        'ts'``) -- the streams whose presence makes an event *canonical* (gives
        it a ``pulseId``).  Empty if the run declares no timing detector, in
        which case no clamp is applied and every assembled event is kept.

        Mirrors how :attr:`psdata.stream.Event.pulseId` sources its value (the
        ``raw`` alg of the ``det_type='ts'`` detector, ``stream.py``
        ``_read_pulse_id``), so the clamp drops exactly the events whose
        ``pulseId`` would be ``None`` for want of that stream."""
        rc = self.run_config
        names = rc.find_detector_by_type(_s._TIMING_DET_TYPE)
        if not names:
            return frozenset()
        det = rc.detector(names[0])
        alg = "raw" if "raw" in det.algs else None
        if alg is None:
            return frozenset()
        return frozenset(det.streams_for(alg))

    def _merge_streams(self, per_stream, include_shutdown_tail=False):
        """Combine the per-stream ``(ts, chunk_path, off, size)`` records into
        the unified ascending ``timestamps`` / ``entries`` index by
        exact-timestamp merge (an event's identity is its 64-bit ts, shared
        across streams).

        By default the merge is clamped to the *canonical* event set: an event
        is kept only if it carries the timing/master stream (see
        :meth:`_timing_streams`), which matches the SMD/``smdwriter`` event set
        exactly and excludes the ragged DAQ end-of-run *shutdown tail* (events
        some detector streams kept writing after the timing/master streams
        closed; ``pulseId=None``).  Pass ``include_shutdown_tail=True`` to keep
        that tail -- the full set of physical L1Accepts on disk."""
        # gather the union of timestamps, then for each ts collect the streams
        merged = {}   # ts -> {stream: (chunk_path, off, size)}
        for stream, recs in per_stream.items():
            for ts, chunk_path, off, size in recs:
                merged.setdefault(ts, {})[stream] = (chunk_path, off, size)
        timing_streams = (frozenset() if include_shutdown_tail
                          else self._timing_streams())
        for ts in sorted(merged):
            entry = merged[ts]
            if timing_streams and not (timing_streams & entry.keys()):
                continue   # shutdown-tail event: lacks the timing/master stream
            self.timestamps.append(ts)
            self.entries.append(entry)

    # -- lookup ------------------------------------------------------------
    @property
    def n_events(self):
        return len(self.timestamps)

    def _position_of(self, ts):
        """Return the event position ``k`` whose timestamp is exactly ``ts``,
        or raise ``KeyError``."""
        i = bisect.bisect_left(self.timestamps, ts)
        if i < len(self.timestamps) and self.timestamps[i] == ts:
            return i
        hint = ""
        if not self.include_shutdown_tail:
            hint = (" -- this index is clamped to the canonical event set; the "
                    "timestamp may belong to a shutdown-tail event excluded by "
                    "default (rebuild with include_shutdown_tail=True for it)")
        raise KeyError(f"no indexed event with timestamp {ts}{hint}")

    def _bd_fd(self, chunk_path):
        """Return an open read fd for ``chunk_path``, cached across reads.

        Keyed by path (not stream) so multi-chunk reads keep one fd per chunk
        file the run rolls through.
        """
        fd = self._bd_fds.get(chunk_path)
        if fd is None:
            fd = os.open(chunk_path, os.O_RDONLY)
            self._bd_fds[chunk_path] = fd
        return fd

    def read_event(self, ts):
        """Read the event at exact timestamp ``ts`` by random access.

        Performs one ``os.pread`` per contributing stream at the indexed
        ``(offset, size)`` -- never scans the bigdata file -- and assembles an
        :class:`psdata.stream.Event` identical to the one the streaming path
        would yield for the same event.

        Raises ``KeyError`` if ``ts`` is not an indexed L1Accept event.
        """
        k = self._position_of(ts)
        return self.read_event_at(k)

    def read_event_at(self, k):
        """Read the ``k``-th L1Accept event (0-based, ascending ts) by random
        access.  Returns a :class:`psdata.stream.Event`."""
        if not (0 <= k < len(self.timestamps)):
            raise IndexError(f"event position {k} out of range "
                             f"[0, {len(self.timestamps)})")
        ts = self.timestamps[k]
        entry = self.entries[k]

        seg_index = {}
        service = _s.SERVICE_L1ACCEPT
        for stream in sorted(entry):
            chunk_path, offset, size = entry[stream]
            fd = self._bd_fd(chunk_path)
            raw = os.pread(fd, size, offset)
            service = self._assemble_stream_dgram(
                stream, chunk_path, offset, size, raw, ts, seg_index)

        return _s.Event(ts, service, self.run_config, seg_index)

    def _assemble_stream_dgram(self, stream, chunk_path, offset, size, raw, ts,
                               seg_index):
        """Validate one stream's ``pread``-ed bigdata dgram bytes for event
        ``ts`` and index its ShapesData into ``seg_index`` (in place).  Returns
        the dgram's service.

        Shared by :meth:`read_event_at` (which ``pread``s inline) and
        :meth:`read_events` (which ``pread``s in a coalesced batch first) so a
        batched read is **byte-identical** to a serial one -- the only thing
        that differs between the two is the *order* the ``pread``s are issued,
        never the bytes that are parsed.
        """
        if len(raw) != size:
            raise RuntimeError(
                f"stream {stream}: short read at offset {offset} of "
                f"{os.path.basename(chunk_path)} (wanted {size}, "
                f"got {len(raw)})")
        buf = bytearray(raw)
        h = _f.parse_dgram_header(buf, 0)
        if h["ts"] != ts:
            raise RuntimeError(
                f"stream {stream}: bigdata dgram ts {h['ts']} != indexed "
                f"ts {ts} (offset {offset} of "
                f"{os.path.basename(chunk_path)}); chunk roll mismatch")
        snap_h = dict(h)
        snap_h["_off"] = 0
        snap_h["_total"] = size
        tables = self.run_config.raw_tables[stream]
        _s._index_dgram(buf, 0, snap_h, tables, seg_index)
        return h["service"]

    # -- batch random read (US-009) ---------------------------------------
    def read_events(self, ks):
        """Read many events by position in ONE coalesced call.

        Equivalent to ``[self.read_event_at(k) for k in ks]`` but issues its
        ``os.pread``s grouped by bigdata chunk file and in **ascending offset
        order within each file**, so the reads walk each chunk forward (a
        kernel-readahead-friendly, seek-minimising pattern) instead of jumping
        around per event.  Returns the assembled events in the **same order as
        ``ks``** (which may be arbitrary / shuffled / repeated).

        I/O accounting is unchanged from the serial path: exactly one
        ``os.pread`` per (event, contributing-stream) pair -- no extra scan, no
        per-call SMD rescan -- and the lazy :meth:`_bd_fd` cache opens each
        chunk file at most once (one ``os.open`` per distinct chunk path,
        reused across the whole batch).

        Missing-segment policy: identical to :meth:`read_event_at`.  The
        per-event missing-segment rule (a detector whose received segment set
        does not match its declared set) is applied lazily inside
        :class:`psdata.stream.Event` on field access, so an incomplete event is
        returned here as an :class:`~psdata.stream.Event` exactly as
        ``read_event_at`` would return it -- never dropped or raised here.

        Parameters
        ----------
        ks : sequence[int]
            Event positions (0-based).  Each is validated against the index
            range; an out-of-range position raises ``IndexError``.

        Returns
        -------
        list[psdata.stream.Event]
            One per requested ``k``, in ``ks`` order.
        """
        ks = [int(k) for k in ks]
        n = len(self.timestamps)
        for k in ks:
            if not (0 <= k < n):
                raise IndexError(
                    f"event position {k} out of range [0, {n})")

        # Collect every required read as (chunk_path, offset, size, k, stream),
        # then group by chunk file and read each file's dgrams in ascending
        # offset order.  Distinct positions in `ks` are de-duplicated for the
        # actual reads (a repeated k is read once and reused) while the returned
        # list still honours every entry in `ks`.
        order = {}                       # k -> position-in-output (first seen)
        reads_by_chunk = {}              # chunk_path -> list[(offset, size, k, stream)]
        for k in ks:
            if k in order:
                continue
            order[k] = len(order)
            for stream in sorted(self.entries[k]):
                chunk_path, offset, size = self.entries[k][stream]
                reads_by_chunk.setdefault(chunk_path, []).append(
                    (offset, size, k, stream))

        # Per-event accumulators, filled as the coalesced reads complete.
        seg_indexes = {k: {} for k in order}
        services = {k: _s.SERVICE_L1ACCEPT for k in order}

        for chunk_path in sorted(reads_by_chunk):
            fd = self._bd_fd(chunk_path)
            for offset, size, k, stream in sorted(reads_by_chunk[chunk_path]):
                raw = os.pread(fd, size, offset)
                services[k] = self._assemble_stream_dgram(
                    stream, chunk_path, offset, size, raw,
                    self.timestamps[k], seg_indexes[k])

        built = {
            k: _s.Event(self.timestamps[k], services[k], self.run_config,
                        seg_indexes[k])
            for k in order
        }
        return [built[k] for k in ks]

    def read_stack(self, ks, det, field="raw", alg="raw"):
        """Batch-read events ``ks`` and stack one detector's segment arrays into
        ONE preallocated buffer of shape ``(len(ks), n_seg, *seg_shape)``.

        Reads are coalesced exactly as in :meth:`read_events` (grouped by chunk
        file, ascending offset, one ``pread`` per event/stream, fds reused).
        The per-event stack is ``Event.stack(det, field, alg)`` -- so a single
        row ``out[i]`` is byte-identical to ``read_event_at(ks[i]).stack(det)``.

        The output dtype, segment count ``n_seg``, and per-segment shape are
        taken from the **first requested event's** stack (the segment shape is
        only known at event time -- it is not in the run config), then the whole
        buffer is allocated once and every row written in place.

        Missing-segment policy (justification): a numeric stacked buffer has no
        natural sentinel for "this event was missing a segment" -- you cannot
        put ``None`` in a row of a uniform ndarray.  ``read_stack`` therefore
        **raises** ``ValueError`` if any requested event is incomplete for
        ``det`` (i.e. ``Event.stack`` returns ``None``, the existing
        missing-segment rule), naming the offending position.  This applies the
        same rule as everywhere else (incomplete -> not valid data) but surfaces
        it eagerly, because the caller asked for a dense array.  Callers who
        want to tolerate gaps should use :meth:`read_events` and handle the
        ``None`` rows themselves.

        Parameters
        ----------
        ks : sequence[int]
            Event positions (0-based); arbitrary order is fine.
        det : str
            Detector name (e.g. ``"jungfrau"``).
        field, alg : str
            Field / algorithm to stack (default ``raw`` / ``raw``).

        Returns
        -------
        numpy.ndarray
            Shape ``(len(ks), n_seg, *seg_shape)``, ``out[i]`` the stack for
            ``ks[i]``.

        Raises
        ------
        ValueError
            If ``ks`` is empty, or any requested event is missing a segment for
            ``det`` (so a dense stack cannot represent it).
        """
        ks = [int(k) for k in ks]
        if not ks:
            raise ValueError("read_stack requires at least one event position")

        events = self.read_events(ks)

        # Shape/dtype come from the first event's stack (segment shape is only
        # known at event time).  Allocate ONE buffer and fill it row by row.
        out = None
        for i, evt in enumerate(events):
            st = evt.stack(det, field=field, alg=alg)
            if st is None:
                raise ValueError(
                    f"event position {ks[i]} is missing a segment for "
                    f"detector {det!r} (alg={alg!r}, field={field!r}); a dense "
                    f"read_stack cannot represent it -- use read_events to "
                    f"handle incomplete events")
            if out is None:
                out = np.empty((len(events),) + st.shape, dtype=st.dtype)
            elif st.shape != out.shape[1:]:
                raise ValueError(
                    f"event position {ks[i]}: {det!r} stack shape {st.shape} "
                    f"!= first event's {out.shape[1:]}")
            out[i] = st
        return out

    # -- resource management ----------------------------------------------
    def close(self):
        for fd in self._bd_fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._bd_fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        tail = ", shutdown_tail" if self.include_shutdown_tail else ""
        return (f"RunIndex(n_events={self.n_events}, "
                f"streams={sorted(self.bd_files)}, "
                f"source={self.scan_source!r}{tail}, "
                f"build_seconds={self.build_seconds:.3f}, "
                f"scan_MB={self.scan_bytes_read / 1e6:.1f})")

    # -- serialization & disk persistence (US-008; disk format IDX-02) ----
    #
    # The once-built index can be (a) saved to a single file and reloaded
    # instantly with NO SMD rescan (a single-process benefit), and (b) shipped
    # to parallel workers in-memory via pickle / ``to_dict``.  Both paths share
    # ONE state-stripping helper (``_persist_state``) so they cannot drift.
    #
    # TWO DIFFERENT trust boundaries, TWO DIFFERENT formats:
    #   * (a) the DISK artifact (:meth:`save`/:meth:`load`) is written once and
    #     re-read by many, possibly OTHER users', jobs off a shared analysis
    #     filesystem (``/sdf``).  ``pickle.load`` executes arbitrary code baked
    #     into the file it reads, so a bare-pickle index on a shared path is a
    #     remote-code-execution vector (any user who can write the directory a
    #     victim's job loads from gets code execution in that job).  The disk
    #     format is therefore NOT pickle: it is a self-describing, versioned,
    #     checksummed container (magic + version + a JSON header + a JSON
    #     payload) decoded by a strict WHITELIST codec (:func:`_index_encode` /
    #     :func:`_index_decode`) that can reconstruct ONLY plain data and the
    #     three psdata config classes -- never an arbitrary callable.  See
    #     :meth:`save` / :meth:`load`.
    #   * (b) the IN-MEMORY ship-to-workers path (``to_dict``/``from_dict`` and
    #     the pickle protocol ``__getstate__``/``__setstate__``) stays pickle:
    #     it is a trusted, in-process hand-off (e.g. Ray's object store) of an
    #     object THIS process just built -- not an untrusted file off disk -- so
    #     the RCE surface the disk format closes does not apply to it.
    #
    # THE gotcha (load-bearing): ``_bd_fds`` caches raw OS file-descriptor
    # integers from ``os.open``.  ``pickle`` does NOT refuse a bare int, so an
    # index serialized *after any read* would otherwise carry stale fds; loading
    # it elsewhere and reading raises ``OSError(9, 'Bad file descriptor')`` or --
    # if those fd numbers were reused -- ``pread``s the WRONG file and returns
    # silent garbage.  ``_ts_to_k`` is likewise a per-process cache.  Both are
    # therefore EXCLUDED from the persisted state and re-initialised empty on
    # reconstruction, so fds reopen lazily on the first read in the new process.
    #
    # The persisted state is exactly: timestamps, entries, env_records,
    # bd_files, chunk_files, multichunk_streams, run_config, build_seconds,
    # smd_bytes_read, scan_source, scan_bytes_read, include_shutdown_tail -- all
    # plain-Python / numpy-dtype values (no fds, no live C objects), so plain
    # ``pickle.dumps(idx)`` is safe once the two per-process caches are stripped.
    # (``env_records`` values are plain (int, str, int, int) tuples -- no fds:
    # the env store's own fd cache lives on the transient EnvStore, not here.)

    _PERSIST_FIELDS = (
        "timestamps",
        "entries",
        "env_records",
        "bd_files",
        "chunk_files",
        "multichunk_streams",
        "run_config",
        "build_seconds",
        "smd_bytes_read",
        "scan_source",
        "scan_bytes_read",
        "include_shutdown_tail",
    )

    # Defaults for fields absent from an older persisted blob (back-compat:
    # indexes pickled before the bigdata-scan source existed have no
    # ``scan_source``/``scan_bytes_read`` -- treat them as SMD-built; likewise a
    # blob predating the clamp has no ``include_shutdown_tail`` -- it was built
    # before the tail could be kept, so it is canonical: False).
    _PERSIST_DEFAULTS = {
        "scan_source": "smd",
        "scan_bytes_read": 0,
        "include_shutdown_tail": False,
        # An index pickled before the env store existed has no env_records --
        # load it as an empty mapping (no env lookups, but everything else
        # works).  A fresh dict is installed per instance in _restore_state so
        # the mutable default is never shared.
        "env_records": {},
    }

    def _persist_state(self):
        """Return the picklable state dict -- the single source of truth for
        what is persisted/serialized.  Deliberately OMITS ``_bd_fds`` and
        ``_ts_to_k`` (per-process caches; see the class note above)."""
        return {name: getattr(self, name) for name in self._PERSIST_FIELDS}

    def _restore_state(self, state):
        """Populate ``self`` from a ``_persist_state`` dict and re-init the
        per-process caches empty so fds reopen lazily on the first read."""
        for name in self._PERSIST_FIELDS:
            if name in state:
                setattr(self, name, state[name])
            else:
                setattr(self, name, self._PERSIST_DEFAULTS.get(name))
        # back-fill: an SMD-built old blob's scan_bytes_read == its smd bytes
        if "scan_bytes_read" not in state:
            self.scan_bytes_read = self.smd_bytes_read
        # A blob predating the env store has no env_records -- install a FRESH
        # empty dict (not the shared _PERSIST_DEFAULTS one) so it is never
        # aliased across restored instances.
        if "env_records" not in state:
            self.env_records = {}
        self._bd_fds = {}        # MUST be a fresh empty cache -- never the
        self._ts_to_k = None     # serialized one (would carry stale fds)
        return self

    # in-memory facet: ship to workers --------------------------------------
    def to_dict(self):
        """Return a plain ``dict`` of the index's persisted state (ship this to
        workers via e.g. Ray's object store; reconstruct with
        :meth:`from_dict`).  Shares the state-stripping helper with
        ``save``/pickle, so it is fd-safe by construction."""
        return self._persist_state()

    @classmethod
    def from_dict(cls, state):
        """Reconstruct a :class:`RunIndex` from a :meth:`to_dict` payload
        WITHOUT rescanning SMD.  Fds reopen lazily on the first read."""
        idx = cls.__new__(cls)
        return idx._restore_state(state)

    # pickle protocol: plain ``pickle.dumps(idx)`` is fd-safe -------------
    def __getstate__(self):
        return self._persist_state()

    def __setstate__(self, state):
        self._restore_state(state)

    # disk persistence ------------------------------------------------------
    def save(self, path):
        """Write the built index to a single file at ``path`` so it can be
        reloaded later (or in another process) WITHOUT rescanning SMD.

        On-disk format (IDX-02): a self-describing, versioned, checksummed
        container -- explicitly **not** a pickle, because the index is a
        shareable artifact re-read off a multi-user filesystem and
        ``pickle.load`` runs arbitrary code baked into the file (an RCE vector).
        Layout::

            magic   = b"PSDATIDX"                         (8 bytes)
            version = uint32 LE (== _INDEX_FORMAT_VERSION)
            hlen    = uint32 LE
            header  = <hlen> bytes UTF-8 JSON
                      {"checksum_algo","checksum","payload_len"}
            payload = <payload_len> bytes UTF-8 JSON

        The ``payload`` is :meth:`_persist_state` rendered by
        :func:`_index_encode` -- a strict, type-tagged, self-describing encoding
        of plain data (and only the three psdata config classes), string-pooled
        so repeated chunk paths cost one entry.  ``checksum`` is the
        ``sha256`` of the payload bytes, verified on load.  Only the fd-safe
        ``_persist_state`` is written.

        The payload is made RELOCATABLE (IDX-03) by
        :func:`_index_portablize_paths`: file paths are stored relative to their
        common root plus a ``portability`` record, so the index survives being
        copied to another mount/host/container (resolved by :meth:`load`).  This
        transforms only the throwaway encoder output -- the live index keeps its
        absolute paths, and a reload from the original location is byte-exact.
        """
        payload = json.dumps(
            _index_portablize_paths(_index_encode(self._persist_state())),
            separators=(",", ":")).encode("utf-8")
        digest = hashlib.new(_INDEX_CHECKSUM_ALGO, payload).hexdigest()
        header = json.dumps(
            {"checksum_algo": _INDEX_CHECKSUM_ALGO,
             "checksum": digest,
             "payload_len": len(payload)},
            separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as fh:
            fh.write(_INDEX_MAGIC)
            fh.write(struct.pack("<I", _INDEX_FORMAT_VERSION))
            fh.write(struct.pack("<I", len(header)))
            fh.write(header)
            fh.write(payload)

    @classmethod
    def load(cls, path, dir=None):
        """Reload an index written by :meth:`save`.  Opens ONLY the index file
        (no SMD files, no rescan): ``smd_bytes_read`` is whatever the original
        build measured, but no new SMD I/O happens here.  Fds into the bigdata
        files reopen lazily on the first :meth:`read_event_at`.

        ``dir`` (optional) is the directory that holds the run's xtc2 files at
        THIS location -- pass it to relocate a portable index whose data now
        lives under a different prefix than where it was built (e.g. built under
        ``/sdf/...`` on host A, loaded under ``/data/...`` on host B).  When
        omitted, the paths resolve to the original build location if the files
        are there, else to the index file's own directory if the data was
        shipped beside it, else the original absolute paths are reproduced (a
        genuine miss then surfaces at read time -- never a wrong file).  See
        :func:`_index_resolve_paths`.

        The magic + version + checksum are all verified before any content is
        interpreted, and the payload is decoded by the whitelist codec
        (:func:`_index_decode`) -- so a corrupt, truncated, format-drifted, or
        (crucially) an old bare-**pickle** index is REFUSED with a clear error
        instead of being trusted.  ``load`` never executes code embedded in the
        file: unlike ``pickle.load`` there is no path to an arbitrary callable.
        """
        with open(path, "rb") as fh:
            magic = fh.read(len(_INDEX_MAGIC))
            if magic != _INDEX_MAGIC:
                raise ValueError(
                    "%r is not a psdata index file: expected magic %r, found "
                    "%r.  A bare-pickle index written by an older psdata is "
                    "refused on purpose (loading a pickle executes arbitrary "
                    "code embedded in the file -- an RCE vector on a shared "
                    "filesystem); rebuild it with build_index(...).save(...)."
                    % (path, _INDEX_MAGIC, magic))
            vbytes = fh.read(4)
            if len(vbytes) < 4:
                raise ValueError(
                    "%r: truncated psdata index (no format version)" % (path,))
            version = struct.unpack("<I", vbytes)[0]
            if version not in _INDEX_SUPPORTED_VERSIONS:
                raise ValueError(
                    "%r: unsupported psdata index format version %d -- this "
                    "psdata reads versions %s.  Rebuild the index with the "
                    "current psdata (build_index(...).save(...))."
                    % (path, version, sorted(_INDEX_SUPPORTED_VERSIONS)))
            hlen_bytes = fh.read(4)
            if len(hlen_bytes) < 4:
                raise ValueError(
                    "%r: truncated psdata index (no header length)" % (path,))
            hlen = struct.unpack("<I", hlen_bytes)[0]
            header_bytes = fh.read(hlen)
            if len(header_bytes) < hlen:
                raise ValueError(
                    "%r: truncated psdata index (short header)" % (path,))
            try:
                header = json.loads(header_bytes.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as e:
                raise ValueError(
                    "%r: corrupt psdata index header (%s)" % (path, e))
            algo = header.get("checksum_algo")
            want_digest = header.get("checksum")
            plen = header.get("payload_len")
            if not isinstance(algo, str) or algo not in _INDEX_ALLOWED_CHECKSUM_ALGOS:
                raise ValueError(
                    "%r: psdata index header names an unsupported checksum "
                    "algorithm %r -- this psdata accepts only %s."
                    % (path, algo, sorted(_INDEX_ALLOWED_CHECKSUM_ALGOS)))
            payload = fh.read()
        if not isinstance(plen, int) or len(payload) != plen:
            raise ValueError(
                "%r: psdata index payload length mismatch (header says %r "
                "bytes, file holds %d) -- the file is truncated or padded; "
                "rebuild the index." % (path, plen, len(payload)))
        got_digest = hashlib.new(algo, payload).hexdigest()
        if not isinstance(want_digest, str) or got_digest != want_digest:
            raise ValueError(
                "%r: psdata index integrity check FAILED (%s mismatch: header "
                "%r, computed %r) -- the file is corrupt or was modified; "
                "rebuild the index." % (path, algo, want_digest, got_digest))
        try:
            doc = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            raise ValueError(
                "%r: corrupt psdata index payload (%s)" % (path, e))
        # Re-absolutize the relocatable paths (IDX-03) against the resolved base
        # BEFORE decoding, so the reconstructed state carries absolute paths and
        # the in-memory index is byte-identical to the original at its home.
        if isinstance(doc, dict):
            _index_resolve_paths(doc, path, dir)
        state = _index_decode(doc)
        idx = cls.__new__(cls)
        return idx._restore_state(state)


# ==========================================================================
# SMD scan: read one SMD file's L1Accept records, following the chunk roll
# ==========================================================================
def _scan_smd_stream(path, bd_c000_path, read_chunk, env_out=None):
    """Scan one SMD file and return ``(records, bytes_read, chunk_paths)``.

    If ``env_out`` (a dict) is given, the env (slow-data) transitions found in
    this SMD file are bucketed into it as
    ``{store_name: [(ts, smd_path, offset, size), ...]}`` (SlowUpdate ->
    ``epics``; BeginStep/BeginRun -> ``scan``), pointing at the SMD file itself
    (the env dgram payload is carried inline there).  This is additive -- it
    never enters ``records`` and leaves the 3-tuple return unchanged, so
    existing callers are unaffected.

    ``records`` is a list of ``(ts, chunk_path, intOffset, intDgramSize)`` for
    every ``L1Accept``: ``chunk_path`` is the bigdata chunk file the offset
    indexes into -- ``bd_c000_path`` until an ``Enable`` rolls the chunk, then
    the next chunk file (resolved from the ``chunkinfo`` carried on that
    Enable).  ``chunk_paths`` is the ordered list of distinct chunk files this
    stream rolled through (``[c000]`` if it never rolled).

    Chunk-roll rule (mirrors ``event_manager.py`` ~208-227 / ``_open_new_bd_file``
    ~248): on each Enable dgram that carries a ``chunkinfo`` whose ``chunkid``
    exceeds the current one, the active bigdata file becomes
    ``<dir of c000>/<chunkinfo.filename>`` and subsequent ``intOffset`` values
    (which restart at 0 in the new chunk) index that file.

    Reads the SMD file in a growing ``os.pread`` window -- the SMD files are
    small (a few MB), so this is the only I/O the index build does; the bigdata
    files are never opened here.

    Only ``SERVICE_L1ACCEPT`` (service 12) dgrams are indexed -- consistent with
    the bigdata path -- and EOB L1Accepts (``SERVICE_L1ACCEPT_EOB``, service 11,
    which ``stream.py``'s ``_EVENT_SERVICES`` does yield while streaming) are NOT
    indexed; revisit if EOB-event random access is needed.
    """
    bd_dir = os.path.dirname(bd_c000_path) if bd_c000_path else None
    cur_chunk_id = 0
    cur_chunk_path = bd_c000_path
    chunk_paths = [bd_c000_path] if bd_c000_path is not None else []

    fd = os.open(path, os.O_RDONLY)
    try:
        # Configure sits at offset 0; read enough to parse it, then grow.
        buf = bytearray(os.pread(fd, max(read_chunk, _f._CONFIG_READ), 0))
        bytes_read = len(buf)
        _cfg, tables, cfg_end = _f.parse_configure(buf)

        smdinfo_key = _find_smdinfo_key(tables)
        if smdinfo_key is None:
            raise RuntimeError(
                f"{os.path.basename(path)} has no smdinfo table -- is it an "
                f"SMD (smalldata) file?")
        table = tables[smdinfo_key]
        has_chunkinfo = any(
            t["det_name"] == _CHUNKINFO_DET and t["alg_name"] == _CHUNKINFO_ALG
            for t in tables.values())

        records = []
        off = cfg_end
        while True:
            if off + _f.DGRAM_HDR > len(buf):
                more = os.pread(fd, read_chunk, len(buf))
                if not more:
                    break
                buf.extend(more)
                bytes_read += len(more)
                continue
            h = _f.parse_dgram_header(buf, off)
            total = _f.XTC_HDR + h["extent"]
            if off + total > len(buf):
                more = os.pread(fd, max(read_chunk, total), len(buf))
                if not more:
                    break   # truncated final dgram -> stop (clean EOF)
                buf.extend(more)
                bytes_read += len(more)
                continue

            if h["service"] == _f.SERVICE_L1ACCEPT:
                info = _read_smdinfo(buf, off, h, smdinfo_key, table)
                if info is not None:
                    intoff, intsize = info
                    records.append((h["ts"], cur_chunk_path, intoff, intsize))
            elif (h["service"] == _f.SERVICE_ENABLE and has_chunkinfo
                  and bd_dir is not None):
                # Follow the chunk roll: a new chunkid names the next bigdata
                # file; offsets after this Enable index into it.
                roll = _read_chunkinfo(buf, off, h, tables)
                if roll is not None:
                    new_id, new_name = roll
                    if new_id > cur_chunk_id:
                        cur_chunk_id = new_id
                        cur_chunk_path = os.path.join(bd_dir, new_name)
                        chunk_paths.append(cur_chunk_path)
            elif env_out is not None and h["service"] in _ENV_SERVICE_STORE:
                # Env (slow-data) transition: the dgram payload is carried inline
                # in THIS SMD file (byte-identical to the bigdata copy but at a
                # different offset), so the env record points at the SMD file.
                # Additive only -- never enters the L1Accept records.
                store = _ENV_SERVICE_STORE[h["service"]]
                env_out.setdefault(store, []).append(
                    (h["ts"], path, off, total))
            off += total
        return records, bytes_read, chunk_paths
    finally:
        os.close(fd)


def _read_chunkinfo(buf, dg_off, dg_hdr, tables):
    """Read the ``chunkinfo`` (``chunkid``, ``filename``) from one Enable
    dgram, or ``None`` if this Enable carries none (the run-start Enable does
    not).  ``filename`` is decoded to a ``str`` (the next chunk's basename)."""
    cid = _f.read_dgram_field(buf, dg_off, dg_hdr, tables,
                              _CHUNKINFO_DET, _CHUNKINFO_ALG, _CHUNK_ID_FIELD)
    if cid is None:
        return None
    fn = _f.read_dgram_field(buf, dg_off, dg_hdr, tables, _CHUNKINFO_DET,
                             _CHUNKINFO_ALG, _CHUNK_FILENAME_FIELD)
    if fn is None:
        return None
    return int(cid), _decode_chunk_filename(fn)


# ==========================================================================
# Bigdata scan: rebuild one stream's records from the bigdata files alone
# (the smdwriter algorithm in pure Python -- no SMD artifact needed)
# ==========================================================================
def _enumerate_bd_chunks(bd_c000_path):
    """Ordered bigdata chunk files for a stream, from its ``c000`` path, by the
    ``-c000``/``-c001``/... filename convention (stops at the first missing
    chunk id).  A single-chunk stream returns ``[c000]``.

    Enumerating by the on-disk filename convention -- rather than following the
    ``chunkinfo`` carried on each Enable (the SMD path's mechanism) -- keeps the
    bigdata scan self-contained: it needs only the bigdata files themselves, no
    chunkinfo and no SMD.  The chunk filenames ``chunkinfo`` would name are
    exactly these ``-c00N`` siblings in the same directory, so the set walked is
    identical to the one the SMD-following path rolls through.
    """
    d = os.path.dirname(bd_c000_path)
    base = os.path.basename(bd_c000_path)
    m = re.search(r"-c(\d+)\.xtc2$", base)
    if not m:
        return [bd_c000_path]
    prefix = base[:m.start()]            # '<exp>-rNNNN-sMMM'
    width = len(m.group(1))              # zero-pad width (3 -> c000)
    chunks = []
    cid = 0
    while True:
        cand = os.path.join(d, f"{prefix}-c{cid:0{width}d}.xtc2")
        if not os.path.exists(cand):
            break
        chunks.append(cand)
        cid += 1
    return chunks if chunks else [bd_c000_path]


def _scan_bigdata_stream(bd_c000_path, env_out=None):
    """Build one stream's L1Accept ``(ts, chunk_path, intOffset, intDgramSize)``
    records by walking the BIGDATA chunk files directly -- no SMD file needed.

    The ``smdwriter`` algorithm in pure Python: within each chunk file the dgram
    offsets restart at 0, so walk from 0 reading each dgram's 24-byte header,
    record the running cursor at every ``L1Accept`` (which equals the SMD's
    ``intOffset``) and its on-disk size ``XTC_HDR + extent`` (the SMD's
    ``intDgramSize``), and advance the cursor by that size across EVERY dgram
    -- Configure / BeginRun / Enable / SlowUpdate included, because they occupy
    bigdata space and shift every later offset.  Payloads are seeked past
    (only the header is ``pread``), so the cost is one ``pread`` per dgram, not
    the file size.

    Returns ``(records, bytes_read, chunk_paths)`` in the SAME shape as
    :func:`_scan_smd_stream`, so :meth:`RunIndex.build_from_bigdata` and
    :meth:`RunIndex.build` feed :meth:`RunIndex._merge_streams` identically and
    produce byte-for-byte identical indexes.  If ``env_out`` (a dict) is given it
    is filled with ``{store_name: [(ts, chunk_path, offset, size), ...]}`` for
    the env transitions (SlowUpdate -> ``epics``; BeginStep/BeginRun ->
    ``scan``), pointing at the bigdata chunk the bytes live in; additive, never
    in ``records``, and it leaves the 3-tuple return unchanged.  The env
    timestamps it yields are identical to the SMD path's (the same broadcast
    transition dgrams), only the file/offset differ.

    Only ``SERVICE_L1ACCEPT`` (service 12) dgrams are indexed -- consistent with
    the SMD path -- and EOB L1Accepts (``SERVICE_L1ACCEPT_EOB``, service 11,
    which ``stream.py``'s ``_EVENT_SERVICES`` does yield while streaming) are NOT
    indexed; revisit if EOB-event random access is needed.
    """
    records = []
    chunk_paths = []
    bytes_read = 0
    for chunk_path in _enumerate_bd_chunks(bd_c000_path):
        chunk_paths.append(chunk_path)
        fd = os.open(chunk_path, os.O_RDONLY)
        try:
            filesize = os.fstat(fd).st_size
            cursor = 0
            while cursor + _f.DGRAM_HDR <= filesize:
                hdr = os.pread(fd, _f.DGRAM_HDR, cursor)
                bytes_read += len(hdr)
                if len(hdr) < _f.DGRAM_HDR:
                    break                       # truncated tail -> clean stop
                h = _f.parse_dgram_header(hdr, 0)
                total = _f.XTC_HDR + h["extent"]
                if cursor + total > filesize:
                    break                       # truncated final dgram
                if h["service"] == _f.SERVICE_L1ACCEPT:
                    records.append((h["ts"], chunk_path, cursor, total))
                elif env_out is not None and h["service"] in _ENV_SERVICE_STORE:
                    # Env (slow-data) transition: bytes live in THIS bigdata
                    # chunk at ``cursor``; only its 24-byte header was read, the
                    # payload is decoded lazily at lookup.  Additive only.
                    store = _ENV_SERVICE_STORE[h["service"]]
                    env_out.setdefault(store, []).append(
                        (h["ts"], chunk_path, cursor, total))
                cursor += total
        finally:
            os.close(fd)
    return records, bytes_read, chunk_paths


# ==========================================================================
# Convenience: discover SMD files & build the index from a run directory
# ==========================================================================
def smd_files_for(stream_files):
    """Given a run's bigdata ``stream_files`` (``{stream: path}`` or the other
    accepted forms), return the matching SMD files ``{stream: smd_path}`` by
    the standard layout (``<dir>/smalldata/<base>.smd.xtc2`` for a bigdata
    ``<dir>/<base>.xtc2``).

    Resolving by filename convention keeps the path pattern in one place; the
    caller can also pass an explicit SMD-file mapping straight to
    :meth:`RunIndex.build`.
    """
    items = _f._normalize_stream_files(stream_files)
    out = {}
    for stream, bd_path in items:
        d = os.path.dirname(bd_path)
        base = os.path.basename(bd_path)
        root, ext = os.path.splitext(base)         # ('...-c000', '.xtc2')
        smd_path = os.path.join(d, "smalldata", root + ".smd" + ext)
        out[stream] = smd_path
    return out


def build_index(stream_files, run_config=None, smd_files=None, source="auto",
                include_shutdown_tail=False):
    """Build a :class:`RunIndex` for a run.

    Parameters
    ----------
    stream_files : see :func:`psdata.format.discover`
        The run's bigdata xtc2 stream files.
    run_config : psdata.format.RunConfig, optional
        Pre-discovered config for ``stream_files``; discovered here if omitted.
    smd_files : optional
        Explicit SMD-file mapping.  If omitted, resolved from ``stream_files``
        via :func:`smd_files_for` (the standard ``smalldata/`` layout).  Ignored
        when ``source="bigdata"``.
    source : {"auto", "smd", "bigdata"}
        Where the random-access index comes from:

        * ``"smd"``     -- scan the small ``.smd.xtc2`` sidecars
          (:meth:`RunIndex.build`).  Fast, but requires the SMD artifact;
          raises if a sidecar is missing.
        * ``"bigdata"`` -- scan the bigdata dgram headers directly
          (:meth:`RunIndex.build_from_bigdata`), needing NO SMD artifact.
        * ``"auto"`` (default) -- use the SMD sidecars when *all* are present
          (fast path), else fall back to the bigdata scan.  This keeps the fast
          path when SMD exists while removing the hard *dependency* on it.
    include_shutdown_tail : bool
        Only affects the bigdata scan (``source="bigdata"`` or an ``"auto"``
        fallback).  ``False`` (default) clamps the index to the canonical event
        set -- byte-identical to the SMD-built index.  ``True`` additionally
        keeps the ragged DAQ end-of-run shutdown tail (events lacking the
        timing/master stream, ``pulseId=None``); see
        :meth:`RunIndex.build_from_bigdata`.  Ignored by the SMD path (the
        sidecars never recorded that tail).

    Returns
    -------
    RunIndex
    """
    if run_config is None:
        run_config = _f.discover(stream_files)
    if source not in ("auto", "smd", "bigdata"):
        raise ValueError(
            f"source must be 'auto', 'smd', or 'bigdata' (got {source!r})")
    if source == "bigdata":
        return RunIndex.build_from_bigdata(
            run_config, include_shutdown_tail=include_shutdown_tail)
    if smd_files is None:
        smd_files = smd_files_for(stream_files)
    if source == "smd":
        return RunIndex.build(smd_files, run_config)
    # source == "auto": SMD fast path iff every sidecar is present on disk,
    # else build the identical index from the bigdata files alone.
    items = _f._normalize_stream_files(smd_files)
    if items and all(os.path.exists(p) for _stream, p in items):
        return RunIndex.build(smd_files, run_config)
    return RunIndex.build_from_bigdata(
        run_config, include_shutdown_tail=include_shutdown_tail)


# ==========================================================================
# Import-purity self check (delegates to the format module's checker)
# ==========================================================================
def assert_no_framework_imports():
    """Raise AssertionError if a framework leaked into ``sys.modules``."""
    _f.assert_no_framework_imports()
