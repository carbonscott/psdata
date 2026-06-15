"""psdata -- minimal clean-room pure-Python xtc2 reader.

This is the data-access layer only: raw detector arrays + event identity,
with no MPI, no framework, and no calibration.  Importing it pulls in only
numpy (no psana / mpi4py / h5py).

US-001 delivers the parse core and generic discovery in :mod:`psdata.format`.
US-002 adds multi-stream event assembly in :mod:`psdata.stream`: an exact
64-bit-timestamp k-way merge that yields :class:`~psdata.stream.Event` objects
in ascending timestamp order, each exposing ``timestamp``, ``pulseId``, and
raw detector arrays.
US-003 adds a random-access-by-event index in :mod:`psdata.index`: a
``timestamp -> {stream: (offset, size)}`` index built by scanning only the
small SMD files, with :meth:`~psdata.index.RunIndex.read_event` /
:meth:`~psdata.index.RunIndex.read_event_at` serving an arbitrary event by a
single ``os.pread`` per stream -- no sequential bigdata scan.
US-005 adds the public "open a run" surface in :mod:`psdata.run`: open a run by
``exp`` / ``run`` / ``dir`` (the standard file layout) or by an explicit
stream-file list, then stream / random-access / introspect it through one
:class:`~psdata.run.Run` handle.

    import psdata

    # open by exp/run/dir (files resolved by the standard layout) ...
    r = psdata.open(exp="mfx100848724", run=51,
                    dir="/sdf/data/lcls/ds/prj/public01/xtc")
    # ... or by an explicit per-stream file list
    r = psdata.open(files=["/path/...-s000-c000.xtc2", ...])

    for evt in r.events():                        # forward stream, ts order
        ts  = evt.timestamp
        pid = evt.pulseId
        jf  = evt.stack("jungfrau")               # (32,512,1024) or None

    evt = r.read_event(ts)                        # random access by timestamp
    evt = r.read_event_at(1000)                   # ... or by event position
    r.detector_names()                            # introspect detectors/fields

The lower-level functional API used internally is still available::

    rc   = psdata.discover(stream_files)          # detectors / fields / segs
    evts = psdata.events(stream_files)            # stream in ts order
    ridx = psdata.build_index(stream_files, run_config=rc)   # scans SMD only

Importing psdata pulls in only numpy -- no psana / mpi4py / h5py.

Calibration is a deliberately SEPARATE, optional layer: ``psdata.calib`` (US-006)
snapshots a detector's calibration constants once (the only DB dependency) and
reloads them offline with numpy only.  It is NOT imported here, so importing the
reader never pulls it (or psana) in; ``import psdata.calib`` explicitly when you
want it.
"""

from . import format  # noqa: F401  (re-export the parse core)
from . import stream   # noqa: F401  (re-export the streaming layer)
from . import index    # noqa: F401  (re-export the random-access layer)
from . import run      # noqa: F401  (re-export the public run surface)
from .format import discover, filter_c000, stream_index_of, decode_damage
from .stream import Event, events
from .index import RunIndex, build_index, smd_files_for
from .run import Run, open

__all__ = ["format", "stream", "index", "run",
           "open", "Run", "discover", "events", "Event",
           "RunIndex", "build_index", "smd_files_for",
           "filter_c000", "stream_index_of", "decode_damage"]
