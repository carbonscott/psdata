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

    import psdata
    rc = psdata.discover(stream_files)            # detectors / fields / segs
    for evt in psdata.events(stream_files):       # stream in ts order
        ts = evt.timestamp
        pid = evt.pulseId
        jf = evt.stack("jungfrau")                # (32,512,1024) or None

    ridx = psdata.build_index(stream_files, run_config=rc)   # scans SMD only
    evt = ridx.read_event(ts)                     # random access by timestamp
    evt = ridx.read_event_at(1000)                # ... or by event position
"""

from . import format  # noqa: F401  (re-export the parse core)
from . import stream   # noqa: F401  (re-export the streaming layer)
from . import index    # noqa: F401  (re-export the random-access layer)
from .format import discover
from .stream import Event, events
from .index import RunIndex, build_index, smd_files_for

__all__ = ["format", "stream", "index", "discover", "events", "Event",
           "RunIndex", "build_index", "smd_files_for"]
