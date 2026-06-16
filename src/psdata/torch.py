#!/usr/bin/env python3
"""psdata.torch -- OPTIONAL torch ``Dataset`` adapter (US-011, demonstrator).

A map-style ``torch.utils.data.Dataset`` over a psdata run: random access by
event position becomes ``dataset[k]``, so the existing serializable, framework-
free index (US-008) and batch read (US-009) plug straight into a
``torch.utils.data.DataLoader`` -- one more *consumer* of the same generic
primitives, alongside the Ray cube (US-010).

This module is **NOT** imported by ``import psdata`` -- only ``import
psdata.torch`` pulls torch in -- and ``torch`` is imported **lazily inside the
functions/methods** that need it, exactly mirroring ``psdata.calib.snapshot``'s
lazy-psana discipline.  So ``import psdata`` stays numpy-only; install the
optional dependency with ``pip install psdata[torch]``.

----------------------------------------------------------------------------
THE fork-safety problem (load-bearing; same root cause as US-008's serialize
gotcha)
----------------------------------------------------------------------------
``DataLoader(num_workers=N)`` spawns worker *processes*; on Linux the default
start method is **fork** (confirmed from torch's source).  A forked child
inherits a *copy* of the dataset object -- and with it the parent's
``RunIndex._bd_fds`` cache of raw OS file-descriptor **integers** (from
``os.open``).  Those fd numbers are only valid in the parent: ``os.pread``-ing
them in a child either raises ``OSError(9, 'Bad file descriptor')`` or -- if the
number was reused for some other file -- silently reads the WRONG bytes.

The fix is the same discipline US-008 already established for serialization:
**drop the inherited fd cache so each process reopens its own fds lazily.**  We
do it with a *pid guard*: the dataset remembers the pid in which its index's
fd cache is valid; on every ``__getitem__`` (and in the optional
``worker_init_fn``) it checks ``os.getpid()`` and, if it changed (i.e. we are in
a forked worker), it **clears** ``index._bd_fds`` -- crucially *without*
``os.close()``, because those fds belong to the parent and closing them would
corrupt the parent's reads.  ``RunIndex._bd_fd`` then re-``os.open``s fresh,
process-local fds on the next read.  This works whether or not the user wires up
``worker_init_fn``, so correctness does not depend on the caller remembering it.
"""

import os

__all__ = ["XTCDataset", "worker_init_fn"]


def _as_index(run_or_index):
    """Accept either a :class:`psdata.run.Run` (build/return its
    :class:`~psdata.index.RunIndex`) or a ``RunIndex`` directly.

    Imported lazily so this module has no import-time psdata-internal coupling
    beyond what ``import psdata.torch`` already implies; numpy-only.
    """
    from psdata.index import RunIndex
    if isinstance(run_or_index, RunIndex):
        return run_or_index
    build_index = getattr(run_or_index, "build_index", None)
    if callable(build_index):                 # a psdata.run.Run facade
        return build_index()
    raise TypeError(
        "XTCDataset expects a psdata.run.Run or a psdata.index.RunIndex, "
        f"got {type(run_or_index).__name__}")


def _drop_inherited_fds(index):
    """Forget the index's cached bigdata fds **without closing them**.

    Used after a fork: the cached ints are the *parent's* fds and are invalid
    here, but closing them would close the parent's real descriptors.  We just
    empty the dict so :meth:`RunIndex._bd_fd` reopens process-local fds lazily.
    """
    # _bd_fds is a plain dict on RunIndex; replace it with a fresh empty one.
    index._bd_fds = {}


class XTCDataset:
    """Map-style ``torch.utils.data.Dataset`` over a psdata run.

    ``len(ds) == index.n_events``; ``ds[k]`` random-accesses the ``k``-th event
    (``RunIndex.read_event_at``), stacks one detector's segments
    (``Event.stack(detector, field, alg)``), applies an optional user
    ``transform``, and returns a ``torch`` tensor (or whatever ``transform``
    returns).

    Parameters
    ----------
    run_or_index : psdata.run.Run | psdata.index.RunIndex
        An open run (its index is built on demand) or an already-built index
        (e.g. one reloaded with ``RunIndex.load`` / shipped via ``to_dict`` --
        US-008).  Either way no SMD rescan happens per worker.
    detector : str
        Detector name to stack, e.g. ``"jungfrau"``.
    field, alg : str
        Field / algorithm passed to ``Event.stack`` (default ``raw`` / ``raw``).
    transform : callable | None
        Optional ``ndarray -> Any`` map applied before tensor conversion.  If it
        returns something that is not an ndarray (e.g. already a tensor) it is
        passed through unchanged; otherwise the ndarray is wrapped with
        ``torch.from_numpy``.
    to_tensor : bool
        If ``True`` (default) the result is converted to a ``torch`` tensor when
        it is still an ndarray.  Set ``False`` to return raw ndarrays.

    Missing-segment policy
    ----------------------
    Mirrors the rest of psdata: ``Event.stack`` returns ``None`` for an event
    that is missing a segment for ``detector``.  Because a ``DataLoader``'s
    default collate cannot batch ``None`` next to tensors, ``__getitem__``
    **raises** ``ValueError`` naming the offending position (the same eager rule
    as ``RunIndex.read_stack``).  Filter incomplete positions out of your
    sampler, or pass a ``transform`` that tolerates ``None`` and a custom
    ``collate_fn``.
    """

    def __init__(self, run_or_index, detector, field="raw", alg="raw",
                 transform=None, to_tensor=True):
        self.index = _as_index(run_or_index)
        self.detector = detector
        self.field = field
        self.alg = alg
        self.transform = transform
        self.to_tensor = to_tensor
        # The pid in which self.index._bd_fds is valid.  Set at construction;
        # re-synced lazily after a fork (see _ensure_fork_safe_fds).
        self._fds_pid = os.getpid()

    # -- torch Dataset map-style contract ---------------------------------
    def __len__(self):
        return self.index.n_events

    def _ensure_fork_safe_fds(self):
        """If we are in a different process than the one whose fds the index
        cached (i.e. a forked DataLoader worker), drop the inherited fds so the
        next read reopens process-local ones.  Idempotent and cheap (one
        ``os.getpid()``)."""
        pid = os.getpid()
        if pid != self._fds_pid:
            _drop_inherited_fds(self.index)
            self._fds_pid = pid

    def __getitem__(self, k):
        self._ensure_fork_safe_fds()
        evt = self.index.read_event_at(int(k))
        stack = evt.stack(self.detector, field=self.field, alg=self.alg)
        if stack is None:
            raise ValueError(
                f"event position {int(k)} is missing a segment for detector "
                f"{self.detector!r} (alg={self.alg!r}, field={self.field!r}); "
                f"the default DataLoader collate cannot batch a None -- filter "
                f"it from your sampler or supply a tolerant collate_fn")
        sample = stack if self.transform is None else self.transform(stack)
        if self.to_tensor:
            sample = _to_tensor(sample)
        return sample

    def __repr__(self):
        return (f"XTCDataset(detector={self.detector!r}, "
                f"n_events={len(self)}, field={self.field!r}, "
                f"alg={self.alg!r})")


def _to_tensor(x):
    """Convert ``x`` to a ``torch`` tensor if it is still a numpy ndarray;
    otherwise pass it through (the transform may already have produced a
    tensor).  torch is imported lazily here so module import stays light and
    ``import psdata`` never reaches this code path."""
    import numpy as np
    if isinstance(x, np.ndarray):
        import torch
        # from_numpy shares memory with the ndarray; the ndarray is freshly
        # allocated per event by Event.stack (np.empty), so this is safe.
        return torch.from_numpy(x)
    return x


def worker_init_fn(worker_id):
    """``DataLoader(worker_init_fn=...)`` hook that makes the dataset's index
    fork-safe in this worker process.

    Wiring this in is OPTIONAL: ``XTCDataset.__getitem__`` already drops the
    inherited fds on its first call in a new process (the pid guard).  This hook
    is provided for users who prefer the explicit, documented torch mechanism --
    it forces the reopen once at worker start rather than on first read.  It
    reaches the dataset via ``torch.utils.data.get_worker_info().dataset`` (the
    per-worker *copy* of the dataset), so it works for any ``XTCDataset``.
    """
    import torch
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    ensure = getattr(ds, "_ensure_fork_safe_fds", None)
    if callable(ensure):
        ensure()


def assert_no_framework_imports():
    """Raise AssertionError if a forbidden framework leaked into
    ``sys.modules`` from importing this module's *non-torch* surface.

    ``psdata.torch`` legitimately imports torch (that's its whole point), so
    torch is NOT forbidden here; this keeps psdata's original reader contract
    (no psana / mpi4py / h5py) for the adapter.
    """
    import sys
    forbidden = ("psana", "mpi4py", "h5py")
    leaked = [m for m in forbidden if m in sys.modules]
    assert not leaked, (
        f"psdata.torch must not import {forbidden}; found {leaked}")
