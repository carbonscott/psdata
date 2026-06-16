#!/usr/bin/env python3
"""US-011 -- torch-style ``XTCDataset`` adapter (demonstrator).

A map-style ``torch.utils.data.Dataset`` over a psdata run: ``ds[k]`` random-
accesses the ``k``-th event via the SAME index machinery US-008 made
serializable and US-009 made batchable.  This suite checks three things:

  1. **Import purity** (always runs, no torch needed): a fresh subprocess
     asserts ``import psdata`` does NOT pull torch into ``sys.modules`` -- only
     ``import psdata.torch`` does.  Mirrors ``test_persist_us008``'s purity check
     and the lazy-import discipline of ``psdata.calib.snapshot``.

  2. **Correctness** (needs torch): ``ds[k]`` is byte-identical to
     ``read_event_at(k).stack(detector)``, returned as a ``torch`` tensor.

  3. **Fork-safety** (needs torch): the load-bearing one.  ``DataLoader(
     num_workers=2)`` forks worker processes that inherit the parent's
     ``RunIndex._bd_fds`` cache of raw OS fd integers (invalid in the child).
     Two checks:
       (a) a real ``DataLoader(num_workers=2)`` pass yields events byte-
           identical to the serial reference (the smoke test the story asks
           for), and the PARENT's fds still work afterwards (workers did not
           close them);
       (b) a DETERMINISTIC ``os.fork`` test that forces fd-number reuse in the
           child -- proving that reading off the stale cache corrupts/raises,
           while the adapter's drop-the-fds reopen recovers exactly.  This makes
           the hazard reproducible rather than relying on the OS not reusing fd
           numbers.

The torch-dependent tests SKIP (not fail) when torch is not installed, per the
story; the import-purity test runs regardless.

Run (on sdfiana025, from the repo root):
    PYTHONPATH=src .venv/bin/python tests/test_torch_us011.py
or via the suite:
    bash run_tests.sh tests/test_torch_us011.py
"""

import glob
import os
import subprocess
import sys

import numpy as np

# --- locate the package (parent of this tests dir) --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")  # .../<repo>/src (holds psdata/)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Reference dataset -- single-chunk primary run (kept in the TEST, never the lib).
EXP = "mfx100848724"
RUN = 51
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
JUNGFRAU = "jungfrau"


def _have_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _stream_files(directory, exp, run):
    paths = sorted(glob.glob(f"{directory}/{exp}-r{run:04d}-s*-c000.xtc2"))
    assert paths, f"no stream files found under {directory} for {exp} r{run}"
    files = {}
    for p in paths:
        base = os.path.basename(p)
        sidx = int(base.split("-s")[1].split("-")[0])
        files[sidx] = p
    return files


def _build_index():
    import psdata
    files = _stream_files(DIR, EXP, RUN)
    rc = psdata.discover(files)
    return psdata.build_index(files, run_config=rc)


# ==========================================================================
# 1. Import purity -- ALWAYS runs (does not need torch installed)
# ==========================================================================
def test_import_purity_subprocess():
    """`import psdata` must NOT import torch; only `import psdata.torch` does."""
    # (a) plain `import psdata` does not pull torch in.
    code_a = (
        "import sys, psdata; "
        "print('BAD' if 'torch' in sys.modules else 'OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code_a],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": _SRC},
    )
    assert proc.returncode == 0, f"`import psdata` failed:\n{proc.stderr}"
    assert proc.stdout.strip() == "OK", \
        f"`import psdata` leaked torch into sys.modules: {proc.stdout.strip()}"

    # (b) the adapter module itself must not leak psana/mpi4py/h5py (its own
    #     reader contract) -- importing psdata.torch IS allowed to import torch.
    code_b = (
        "import sys, psdata.torch as t; "
        "t.assert_no_framework_imports(); "
        "bad=[m for m in ('psana','mpi4py','h5py') if m in sys.modules]; "
        "print('BAD' if bad else 'OK', bad)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code_b],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": _SRC},
    )
    # If torch is not installed, `import psdata.torch` succeeds (torch is lazy
    # inside functions) -- the module body imports only `os`. So this check is
    # valid with or without torch installed.
    assert proc.returncode == 0, \
        f"`import psdata.torch` framework check failed:\n{proc.stderr}"
    assert proc.stdout.startswith("OK"), \
        f"psdata.torch leaked a forbidden framework: {proc.stdout.strip()}"
    print("[purity] `import psdata` stays numpy-only; psdata.torch is "
          "framework-free (no psana/mpi4py/h5py)")


# ==========================================================================
# 2. Correctness -- ds[k] == read_event_at(k).stack(det), as a torch tensor
# ==========================================================================
def test_getitem_byte_identical():
    if not _have_torch():
        print("[skip] correctness: torch not installed")
        return
    import torch
    from psdata.torch import XTCDataset

    idx = _build_index()
    ds = XTCDataset(idx, JUNGFRAU)
    assert len(ds) == idx.n_events, "len(ds) must equal index.n_events"

    for k in (0, 1, 17, 999, idx.n_events - 1):
        t = ds[k]
        assert isinstance(t, torch.Tensor), f"ds[{k}] is not a tensor"
        ref = idx.read_event_at(k).stack(JUNGFRAU)
        assert ref is not None, f"reference event {k} unexpectedly missing"
        assert t.shape == tuple(ref.shape), \
            f"ds[{k}] shape {tuple(t.shape)} != {ref.shape}"
        assert np.array_equal(t.numpy(), ref), \
            f"ds[{k}] not byte-identical to read_event_at({k}).stack"
    print(f"[ok] ds[k] byte-identical to read_event_at(k).stack as a tensor "
          f"({len(ds)} events)")

    # transform is applied before tensor conversion.
    ds2 = XTCDataset(idx, JUNGFRAU,
                     transform=lambda a: a.astype(np.float32) / 2.0)
    out = ds2[0]
    assert out.dtype == torch.float32
    assert np.allclose(out.numpy(),
                       idx.read_event_at(0).stack(JUNGFRAU).astype(np.float32) / 2.0)
    print("[ok] user transform applied before tensor conversion")


# ==========================================================================
# 3a. Fork-safety -- real DataLoader(num_workers=2) yields correct events
# ==========================================================================
def test_dataloader_num_workers_fork_safe():
    if not _have_torch():
        print("[skip] DataLoader fork-safety: torch not installed")
        return
    import torch
    from torch.utils.data import DataLoader, Subset
    from psdata.torch import XTCDataset, worker_init_fn

    idx = _build_index()

    # Populate the PARENT's fd cache BEFORE forking workers -- this is what
    # makes the gotcha real (stale parent fds get inherited by the children).
    ref0 = idx.read_event_at(0).stack(JUNGFRAU)
    assert idx._bd_fds, "expected parent fd cache to be populated by a read"
    parent_fds_before = dict(idx._bd_fds)

    ks = list(range(0, 64))
    ref = np.stack([idx.read_event_at(k).stack(JUNGFRAU) for k in ks])

    ds = XTCDataset(idx, JUNGFRAU)
    sub = Subset(ds, ks)   # keep DataLoader's default sampler order == ks
    dl = DataLoader(sub, batch_size=8, num_workers=2, shuffle=False,
                    worker_init_fn=worker_init_fn)
    got = np.concatenate([b.numpy() for b in dl], axis=0)
    assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"
    assert np.array_equal(got, ref), \
        "DataLoader(num_workers=2) produced corrupt/incorrect events"

    # The workers must NOT have closed the parent's fds (they only DROPPED their
    # inherited copies).  The parent can still read after the workers finished.
    assert idx._bd_fds == parent_fds_before, \
        "parent fd cache changed -- a worker mutated the parent's fds"
    again = idx.read_event_at(0).stack(JUNGFRAU)
    assert np.array_equal(again, ref0), \
        "parent read corrupted after DataLoader workers ran"
    print("[ok] DataLoader(num_workers=2) byte-identical to serial; parent "
          "fds intact and still readable after workers finished")


# ==========================================================================
# 3b. Fork-safety -- DETERMINISTIC: force fd-number reuse, prove drop-reopen
# ==========================================================================
def test_fork_fd_reuse_is_recovered():
    """Reproduce the gotcha deterministically (no reliance on the OS happening
    to reuse fd numbers): in a forked child, close the inherited fds and grab
    those numbers for /dev/null, so the cached ints now point at the WRONG file.
    Reading off the stale cache must fail (raise or wrong data); dropping the
    cache (what the adapter does after fork) and reopening must recover exactly.
    """
    if not _have_torch():
        # This test exercises the fd discipline, not torch directly, but it is
        # the deterministic backstop for the DataLoader test -- group it with
        # the torch-dependent block so the suite has one clear skip story.
        print("[skip] deterministic fd-reuse: grouped with torch tests")
        return
    from psdata.torch import _drop_inherited_fds

    idx = _build_index()
    ref0 = idx.read_event_at(0).stack(JUNGFRAU)   # populate parent fds
    parent_fds = sorted(idx._bd_fds.values())
    assert parent_fds, "expected parent fd cache to be populated"

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                   # ---- child ----
        os.close(r)
        try:
            # Free the inherited fd numbers, then occupy them with /dev/null so
            # a pread on the cached int reads the WRONG (empty) file.
            for fd in parent_fds:
                os.close(fd)
            hog = [os.open(os.devnull, os.O_RDONLY) for _ in parent_fds]  # noqa: F841
            # Stale cache -> corruption.  Either a raise or a non-matching/None.
            noguard_failed = False
            try:
                bad = idx.read_event_at(0).stack(JUNGFRAU)
                noguard_failed = (bad is None) or (not np.array_equal(bad, ref0))
            except Exception:
                noguard_failed = True
            # The adapter's fix: drop the inherited fds (NO os.close -- already
            # closed) and let _bd_fd reopen process-local ones.
            _drop_inherited_fds(idx)
            recovered = idx.read_event_at(0).stack(JUNGFRAU)
            ok = noguard_failed and np.array_equal(recovered, ref0)
            os.write(w, b"OK" if ok else
                     ("FAIL noguard_failed=%s recovered_ok=%s"
                      % (noguard_failed,
                         np.array_equal(recovered, ref0))).encode())
        except Exception as e:   # pragma: no cover - surface child errors
            os.write(w, ("CHILD-ERR %s: %s" % (type(e).__name__, e)).encode())
        finally:
            os._exit(0)
    else:                                          # ---- parent ----
        os.close(w)
        out = b""
        while True:
            c = os.read(r, 4096)
            if not c:
                break
            out += c
        os.close(r)
        os.waitpid(pid, 0)
        msg = out.decode()
        assert msg == "OK", f"deterministic fd-reuse fork test failed: {msg!r}"

    # Parent's own fds are untouched and still read correctly after the fork.
    assert np.array_equal(idx.read_event_at(0).stack(JUNGFRAU), ref0), \
        "parent corrupted after deterministic fork test"
    print("[ok] deterministic fd-reuse in a forked child corrupts the stale "
          "cache; drop+reopen recovers byte-identical; parent unaffected")


if __name__ == "__main__":
    test_import_purity_subprocess()
    print("[ok] import purity (subprocess)")
    test_getitem_byte_identical()
    test_dataloader_num_workers_fork_safe()
    test_fork_fd_reuse_is_recovered()
    print("\nALL US-011 TESTS PASSED" +
          ("" if _have_torch() else " (torch-dependent tests SKIPPED)"))
