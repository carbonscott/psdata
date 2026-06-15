# psdata

A minimal, clean-room, **pure-Python** reader for LCLS-II **xtc2** files.

`psdata` is the *data-access layer only*: it turns xtc2 stream files into numpy
arrays — **raw detector frames plus event identity (timestamp, pulseId)** — with
**no MPI, no framework, and no calibration**. It parses the xtc2 binary format
directly with `struct` + `numpy.frombuffer`; importing it pulls in **only
numpy** (no `psana`, no `mpi4py`, no `h5py`).

Calibration / geometry (turning raw frames into a calibrated 2-D HDR image) is
deliberately a *separate, optional* layer and is **not** part of this reader.

## Public API

Open a run, then stream it, random-access it by timestamp, or introspect it.

```python
import psdata

# Open a run -- either by exp / run / dir (standard file layout) ...
r = psdata.open(exp="mfx100848724", run=51,
                dir="/sdf/data/lcls/ds/prj/public01/xtc")

# ... or by an explicit list of per-stream xtc2 files
r = psdata.open(files=["/path/...-s000-c000.xtc2",
                       "/path/...-s001-c000.xtc2", ...])

# Introspect: detectors, their fields (name / dtype / rank), and segments
r.detector_names()                 # -> ['epicsinfo', 'jungfrau', 'timing', ...]
det = r.detector("jungfrau")
det.alg_names()                    # -> ['config', 'raw']
det.field_names("raw")             # -> ['raw', ...]
det.segment_ids("raw")             # -> [0, 1, ..., 31]
det.algs["raw"]["raw"]             # FieldInfo(name='raw', dtype=uint16, rank=3)

# Forward streaming -- assembled events in ascending timestamp order
for evt in r.events():
    ts  = evt.timestamp            # 64-bit (sec << 32) | nsec
    pid = evt.pulseId              # timing detector's pulseId (bit 63 masked)
    jf  = evt.stack("jungfrau")    # (32, 512, 1024) uint16 stack, or None
    # evt.raw("jungfrau")          # -> {segment: ndarray} or None
    # evt.as_dict()                # -> {det_name: {segment: ndarray} | None}

# Random access by event -- builds an index from the small SMD files only
evt = r.read_event(ts)             # by exact 64-bit timestamp
evt = r.read_event_at(1000)        # ... or by event position (0-based, ts order)

r.close()                          # release index file descriptors
```

`psdata.open(...)` returns a [`Run`](run.py). A `Run` is a thin facade over the
three layers below; use it as a context manager (`with psdata.open(...) as r:`)
to auto-release the random-access index's file descriptors.

### What an event exposes

| accessor | returns |
| --- | --- |
| `evt.timestamp` | 64-bit event timestamp `(sec << 32) \| nsec` |
| `evt.pulseId` | the timing detector's `pulseId` (LCLS-1 bit 63 masked off), or `None` |
| `evt.stack(name)` | `(n_segments, *seg_shape)` array ordered by segment id, or `None` |
| `evt.raw(name)` | `{segment: ndarray}`, or `None` if a segment is missing |
| `evt.as_dict()` | `{det_name: {segment: ndarray} \| None}` for every real detector |
| `evt.damage(name)` | `{segment: (damage_id, userbits)}` per-segment damage |

A detector whose received segment set does not match its full declared set is
returned as `None` for that event (psana's missing-segment rule). Raw arrays are
never silently dropped on damage; damage is *surfaced* via `evt.damage(...)`.

### File-layout convention

`open(exp=, run=, dir=)` resolves the run's files by the standard LCLS-II layout
— this is the only place path patterns live:

* bigdata streams: `{dir}/{exp}-r{run:04d}-s{stream:03d}-c000.xtc2`
* SMD index: `{dir}/smalldata/{base}.smd.xtc2` for each bigdata `{dir}/{base}.xtc2`

Only the first chunk (`c000`) of each stream is opened; later chunks (`c001`, …)
are followed automatically via the `chunkinfo` roll on each `Enable` transition.

### Lower-level functional API

The `Run` facade is built on a functional API that is also public:

```python
rc   = psdata.discover(stream_files)                 # -> RunConfig
evts = psdata.events(stream_files, run_config=rc)    # stream in ts order
ridx = psdata.build_index(stream_files, run_config=rc)   # -> RunIndex (SMD scan)
evt  = ridx.read_event(ts)
```

`stream_files` accepts a `{index: path}` dict, `(index, path)` pairs, or a plain
list of paths.

## Modules

| module | layer | story |
| --- | --- | --- |
| [`format.py`](format.py) | xtc2 parse core + generic detector/segment discovery | US-001 |
| [`stream.py`](stream.py) | multi-stream event assembly (exact 64-bit-ts k-way merge) | US-002 |
| [`index.py`](index.py) | random-access-by-event index (scans SMD only) + multi-chunk roll | US-003 / US-004 |
| [`run.py`](run.py) | public `open()` / `Run` surface | US-005 |
| [`calib/snapshot.py`](calib/snapshot.py) | **separate** layer: one-time calibration-constant snapshot + pinning | US-006 |

## Optional: calibration-constant snapshot (`psdata.calib`)

Calibration is **not** part of the reader. `psdata.calib` is a *separate*
sub-package: a one-time step that snapshots a detector's calibration constants
(the only network/DB dependency) so a later calibrated-image render can run
fully offline. Importing the reader (`import psdata`) does **not** import it, so
the reader stays numpy-only.

```python
# --- one-time snapshot (needs psana; run under psconda.sh) ---
from psdata.calib import snapshot_calib
snap_dir = snapshot_calib(exp="mfx100848724", run=51,
                          dir="/sdf/data/lcls/ds/prj/public01/xtc",
                          detname="jungfrau", out_dir="/some/cache")
# writes {out_dir}/jungfrau_r0051/ : pedestals/pixel_gain/pixel_offset/...npy,
# mask.npy, geometry.txt, manifest.json

# --- reload offline (pure numpy, NO psana) ---
from psdata.calib import load_snapshot
snap = load_snapshot(snap_dir)
snap.pedestals          # (3,32,512,1024) f32  -- leading axis = 3 gain stages
snap.pixel_gain         # (3,32,512,1024) f32
snap.pixel_offset       # (3,32,512,1024) f32  (None if absent -> treat as 0)
snap.mask               # (32,512,1024)   u8   -- det.raw._mask(status=True)
snap.geometry           # ~5-8 KB geometry text
snap.calibconst()       # rebuilds psana's {ctype: (array, meta)} dict
```

Snapshots are **pinned by `(detector_uniqueid, run)`** and retain each
constant's validity metadata (`run` / `run_end` / `version`). A reload
reproduces the exact arrays psana's `det.raw._calibconst` returns
(`np.array_equal`). **Staleness is silent:** a constant's validity *run* can
differ from the pin run (e.g. pedestals valid from run 49, pin run 51); reusing
a snapshot outside a constant's validity range gives wrong-but-silent results.
`snap.validity(ctype)` and `snap.is_valid_for_run(run)` expose the ranges for an
opt-in check; the package never refuses a stale reload.

## Environment

Work runs on host **`sdfiana025`**. The reader itself needs **only numpy** — no
psana, no special environment.

The acceptance tests that cross-check against psana
([`tests/test_regression_us005.py`](tests/test_regression_us005.py) and the
optional [`tests/test_calib_us006.py`](tests/test_calib_us006.py)) are the
*only* parts that need a working **psana**, which they use purely to generate
ground truth to compare against (and, for US-006, to take the one-time
snapshot). Use the production install:

```bash
source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
bash psdata/run_tests.sh                       # run the full acceptance suite
bash psdata/run_tests.sh psdata/tests/test_regression_us005.py   # just US-005
bash psdata/run_tests.sh psdata/tests/test_calib_us006.py        # calib snapshot
```

`run_tests.sh` prepends an isolated package-parent dir to `PYTHONPATH` so
`import psdata` resolves to this package while `import psana` resolves to the
production env (the repo root holds a single-file `psdata.py` reference and an
unbuilt `psana/` clone, both of which would otherwise shadow the real ones).

## Reference dataset

The acceptance tests cross-check against:

* **exp** `mfx100848724`, **run** `51`,
  **dir** `/sdf/data/lcls/ds/prj/public01/xtc`
* detector **`jungfrau`** (Jungfrau 8M, 3-gain auto-ranging) — raw frame is
  `(32, 512, 1024)` uint16, byte-identical to psana's `det.raw.raw(evt)`.

This run is single-chunk and clean; a multi-chunk run
(`mfx101343025` run 35, `mfx101589626` run 31) is used to verify the chunk roll.
