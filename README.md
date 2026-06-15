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

## Environment

Work runs on host **`sdfiana025`**. The reader itself needs **only numpy** — no
psana, no special environment.

The regression test ([`tests/test_regression_us005.py`](tests/test_regression_us005.py))
is the *only* part that needs a working **psana**, which it uses purely to
generate ground truth to compare against. Use the production install:

```bash
source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
bash psdata/run_tests.sh                       # run the full acceptance suite
bash psdata/run_tests.sh psdata/tests/test_regression_us005.py   # just US-005
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
