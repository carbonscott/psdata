# psdata

A minimal, clean-room, **pure-Python** reader for LCLS-II **xtc2** files.

`psdata` is the *data-access layer only*: it turns xtc2 stream files into numpy
arrays — **raw detector frames plus event identity (timestamp, pulseId)** — with
**no MPI, no framework, and no calibration**. It parses the xtc2 binary format
directly with `struct` + `numpy.frombuffer`; importing it pulls in **only
numpy** (no `psana`, no `mpi4py`, no `h5py`).

Calibration / geometry (turning raw frames into a calibrated 2-D HDR image) is
deliberately a *separate, optional* layer and is **not** part of this reader.

## Install

Standalone and [uv](https://docs.astral.sh/uv/)-runnable. The reader needs only
numpy:

```bash
git clone <this-repo> psdata && cd psdata
uv venv
uv pip install -e .             # core reader (numpy only)
uv pip install -e ".[test]"     # + pytest for the acceptance suite (optional)
```

```python
import psdata
r = psdata.open(exp="mfx100848724", run=51,
                dir="/sdf/data/lcls/ds/prj/public01/xtc")
```

psana is needed only by the cross-check tests and is deliberately **not** a
pip/uv dependency — see [Environment](#environment).

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

`psdata.open(...)` returns a [`Run`](src/psdata/run.py). A `Run` is a thin facade over the
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
| [`format.py`](src/psdata/format.py) | xtc2 parse core + generic detector/segment discovery | US-001 |
| [`stream.py`](src/psdata/stream.py) | multi-stream event assembly (exact 64-bit-ts k-way merge) | US-002 |
| [`index.py`](src/psdata/index.py) | random-access-by-event index (scans SMD only) + multi-chunk roll; serializable / disk-persisted (`save`/`load`, `to_dict`/`from_dict`) + batch read (`read_events`/`read_stack`) | US-003 / US-004 / US-008 / US-009 |
| [`run.py`](src/psdata/run.py) | public `open()` / `Run` surface | US-005 |
| [`calib/snapshot.py`](src/psdata/calib/snapshot.py) | **separate** layer: one-time calibration-constant snapshot + pinning | US-006 |
| [`hdr/`](src/psdata/hdr) | **separate** layer: standalone offline calibrated 2-D HDR image render | US-007 |

## Examples

Runnable demonstrators live in [`examples/`](examples). They import `psdata`
and nothing heavier unless a demonstrator's own optional dependency is declared.

| example | what it shows |
| --- | --- |
| [`fetch_raw_adu.py`](examples/fetch_raw_adu.py) | walk a run, pull raw ADU + event identity (numpy-only) |
| [`cube_ray_shared_index.py`](examples/cube_ray_shared_index.py) | a GroupBy-Aggregate **cube** parallelized with **Ray**: the driver builds the `RunIndex` **once** and ships it to the workers via Ray's object store (`to_dict()`/`from_dict()`, US-008); each worker **batch-reads** its slice (`read_events`, US-009) and bins by a **real** per-event key (`pulseId`) — **no per-worker SMD rescan**, in contrast to a psana `DataSource`+`build_table` per worker |

The cube demonstrator needs Ray (the `demo` extra, `pip install -e .[demo]`),
which is **not** a core dependency — `import psdata` stays numpy-only. Ray is
imported only by the example. Run it via the env-setup wrapper:

```bash
examples/run_cube_ray.sh                      # 800 evt, workers 1/4/16, +rescan contrast
examples/run_cube_ray.sh --check              # assert parallel cube == serial cube, exit
```

Because the jungfrau frame is ~33.6 MB/event, the cube is **I/O-bandwidth-bound**
on this detector, so its absolute `evt/s` is gated by shared-filesystem read
bandwidth, not by the index; the index's payoff is the **rescan tax** the run
reports — the per-worker SMD scan (~one index build) the shared index removes,
which is what capped the psana prototype's scaling.

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

## Optional: standalone calibrated 2-D HDR image (`psdata.hdr`)

Like `psdata.calib`, the calibrated-image render is a *separate, optional*
layer — **not** part of the reader, and **not** imported by `import psdata`.
`psdata.hdr` turns a raw detector stack into the calibrated 2-D HDR image
**fully offline at render time** (only numpy — no web calib DB, no MPI, no psana
framework), building on a `psdata.calib` snapshot.

```python
# --- one-time prep (needs psana; run under psconda.sh) ---
from psdata.calib import snapshot_calib
from psdata.hdr.geometry import cache_pixel_indexes_for_snapshot
snap_dir = snapshot_calib(exp="mfx100848724", run=51,
                          dir="/sdf/data/lcls/ds/prj/public01/xtc",
                          detname="jungfrau", out_dir="/some/cache")
cache_pixel_indexes_for_snapshot(snap_dir)   # derive ix/iy from geometry.txt
# (GeometryAccess; written as pixel_index_ix.npy / pixel_index_iy.npy)

# --- offline render (pure numpy, NO psana) ---
from psdata.calib import load_snapshot
from psdata.hdr import HDRImager
imager = HDRImager(load_snapshot(snap_dir))
calib  = imager.calib(raw_stack)             # (32,512,1024) f32 == det.raw.calib
image  = imager.image(calib)                 # (4216,4432)   f32 == det.raw.image
# or in one step:
calib, image = imager.render(raw_stack)
```

The pipeline is **byte-exact** versus psana for the reference Jungfrau dataset:
`calib` matches `det.raw.calib(evt)` and `image` matches `det.raw.image(evt)`
with `max|diff| == 0`. Internals are vendored framework-free numpy:

* **gain decode** ([`hdr/jungfrau.py`](src/psdata/hdr/jungfrau.py)) — psana
  `UtilsJungfrau.calib_jungfrau`: gain bits = `raw >> 14` (stage map
  `{0→0, 1→1, 3→2}`, code `2` is bad → 0); `adc = raw & 0x3fff`;
  `calib = (adc − (pedestals+pixel_offset)[stage]) / pixel_gain[stage] · mask`.
* **image remap** ([`hdr/image.py`](src/psdata/hdr/image.py)) — psana
  `UtilsAreaDetector` (`mapmode=2`, `fillholes=True`): scatter into the
  `(image_row, image_col)` grid, take max on overlapping bins, fill single-bin
  holes with the min of four neighbours.
* **geometry** ([`hdr/geometry.py`](src/psdata/hdr/geometry.py)) — the per-pixel
  `(ix, iy)` index maps are derived once from the snapshot's geometry text via
  psana's pure-numpy `GeometryAccess.get_pixel_coord_indexes` (byte-identical
  to `det.raw._pixel_coord_indexes()`) and cached into the snapshot, so the
  render *apply* needs no `GeometryAccess` — that one psana touch is lazy and
  snapshot-time only.

The render is **per-detector-type** (gain decode + geometry differ by
detector); today Jungfrau is wired in. The raw reader (US-001…US-005), by
contrast, is detector-universal.

## Environment

Work runs on host **`sdfiana025`**. The reader itself needs **only numpy** — no
psana, no special environment; a plain `uv pip install -e .` is enough.

The acceptance tests that cross-check against psana
([`tests/test_regression_us005.py`](tests/test_regression_us005.py) and the
optional [`tests/test_calib_us006.py`](tests/test_calib_us006.py) /
[`tests/test_hdr_us007.py`](tests/test_hdr_us007.py)) are the *only* parts that
need a working **psana**, which they use purely to generate ground truth to
compare against (and, for US-006/US-007, to take the one-time snapshot + derive
the geometry index maps).

**psana is the SLAC production conda build, not a pip/uv package** — so this
project intentionally has no `[psana]` extra. Source it from the production
install, which enters via `PYTHONPATH` (prepend, never replace):

```bash
source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
bash run_tests.sh                                   # full acceptance suite
bash run_tests.sh tests/test_regression_us005.py    # just US-005 (byte-exact)
bash run_tests.sh tests/test_calib_us006.py         # calib snapshot
bash run_tests.sh tests/test_hdr_us007.py           # offline HDR render
```

`run_tests.sh` prepends this project's `src/` dir to `PYTHONPATH` so
`import psdata` resolves to the package here while `import psana` resolves to the
production env. Because the package now lives under `src/`, the project root no
longer shadows either import, so the old `.pkgroot`/`.rundir` symlink workaround
is gone.

## Reference dataset

The acceptance tests cross-check against:

* **exp** `mfx100848724`, **run** `51`,
  **dir** `/sdf/data/lcls/ds/prj/public01/xtc`
* detector **`jungfrau`** (Jungfrau 8M, 3-gain auto-ranging) — raw frame is
  `(32, 512, 1024)` uint16, byte-identical to psana's `det.raw.raw(evt)`.

This run is single-chunk and clean; a multi-chunk run
(`mfx101343025` run 35, `mfx101589626` run 31) is used to verify the chunk roll.
