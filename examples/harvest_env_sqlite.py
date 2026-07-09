#!/usr/bin/env python
"""Harvest a run's env store (epics + scan slow data) into SQLite -- no psana.

Machine metadata in LCLS-II is written on *transition* dgrams, not on the
per-event ``L1Accept``:

  * ``epics`` slow-control values ride on every ``SlowUpdate`` (one dgram per
    slow tick -- 84 for this run);
  * ``scan`` step fields (``step_value`` / ``step_docstring``) ride on every
    ``BeginStep`` (one per scan step -- 5 for this run).

This script walks those transition dgrams through psdata's env-store random
access (:class:`psdata.envstore.EnvStore`) and ingests one row per
``(SlowUpdate dgram, epics variable)`` -- plus a handful of scan-step rows -- into
a single SQLite table.  It is per-run metadata, **not** per-event: it never
touches an L1Accept.

psdata is the numpy-only data-access layer, so this harvester imports psdata,
numpy, and the standard library (``sqlite3`` included) and *nothing* else -- no
psana, no MPI, no h5py.  The strongest proof of that is that it runs to
completion under a numpy-only venv where psana is not installed at all.

Run from the repo root with the project's venv::

    .venv/bin/python examples/harvest_env_sqlite.py \
        --exp rixx45619 --run 122 \
        --dir /sdf/data/lcls/ds/prj/public01/xtc \
        --db /tmp/env_r122.sqlite

Schema (table ``env_samples``)
------------------------------
One row per sampled variable at one env-dgram timestamp:

  run                the run number
  store              'epics' | 'scan'
  step               0-based scan step the sample falls in (see below); -1 if
                     the sample precedes the first BeginStep
  ts                 the env dgram's own timestamp (SlowUpdate / BeginStep).
                     A uint64; stored as INTEGER when every ts <= 2**63-1
                     (verified true for this run) -- otherwise as a decimal TEXT
                     string, so nothing is silently truncated into signed 64-bit.
  stream             the owning stream whose dgram carried the payload bytes
  var                the internal variable name (e.g. 'AT1K0_photon_energy')
  pv                 the real EPICS PV name (e.g. 'AT1K0:GAS:PhotonEnergy_RBV').
                     '' for a var with no epicsinfo mapping (e.g. 'StaleFlags')
                     and for every scan var -- such rows are kept, not dropped.
  value              scalar numeric value (float or int), NUMERIC affinity so an
                     int stays INTEGER and a float stays REAL; NULL for
                     str / array / none.
  value_text         string value (NULL otherwise)
  value_json         a JSON list for a rank>=1 array value, e.g. StaleFlags
                     (NULL otherwise)
  value_type         'float' | 'int' | 'str' | 'array' | 'none' -- says which
                     value column to read (nothing is coerced across kinds)
  src_xtc            absolute path of the xtc2/smd file the bytes were read from
  extractor_version  version string of this harvester

The ``step`` of a sample is derived from the ``scan`` store's BeginStep
timestamps with the same as-of rule psana uses for random access,
``searchsorted(begin_step_ts, ts, side='right') - 1``; a sample before the first
BeginStep gets step ``-1`` (none occur in this run).  The scan rows themselves
carry their own step index (step_value 10000->0, 30000->1, ...).

Value representation, explicitly:
  * float / int scalar  -> ``value``      (value_type 'float' / 'int')
  * str (charstr)       -> ``value_text`` (value_type 'str')
  * rank>=1 array       -> ``value_json`` as a JSON list (value_type 'array')
  * None (no value)     -> all three NULL (value_type 'none'); reported, never
                           silently coerced.
"""

import argparse
import json
import os
import sqlite3
import time

import numpy as np

import psdata

# ---- defaults (the golden dataset) --------------------------------------
EXP = "rixx45619"
RUN = 122
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"

TABLE = "env_samples"
EXTRACTOR_VERSION = "psdata.harvest_env_sqlite/1.0"

# SQLite INTEGER is signed 64-bit; env timestamps are uint64.  We store ts as
# INTEGER only when every value fits, else as a decimal TEXT string.
INT64_MAX = 2 ** 63 - 1

# Column order used by every inserted row and by the INSERT statement.
COLUMNS = ("run", "store", "step", "ts", "stream", "var", "pv",
           "value", "value_text", "value_json", "value_type",
           "src_xtc", "extractor_version")


# ---- value / step helpers -----------------------------------------------
def classify(val):
    """Map a decoded env value to ``(value_type, value, value_text, value_json)``.

    Scalars go to ``value`` (float/int), strings to ``value_text``, rank>=1
    arrays to ``value_json`` (a JSON list); ``None`` -> all NULL.  Nothing is
    coerced across kinds -- ``value_type`` records which column is populated.
    """
    if val is None:
        return "none", None, None, None
    if isinstance(val, np.ndarray):
        return "array", None, None, json.dumps(np.asarray(val).ravel().tolist())
    if isinstance(val, (bytes, bytearray)):
        return "str", None, bytes(val).decode("latin-1"), None
    if isinstance(val, str):
        return "str", None, val, None
    if isinstance(val, (bool, np.bool_)):
        return "int", int(val), None, None
    if isinstance(val, (float, np.floating)):
        return "float", float(val), None, None
    if isinstance(val, (int, np.integer)):
        return "int", int(val), None, None
    # Unexpected kind: keep it as its repr rather than dropping or coercing it.
    return "str", None, repr(val), None


def step_of(ts, begin_step_ts):
    """0-based scan step ``ts`` falls in, via the as-of rule
    ``searchsorted(begin_step_ts, ts, 'right') - 1``; -1 before the first
    BeginStep."""
    if len(begin_step_ts) == 0:
        return -1
    return int(np.searchsorted(begin_step_ts, np.uint64(int(ts)),
                               side="right")) - 1


def _owning_streams(store):
    """Sorted stream indices that own (carry the payload for) ``store``'s vars.

    Only the owning stream's copy of a broadcast transition dgram carries the
    container payload, so we harvest from those streams alone (mirrors psana's
    per-stream env managers)."""
    streams = set()
    for var in store.var_names():
        s = store.owning_stream(var)
        if s is not None:
            streams.add(int(s))
    return sorted(streams)


# ---- harvesting ----------------------------------------------------------
def collect_scan(run, scan, env_records):
    """Rows for the scan store + the ascending BeginStep timestamp array.

    Iterates the scan store's owning-stream records; a dgram that does not carry
    the scan container (the BeginRun the store is also fed) yields ``None`` and
    is skipped.  Emits one row each for ``step_value`` and ``step_docstring`` per
    BeginStep."""
    recs_by_stream = env_records.get("scan", {})
    owning = _owning_streams(scan)

    # First pass: the BeginStep timestamps (dgrams that actually carry a value).
    begin = []
    for stream in owning:
        for (ts, path, off, size) in recs_by_stream.get(stream, ()):
            if scan.value("step_value", ts) is not None:
                begin.append(int(ts))
    begin_step_ts = np.array(sorted(set(begin)), dtype=np.uint64)

    # Second pass: emit the scan-step rows.
    rows = []
    for stream in owning:
        for (ts, path, off, size) in recs_by_stream.get(stream, ()):
            if scan.value("step_value", ts) is None:
                continue                       # BeginRun -- no scan container
            step = step_of(ts, begin_step_ts)
            for var in scan.var_names():
                val = scan.value(var, ts)
                vtype, vnum, vtext, vjson = classify(val)
                rows.append((run, "scan", step, int(ts), int(stream), var,
                             scan.pv_name(var), vnum, vtext, vjson, vtype,
                             path, EXTRACTOR_VERSION))
    return rows, begin_step_ts


def collect_epics(run, epics, env_records, begin_step_ts):
    """One row per (SlowUpdate dgram, epics variable).

    Walks the epics store's owning-stream SlowUpdate records; for each dgram
    reads every variable as-of that dgram's own timestamp (an exact hit, so the
    value is the one this SlowUpdate carries) and records it with the dgram's
    step and source file."""
    recs_by_stream = env_records.get("epics", {})
    owning = _owning_streams(epics)
    varlist = epics.var_names()

    rows = []
    for stream in owning:
        for (ts, path, off, size) in recs_by_stream.get(stream, ()):
            step = step_of(ts, begin_step_ts)
            for var in varlist:
                val = epics.value(var, ts)
                vtype, vnum, vtext, vjson = classify(val)
                rows.append((run, "epics", step, int(ts), int(stream), var,
                             epics.pv_name(var), vnum, vtext, vjson, vtype,
                             path, EXTRACTOR_VERSION))
    return rows


def _create_table(conn, ts_coltype):
    """(Re)create the env_samples table -- idempotent: drop then create."""
    conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
    conn.execute(f"""
        CREATE TABLE {TABLE} (
            run                INTEGER NOT NULL,
            store              TEXT    NOT NULL,
            step               INTEGER NOT NULL,
            ts                 {ts_coltype} NOT NULL,
            stream             INTEGER NOT NULL,
            var                TEXT    NOT NULL,
            pv                 TEXT    NOT NULL,
            value              NUMERIC,
            value_text         TEXT,
            value_json         TEXT,
            value_type         TEXT    NOT NULL,
            src_xtc            TEXT    NOT NULL,
            extractor_version  TEXT    NOT NULL,
            PRIMARY KEY (run, store, var, ts)
        )
    """)
    conn.execute(f"CREATE INDEX ix_{TABLE}_pv_ts ON {TABLE} (pv, ts)")
    conn.execute(f"CREATE INDEX ix_{TABLE}_store_step ON {TABLE} (store, step)")


def harvest(exp, run, dir, db_path):
    """Harvest the env store of one run into ``db_path``; return a summary dict."""
    t0 = time.time()
    n_none = 0
    with psdata.open(exp=exp, run=run, dir=dir) as r:
        # build_index() records the env dgram offsets with zero extra I/O; env
        # stores wrap those records.
        env_records = r.build_index().env_records
        epics = r.env_store("epics")
        scan = r.env_store("scan")

        scan_rows, begin_step_ts = collect_scan(run, scan, env_records)
        epics_rows = collect_epics(run, epics, env_records, begin_step_ts)

    rows = scan_rows + epics_rows
    n_none = sum(1 for row in rows if row[COLUMNS.index("value_type")] == "none")

    # uint64 ts -> INTEGER when every value fits signed 64-bit, else TEXT.
    ts_idx = COLUMNS.index("ts")
    max_ts = max(row[ts_idx] for row in rows)
    ts_fits = max_ts <= INT64_MAX
    ts_coltype = "INTEGER" if ts_fits else "TEXT"
    if not ts_fits:
        rows = [row[:ts_idx] + (str(row[ts_idx]),) + row[ts_idx + 1:]
                for row in rows]

    conn = sqlite3.connect(db_path)
    try:
        _create_table(conn, ts_coltype)
        placeholders = ", ".join("?" for _ in COLUMNS)
        conn.executemany(
            f"INSERT OR REPLACE INTO {TABLE} ({', '.join(COLUMNS)}) "
            f"VALUES ({placeholders})", rows)
        conn.commit()

        n_rows = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        n_pv = conn.execute(
            f"SELECT COUNT(DISTINCT pv) FROM {TABLE} WHERE pv != ''"
        ).fetchone()[0]
        n_var = conn.execute(
            f"SELECT COUNT(DISTINCT var) FROM {TABLE}").fetchone()[0]
        steps = [r[0] for r in conn.execute(
            f"SELECT DISTINCT step FROM {TABLE} ORDER BY step")]
        n_ts = conn.execute(
            f"SELECT COUNT(DISTINCT ts) FROM {TABLE}").fetchone()[0]
    finally:
        conn.close()

    return {
        "db_path": os.path.abspath(db_path),
        "n_rows": n_rows,
        "n_epics_rows": len(epics_rows),
        "n_scan_rows": len(scan_rows),
        "n_none": n_none,
        "distinct_pvs": n_pv,
        "distinct_vars": n_var,
        "distinct_ts": n_ts,
        "steps": steps,
        "max_ts": max_ts,
        "ts_coltype": ts_coltype,
        "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", default=EXP, help="experiment id")
    ap.add_argument("--run", type=int, default=RUN, help="run number")
    ap.add_argument("--dir", default=DIR, help="dir holding the run's xtc2 files")
    ap.add_argument("--db", default=None,
                    help="output SQLite path (default /tmp/env_r<run>.sqlite)")
    args = ap.parse_args()

    db_path = args.db or f"/tmp/env_r{args.run}.sqlite"
    s = harvest(args.exp, args.run, args.dir, db_path)

    print(f"harvested exp={args.exp} run={args.run} -> {s['db_path']}")
    print(f"  rows inserted : {s['n_rows']}  "
          f"(epics {s['n_epics_rows']}, scan {s['n_scan_rows']}, "
          f"unrepresentable/none {s['n_none']})")
    print(f"  distinct PVs  : {s['distinct_pvs']}  (real EPICS PV names, pv != '')")
    print(f"  distinct vars : {s['distinct_vars']}")
    print(f"  distinct steps: {s['steps']}")
    print(f"  distinct env dgram ts: {s['distinct_ts']}")
    print(f"  ts stored as  : {s['ts_coltype']}  "
          f"(max ts {s['max_ts']} {'<=' if s['ts_coltype']=='INTEGER' else '>'} "
          f"2**63-1={INT64_MAX})")
    print(f"  extractor     : {EXTRACTOR_VERSION}")
    print(f"  elapsed       : {s['elapsed_s']:.2f}s")


if __name__ == "__main__":
    main()
