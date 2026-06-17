#!/usr/bin/env python
"""Fetch raw ADU data with psdata.

Walks the first few events of a run and, for each, pulls the raw detector
frame -- raw ADU straight off the xtc2 stream, *no* calibration -- together
with the event identity (timestamp, pulseId).

psdata is the numpy-only data-access layer: this script imports psdata and
numpy and nothing else (no psana, no MPI, no framework).

Run from the repo root with the project's venv:

    .venv/bin/python examples/fetch_raw_adu.py
"""

import numpy as np

import psdata

EXP = "mfx100848724"
RUN = 51
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
DET = "jungfrau"
N_EVT = 5  # how many events to walk


def main():
    with psdata.open(exp=EXP, run=RUN, dir=DIR) as r:
        print(f"opened exp={EXP} run={RUN} det={DET}")
        print(f"detectors: {r.detector_names()}")
        print()

        for i, evt in enumerate(r.events()):
            if i >= N_EVT:
                break

            raw = evt.stack(DET)  # (32, 512, 1024) uint16 raw ADU, or None
            if raw is None:
                print(f"[{i}] ts={evt.timestamp}  pulseId={evt.pulseId}  "
                      f"{DET}: <no full segment set>")
                continue

            # In a raw Jungfrau frame the low 14 bits are the ADC count and the
            # top 2 bits are the gain stage -- splitting them out is the only
            # interpretation applied here; the array itself is untouched.
            adc = raw & 0x3FFF
            gain = raw >> 14
            print(f"[{i}] ts={evt.timestamp}  pulseId={evt.pulseId}")
            print(f"      raw  shape={raw.shape} dtype={raw.dtype}  "
                  f"min={raw.min()} max={raw.max()}")
            print(f"      adc  (raw & 0x3fff)  min={adc.min()} max={adc.max()} "
                  f"mean={adc.mean():.1f}")
            print(f"      gain (raw >> 14)     stages_present={np.unique(gain).tolist()}")
            print(f"      seg0 corner raw[0,0,:6]={raw[0, 0, :6].tolist()}")
            print()


if __name__ == "__main__":
    main()
