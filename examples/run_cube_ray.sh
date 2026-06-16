#!/usr/bin/env bash
# Run the Ray shared-index cube demonstrator (examples/cube_ray_shared_index.py).
#
# Ray is a DEMONSTRATOR-only dependency -- it is NOT in psdata's core deps and
# `import psdata` stays numpy-only.  Ray is also not in the psdata .venv nor in
# the production psana conda.  The owner's Ray prototype ships a uv venv with
# Ray installed at cube_prototype/.venv (ray 2.51.2 on the psana python); we
# reuse that interpreter and just prepend THIS repo's src/ so `import psdata`
# resolves to the numpy-only reader (psdata pulls in only numpy, so layering it
# on the Ray interpreter does not violate the framework-free contract).
#
# Usage (on sdfiana025, from the repo root):
#   examples/run_cube_ray.sh                     # default: 2000 evt, workers 1/4/16
#   examples/run_cube_ray.sh --events 4000 --bins 16
#   examples/run_cube_ray.sh --workers 1 4
#   examples/run_cube_ray.sh --check             # assert parallel==serial, then exit
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/src"

# The owner's Ray venv (ray 2.51.2).  Override with RAY_VENV=/path/to/venv if the
# cube_prototype venv has moved.
RAY_VENV="${RAY_VENV:-/sdf/data/lcls/ds/prj/prjcwang31/results/cube_prototype/.venv}"
RAY_PY="$RAY_VENV/bin/python"

if [[ ! -x "$RAY_PY" ]]; then
  echo "ERROR: Ray interpreter not found at $RAY_PY" >&2
  echo "Set RAY_VENV to a venv that has 'ray' installed (the demo [demo] extra)," >&2
  echo "or: uv venv .demo-venv && uv pip install --python .demo-venv ray && \\" >&2
  echo "    RAY_VENV=\$PWD/.demo-venv examples/run_cube_ray.sh" >&2
  exit 1
fi

# The cube_prototype venv is a thin uv venv over the psana conda python with
# `include-system-site-packages = false`, so numpy (which lives in that conda
# env) is reached via PYTHONPATH -- the same trick the owner's run_cube.sh uses.
# We add ONLY the conda site-packages (numpy), NOT the psana rel dir, so psana
# is never importable here: the demo needs numpy, not the framework.
CONDA_SITE="${CONDA_SITE:-/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps_20241122/lib/python3.9/site-packages}"

# Prepend psdata's src/ (numpy-only) so `import psdata` resolves to this repo,
# then the conda site-packages so `import numpy` resolves.
export PYTHONPATH="$SRC:$CONDA_SITE${PYTHONPATH:+:$PYTHONPATH}"
# Quiet Ray's GPU-accelerator probe on CPU nodes (matches the owner's run_cube.sh).
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

exec "$RAY_PY" "$REPO/examples/cube_ray_shared_index.py" "$@"
