#!/usr/bin/env python3
"""tools/check_purity.py -- canonical import-purity gate for psdata.

Project guardrail: no ``psdata`` module may import ``psana``, ``mpi4py``,
``h5py``, or ``xtcdata`` at runtime.  Allowed runtime deps: numpy + the
Python standard library (``sqlite3`` included).

Today that guardrail is convention-by-repetition: ``assert_no_framework_imports``
defined in ``src/psdata/format.py`` and re-exported from ``stream.py``,
``run.py``, ``index.py``, plus a scattering of per-test-file
``test_import_purity_*`` functions.  This script is the single canonical gate:
it discovers every module under ``src/psdata/`` dynamically (so new modules,
e.g. ``envstore.py``, are covered automatically the moment they exist) and
checks each one two independent ways:

1. **Dynamic check** -- import the module *alone* in a **fresh interpreter
   subprocess** (so a leak from one module can't be masked by another module
   already populating ``sys.modules`` in-process), then inspect
   ``sys.modules`` for the forbidden top-level packages.  Matching is on the
   top-level package name, so a submodule import such as ``psana.psexp``
   still counts as ``psana`` leaking.

2. **Static check** -- walk the AST of the module's source and flag any
   ``import`` / ``from ... import`` (including a bare ``importlib.import_module(...)``
   / ``__import__(...)`` call with a literal string argument) naming a
   forbidden package.  This catches an import that is lazy or guarded behind
   a branch that the dynamic check's plain ``import`` wouldn't exercise.

Usage:
    python tools/check_purity.py
        # checks every src/psdata/*.py module

    python tools/check_purity.py examples/harvest_env_sqlite.py [more.py ...]
        # ALSO checks extra script/module targets (e.g. a harvester script),
        # in addition to the src/psdata/*.py sweep.

A target may fail to import for a reason unrelated to purity: an *allowed but
optional* runtime dependency (currently just ``torch``, lazily imported by
``psdata/torch.py``) might not be installed in the current venv.  That is
reported as SKIPPED, not FAIL -- absence of an allowed optional dependency is
not a purity violation.

Exit code: 0 if PURITY PASS, 1 if PURITY FAIL.
"""
import ast
import glob
import json
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FORBIDDEN = ("psana", "mpi4py", "h5py", "xtcdata")

# Known-allowed *optional* runtime deps.  Their absence must never fail the
# gate -- only their presence in the forbidden set would.  (torch is imported
# lazily, inside functions, by psdata/torch.py; it is not forbidden, but may
# simply not be installed in a numpy-only venv.)
OPTIONAL_ALLOWED = ("torch",)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
PSDATA_SRC = os.path.join(SRC_DIR, "psdata")

SUBPROCESS_TIMEOUT_S = 120


def _top_level(dotted_name):
    """'psana.psexp' -> 'psana'; 'numpy' -> 'numpy'."""
    return dotted_name.split(".", 1)[0]


def discover_psdata_modules():
    """Glob every src/psdata/*.py -> [(import_name, file_path), ...], sorted.

    __init__.py maps to the package import name "psdata" itself; every other
    file "<mod>.py" maps to "psdata.<mod>". Discovering by glob (rather than
    a hardcoded list) means a new module (e.g. envstore.py) is picked up
    automatically the moment it exists on disk -- no edit to this script
    required.
    """
    mods = []
    for path in sorted(glob.glob(os.path.join(PSDATA_SRC, "*.py"))):
        base = os.path.basename(path)
        import_name = "psdata" if base == "__init__.py" else "psdata." + base[:-3]
        mods.append((import_name, path))
    return mods


# ---------------------------------------------------------------------------
# Static check: AST walk for forbidden import statements (incl. lazy/guarded
# imports that a plain dynamic `import` wouldn't necessarily trigger).
# ---------------------------------------------------------------------------
def static_check(path):
    """Return the set of forbidden top-level package names that this source
    file names in an import statement (or a literal-string
    importlib.import_module/__import__ call), anywhere in the file --
    including inside function bodies (lazy imports) and conditional branches
    (guarded imports)."""
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src, filename=path)
    hits = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = _top_level(alias.name)
                if top in FORBIDDEN:
                    hits.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # "from psana import DataSource" / "from xtcdata.dgram import X"
                # (node.module never carries the leading dots; those live in
                # node.level, so this also catches "from .xtcdata import X".)
                top = _top_level(node.module)
                if top in FORBIDDEN:
                    hits.add(top)
            elif node.level > 0:
                # bare "from . import <name>" -- guard against a local name
                # that shadows a forbidden package name.
                for alias in node.names:
                    if alias.name in FORBIDDEN:
                        hits.add(alias.name)
        elif isinstance(node, ast.Call):
            func = node.func
            is_dynamic_import_call = (
                (isinstance(func, ast.Name) and func.id == "__import__")
                or (isinstance(func, ast.Attribute) and func.attr == "import_module")
            )
            if is_dynamic_import_call and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    top = _top_level(arg0.value)
                    if top in FORBIDDEN:
                        hits.add(top)
    return hits


# ---------------------------------------------------------------------------
# Dynamic check: import the target ALONE in a fresh interpreter subprocess.
# ---------------------------------------------------------------------------
_SUBPROCESS_SRC = r'''
import sys, json
forbidden = {forbidden!r}
optional = {optional!r}
import_name = {import_name!r}
file_path = {file_path!r}
result = {{"ok": True, "leaked": [], "error": None, "skipped_optional": None}}
try:
    if import_name:
        import importlib
        importlib.import_module(import_name)
    else:
        # Standalone script/module target given by file path: import it AS A
        # MODULE (not run it), so a "if __name__ == '__main__':" guard in the
        # target does not fire -- module __name__ here is "_purity_target",
        # never "__main__".
        import importlib.util
        spec = importlib.util.spec_from_file_location("_purity_target", file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_purity_target"] = mod
        spec.loader.exec_module(mod)
except ImportError as e:
    missing_top = (getattr(e, "name", None) or "").split(".")[0]
    if missing_top and missing_top in optional:
        result["skipped_optional"] = missing_top
    else:
        result["ok"] = False
        result["error"] = "{{}}: {{}}".format(type(e).__name__, e)
except Exception as e:
    result["ok"] = False
    result["error"] = "{{}}: {{}}".format(type(e).__name__, e)
result["leaked"] = sorted({{m.split(".", 1)[0] for m in sys.modules}} & set(forbidden))
print("PURITY_RESULT_JSON:" + json.dumps(result))
'''


def dynamic_check(import_name, file_path):
    """Import (import_name, if set, else file_path-as-module) alone in a
    fresh subprocess. Returns dict(ok, leaked, error, skipped_optional)."""
    src = _SUBPROCESS_SRC.format(
        forbidden=list(FORBIDDEN), optional=list(OPTIONAL_ALLOWED),
        import_name=import_name, file_path=file_path)
    env = dict(os.environ)
    pypath = [SRC_DIR]
    existing = env.get("PYTHONPATH")
    if existing:
        pypath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pypath)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", src], capture_output=True, text=True,
            timeout=SUBPROCESS_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "leaked": [],
                "error": "subprocess timed out after "
                         f"{SUBPROCESS_TIMEOUT_S}s",
                "skipped_optional": None}
    for line in proc.stdout.splitlines():
        if line.startswith("PURITY_RESULT_JSON:"):
            return json.loads(line[len("PURITY_RESULT_JSON:"):])
    return {"ok": False, "leaked": [],
            "error": f"subprocess produced no result (rc={proc.returncode}); "
                     f"stderr:\n{(proc.stdout + proc.stderr)[-2000:]}",
            "skipped_optional": None}


def check_target(import_name, file_path):
    """Run static + dynamic checks on one target. Returns a result dict:
    {"label", "status" in {"PASS","FAIL","SKIPPED"}, "detail"}."""
    if import_name:
        label = import_name
    elif file_path.startswith(REPO_ROOT + os.sep):
        label = os.path.relpath(file_path, REPO_ROOT)
    else:
        label = file_path  # outside the repo (e.g. a /tmp throwaway) -- show absolute
    static_hits = static_check(file_path)
    if static_hits:
        return {"label": label, "status": "FAIL",
                "detail": "AST: forbidden import(s) found in source: "
                          f"{sorted(static_hits)}"}

    dyn = dynamic_check(import_name, file_path)
    if not dyn["ok"]:
        return {"label": label, "status": "FAIL",
                "detail": f"subprocess import error: {dyn['error']}"}
    if dyn["leaked"]:
        return {"label": label, "status": "FAIL",
                "detail": f"leaked into sys.modules: {dyn['leaked']}"}
    if dyn["skipped_optional"]:
        return {"label": label, "status": "SKIPPED",
                "detail": "optional (allowed) dependency "
                          f"'{dyn['skipped_optional']}' not installed in "
                          "this venv -- not a purity failure"}
    return {"label": label, "status": "PASS", "detail": "clean"}


def main(argv):
    targets = discover_psdata_modules()

    for arg in argv:
        path = os.path.abspath(arg)
        if not os.path.isfile(path):
            print(f"error: extra target not found: {arg}", file=sys.stderr)
            return 1
        targets.append((None, path))  # None import_name => standalone script/module

    if not targets:
        print("error: no targets discovered (expected src/psdata/*.py to exist)",
              file=sys.stderr)
        return 1

    results = [check_target(import_name, path) for import_name, path in targets]

    offenders = []
    n_skipped = 0
    for r in results:
        print(f"[{r['status']:>7}] {r['label']:<40} {r['detail']}")
        if r["status"] == "FAIL":
            offenders.append(r["label"])
        elif r["status"] == "SKIPPED":
            n_skipped += 1

    n_checked = len(results)
    if not offenders:
        skip_note = f" ({n_skipped} skipped: optional dep missing)" if n_skipped else ""
        print(f"PURITY PASS: {n_checked} modules, 0 forbidden imports "
              f"(psana, mpi4py, h5py, xtcdata){skip_note}")
        return 0
    print(f"PURITY FAIL: {n_checked} modules, {len(offenders)} forbidden "
          f"import(s) in: {', '.join(offenders)} "
          "(forbidden set: psana, mpi4py, h5py, xtcdata)")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
