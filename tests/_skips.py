#!/usr/bin/env python3
"""Machine-readable skip protocol for the psdata acceptance suite (HYG-03).

Why this exists
---------------
Every test file in this suite is a plain script: it runs its checks and exits 0
on success.  Individual checks used to "skip" by *printing a message and
returning* -- the file still exited 0, so ``run_tests.sh`` (which tallies
pass/fail purely from the exit code) scored the skipped check as a **PASS**, and
no skip count appeared anywhere.  A recorded "12 passed" run concealed three
skips, two of them on the exact paths where the reader's worst data-loss bug was
later found.  A conformance suite that exits 0 when its oracle tests do not run
is worthless.

The protocol
------------
A check that cannot run calls :func:`skip` instead of ``print``.  That emits one
line on stdout::

    ##SKIP## <name> :: <reason>

``run_tests.sh`` tees each test's output, counts those marker lines, prints an
explicit skip count in the final tally, and **fails the suite** unless every
emitted skip ``name`` appears in ``tests/skips_allowed.txt`` (an allowlist of
``name :: justification`` records).  A skip is not a pass.

Rules for adding a skip
-----------------------
* ``name`` is a stable, unique slug (``[a-z0-9_]+``) -- it is the allowlist key,
  so renaming it silently un-allowlists the skip (fail-closed, on purpose).
* ``reason`` says what could not run and why, in the *observed* environment.
* Adding a name to ``tests/skips_allowed.txt`` requires a real justification.
  "The oracle's ground truth is missing" is NOT a justification -- capture the
  ground truth.  The only currently-justified skips are the ``torch`` ones:
  torch is a declared *optional* extra and is genuinely absent from the
  production psconda env.

This module imports nothing but the standard library; it must stay importable
from a test that has not yet imported psdata (or anything else).
"""

import sys

SKIP_MARKER = "##SKIP##"
SEP = "::"


def skip(name, reason):
    """Emit a machine-readable skip record for the check ``name``.

    The runner counts these; a skip whose ``name`` is not in
    ``tests/skips_allowed.txt`` fails the suite (HYG-03).

    Parameters
    ----------
    name : str
        Stable unique slug identifying the skipped check (the allowlist key).
    reason : str
        Why the check could not run in this environment.

    Returns
    -------
    None
        So a caller can ``return skip(...)`` and keep its early-return control
        flow unchanged.
    """
    name = str(name).strip()
    if not name:
        raise ValueError("skip() needs a non-empty name (it is the allowlist key)")
    if SEP in name or any(c.isspace() for c in name):
        raise ValueError(
            "skip() name must be a whitespace-free slug without %r: %r"
            % (SEP, name))
    # one line, one record: collapse any newline in the reason.
    reason = " ".join(str(reason).split())
    print("%s %s %s %s" % (SKIP_MARKER, name, SEP, reason))
    sys.stdout.flush()
    return None
