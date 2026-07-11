#!/usr/bin/env python3
"""HYG-03 regression: the suite runner must not score a SKIP as a PASS.

The bug
-------
``run_tests.sh`` used to tally pass/fail purely from each test file's exit code.
Individual checks inside a file, however, "skip" by printing a message and
returning -- the file still exits 0.  So a skipped check scored as a **PASS**,
and no skip count appeared anywhere: a recorded "12 passed" run concealed three
skips, two of them the psana oracles on the very code path where the reader's
worst bug (silent data loss on 3.3% of a run) was later found.

The contract this test pins
---------------------------
1. A test that emits an **unlisted** skip record (``##SKIP## <name> :: <reason>``)
   makes the runner **exit nonzero**, even though the test file itself exited 0.
   *This is the discriminator*: on the parent commit the runner exits 0 here,
   because it only ever looked at the exit code.
2. The final summary carries an **explicit skip count** ("N passed, M failed,
   S skipped"), and names each skip with its reason.
3. A skip whose name IS in ``tests/skips_allowed.txt`` (with a justification)
   does **not** fail the suite -- justified skips stay possible, they just have
   to be declared.
4. A genuinely failing test file still fails the suite (no regression in the
   original behaviour), and a clean run still exits 0.
4b. An UNMARKED skip -- the OLD idiom, a printed ``[skip]`` / ``[ skip ]`` /
   ``skipping`` / ``skipped`` line OR a standalone uppercase ``SKIP`` token
   (``SKIP ...`` / ``SKIP:``), exiting 0 with no ``##SKIP##`` marker -- also
   fails the suite.  This backstops a future author reverting to the idiom this
   repo actually used (old ``test_config_us010`` / ``test_uniqueid_us011`` wrote
   ``print("SKIP ...")``).  The word boundary means ``PSDATA_SKIP_SLOW`` is not
   flagged.
4c. A run that executes zero test files fails ("ran 0 tests") -- no vacuous green.
5. The default TESTS list and the on-disk ``tests/test_*.py`` files agree BOTH
   ways: an on-disk test not registered never runs; a registered entry with no
   file on disk is named clearly (not left to degrade into a runtime FAILED).
6. An allowlist entry with an empty justification (``name ::``) is malformed and
   does not rubber-stamp a skip.

This test is SELF-CONTAINED: no psana, no psdata, no SLAC data.  It synthesizes
fake test scripts in a temp dir and drives the real ``run_tests.sh`` over them,
so it can be run from any cwd, on any machine::

    python3 tests/test_runner_hygiene_hyg03.py
"""

import os
import re
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
RUN_TESTS = os.path.join(_REPO, "run_tests.sh")
SKIPS_ALLOWED = os.path.join(_HERE, "skips_allowed.txt")

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _skips import SKIP_MARKER   # noqa: E402  (the protocol under test)

UNLISTED = "hyg03_selftest_unlisted_skip"   # deliberately NOT in the allowlist


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _allowed_names():
    """The skip names declared in tests/skips_allowed.txt ('name :: why')."""
    names = []
    with open(SKIPS_ALLOWED) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "::" not in line:
                continue
            names.append(line.split("::", 1)[0].strip())
    return names


def _write(path, body):
    with open(path, "w") as f:
        f.write(body)
    os.chmod(path, 0o755)
    return path


def _fake_skipping_test(tmp, fname, skip_name):
    """A fake test script that skips one check and exits 0 -- exactly the shape
    the real tests have (print a marker, return, exit 0)."""
    return _write(os.path.join(tmp, fname), (
        "#!/usr/bin/env python3\n"
        "print('[ok] a check that really ran')\n"
        "print('%s %s :: synthetic skip emitted by the HYG-03 self-test')\n"
        "print('ALL CHECKS PASSED')\n"
        "raise SystemExit(0)\n" % (SKIP_MARKER, skip_name)))


def _fake_clean_test(tmp, fname="fake_clean.py"):
    # NB: the body must not print any skip-looking word, or the runner's
    # unmarked-skip check would (correctly) redden this deliberately-clean file.
    return _write(os.path.join(tmp, fname), (
        "#!/usr/bin/env python3\n"
        "print('[ok] a clean check ran')\n"
        "raise SystemExit(0)\n"))


def _fake_bare_skip_test(tmp, fname="fake_bare_skip.py"):
    """A fake test that skips a check the OLD way -- a printed '[skip]' line and
    ``return``, exiting 0 with NO ##SKIP## marker.  This is the pre-fix idiom the
    unmarked-skip check exists to catch."""
    return _write(os.path.join(tmp, fname), (
        "#!/usr/bin/env python3\n"
        "print('[skip] no psana in this env')\n"   # old idiom, no marker
        "print('ALL CHECKS PASSED')\n"
        "raise SystemExit(0)\n"))


def _fake_skip_token_test(tmp, fname="fake_skip_token.py"):
    """The OTHER pre-fix idiom, and the one this repo actually used: a printed
    'SKIP ...' line (uppercase token) and exit 0, no ##SKIP## marker.  The old
    test_config_us010 / test_uniqueid_us011 wrote exactly this."""
    return _write(os.path.join(tmp, fname), (
        "#!/usr/bin/env python3\n"
        "print('SKIP in-proc purity: a sibling test already imported psana')\n"
        "print('ALL CHECKS PASSED')\n"
        "raise SystemExit(0)\n"))


def _fake_prose_marker_test(tmp, fname="fake_prose_marker.py"):
    """A GREEN test whose summary merely MENTIONS the marker in prose -- exactly
    like test_torch_us011's real line 'see the ##SKIP## records above' -- with NO
    real skip record.  The marker appears mid-line (not at line start) and there
    is no '::', so the collector must NOT count it as a skip."""
    return _write(os.path.join(tmp, fname), (
        "#!/usr/bin/env python3\n"
        "print('ALL CHECKS PASSED (done; see the %s records above)')\n"
        "raise SystemExit(0)\n" % SKIP_MARKER))


def _fake_failing_test(tmp, fname="fake_fail.py"):
    return _write(os.path.join(tmp, fname), (
        "#!/usr/bin/env python3\n"
        "print('[FAIL] synthetic failure')\n"
        "raise SystemExit(3)\n"))


def _run_suite(*test_paths):
    """Invoke the real run_tests.sh over explicit test files (its documented
    ``$@`` form).  Returns (returncode, combined_output).  Run from / so a cwd
    dependency in the runner would show up."""
    proc = subprocess.run(
        ["bash", RUN_TESTS] + list(test_paths),
        cwd="/", capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _summary_counts(out):
    """Parse the 'N passed, M failed, S skipped' tally.  Returns (n, m, s)."""
    m = re.search(r"^(\d+) passed, (\d+) failed, (\d+) skipped\s*$",
                  out, re.MULTILINE)
    assert m, ("no explicit 'N passed, M failed, S skipped' summary in the "
               "runner output -- a suite with no skip count cannot tell a skip "
               "from a pass (HYG-03).  Output was:\n" + out)
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# --------------------------------------------------------------------------
# 1. THE DISCRIMINATOR: an unlisted skip fails the suite
# --------------------------------------------------------------------------
def test_unlisted_skip_fails_the_suite():
    """A test file that skips a check and exits 0 must NOT be scored as a pass.

    On the parent commit the runner returns 0 here (it only read the exit code);
    on the fixed runner it returns nonzero and says which skip was unjustified.
    """
    assert UNLISTED not in _allowed_names(), \
        "the self-test's sentinel skip name must not be allowlisted"
    with tempfile.TemporaryDirectory() as tmp:
        skipper = _fake_skipping_test(tmp, "fake_skip.py", UNLISTED)
        clean = _fake_clean_test(tmp)
        rc, out = _run_suite(skipper, clean)

    assert rc != 0, (
        "run_tests.sh exited 0 on an UNJUSTIFIED SKIP -- a skip is being scored "
        "as a pass (HYG-03).  Output was:\n" + out)
    n_pass, n_fail, n_skip = _summary_counts(out)
    assert n_skip == 1, f"expected 1 skip counted, summary said {n_skip}\n{out}"
    assert n_fail == 0, (
        f"the skipping file exits 0, so it is not a FAILED file; summary said "
        f"{n_fail} failed\n{out}")
    assert UNLISTED in out, "the runner must name the offending skip\n" + out
    assert "UNJUSTIFIED SKIP" in out, (
        "the runner must say the skip was unjustified\n" + out)
    # the skip's reason is surfaced, not just its name
    assert "synthetic skip emitted by the HYG-03 self-test" in out, \
        "the runner must print each skip's reason\n" + out
    # (this test's own stdout is itself scanned by the outer suite's
    #  unmarked-skip check, so do not print 'S skipped' or a standalone
    #  uppercase SKIP token here -- keep the words lowercase)
    print(f"[ok] unlisted skip -> runner exit {rc} (nonzero); summary carried "
          f"an explicit skip count (S={n_skip}); the unjustified-skip flag "
          f"named it")


# --------------------------------------------------------------------------
# 2. an ALLOWLISTED skip does not fail the suite (justified skips stay possible)
# --------------------------------------------------------------------------
def test_allowlisted_skip_passes_but_is_counted():
    """A skip declared in tests/skips_allowed.txt exits 0 -- but is still COUNTED
    and printed with its justification (visible, not silent)."""
    allowed = _allowed_names()
    assert allowed, "tests/skips_allowed.txt declares no skips at all"
    name = allowed[0]          # whatever the allowlist actually declares
    with tempfile.TemporaryDirectory() as tmp:
        skipper = _fake_skipping_test(tmp, "fake_skip_ok.py", name)
        clean = _fake_clean_test(tmp)
        rc, out = _run_suite(skipper, clean)

    assert rc == 0, (
        f"an allowlisted skip ({name}) must not fail the suite; runner exited "
        f"{rc}\n{out}")
    n_pass, n_fail, n_skip = _summary_counts(out)
    assert (n_pass, n_fail, n_skip) == (2, 0, 1), \
        f"expected 2 passed, 0 failed, 1 skipped; got {(n_pass, n_fail, n_skip)}\n{out}"
    assert "UNJUSTIFIED SKIP" not in out, \
        "an allowlisted skip must not be reported as unjustified\n" + out
    assert name in out and "justification:" in out, \
        "the runner must print the skip with its justification\n" + out
    print(f"[ok] allowlisted skip {name!r} -> runner exit 0, still counted "
          f"(skip count S={n_skip}) and printed with its justification")


# --------------------------------------------------------------------------
# 3. no regression: a clean run passes, a failing file still fails
# --------------------------------------------------------------------------
def test_clean_run_and_failure_still_work():
    with tempfile.TemporaryDirectory() as tmp:
        clean = _fake_clean_test(tmp)
        rc, out = _run_suite(clean)
        assert rc == 0, f"a clean run must exit 0; got {rc}\n{out}"
        assert _summary_counts(out) == (1, 0, 0), out

        failing = _fake_failing_test(tmp)
        rc, out = _run_suite(clean, failing)
        assert rc != 0, f"a failing test file must fail the suite; got {rc}\n{out}"
        n_pass, n_fail, n_skip = _summary_counts(out)
        assert (n_pass, n_fail, n_skip) == (1, 1, 0), \
            f"expected 1 passed, 1 failed, 0 skipped; got {(n_pass, n_fail, n_skip)}\n{out}"
    print("[ok] clean run exits 0; a failing test file still fails the suite "
          "(exit code tally preserved)")


# --------------------------------------------------------------------------
# 4. a skipping test's OWN output still reaches the terminal (tee, not swallow)
# --------------------------------------------------------------------------
def test_runner_still_streams_test_output():
    with tempfile.TemporaryDirectory() as tmp:
        skipper = _fake_skipping_test(tmp, "fake_skip.py", UNLISTED)
        rc, out = _run_suite(skipper)
    assert "[ok] a check that really ran" in out, \
        "the runner must still stream each test's own output\n" + out
    assert f"### running {skipper}" in out, \
        "the runner must still announce each test file\n" + out
    print("[ok] test output is still streamed (tee), not swallowed by the capture")


# --------------------------------------------------------------------------
# 4b. THE SECOND DISCRIMINATOR: an UNMARKED (old-idiom) skip fails the suite
# --------------------------------------------------------------------------
def test_bare_unmarked_skip_fails_the_suite():
    """A test that skips the OLD way -- prints a bare '[skip]' line and exits 0,
    with NO ##SKIP## marker -- must NOT score as a pass.

    The ##SKIP## protocol only reclassifies a run when the marker appears; a
    future author who reverts to the old idiom would slip a silent skip past it.
    On the parent commit this exits 0 (there is no such check at all); on the
    fixed runner it exits nonzero and tells the author to route the skip through
    ``skip(name, reason)``.  A fresh discriminator, independent of the marker path.
    """
    with tempfile.TemporaryDirectory() as tmp:
        bare = _fake_bare_skip_test(tmp, "fake_bare_skip.py")
        clean = _fake_clean_test(tmp)
        rc, out = _run_suite(bare, clean)

    assert rc != 0, (
        "run_tests.sh exited 0 on a bare unmarked skip line -- an old-idiom "
        "skip is being scored as a pass (HYG-03).  Output was:\n" + out)
    assert "UNMARKED SKIP" in out, (
        "the runner must flag the unmarked skip line\n" + out)
    assert bare in out, "the runner must name the offending test file\n" + out
    # it must point the author at the sanctioned protocol
    assert "skip(" in out, (
        "the runner must tell the author to route it through skip(name, reason)"
        "\n" + out)
    # a bare skip is NOT a ##SKIP## record, so it must not inflate the skip count
    n_pass, n_fail, n_skip = _summary_counts(out)
    assert n_skip == 0, (
        f"an unmarked skip must not be counted as a ##SKIP## record; summary "
        f"said {n_skip}\n{out}")
    print(f"[ok] a bare unmarked skip line reddens the run (exit {rc}); runner "
          f"names the file and points the author at skip(name, reason)")


# --------------------------------------------------------------------------
# 4b'. the uppercase 'SKIP ...' idiom -- the one THIS REPO actually used -- too
# --------------------------------------------------------------------------
def test_bare_skip_token_fails_the_suite():
    """A test printing an old-style uppercase 'SKIP ...' line and exiting 0 (no
    ##SKIP## marker) must redden the run.

    This is the exact pre-fix idiom of this repo's own test_config_us010 /
    test_uniqueid_us011 -- 'print("SKIP in-proc purity: ...")'.  The first cut of
    the unmarked-skip regex (\\[skip\\]|skipping|SKIPPED) MISSED it: 'SKIP ' is
    not '[skip]'/'skipping'/'SKIPPED'.  The broadened scan (a standalone
    uppercase SKIP token) catches it.  Fails on the parent AND on that first cut.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tok = _fake_skip_token_test(tmp, "fake_skip_token.py")
        clean = _fake_clean_test(tmp)
        rc, out = _run_suite(tok, clean)

    assert rc != 0, (
        "run_tests.sh exited 0 on a bare 'SKIP ...' line -- the repo's own "
        "pre-fix idiom is being scored as a pass (HYG-03).  Output was:\n" + out)
    assert "UNMARKED SKIP" in out, "the runner must flag the line\n" + out
    assert tok in out, "the runner must name the offending file\n" + out
    n_pass, n_fail, n_skip = _summary_counts(out)
    assert n_skip == 0, (
        f"an unmarked skip must not be counted as a ##SKIP## record; summary "
        f"said {n_skip}\n{out}")
    print(f"[ok] an uppercase skip-token line reddens the run (exit {rc}) -- the "
          f"repo's own pre-fix idiom no longer slips through")


# --------------------------------------------------------------------------
# 4b''. a PROSE mention of the marker is NOT a skip record (collector anchored)
# --------------------------------------------------------------------------
def test_prose_marker_mention_is_not_counted():
    """A GREEN test whose output merely MENTIONS the marker in prose -- exactly
    test_torch_us011's real summary line 'see the ##SKIP## records above' -- must
    NOT be collected as a skip record.

    On the parent (marker collected with an unanchored `grep -F`) that prose line
    parses to a phantom skip named 'records above)', which is not allowlisted ->
    UNJUSTIFIED SKIP -> the whole suite falsely exits 1 (this was the only thing
    keeping the merged suite red).  The anchored collector -- the marker must
    start the line and a real record carries '::' -- ignores it: 0 skips, exit 0.
    """
    with tempfile.TemporaryDirectory() as tmp:
        prose = _fake_prose_marker_test(tmp, "fake_prose_marker.py")
        clean = _fake_clean_test(tmp)
        rc, out = _run_suite(prose, clean)

    n_pass, n_fail, n_skip = _summary_counts(out)
    assert n_skip == 0, (
        f"a prose mention of the marker was miscounted as a skip record "
        f"(S={n_skip}) -- the collector is not anchored to line start (HYG-03).  "
        f"Output was:\n{out}")
    assert rc == 0, (
        f"a green run whose output merely mentions the marker in prose must exit "
        f"0; runner exited {rc}\n{out}")
    assert "UNJUSTIFIED SKIP" not in out, (
        "a prose marker mention must not raise an unjustified-skip failure\n" + out)
    assert n_pass == 2 and n_fail == 0, \
        f"expected 2 passed, 0 failed; got {n_pass} passed, {n_fail} failed\n{out}"
    print("[ok] a prose mention of the marker is not counted as a skip record "
          "(collector anchored to line start); the run stays green")


# --------------------------------------------------------------------------
# 4c. a vacuous green -- zero test files executed -- must fail, not pass
# --------------------------------------------------------------------------
def test_zero_tests_fails():
    """A run that executes zero test files must FAIL, not silently report
    success on nothing.  Reached via PSDATA_NO_DEFAULT_TESTS=1 with no args (the
    only way to express 'run nothing' -- otherwise no-args means the default
    suite)."""
    proc = subprocess.run(
        ["bash", RUN_TESTS], cwd="/", capture_output=True, text=True,
        env={**os.environ, "PSDATA_NO_DEFAULT_TESTS": "1"},
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, \
        f"a 0-test run must fail; runner exited {proc.returncode}\n{out}"
    assert "ran 0 tests" in out, \
        "the runner must say it ran 0 tests\n" + out
    print(f"[ok] a 0-test run fails (exit {proc.returncode}, 'ran 0 tests') -- "
          f"no vacuous green")


# --------------------------------------------------------------------------
# 5. the default TESTS list and the on-disk test files agree BOTH ways
# --------------------------------------------------------------------------
# test_gate02_gated_forward.py is registered here but arrives via the (already
# merged to main) GATE-02 change, so it is absent on THIS branch and present
# post-merge.  Exempt it from the registered->on-disk direction so this branch
# stays green, while still failing clearly for any OTHER genuinely-missing file.
_PENDING_MERGE = {"test_gate02_gated_forward.py"}


def test_default_suite_matches_disk_both_ways():
    """A completeness check in BOTH directions (aligning with HYG-05):

      * on-disk -> registered: a test file that exists but is not in the default
        TESTS list never runs -- the same disease as a skip scored as a pass;
      * registered -> on-disk: a TESTS entry with no file on disk (deleted or
        renamed) must fail with a clear message, not degrade to a runtime FAILED.
    """
    with open(RUN_TESTS) as f:
        runner = f.read()

    on_disk = sorted(n for n in os.listdir(_HERE)
                     if n.startswith("test_") and n.endswith(".py"))
    # registered entries = the 'tests/<name>.py' paths named in run_tests.sh
    registered = sorted(set(re.findall(r"tests/(test_[A-Za-z0-9_]+\.py)", runner)))

    # direction 1: on-disk => registered
    unregistered = [n for n in on_disk if n not in registered]
    assert not unregistered, (
        "these test files exist on disk but are NOT registered in run_tests.sh's "
        "default TESTS list, so the suite never runs them: %s" % unregistered)

    # direction 2: registered => on-disk
    absent = [n for n in registered
              if not os.path.exists(os.path.join(_HERE, n))]
    unexpected = [n for n in absent if n not in _PENDING_MERGE]
    assert not unexpected, (
        "these files are registered in run_tests.sh's TESTS list but do not "
        "exist on disk (deleted? renamed? typo?): %s" % unexpected)
    if absent:
        print(f"[note] registered but not yet on this branch (arrives via a "
              f"pending merge, exempted): {absent}")
    print(f"[ok] default suite and disk agree both ways "
          f"({len(on_disk)} on disk, {len(registered)} registered)")


# --------------------------------------------------------------------------
# 6. an allowlist entry with an EMPTY justification does not rubber-stamp a skip
# --------------------------------------------------------------------------
def test_empty_justification_is_not_allowed():
    """A 'name ::' allowlist line with nothing after the '::' is malformed and
    must NOT justify a skip -- the skip stays unjustified and fails the suite
    (Finding 6).  Driven via the PSDATA_SKIPS_ALLOWED override so the real
    allowlist is untouched."""
    name = "hyg03_selftest_empty_just"
    with tempfile.TemporaryDirectory() as tmp:
        allow = os.path.join(tmp, "skips_allowed.txt")
        with open(allow, "w") as f:
            f.write("# malformed on purpose: empty justification below\n")
            f.write("%s ::\n" % name)
        skipper = _fake_skipping_test(tmp, "fake_skip_empty.py", name)
        clean = _fake_clean_test(tmp)
        proc = subprocess.run(
            ["bash", RUN_TESTS, skipper, clean], cwd="/",
            capture_output=True, text=True,
            env={**os.environ, "PSDATA_SKIPS_ALLOWED": allow},
        )
        out = proc.stdout + proc.stderr

    assert proc.returncode != 0, (
        "an empty-justification allowlist entry must not justify a skip; runner "
        "exited %d\n%s" % (proc.returncode, out))
    assert "UNJUSTIFIED SKIP" in out, \
        "the skip should be reported as unjustified\n" + out
    assert "malformed" in out.lower(), \
        "the runner should warn that the allowlist entry is malformed\n" + out
    print("[ok] an empty-justification allowlist entry is rejected as malformed; "
          "the skip stays unjustified")


def main():
    print("=" * 72)
    print("HYG-03 regression: a skip is not a pass")
    print("=" * 72)
    test_unlisted_skip_fails_the_suite()
    test_allowlisted_skip_passes_but_is_counted()
    test_clean_run_and_failure_still_work()
    test_runner_still_streams_test_output()
    test_bare_unmarked_skip_fails_the_suite()
    test_bare_skip_token_fails_the_suite()
    test_prose_marker_mention_is_not_counted()
    test_zero_tests_fails()
    test_empty_justification_is_not_allowed()
    test_default_suite_matches_disk_both_ways()
    print("\nALL HYG-03 RUNNER-HYGIENE CHECKS PASSED")


if __name__ == "__main__":
    main()
