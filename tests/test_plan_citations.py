"""The engine must not cite a document that lives only in a private repository.

The private plan is not in this repository. A pointer to it sends a reader of the
public source to a file they cannot open, and it pins the code to a private draft
whose section numbers can move without the code changing. Behaviour belongs in the
engine's own docstrings and tests.

This test is the guard. It walks every Python file under `src/` and `tests/` and
fails on the two spellings that have actually appeared: the private filename, and
its section marker. Ordinary English uses of "plan" (a memory plan, a training
plan) are a different word, are written in lower case, and are left alone.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The private document by name, and the section marker that is its numbering.
#: The marker has no other use in this codebase, so its presence is always a
#: citation. It is spelled with an escape here so this file does not match itself.
CITATION = re.compile(r"PLAN\.md|\u00a7|PLAN[\s_]+[0-9]")


def _engine_sources() -> list[Path]:
    """Every Python file in the engine package and its tests."""
    sources: list[Path] = []
    for root in ("src", "tests"):
        sources.extend(sorted((REPO_ROOT / root).rglob("*.py")))
    return sources


def _citations_in(path: Path) -> list[str]:
    """Lines of `path` that cite the private document."""
    found: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if CITATION.search(line):
            found.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    return found


def test_sources_exist() -> None:
    """A guard over an empty tree proves nothing, so the tree must be found."""
    assert _engine_sources(), "no engine sources were found; the guard is not watching"


def test_no_engine_file_cites_the_private_plan() -> None:
    """No file in the engine may name or section-number the private document."""
    offenders: list[str] = []
    for path in _engine_sources():
        offenders.extend(_citations_in(path))

    if offenders:
        listing = "\n".join(offenders)
        pytest.fail(
            f"{len(offenders)} citation(s) of the private document in the engine. "
            f"The public source must stand on its own:\n{listing}"
        )


def test_the_guard_catches_a_citation() -> None:
    """The guard is only evidence if it fails on the thing it forbids."""
    sample = REPO_ROOT / "tests" / "_guard_selftest_sample.py"
    sample.write_text(
        '"""A docstring citing the private doc.\n\nSee \u00a7' + "16.5.\n" + '"""\n'
    )
    try:
        assert _citations_in(sample), "the guard missed a citation it must catch"
    finally:
        sample.unlink()
