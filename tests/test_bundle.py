from pathlib import Path

import pytest

from mekoy.bundle import read_shots, write_bundle
from mekoy.dataset import TaskExample
from mekoy.errors import CompileError
from mekoy.spec import OwnershipFlags, Slos, SystemSpec, load_spec

_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"merchant": {"type": "string"}},
}


def _spec() -> SystemSpec:
    return SystemSpec(
        task="Extract receipt header and line items from already-text.",
        json_schema=_SCHEMA,
        slos=Slos(quality=0.9, cost_per_doc=0.003, latency_ms=240.0),
        model_id="qwen2.5:7b",
        k_shot=4,
        retries=1,
        ownership=OwnershipFlags(runtime_owned=True, downloadable=True),
    )


def _examples() -> tuple[TaskExample, ...]:
    """The labeled rows the harness shows. A k_shot above zero needs them."""
    return tuple(
        TaskExample(text=f"receipt {i}", outcome={"merchant": f"shop {i}"})
        for i in range(4)
    )


def test_write_bundle_round_trip(tmp_path: Path) -> None:
    spec = _spec()
    report = "winner: k_shot=4 retries=1 quality=0.900 schema=1.000\n"
    dest = write_bundle(tmp_path / "sys", spec, report, examples=_examples())
    assert dest.is_dir()
    assert load_spec(dest) == spec
    assert (dest / "report.txt").read_text(encoding="utf-8") == report
    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "mekoy invoke" in readme
    assert "qwen2.5:7b" in readme
    assert "--k-shot 4" in readme
    assert "--retries 1" in readme


def test_write_bundle_is_idempotent(tmp_path: Path) -> None:
    spec = _spec()
    dest = tmp_path / "sys"
    first = write_bundle(dest, spec, "winner\n", examples=_examples())
    second = write_bundle(dest, spec, "winner\n", examples=_examples())
    assert first == second
    names = sorted(path.name for path in dest.iterdir())
    assert names == [
        "README.md",
        "brief.md",
        "checks.md",
        "docker-compose.yml",
        "examples.jsonl",
        "report.txt",
        "spec.json",
    ]


def test_write_bundle_rejects_a_file(tmp_path: Path) -> None:
    target = tmp_path / "not-a-dir"
    _ = target.write_text("nope\n", encoding="utf-8")
    with pytest.raises(CompileError, match="not a directory"):
        _ = write_bundle(target, _spec(), "winner\n", examples=_examples())


def test_a_bundle_carries_the_job_the_examples_and_the_checks(tmp_path: Path) -> None:
    """A recipient must be able to run it, understand it, and check it.

    The bundle used to carry a spec, a report, a compose file, and a README - no brief,
    no labeled rows, and no checks - which made it a pointer to a model rather than a
    System. The harness is the System.
    """
    dest = write_bundle(tmp_path / "sys", _spec(), "winner\n", examples=_examples())

    brief = (dest / "brief.md").read_text(encoding="utf-8")
    assert "Extract receipt header" in brief
    assert "`merchant`" in brief, "the fields it returns must be named"

    checks = (dest / "checks.md").read_text(encoding="utf-8")
    assert "0.900" in checks, "the gate it was held to must travel with it"
    assert "schema" in checks

    rows = (dest / "examples.jsonl").read_text(encoding="utf-8")
    assert len([line for line in rows.splitlines() if line.strip()]) == 4

    spec = load_spec(dest)
    assert spec.spec_version >= 1, "an unversioned spec cannot detect a format change"


def test_a_bundle_refuses_to_ship_a_harness_without_its_examples(
    tmp_path: Path,
) -> None:
    """Shipping `k_shot=4` and none of the four would run a different System."""
    with pytest.raises(CompileError, match="part of the harness"):
        _ = write_bundle(tmp_path / "sys", _spec(), "winner\n")
    assert not (tmp_path / "sys" / "examples.jsonl").exists()


def test_the_readme_install_instruction_resolves_to_a_real_repository(
    tmp_path: Path,
) -> None:
    """The old line told a stranger to install an unpublished package.

    `uv tool install mekoy` fails because the package is not on PyPI, and the
    fallback on the same line was a literal `<repo>` placeholder with no URL to fill it
    in. A bundle that cannot be installed is not a System anyone owns.
    """
    dest = write_bundle(tmp_path / "sys", _spec(), "winner\n", examples=_examples())
    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "<repo>" not in readme
    assert "uv tool install" not in readme, "the package is not published"
    assert "https://github.com/" in readme
    assert "git clone" in readme


def test_a_bundle_without_examples_says_so(tmp_path: Path) -> None:
    """A zero-shot System is fine; the README must not imply examples exist."""
    spec = _spec().model_copy(update={"k_shot": 0})
    dest = write_bundle(tmp_path / "sys", spec, "winner\n")
    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "carries no examples" in readme
    assert not (dest / "examples.jsonl").exists()


def test_a_bundle_records_the_rows_the_harness_actually_shows(tmp_path: Path) -> None:
    """The rows the model saw are the head of the TRAIN split, not the head of the file.

    A verifier that takes the first `k_shot` rows of the corpus is testing a different
    System: on the restaurant corpus that guess shared one row in eight with the truth,
    and it made a correct bundle reproduce as 0.913 against its own 0.923.
    """
    corpus = tuple(
        TaskExample(text=f"row {i}", outcome={"merchant": str(i)}) for i in range(6)
    )
    # The compile shows these, drawn from a split whose order is not the file's order.
    shown = (corpus[4], corpus[2], corpus[5])

    spec = _spec().model_copy(update={"k_shot": len(shown)})
    dest = write_bundle(
        tmp_path / "sys", spec, "winner\n", examples=corpus, shots=shown
    )

    recorded = read_shots(dest)
    assert [r.text for r in recorded] == [r.text for r in shown]
    assert len(recorded) == len(shown), "exactly what the harness shows, not the pool"
    assert [r.text for r in recorded] != [r.text for r in corpus[: len(shown)]]


def test_verification_refuses_when_the_shots_were_never_recorded(
    tmp_path: Path,
) -> None:
    """Not knowing which rows were shown is not the same as knowing none were."""
    spec = _spec()
    dest = write_bundle(tmp_path / "sys", spec, "winner\n", examples=_examples())
    assert read_shots(dest) == (), "an older bundle records no shots"
