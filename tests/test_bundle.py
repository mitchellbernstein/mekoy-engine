from pathlib import Path

import pytest

from mekoy.bundle import write_bundle
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


def test_write_bundle_round_trip(tmp_path: Path) -> None:
    spec = _spec()
    report = "winner: k_shot=4 retries=1 quality=0.900 schema=1.000\n"
    dest = write_bundle(tmp_path / "sys", spec, report)
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
    first = write_bundle(dest, spec, "winner\n")
    second = write_bundle(dest, spec, "winner\n")
    assert first == second
    names = sorted(path.name for path in dest.iterdir())
    assert names == [
        "README.md",
        "docker-compose.yml",
        "report.txt",
        "spec.json",
    ]


def test_write_bundle_rejects_a_file(tmp_path: Path) -> None:
    target = tmp_path / "not-a-dir"
    _ = target.write_text("nope\n", encoding="utf-8")
    with pytest.raises(CompileError, match="not a directory"):
        _ = write_bundle(target, _spec(), "winner\n")
