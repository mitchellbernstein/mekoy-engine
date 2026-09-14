from pathlib import Path

import pytest
from pydantic import ValidationError

from mekoy.errors import CompileError
from mekoy.spec import OwnershipFlags, Slos, SystemSpec, load_spec, write_spec

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


def _payload() -> dict[str, object]:
    return {
        "task": "Extract receipt header and line items from already-text.",
        "schema": _SCHEMA,
        "slos": {"quality": 0.9, "cost_per_doc": 0.003, "latency_ms": 240.0},
        "model_id": "qwen2.5:7b",
        "k_shot": 4,
        "retries": 1,
        "ownership": {"runtime_owned": True, "downloadable": True},
    }


def test_spec_round_trip_json(tmp_path: Path) -> None:
    spec = _spec()
    path = write_spec(tmp_path / "spec.json", spec)
    loaded = load_spec(path)
    assert loaded == spec
    raw = path.read_text(encoding="utf-8")
    assert '"schema"' in raw
    assert "teacher" not in raw
    assert "gpt-" not in raw
    assert "claude" not in raw


def test_spec_loads_from_directory(tmp_path: Path) -> None:
    spec = _spec()
    _ = write_spec(tmp_path / "spec.json", spec)
    assert load_spec(tmp_path) == spec


def test_spec_is_frozen() -> None:
    spec = _spec()
    with pytest.raises(ValidationError):
        spec.__setattr__("task", "nope")


def test_spec_rejects_closed_model_fields() -> None:
    with pytest.raises(ValidationError):
        _ = SystemSpec.model_validate({**_payload(), "teacher_model": "gpt-4"})


def test_spec_rejects_negative_k_shot() -> None:
    with pytest.raises(ValidationError):
        _ = SystemSpec.model_validate({**_payload(), "k_shot": -1})


def test_load_spec_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CompileError, match="spec not found"):
        _ = load_spec(tmp_path / "missing.json")
