from dataclasses import fields
from pathlib import Path

import pytest
from pydantic import ValidationError

from mekoy.bundle import spec_for
from mekoy.compile import CompileReport, Trial
from mekoy.errors import CompileError
from mekoy.search import HarnessConfig
from mekoy.spec import (
    SPEC_VERSION,
    OwnershipFlags,
    Slos,
    SystemSpec,
    load_spec,
    write_spec,
)

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


def _base_spec(**over: object) -> SystemSpec:
    """A spec with the required fields filled, so a test can vary one thing."""
    fields: dict[str, object] = {
        "task": "restaurant call extraction",
        "json_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
        "slos": Slos(quality=0.9, cost_per_doc=0.0, latency_ms=1.0),
        "model_id": "qwen2.5:7b",
        "k_shot": 4,
        "retries": 0,
        "ownership": OwnershipFlags(runtime_owned=True, downloadable=True),
    }
    fields.update(over)
    return SystemSpec(**fields)


def test_a_spec_carries_the_whole_harness() -> None:
    """A bundle that loses an axis hands over a System nobody measured.

    This is the defect `verify-bundle` surfaced on its first use. The restaurant winner
    is `k=4 r=0 schema strict`; a spec recording only `k_shot` and `retries` exported it
    as `k=4 r=0`, so a recipient got the default brief and a different score than the
    one they were quoted.
    """
    strict = _base_spec(
        spec_version=SPEC_VERSION, prompt="strict", schema_constrained=True
    )
    default = _base_spec(spec_version=SPEC_VERSION)

    assert strict.harness().prompt == "strict"
    assert strict.harness().schema is True
    assert strict.harness() != default.harness(), "the axes must reach the harness"
    assert strict.carries_harness


def test_a_spec_from_before_the_harness_was_recorded_is_not_verifiable() -> None:
    """The honest answer is "cannot check", never a default scored as if it were real.

    Every bundle written before the axes existed parses fine and silently defaults all
    five, which is indistinguishable from a System that genuinely chose the defaults.
    """
    old = _base_spec(spec_version=1)
    assert not old.carries_harness


def test_spec_for_writes_every_axis_the_search_can_turn() -> None:
    """The writer is where the loss happened, so the writer is where it is pinned."""
    config = HarnessConfig(
        k_shot=4,
        retries=0,
        constrained=True,
        prompt="strict",
        schema=True,
        consistency=1,
    )
    winner = Trial(config=config, scores=())
    report = CompileReport(
        winner=winner, trials=(winner,), test=winner, stopped_early=False
    )
    spec = spec_for(report, task="job", model_id="qwen2.5:7b")

    assert spec.spec_version == SPEC_VERSION, "an older version makes the reader refuse"
    rebuilt = spec.harness()
    for axis in fields(HarnessConfig):
        if axis.name == "model":
            continue  # the spec's model_id is the model to run, not the search axis
        assert getattr(rebuilt, axis.name) == getattr(config, axis.name), axis.name
