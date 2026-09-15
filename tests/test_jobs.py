"""Jobs defined in data, and the regression that matters.

The point of this file is to let someone add a job without editing the engine. The
risk of that change is that the jobs which already work quietly change behaviour, so
the shipped tasks are pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mekoy.errors import CompileError
from mekoy.jobs import (
    CheckSpec,
    FieldSpec,
    JobDefinition,
    job_from_examples,
    load_job,
    validate_job,
)
from mekoy.outcome import RestaurantOutcome
from mekoy.receipt import Receipt
from mekoy.tasks import BANKING77, RECEIPT, RESTAURANT, task_from_definition


def _manifest(tmp_path: Path) -> Path:
    """A job the engine has never seen, as a labeled file."""

    def row(a: float, b: float, total: float, carrier: str) -> dict[str, object]:
        return {
            "text": f"{carrier} manifest: {a}kg + {b}kg = {total}kg",
            "outcome": {
                "carrier": carrier,
                "weight_a": a,
                "weight_b": b,
                "total_weight": total,
            },
        }

    rows = [
        row(12.0, 5.5, 17.5, "DHL"),
        row(3.0, 4.0, 7.0, "FedEx"),
        row(20.0, 1.5, 21.5, "UPS"),
        row(8.0, 2.0, 10.0, "USPS"),
    ]
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_a_job_can_be_inferred_from_examples_alone(tmp_path: Path) -> None:
    """Asking a user to hand-write a schema is asking them to do the compiler's job."""
    definition = job_from_examples(
        _manifest(tmp_path), name="manifest", task="Read a shipping manifest."
    )
    assert definition is not None
    assert definition.names() == (
        "carrier",
        "weight_a",
        "weight_b",
        "total_weight",
    )
    assert validate_job(definition) == ()


def test_inference_produces_a_required_check(tmp_path: Path) -> None:
    """Without one, nothing is ever rejected and the gate is decorative."""
    definition = job_from_examples(
        _manifest(tmp_path), name="manifest", task="Read a shipping manifest."
    )
    assert definition is not None
    assert any(check.kind == "required" for check in definition.checks)


def test_inference_finds_the_arithmetic_the_labels_already_satisfy(
    tmp_path: Path,
) -> None:
    """A check the user's own labels fail would mark every candidate broken."""
    definition = job_from_examples(
        _manifest(tmp_path), name="manifest", task="Read a shipping manifest."
    )
    assert definition is not None
    arithmetic = [c for c in definition.checks if c.kind == "arithmetic"]
    assert arithmetic, "12.0 + 5.5 = 17.5 should be found"
    assert set(arithmetic[0].parts) == {"weight_a", "weight_b"}
    assert arithmetic[0].total == "total_weight"


def test_a_data_job_rejects_arithmetic_that_does_not_add_up(tmp_path: Path) -> None:
    """The gate has to actually run against an answer, not just exist."""
    definition = job_from_examples(
        _manifest(tmp_path), name="manifest", task="Read a shipping manifest."
    )
    assert definition is not None
    task = task_from_definition(definition)

    good = {"carrier": "DHL", "weight_a": 12.0, "weight_b": 5.5, "total_weight": 17.5}
    bad = {"carrier": "DHL", "weight_a": 12.0, "weight_b": 5.5, "total_weight": 40.0}
    assert task.gate(good) == ()
    reasons = task.gate(bad)
    assert reasons
    assert "!=" in reasons[0]


def test_a_data_job_rejects_a_missing_required_field(tmp_path: Path) -> None:
    """An empty field is a gate reason, not a parse error."""
    definition = job_from_examples(
        _manifest(tmp_path), name="manifest", task="Read a shipping manifest."
    )
    assert definition is not None
    task = task_from_definition(definition)
    reasons = task.gate(
        {"carrier": "", "weight_a": 1.0, "weight_b": 1.0, "total_weight": 2.0}
    )
    assert any("carrier" in r for r in reasons)


def test_a_data_job_scores_dict_labels(tmp_path: Path) -> None:
    """A data job's labels arrive as dicts; scoring them as objects would fail."""
    definition = job_from_examples(
        _manifest(tmp_path), name="manifest", task="Read a shipping manifest."
    )
    assert definition is not None
    task = task_from_definition(definition)
    gold = {"carrier": "DHL", "weight_a": 12.0, "weight_b": 5.5, "total_weight": 17.5}
    same = task.score_pair(gold, dict(gold))
    half = task.score_pair(gold, {**gold, "carrier": "UPS", "weight_a": 1.0})
    assert same.quality == 1.0
    assert half.quality < same.quality


def test_a_definition_naming_an_unknown_field_is_refused() -> None:
    """A typo should be a message, not a run that dies halfway with no explanation."""
    definition = JobDefinition(
        name="broken",
        task="do something",
        fields=(FieldSpec(name="a"),),
        checks=(CheckSpec(kind="required", fields=("a", "nonexistent")),),
    )
    problems = validate_job(definition)
    assert any("nonexistent" in str(p) for p in problems)


def test_an_unknown_check_kind_is_refused() -> None:
    """The check vocabulary is closed on purpose."""
    definition = JobDefinition(
        name="x",
        task="y",
        fields=(FieldSpec(name="a"),),
        checks=(CheckSpec(kind="required", fields=("a",)), CheckSpec(kind="vibes")),
    )
    problems = validate_job(definition)
    assert any("unknown check" in str(p) for p in problems)


def test_a_job_with_no_required_check_is_refused() -> None:
    """A gate that rejects nothing is not a gate."""
    definition = JobDefinition(
        name="x",
        task="y",
        fields=(FieldSpec(name="a"),),
        checks=(CheckSpec(kind="arithmetic", parts=("a",), total="a"),),
    )
    assert any("required check" in str(p) for p in validate_job(definition))


def test_a_definition_round_trips_through_a_file(tmp_path: Path) -> None:
    """The format has to be writable by hand as well as inferred."""
    definition = JobDefinition(
        name="ticket",
        task="Route a support ticket.",
        fields=(FieldSpec(name="queue", phrase=True), FieldSpec(name="priority")),
        checks=(CheckSpec(kind="required", fields=("queue", "priority")),),
    )
    path = tmp_path / "job.json"
    path.write_text(definition.model_dump_json(indent=2))
    assert load_job(path) == definition


def test_a_missing_definition_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(CompileError, match="not found"):
        load_job(tmp_path / "absent.json")


def test_the_shipped_tasks_still_gate_and_score_what_they_did() -> None:
    """The regression this change risks, pinned without a model in the loop.

    Adding a way to describe jobs in data must not move the behaviour of the jobs that
    already work. A network call would make this test slow and flaky, so it pins the two
    things that decide a score - the field list and the gate - against known inputs. The
    end-to-end numbers are verified against a real model in the surface run.
    """
    assert RESTAURANT.scored_fields == (
        "restaurant",
        "intent",
        "status",
        "party_size",
        "when",
        "under_name",
        "booked",
    )
    assert RECEIPT.scored_fields == (
        "merchant",
        "date",
        "address",
        "currency",
        "subtotal",
        "tax",
        "total",
    )
    assert BANKING77.scored_fields == ("label",)
    assert RESTAURANT.retry_ladder == (0, 1)
    assert RECEIPT.retry_ladder == (0, 3)

    # The restaurant gate rejects a claimed booking whose own evidence denies it. That
    # is the failure the field checks cannot catch: every field is well formed and the
    # answer still contradicts the call it quotes.
    contradicted = RestaurantOutcome(
        restaurant="Loro",
        intent="reservation",
        status="confirmed",
        party_size=2,
        when="Friday",
        under_name="Maya",
        evidence="they did not take a reservation, just come by",
        booked=True,
    )
    # A consistent one: availability was checked, nobody took a booking.
    honest = contradicted.model_copy(
        update={
            "intent": "availability",
            "status": "unknown",
            "booked": False,
            "evidence": "a table was free Friday",
        }
    )
    assert RESTAURANT.gate(contradicted) != (), "a denied booking must be rejected"
    assert RESTAURANT.gate(honest) == (), "the honest version must pass"

    # The receipt gate rejects arithmetic that does not add up.
    bad = Receipt(
        merchant="HEB",
        date="2024-01-15",
        address="",
        currency="USD",
        subtotal=7.49,
        tax=0.61,
        service=0.0,
        discount=0.0,
        total=99.0,
        items=[],
    )
    assert RECEIPT.gate(bad) != (), (
        "a total that contradicts the parts must be rejected"
    )
