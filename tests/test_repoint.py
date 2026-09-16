"""The model comparison: same harness, same held-out rows, one axis changed.

The claim the card makes is narrow and checkable, so what is tested here is that the
claim is not overstated: both lanes really run the same rows, the harness comes from the
spec rather than a default, a gap inside the noise band is called a tie, and speed is
reported on its own axis from quality.
"""

import pytest

from mekoy.dataset import ExampleRecord
from mekoy.outcome import RestaurantOutcome
from mekoy.repoint import Measurement, compare_models, format_model_comparison, measure
from mekoy.spec import OwnershipFlags, Slos, SystemSpec
from mekoy.tasks import RESTAURANT

_ROWS = 3


def _spec(**over: object) -> SystemSpec:
    base: dict[str, object] = {
        "spec_version": 2,
        "task": "restaurant call outcome",
        "json_schema": {"type": "object", "properties": {}},
        "slos": Slos(quality=0.9, cost_per_doc=0.01, latency_ms=1000),
        "model_id": "qwen2.5:7b",
        "k_shot": 2,
        "retries": 1,
        "constrained": True,
        "prompt": "strict",
        "schema_constrained": True,
        "ownership": OwnershipFlags(runtime_owned=True, downloadable=True),
    }
    base.update(over)
    return SystemSpec(**base)  # type: ignore[arg-type]


def _rows() -> tuple[ExampleRecord, ...]:
    """Three documents, so a lane's row count is unambiguous."""
    return tuple(
        ExampleRecord(
            text=f"call {i}: a table for {i + 2} Friday under Maya.",
            outcome=RestaurantOutcome(
                restaurant="Uchi",
                intent="reservation",
                status="confirmed",
                party_size=i + 2,
                when="Friday",
                under_name="Maya",
                evidence="a table",
                booked=True,
            ),
        )
        for i in range(_ROWS)
    )


class _PerModel:
    """A completer whose answer is fixed, so each lane's score is predictable."""

    local = True

    def __init__(self, *, answer: str = "good") -> None:
        self.answer = answer
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def complete(
        self,
        *,
        system: str,
        user: str,
        **_rest: object,
    ) -> str:
        self.systems.append(system)
        self.prompts.append(user)
        if self.answer == "bad":
            return "{}"
        return RestaurantOutcome(
            restaurant="Uchi",
            intent="reservation",
            status="confirmed",
            party_size=2,
            when="Friday",
            under_name="Maya",
            evidence="a table",
            booked=True,
        ).model_dump_json()


def test_both_lanes_see_the_same_rows_and_the_spec_harness() -> None:
    """A comparison that ran different rows, or a default harness, proves nothing.

    The model is a property of the completer, not of the call, so a lane is one
    completer; what the measurement must guarantee is that both lanes were handed the
    same test rows and the spec's own prompt variant.
    """
    spec = _spec()
    seven, fourteen = _PerModel(), _PerModel()
    rows = _rows()
    source = measure(
        spec, completer=seven, model="qwen2.5:7b", rows=rows, train=(), task=RESTAURANT
    )
    target = measure(
        spec,
        completer=fourteen,
        model="qwen2.5:14b",
        rows=rows,
        train=(),
        task=RESTAURANT,
    )
    assert source.rows == _ROWS
    assert target.rows == _ROWS
    # One call per document per lane, and both lanes got the same documents.
    assert len(seven.prompts) == len(fourteen.prompts) == _ROWS
    assert seven.prompts == fourteen.prompts
    # The spec's prompt variant, not the shipped default.
    assert all("Work carefully" in system for system in seven.systems)


def test_the_harness_comes_from_the_spec_not_a_default() -> None:
    """The shot count is the observable: a defaulting measure skips the spec."""
    spec = _spec(k_shot=2, prompt="default")
    completer = _PerModel()
    _ = measure(
        spec,
        completer=completer,
        model="qwen2.5:7b",
        rows=_rows()[:1],
        train=_rows(),
        task=RESTAURANT,
    )
    prompt = completer.prompts[0]
    assert prompt.count('"restaurant"') >= 2, "the spec's shots missed the prompt"


def test_a_rejecting_lane_scores_zero() -> None:
    """A lane whose answers fail the gate scores nothing, as in the search."""
    spec = _spec()
    out = measure(
        spec,
        completer=_PerModel(answer="bad"),
        model="qwen2.5:7b",
        rows=_rows(),
        train=(),
        task=RESTAURANT,
    )
    assert out.rejected == _ROWS
    assert out.quality == 0.0


def test_a_gap_inside_the_noise_band_is_a_tie() -> None:
    spec = _spec()
    source = Measurement("a", 0.900, 0.900, 1.0, 54, 0, 1000.0)
    target = Measurement("b", 0.905, 0.905, 1.0, 54, 0, 1000.0)
    out = compare_models(spec, source=source, target=target)
    assert out["quality_winner"] == "tie"
    assert out["delta_quality"] == pytest.approx(0.005)


def test_speed_is_reported_on_its_own_axis_from_quality() -> None:
    """The measured surprise: the larger model was also faster."""
    spec = _spec()
    source = Measurement("qwen2.5:7b", 0.905, 0.847, 0.963, 54, 2, 40287.0)
    target = Measurement("qwen2.5:14b", 0.942, 0.921, 1.0, 54, 0, 27310.0)
    card = format_model_comparison(
        spec=spec, source=source, target=target, source_label="7b", target_label="14b"
    )
    out = compare_models(spec, source=source, target=target)
    assert out["quality_winner"] == "target"
    assert out["faster"] == "target"
    assert "speed and size are not the same axis" in card
    assert "held-out rows: 54" in card
    assert "k=2 r=1 grammar schema-pinned prompt=strict" in card
