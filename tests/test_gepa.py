"""GEPA-light's contract. The optimiser itself is not run here: it needs a model."""

import pytest

from mekoy.banking77 import Intent
from mekoy.errors import CompileError
from mekoy.gepa import (
    DEFAULT_METRIC_CALLS,
    _ollama_base,
    available,
    feedback_for,
    field_feedback,
    optimize,
)
from mekoy.tasks import BANKING77


def test_the_metric_call_cap_sits_in_the_planned_band() -> None:
    """GEPA is capped at 50-150 metric calls."""
    assert 50 <= DEFAULT_METRIC_CALLS <= 150


def test_a_correct_reply_scores_one_with_usable_feedback() -> None:
    score, feedback = feedback_for(
        BANKING77, Intent(label="age_limit"), '{"label": "age_limit"}'
    )
    assert score == 1.0
    assert "agreed" in feedback


def test_a_wrong_label_names_the_disagreement() -> None:
    """A bare 0.0 tells a reflection model nothing it can act on."""
    score, feedback = feedback_for(
        BANKING77, Intent(label="age_limit"), '{"label": "atm_support"}'
    )
    assert score == 0.0
    assert "age_limit" in feedback
    assert "atm_support" in feedback


def test_the_gate_reason_becomes_the_feedback() -> None:
    """The optimiser is pushed by the checks we actually enforce."""
    score, feedback = feedback_for(
        BANKING77, Intent(label="age_limit"), '{"label": "not_an_intent"}'
    )
    assert score == 0.0
    assert "gate" in feedback
    assert "not one of the 77" in feedback


def test_field_feedback_names_every_disagreement() -> None:
    text = field_feedback(BANKING77, Intent(label="a"), Intent(label="b"))
    assert "label" in text
    assert "'a'" in text
    assert "'b'" in text


def test_unparseable_output_scores_zero() -> None:
    score, feedback = feedback_for(BANKING77, Intent(label="age_limit"), "not json")
    assert score == 0.0
    assert "gate" in feedback.lower() or "json" in feedback.lower()


def test_ollama_base_drops_the_openai_compatible_suffix() -> None:
    """litellm appends its own path, so /v1 doubles up and 404s."""
    assert _ollama_base("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434"
    assert _ollama_base("http://127.0.0.1:11434/v1/") == "http://127.0.0.1:11434"
    assert _ollama_base("http://host:1234") == "http://host:1234"


def test_availability_is_reported_not_assumed() -> None:
    assert isinstance(available(), bool)


def _pairs() -> tuple[tuple[str, object], ...]:
    return (("a text", Intent(label="age_limit")),)


def test_setting_both_budgets_is_refused_by_dspy() -> None:
    """Auto and the cap are both wanted; dspy allows exactly one."""
    if not available():
        pytest.skip("dspy extra not installed")
    with pytest.raises(CompileError, match="exactly one"):
        optimize(
            BANKING77,
            trainset=_pairs(),
            valset=_pairs(),
            model_id="m",
            base_url="http://127.0.0.1:11434/v1",
            max_metric_calls=60,
            auto="light",
        )


def test_setting_neither_budget_is_refused() -> None:
    if not available():
        pytest.skip("dspy extra not installed")
    with pytest.raises(CompileError, match="exactly one"):
        optimize(
            BANKING77,
            trainset=_pairs(),
            valset=_pairs(),
            model_id="m",
            base_url="http://127.0.0.1:11434/v1",
            max_metric_calls=None,
            auto=None,
        )


def test_an_empty_trainset_is_refused() -> None:
    if not available():
        pytest.skip("dspy extra not installed")
    with pytest.raises(CompileError, match="trainset"):
        optimize(
            BANKING77,
            trainset=(),
            valset=_pairs(),
            model_id="m",
            base_url="http://127.0.0.1:11434/v1",
        )
