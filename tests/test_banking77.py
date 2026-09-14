"""BANKING77: the classification task class. No network."""

from collections import Counter
from pathlib import Path

from mekoy.banking77 import (
    CATEGORIES,
    Intent,
    _interleave,
    banking_gate,
    banking_prompt,
    score_intent,
)
from mekoy.dataset import load_task_examples, split_examples
from mekoy.tasks import BANKING77, RECEIPT, RESTAURANT, task_for_path
from mekoy.verify import VerifyFail, VerifyOk, parse_and_gate

_CORPUS = Path("examples/banking77/banking77.jsonl")


def test_the_label_set_is_the_published_one() -> None:
    assert len(CATEGORIES) == 77
    assert len(set(CATEGORIES)) == 77


def test_the_prompt_lists_every_intent() -> None:
    """A classifier cannot pick from a set it has not been shown."""
    prompt = banking_prompt()
    for category in CATEGORIES:
        assert category in prompt


def test_gate_rejects_a_label_outside_the_set() -> None:
    assert banking_gate(Intent(label="age_limit")) == ()
    problems = banking_gate(Intent(label="transfer_money_now"))
    assert problems
    assert "not one of the 77" in problems[0]


def test_scoring_is_exact_accuracy() -> None:
    gold = Intent(label="age_limit")
    assert score_intent(gold=gold, pred=Intent(label="age_limit")).quality == 1.0
    assert score_intent(gold=gold, pred=Intent(label="atm_support")).quality == 0.0
    assert score_intent(gold=gold, pred=Intent(label=" atm_support ")).quality == 0.0


def test_classification_has_no_partial_credit() -> None:
    """quality and strict_quality agree: there is no fuzzy form of a label."""
    score = score_intent(gold=Intent(label="a"), pred=Intent(label="b"))
    assert score.quality == score.strict_quality


def test_interleaving_covers_every_class_in_any_prefix() -> None:
    """The bug this exists for: 600 rows of a 77-class task covered five."""
    rows = [(f"text {i}", f"cat_{i % 5}") for i in range(50)]
    ordered = _interleave(rows)
    assert len({label for _, label in ordered[:5]}) == 5


def test_the_generic_gate_validates_a_classification_label() -> None:
    good = parse_and_gate(
        '{"label": "age_limit"}', model=BANKING77.model, gate=BANKING77.gate
    )
    assert isinstance(good, VerifyOk)
    bad = parse_and_gate(
        '{"label": "not_an_intent"}', model=BANKING77.model, gate=BANKING77.gate
    )
    assert isinstance(bad, VerifyFail)
    assert any("not one of the 77" in r for r in bad.reasons)


def test_task_detection_separates_classification_from_extraction() -> None:
    assert task_for_path(_CORPUS) is BANKING77
    assert task_for_path(Path("examples/cord-receipt/cord.jsonl")) is RECEIPT
    assert task_for_path(Path("examples/bucko-restaurant/examples.jsonl")) is RESTAURANT


def test_the_corpus_covers_every_intent() -> None:
    rows = load_task_examples(_CORPUS, BANKING77)
    counts = Counter(r.outcome.label for r in rows)
    assert len(counts) == 77, sorted(counts)
    assert min(counts.values()) >= 5
    assert len(split_examples(rows).test) > 50


def test_a_scalar_label_row_loads_into_the_schema() -> None:
    """Classification rows carry the label as a bare value, not an object."""
    rows = load_task_examples(_CORPUS, BANKING77)
    assert isinstance(rows[0].outcome, Intent)
    assert rows[0].outcome.label in CATEGORIES


def test_every_task_carries_its_own_scorer() -> None:
    """score_pair no longer dispatches on the task name."""
    for task, gold, pred, expected in (
        (BANKING77, Intent(label="a"), Intent(label="a"), 1.0),
        (BANKING77, Intent(label="a"), Intent(label="b"), 0.0),
    ):
        assert task.score_pair(gold, pred).quality == expected


def test_the_banking_task_has_no_free_text_fields() -> None:
    assert BANKING77.phrase_fields == frozenset()
    assert BANKING77.scored_fields == ("label",)
    assert BANKING77.retry_ladder == (0, 1)
