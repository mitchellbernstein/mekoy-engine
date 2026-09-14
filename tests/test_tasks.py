from pathlib import Path

import pytest

from mekoy.compile import Budget, compile_system, format_report
from mekoy.dataset import load_task_examples, split_examples
from mekoy.receipt import Receipt
from mekoy.search import HarnessConfig, SearchSpace, evaluate
from mekoy.tasks import RECEIPT, RESTAURANT, task_for_path
from mekoy.verify import VerifyFail, VerifyOk, parse_and_gate

_CAFE = Path("examples/cord-receipt/hard.jsonl")
_CALLS = Path("examples/bucko-restaurant/examples.jsonl")

_SOUND = (
    '{"merchant":"Uchi","date":"2024-03-14","currency":"USD","subtotal":10.0,'
    '"tax":0.8,"total":10.8,"items":[{"desc":"Taco","qty":2,"unit_price":5.0,'
    '"line_total":10.0}]}'
)
_BROKEN = (
    '{"merchant":"Uchi","date":"2024-03-14","currency":"USD","subtotal":9.0,'
    '"tax":0.8,"total":10.8,"items":[{"desc":"Taco","qty":2,"unit_price":5.0,'
    '"line_total":10.0}]}'
)


class _Echo:
    """Returns the gold for whichever labeled text is in the prompt."""

    local: bool = True

    def __init__(self, pairs: tuple[tuple[str, str], ...]) -> None:
        self._pairs = pairs

    def complete(
        self,
        *,
        system: str,
        user: str,
        constrained: bool = True,
        temperature: float = 0.0,
        schema: dict[str, object] | None = None,
    ) -> str:
        del system, constrained, temperature, schema
        tail = user.rsplit("Text:\n", 1)[-1]
        target = tail.rsplit("\nJSON:", 1)[0]
        for text, raw in self._pairs:
            if text == target:
                return raw
        return "{}"


def test_task_detection_follows_the_row_shape() -> None:
    assert task_for_path(_CALLS) is RESTAURANT
    assert task_for_path(_CAFE) is RECEIPT


def test_receipt_rows_load_through_the_task_loader() -> None:
    rows = load_task_examples(_CAFE, RECEIPT)
    assert rows
    assert isinstance(rows[0].outcome, Receipt)
    assert split_examples(rows).test


def test_the_generic_gate_runs_a_receipt_check() -> None:
    sound = parse_and_gate(_SOUND, model=RECEIPT.model, gate=RECEIPT.gate)
    assert isinstance(sound, VerifyOk)
    broken = parse_and_gate(_BROKEN, model=RECEIPT.model, gate=RECEIPT.gate)
    assert isinstance(broken, VerifyFail)
    assert any("subtotal" in r for r in broken.reasons), broken.reasons


def test_the_restaurant_gate_is_unchanged_by_the_generic_path() -> None:
    ok = (
        '{"restaurant":"Uchi","intent":"availability","status":"confirmed",'
        '"party_size":2,"when":"Friday","under_name":null,'
        '"evidence":"A table for two is open Friday.","booked":false}'
    )
    assert isinstance(
        parse_and_gate(ok, model=RESTAURANT.model, gate=RESTAURANT.gate), VerifyOk
    )


def test_the_space_follows_the_task_ladder() -> None:
    """An arithmetic gate needs more repair passes than a policy violation."""
    receipt_space = SearchSpace.for_task(RECEIPT, train_n=10)
    assert receipt_space.retries == RECEIPT.retry_ladder
    assert receipt_space.prompts == tuple(RECEIPT.prompt_variants)
    call_space = SearchSpace.for_task(RESTAURANT, train_n=100)
    assert call_space.retries == RESTAURANT.retry_ladder
    assert len(call_space.prompts) > 1


def test_evaluate_scores_a_receipt_with_its_own_schema() -> None:
    """The bug this test exists for: the loop defaulted to the restaurant task."""
    rows = load_task_examples(_CAFE, RECEIPT)
    split = split_examples(rows)
    pairs = tuple((r.text, r.outcome.model_dump_json()) for r in rows)
    trial = evaluate(
        _Echo(pairs),
        split.dev,
        HarnessConfig(k_shot=0, retries=0, constrained=True),
        shots=split.train,
        task=RECEIPT,
    )
    assert trial.schema_rate == 1.0, trial.reasons
    assert trial.quality == pytest.approx(1.0)


def test_a_receipt_compile_runs_end_to_end() -> None:
    rows = load_task_examples(_CAFE, RECEIPT)
    split = split_examples(rows)
    pairs = tuple((r.text, r.outcome.model_dump_json()) for r in rows)
    report = compile_system(
        _Echo(pairs), split, SearchSpace.single(), Budget(trials=1), task=RECEIPT
    )
    text = format_report(report)
    assert "training: skipped" in text
    assert report.test.schema_rate == 1.0
    assert report.test.quality == pytest.approx(1.0)


def test_all_rejected_search_says_so() -> None:
    """A 0.000 row with no explanation is worse than saying the gate blocked it."""
    rows = load_task_examples(_CAFE, RECEIPT)
    split = split_examples(rows)
    report = compile_system(
        _Echo(()), split, SearchSpace.single(), Budget(trials=1), task=RECEIPT
    )
    text = format_report(report)
    assert report.test.schema_rate == 0.0
    assert "blocked: every candidate failed the gate" in text
