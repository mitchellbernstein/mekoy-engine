"""Stage 0 wiring. The optimiser itself needs a model, so it is not run here."""

from pathlib import Path

import pytest

from mekoy.bootstrap import DEFAULT_MAX_DEMOS, _kept_pairs
from mekoy.dataset import load_task_examples, split_examples
from mekoy.search import HarnessConfig, SearchSpace, evaluate
from mekoy.tasks import RECEIPT

_CORD = Path("examples/cord-receipt/cord.jsonl")


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


def test_the_bootstrap_axis_is_off_by_default() -> None:
    assert HarnessConfig().bootstrap is False
    assert SearchSpace().bootstrap == (False,)


def test_enabling_the_axis_doubles_the_candidate_pool() -> None:
    base = SearchSpace.for_task(RECEIPT, train_n=10)
    with_boot = SearchSpace(
        k_shots=base.k_shots,
        retries=base.retries,
        constrained=base.constrained,
        prompts=base.prompts,
        bootstrap=(False, True),
    )
    assert len(with_boot.candidates()) == 2 * len(base.candidates())
    assert any(c.bootstrap for c in with_boot.candidates())


def test_the_label_names_the_shot_selection() -> None:
    plain = HarnessConfig(k_shot=4, bootstrap=False).label
    boot = HarnessConfig(k_shot=4, bootstrap=True).label
    assert "boot" not in plain
    assert boot.endswith("boot")


def test_bootstrapped_shots_are_used_instead_of_the_first_k() -> None:
    """The whole point: Stage 0 chooses which rows to show."""
    rows = load_task_examples(_CORD, RECEIPT)
    split = split_examples(rows)
    pairs = tuple((r.text, r.outcome.model_dump_json()) for r in rows)
    # A pool whose first row is deliberately not the first training row.
    pool = ((split.train[3].text, split.train[3].outcome),)
    cfg = HarnessConfig(k_shot=1, retries=0, constrained=True, bootstrap=True)
    trial = evaluate(
        _Echo(pairs),
        split.dev,
        cfg,
        shots=split.train,
        task=RECEIPT,
        bootstrapped=pool,
    )
    assert trial.schema_rate == 1.0, trial.reasons
    assert trial.quality == pytest.approx(1.0)


def test_without_a_pool_the_bootstrap_flag_falls_back_to_the_rows() -> None:
    """An empty pool must not silently evaluate zero shots."""
    rows = load_task_examples(_CORD, RECEIPT)
    split = split_examples(rows)
    pairs = tuple((r.text, r.outcome.model_dump_json()) for r in rows)
    cfg = HarnessConfig(k_shot=2, retries=0, constrained=True, bootstrap=True)
    assert (
        evaluate(
            _Echo(pairs), split.dev, cfg, shots=split.train, task=RECEIPT
        ).schema_rate
        == 1.0
    )


def test_kept_demos_map_back_to_gold_and_dedupe() -> None:
    class _Demo:
        def __init__(self, document: str) -> None:
            self.document = document

    class _Program:
        def __init__(self) -> None:
            self.demos = [_Demo("a"), _Demo("a"), _Demo("b"), _Demo("unknown")]

    gold = {"a": "gold-a", "b": "gold-b"}
    kept = _kept_pairs(_Program(), _Program(), gold, limit=5)
    assert kept == (("a", "gold-a"), ("b", "gold-b"))


def test_kept_demos_respect_the_cap() -> None:
    class _Demo:
        def __init__(self, document: str) -> None:
            self.document = document

    class _Program:
        def __init__(self) -> None:
            self.demos = [_Demo(x) for x in ("a", "b", "c")]

    kept = _kept_pairs(_Program(), _Program(), {"a": 1, "b": 2, "c": 3}, limit=2)
    assert len(kept) == 2


def test_the_demo_cap_matches_the_shot_bracket() -> None:
    """Shots are worth testing at 0 and 4."""
    assert DEFAULT_MAX_DEMOS == 4
