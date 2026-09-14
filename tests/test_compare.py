from pathlib import Path

import pytest

from mekoy.compare import compare_cards, format_comparison
from mekoy.compile import SearchSpace, compile_system
from mekoy.dataset import ExampleRecord, load_examples, split_examples
from mekoy.outcome import RestaurantOutcome
from mekoy.score import score_outcome

_CARD = """winner  k=2 r=1 grammar strict
dev     quality=0.985 schema=1.000 n=19
test    quality=0.940 strict=0.940 schema=1.000 n=19
cost    $0.00000/doc  latency=6425ms/doc
trials  6 candidate(s), stopped_on_slo=False
training: skipped
arms:
  k=2 r=1 grammar strict       dev=0.985 schema=1.000 $0.00000 6436ms
"""

_OTHER = (
    _CARD.replace("quality=0.940", "quality=0.970")
    .replace("latency=6425ms", "latency=1615ms")
    .replace("$0.00000/doc", "$0.01700/doc")
)


def test_compare_cards_reads_the_measured_lines() -> None:
    cmp = compare_cards("local", _CARD, "frontier", _OTHER)
    assert cmp.n_test == 19
    assert cmp.left_quality == pytest.approx(0.940)
    assert cmp.right_quality == pytest.approx(0.970)
    assert cmp.right_cost == pytest.approx(0.017)
    assert cmp.quality_winner == "right"
    assert cmp.faster == "right"


def test_compare_reports_a_tie_inside_the_noise_band() -> None:
    close = _OTHER.replace("quality=0.970", "quality=0.945")
    cmp = compare_cards("a", _CARD, "b", close)
    assert cmp.quality_winner == "tie"
    assert "tie" in format_comparison(cmp)


def test_compare_refuses_different_test_sizes() -> None:
    bigger = _OTHER.replace("n=19", "n=40")
    with pytest.raises(ValueError, match="different test sizes"):
        compare_cards("a", _CARD, "b", bigger)


def test_compare_refuses_a_card_with_no_test_line() -> None:
    with pytest.raises(ValueError, match="no test line"):
        compare_cards("a", "training: skipped\n", "b", _CARD)


class _GoldEcho:
    local: bool = True

    def __init__(self, rows: tuple[ExampleRecord, ...]) -> None:
        self._by_text = {r.text: r.outcome.model_dump_json() for r in rows}

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
        return self._by_text.get(tail.rsplit("\nJSON:", 1)[0], "{}")


def test_two_compiles_of_the_same_input_are_identical() -> None:
    rows = load_examples(Path("examples/bucko-restaurant/examples.jsonl"))
    split = split_examples(rows)
    first = compile_system(_GoldEcho(rows), split, SearchSpace.single())
    second = compile_system(_GoldEcho(rows), split, SearchSpace.single())
    assert first.test.quality == second.test.quality


def test_when_normalization_ignores_time_formatting() -> None:
    """'Sunday at 1:00 PM' and 'Sunday at 1pm' are the same answer."""

    def row(when: str) -> RestaurantOutcome:
        return RestaurantOutcome(
            restaurant="Uchi",
            intent="availability",
            status="confirmed",
            party_size=4,
            when=when,
            under_name=None,
            evidence="open",
            booked=False,
        )

    pairs = [
        ("Sunday at 1:00 PM", "Sunday at 1pm"),
        ("9am Saturday", "Saturday at 9:00am"),
        ("Saturday brunch 10:30", "10:30 AM saturday brunch"),
    ]
    for gold, pred in pairs:
        assert score_outcome(gold=row(gold), pred=row(pred)).quality == 1.0, (
            gold,
            pred,
        )


def test_when_normalization_still_separates_different_times() -> None:
    def row(when: str) -> RestaurantOutcome:
        return RestaurantOutcome(
            restaurant="Uchi",
            intent="availability",
            status="confirmed",
            party_size=4,
            when=when,
            under_name=None,
            evidence="open",
            booked=False,
        )

    assert (
        score_outcome(gold=row("Sunday at 1pm"), pred=row("Sunday at 2pm")).quality
        < 1.0
    )
    assert (
        score_outcome(gold=row("Sunday at 1pm"), pred=row("Saturday at 1pm")).quality
        < 1.0
    )
    assert (
        score_outcome(gold=row("tonight at 8"), pred=row("tomorrow at 8")).quality < 1.0
    )
