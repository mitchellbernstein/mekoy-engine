from mekoy.outcome import RestaurantOutcome
from mekoy.score import score_outcome


def _gold() -> RestaurantOutcome:
    return RestaurantOutcome(
        restaurant="North Loop Bistro",
        intent="availability",
        status="confirmed",
        party_size=4,
        when="Friday 7pm",
        under_name=None,
        evidence="A table is available. No reservation was made.",
        booked=False,
    )


def test_score_outcome_is_perfect_on_identical() -> None:
    gold = _gold()
    scored = score_outcome(gold=gold, pred=gold)
    assert scored.schema_ok is True
    assert scored.field_hits == scored.field_total


def test_score_outcome_drops_on_wrong_intent() -> None:
    gold = _gold()
    pred = gold.model_copy(update={"intent": "reservation"})
    scored = score_outcome(gold=gold, pred=pred)
    assert scored.field_hits < scored.field_total
