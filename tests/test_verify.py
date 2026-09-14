import json

from mekoy.outcome import RestaurantOutcome
from mekoy.verify import VerifyFail, VerifyOk, parse_and_verify


def _gold() -> RestaurantOutcome:
    return RestaurantOutcome(
        restaurant="North Loop Bistro",
        intent="reservation",
        status="confirmed",
        party_size=4,
        when="next Friday at 7:00 PM",
        under_name="Mitchell",
        evidence="The host confirmed a reservation for four.",
        booked=True,
    )


def test_parse_and_verify_accepts_confirmed_booking() -> None:
    result = parse_and_verify(_gold().model_dump_json())
    assert isinstance(result, VerifyOk)
    assert result.outcome.booked is True


def test_parse_and_verify_rejects_availability_booked() -> None:
    bad = _gold().model_copy(update={"intent": "availability", "booked": True})
    result = parse_and_verify(bad.model_dump_json())
    assert isinstance(result, VerifyFail)


def test_parse_and_verify_rejects_invalid_json() -> None:
    result = parse_and_verify("{not json")
    assert isinstance(result, VerifyFail)


def test_gate_rejects_a_booked_claim_the_evidence_denies() -> None:
    """The failure field checks cannot see.

    Observed live: a model returned intent=reservation, status=confirmed,
    booked=true, and quoted "they did not book it" in its own evidence. Every
    field agreed with every other field, so only the text can catch it.
    """
    raw = (
        '{"restaurant":"Home Slice","intent":"reservation","status":"confirmed",'
        '"party_size":2,"when":"tonight at 8","under_name":null,'
        '"evidence":"Host said a table for two is free tonight at 8, '
        'they did not book it.","booked":true}'
    )
    result = parse_and_verify(raw)
    assert isinstance(result, VerifyFail)
    assert any("denies it was taken" in r for r in result.reasons), result.reasons


def test_gate_still_accepts_a_real_confirmation() -> None:
    raw = (
        '{"restaurant":"Home Slice","intent":"reservation","status":"confirmed",'
        '"party_size":2,"when":"tonight at 8","under_name":null,'
        '"evidence":"Host confirmed a reservation for two tonight at 8.",'
        '"booked":true}'
    )
    assert isinstance(parse_and_verify(raw), VerifyOk)


def test_gate_does_not_punish_evidence_without_a_denial() -> None:
    """Gold evidence does not always contain a booking verb; that is not a denial."""
    raw = (
        '{"restaurant":"Matt\'s El Rancho","intent":"reservation",'
        '"status":"confirmed","party_size":4,"when":"Sunday at 1pm",'
        '"under_name":"Bernstein",'
        '"evidence":"Host confirmed patio for four Sunday at 1pm under Bernstein.",'
        '"booked":true}'
    )
    assert isinstance(parse_and_verify(raw), VerifyOk)


def _row(**over: object) -> str:
    base: dict[str, object] = {
        "restaurant": "Uchi",
        "intent": "reservation",
        "status": "confirmed",
        "party_size": 2,
        "when": "Friday",
        "under_name": None,
        "evidence": "Host confirmed a reservation for two Friday.",
        "booked": True,
    }
    base.update(over)
    return json.dumps(base)


def test_gate_rejects_confirmed_when_the_evidence_says_full() -> None:
    """Same hole as a hallucinated booking, one field over."""
    raw = _row(evidence="They were fully booked.", booked=True)
    result = parse_and_verify(raw)
    assert isinstance(result, VerifyFail)
    assert any("nothing was free" in r for r in result.reasons), result.reasons


def test_gate_rejects_unavailable_when_the_evidence_offers_a_table() -> None:
    raw = _row(
        intent="availability",
        status="unavailable",
        booked=False,
        evidence="Staff said a table for two is available Friday.",
    )
    result = parse_and_verify(raw)
    assert isinstance(result, VerifyFail)
    assert any("offers a table" in r for r in result.reasons), result.reasons


def test_gate_rejects_unknown_when_the_evidence_says_it_was_taken() -> None:
    raw = _row(status="unknown", booked=False)
    result = parse_and_verify(raw)
    assert isinstance(result, VerifyFail)
    assert any("evidence says it was taken" in r for r in result.reasons), (
        result.reasons
    )


def test_negated_availability_is_not_read_as_an_offer() -> None:
    """'No table for two is available' must not be read as a table being free."""
    raw = _row(
        intent="availability",
        status="unavailable",
        booked=False,
        evidence="No table for two is available Friday.",
    )
    assert isinstance(parse_and_verify(raw), VerifyOk)


def test_legitimate_status_evidence_still_passes() -> None:
    cases = [
        _row(
            intent="availability",
            status="unavailable",
            booked=False,
            evidence="Fully booked Friday, no table for 2.",
        ),
        _row(
            intent="availability",
            status="confirmed",
            booked=False,
            evidence="A table for 2 is open Friday, but they did not take it.",
        ),
        _row(evidence="Host reserved a table for two Friday under Sam."),
    ]
    for raw in cases:
        assert isinstance(parse_and_verify(raw), VerifyOk), raw


def test_gate_rejects_a_headcount_the_evidence_contradicts() -> None:
    raw = _row(party_size=6, evidence="Host confirmed a table for two Friday.")
    result = parse_and_verify(raw)
    assert isinstance(result, VerifyFail)
    assert any("evidence says" in r for r in result.reasons), result.reasons


def test_gate_accepts_evidence_that_never_names_a_headcount() -> None:
    """Silence is not a contradiction."""
    raw = _row(party_size=4, evidence="Host confirmed the reservation Friday.")
    assert isinstance(parse_and_verify(raw), VerifyOk)


def test_a_time_is_not_read_as_a_headcount() -> None:
    """'for 4 at 7pm' must not make the gate think the party size is 7."""
    raw = _row(
        party_size=4,
        when="Friday at 7pm",
        evidence="Took the reservation for 4 at 7pm Friday.",
    )
    assert isinstance(parse_and_verify(raw), VerifyOk)


def test_spelled_headcounts_are_understood() -> None:
    raw = _row(party_size=3, evidence="Host confirmed a table for three Friday.")
    assert isinstance(parse_and_verify(raw), VerifyOk)
