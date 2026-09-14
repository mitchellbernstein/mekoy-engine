from pathlib import Path

from mekoy.dataset import load_examples
from mekoy.errors import ModelUnreachableError
from mekoy.evalgen import build_scenarios
from mekoy.outcome import RestaurantOutcome
from mekoy.paraphrase import (
    Rendered,
    accepts,
    claims_booking,
    generate,
    label_problems,
    mentions_party_size,
    render_prompt,
    write_corpus,
)
from mekoy.paraphrase import requests_booking as _requests_booking
from mekoy.verify import NUMBER_WORDS as _NUMBER_WORDS

_SCENARIOS = build_scenarios()
_BOOKED = next(s for s in _SCENARIOS if s.booked)
_UNBOOKED = next(s for s in _SCENARIOS if not s.booked and s.status == "confirmed")
_NO_ANSWER = next(s for s in _SCENARIOS if s.status == "unavailable")


class _Scripted:
    local: bool = True

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.prompts: list[str] = []

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
        self.prompts.append(user)
        return self._replies.pop(0)


def test_render_prompt_states_the_whole_label() -> None:
    text = render_prompt(_BOOKED)
    for key in ("restaurant:", "intent:", "status:", "party_size:", "when:", "booked:"):
        assert key in text


def test_accepts_a_transcript_that_asserts_the_gold() -> None:
    text = (
        f"{_BOOKED.restaurant} took the reservation for "
        f"{_BOOKED.party_size} {_BOOKED.when}."
    )
    if _BOOKED.under_name:
        text += f" Under {_BOOKED.under_name}."
    assert accepts(_BOOKED, text) == ()


def test_rejects_a_claim_of_booking_when_nothing_was_booked() -> None:
    bad = (
        f"{_UNBOOKED.restaurant}: a table for {_UNBOOKED.party_size} "
        f"{_UNBOOKED.when} is open, and the reservation is in."
    )
    problems = accepts(_UNBOOKED, bad)
    assert any("claims it was taken" in p for p in problems), problems


def test_an_understated_booking_is_kept_as_a_hard_case() -> None:
    """A row whose transcript never says "booked" is still validly labeled.

    The gold is what happened, not what the model chose to write. Rejecting these
    would bias the corpus toward easy rows and hide exactly the failure we care
    about. Only a contradiction is fatal.
    """
    text = (
        f"{_BOOKED.restaurant} took a table for {mentions_word(_BOOKED.party_size)} "
        f"{_BOOKED.when} under {_BOOKED.under_name}, no mention of a booking."
    )
    assert accepts(_BOOKED, text) == ()

    # Missing facts are still fatal: the party size, time, and name are the row.
    vague = f"{_BOOKED.restaurant}: a table is open."
    assert any("party_size" in p for p in accepts(_BOOKED, vague))


def test_unknown_rendered_as_no_answer_is_a_contradiction() -> None:
    """If the transcript says nobody spoke, the call is not merely unclear."""
    unknown = next(s for s in _SCENARIOS if s.status == "unknown")
    text = (
        f"{unknown.restaurant}: nobody answered, it went to voicemail. "
        f"Not sure about the table for {unknown.party_size}."
    )
    problems = accepts(unknown, text)
    assert any("nobody spoke" in p for p in problems), problems


def test_rejects_a_missing_restaurant() -> None:
    problems = accepts(_NO_ANSWER, "Somewhere did not answer. Voicemail.")
    assert "restaurant name missing" in problems


def test_rejects_a_second_restaurant() -> None:
    text = f"{_UNBOOKED.restaurant} has a table open, unlike Franklin BBQ."
    problems = accepts(_UNBOOKED, text)
    assert any("second restaurant" in p for p in problems), problems


def test_rejects_unavailable_written_as_unclear() -> None:
    text = (
        f"{_NO_ANSWER.restaurant}: the call was unclear, not sure if a table "
        f"for {_NO_ANSWER.party_size} is open."
    )
    problems = accepts(_NO_ANSWER, text)
    assert any("merely unclear" in p for p in problems), problems


def test_rejects_a_dropped_party_size() -> None:
    text = f"{_NO_ANSWER.restaurant} did not answer after four rings. Voicemail."
    problems = accepts(_NO_ANSWER, text)
    assert any("party_size" in p for p in problems), problems


def test_rejects_an_empty_transcript() -> None:
    assert accepts(_BOOKED, "   ") == ("empty transcript",)


def test_generate_retries_then_records_the_reject() -> None:
    good = (
        f"{_BOOKED.restaurant} took the reservation for {_BOOKED.party_size} "
        f"{_BOOKED.when} under {_BOOKED.under_name}."
    )
    completer = _Scripted([good])
    ok, bad = generate(completer, (_BOOKED,), attempts=2)
    assert len(ok) == 1
    assert bad == ()
    assert ok[0].outcome().booked is True

    completer = _Scripted(["nonsense", "still nonsense"])
    ok, bad = generate(completer, (_BOOKED,), attempts=2)
    assert ok == ()
    assert len(bad) == 1
    assert bad[0][0] is _BOOKED
    assert bad[0][1]


def test_written_corpus_round_trips_through_the_loader(tmp_path: Path) -> None:
    """The loader needs three rows; the writer must produce a loadable file."""
    rows_in = tuple(
        Rendered(scenario=s, text=f"note for {s.restaurant}")
        for s in (_BOOKED, _UNBOOKED, _NO_ANSWER)
    )
    path = write_corpus(tmp_path / "p.jsonl", rows_in)
    rows = load_examples(path)
    assert len(rows) == 3
    assert [r.outcome.booked for r in rows] == [
        s.booked for s in (_BOOKED, _UNBOOKED, _NO_ANSWER)
    ]
    assert rows[0].text == f"note for {_BOOKED.restaurant}"


def test_a_model_error_costs_one_attempt_not_the_run() -> None:
    """A local server can time out mid-corpus; that must not abort the corpus."""

    class _Flaky:
        local: bool = True

        def __init__(self) -> None:
            self.calls = 0

        def complete(
            self,
            *,
            system: str,
            user: str,
            constrained: bool = True,
            temperature: float = 0.0,
            schema: dict[str, object] | None = None,
        ) -> str:
            del system, user, constrained, temperature, schema
            self.calls += 1
            if self.calls == 1:
                raise ModelUnreachableError(message="timed out")
            return (
                f"{_BOOKED.restaurant} took the reservation for "
                f"{_BOOKED.party_size} {_BOOKED.when} under {_BOOKED.under_name}."
            )

    completer = _Flaky()
    ok, bad = generate(completer, (_BOOKED,), attempts=2)
    assert len(ok) == 1
    assert bad == ()


def test_negated_claims_are_not_read_as_bookings() -> None:
    """'did not confirm the reservation' is a denial; rejecting it would lose data."""
    assert claims_booking("they did not confirm the reservation") is False
    assert claims_booking("we never took the reservation") is False
    assert claims_booking("couldn't book it") is False
    assert claims_booking("she confirmed a reservation") is True
    assert claims_booking("they took the reservation") is True
    assert claims_booking("the reservation is in") is True


def test_spoken_party_sizes_count() -> None:
    assert mentions_party_size("a table for two tonight", 2) is True
    assert mentions_party_size("party of 4 at 8pm", 4) is True
    assert mentions_party_size("a table for two", 4) is False


def test_accepts_a_sentence_phrased_naturally() -> None:
    """The earlier phrase list missed 'confirmed a reservation'."""
    text = (
        f"hi, {_BOOKED.restaurant} confirmed a reservation for "
        f"{mentions_word(_BOOKED.party_size)} {_BOOKED.when} "
        f"under {_BOOKED.under_name}"
    )
    assert accepts(_BOOKED, text) == ()


def mentions_word(size: int) -> str:
    return _NUMBER_WORDS[size][1]


def test_an_implied_request_is_accepted() -> None:
    """Real calls imply the request; requiring it explicitly rejects real data.

    This is the check that was tried and thrown away: it failed the hand-labeled
    fixture, which is the fixture's whole job.
    """
    reservation = next(s for s in _SCENARIOS if s.intent == "reservation")
    implied = (
        f"{reservation.restaurant} did not answer. Voicemail after eight rings. "
        f"Party of {reservation.party_size} {reservation.when}."
    )
    assert not any(
        "intent" in p or "asks" in p or "requests" in p
        for p in label_problems(reservation.outcome(), implied)
    )


def test_availability_label_with_a_booking_request_is_a_contradiction() -> None:
    """The one intent check that is genuinely a contradiction."""
    availability = next(s for s in _SCENARIOS if s.intent == "availability")
    asked = (
        f"{availability.restaurant}: we called to book a table for "
        f"{availability.party_size} and they said one is open {availability.when}."
    )
    problems = label_problems(availability.outcome(), asked)
    assert any("asks to book one" in p for p in problems), problems


def test_the_gate_checks_a_label_not_a_scenario() -> None:
    """The same gate must be able to audit a hand-labeled row."""
    gold = RestaurantOutcome(
        restaurant="Uchi",
        intent="reservation",
        status="confirmed",
        party_size=4,
        when="Friday at 7pm",
        under_name="Maya",
        evidence="x",
        booked=True,
    )
    good = "uchi took the reservation for four friday at 7pm under maya"
    assert label_problems(gold, good) == ()


def test_negated_requests_are_not_read_as_requests() -> None:
    """'I did not make a reservation' contains a request phrase and denies it."""
    assert _requests_booking("i did not make a reservation") is False
    assert _requests_booking("we never asked to book a table") is False
    assert _requests_booking("didn't book a table") is False
    assert _requests_booking("we called to book a table") is True
    assert _requests_booking("asked them to hold a table") is True


def test_denied_confirmation_counts_as_unclear_language() -> None:
    """Word order varies; the meaning does not."""
    unknown = next(s for s in _SCENARIOS if s.status == "unknown")
    text = (
        f"{unknown.restaurant}: staff never explicitly confirmed whether the "
        f"reservation was made."
    )
    assert not any(
        "does not sound unclear" in p for p in label_problems(unknown.outcome(), text)
    )


def test_free_counts_as_availability_language() -> None:
    """A host saying 'free' is not saying 'open', but means the same."""
    confirmed = next(s for s in _SCENARIOS if s.status == "confirmed" and not s.booked)
    text = (
        f"{confirmed.restaurant}: the host said a table for "
        f"{confirmed.party_size} is free {confirmed.when}."
    )
    assert not any(
        "no availability language" in p
        for p in label_problems(confirmed.outcome(), text)
    )


def test_hand_labeled_fixture_supports_its_own_labels() -> None:
    """The user's own gold must pass the gate, or the gate is wrong.

    Every other test here uses generated text. This one is the fixture a human
    wrote, and it is the reason the intent check was cut down to its
    contradiction case.
    """
    rows = load_examples(Path("examples/bucko-restaurant/examples.jsonl"))
    bad = [
        (row.outcome.restaurant, label_problems(row.outcome, row.text))
        for row in rows
        if label_problems(row.outcome, row.text)
    ]
    assert bad == [], bad


def test_generated_corpus_supports_its_own_labels() -> None:
    rows = load_examples(Path("examples/bucko-restaurant/generated.jsonl"))
    bad = [
        row.outcome.restaurant for row in rows if label_problems(row.outcome, row.text)
    ]
    assert bad == [], bad
