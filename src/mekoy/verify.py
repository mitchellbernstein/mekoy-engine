"""Deterministic Bucko outcome checks. Search never runs without these.

The gate is the last line of defence on the one field that can hurt someone: a
family assistant that reports a reservation that does not exist is worse than one
that reports nothing. So `booked` is checked against the *evidence text*, not just
against the other fields. A model can produce an internally consistent object
whose story the transcript contradicts — `intent=reservation`, `status=confirmed`,
`booked=true` — while its own `evidence` field quotes the restaurant saying they
did not book it. Field checks alone cannot see that.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import assert_never

from pydantic import BaseModel, ValidationError

from mekoy.outcome import RestaurantOutcome

#: Phrases that assert a reservation was actually taken.
_TOOK_IT_RE = re.compile(
    r"(?:"
    r"(?:took|reserved|secured|got)\s+(?:a|the|our|your)?\s*(?:reservation|table)"
    r"|booked\s+(?:it|us|you|a table|the table|a reservation)"
    r"|confirmed\s+(?:a|the|our|your)?\s*(?:reservation|table|booking)"
    r"|reservation\s+(?:is|was)?\s*(?:in|confirmed|set|booked)"
    r"|(?:we|you)(?:'re| are| were)?\s+(?:on the list|on the books|all set)"
    r"|we'?ll hold|holding a table|have you down"
    r")",
)

#: A denial that a booking happened, with room for adverbs: "never explicitly
#: confirmed", "did not book it", "no reservation was made".
_DENIES_RE = re.compile(
    r"(?:did\s+not|didn't|do\s+not|don't|never|not|no)\s+(?:\w+\s+){0,2}(?:book|booked)\b"
    r"|(?:did\s+not|didn't|do\s+not|don't|never|not|no)\s+(?:\w+\s+){0,2}"
    r"(?:take|taken|took|confirm|confirmed|make|made|have|held)\b"
    r"[^.]{0,40}?(?:reservation|table|hold|booking)"
    r"|(?:no|not\s+any)\s+(?:reservation|table|booking|hold)\b"
    r"|(?:reservation|table|booking)\s+(?:was|is|were)\s+not\s+\w+"
)

#: Evidence that nothing was available.
_NOTHING_AVAILABLE_RE = re.compile(
    r"fully\s+booked|completely\s+booked|booked\s+solid|no\s+table|no\s+tables"
    r"|nothing\s+available|not\s+available|no\s+availability|no\s+room"
    r"|no\s+openings|no\s+reservations?\s+available"
)

#: Evidence that a table could be had. Allows intervening words, because a host
#: says "a table for two is available", not "a table is available".
_AVAILABILITY_RE = re.compile(
    r"(?:table|tables|spot|seating|something|room)\b(?:\s+\w+){0,4}\s+"
    r"(?:is|are|was|were)?\s*(?:open|available|free)\b"
    r"|(?:is|are)\s+open\b"
    r"|available\s+for\b"
    r"|can\s+seat\b"
)

#: Spoken forms a party size may take. Public so the eval generators can
#: share one table instead of keeping their own copy.
NUMBER_WORDS: dict[int, tuple[str, ...]] = {
    1: ("1", "one"),
    2: ("2", "two"),
    3: ("3", "three"),
    4: ("4", "four"),
    5: ("5", "five"),
    6: ("6", "six"),
    7: ("7", "seven"),
    8: ("8", "eight"),
    9: ("9", "nine"),
    10: ("10", "ten"),
    11: ("11", "eleven"),
    12: ("12", "twelve"),
}

#: A phrase that names a party size. Anchored on the preposition so a time
#: ("at 7") is not read as a headcount.
_PARTY_RE = re.compile(
    r"(?:for|party\s+of|table\s+of|group\s+of|seating\s+for)\s+(\w+)"
)

#: Tokens that turn a claim into a denial. Checked in the window before a match.
_NEGATIONS = (
    "not",
    "n't",
    "never",
    "no ",
    "didn't",
    "did not",
    "unable",
    "failed to",
    "without",
)
_NEGATION_WINDOW = 30


def unnegated_match(pattern: re.Pattern[str], haystack: str) -> bool:
    """True when `pattern` matches somewhere it is not being denied.

    "did not confirm the reservation" contains the phrase and means the opposite.
    """
    for match in pattern.finditer(haystack):
        window = haystack[max(0, match.start() - _NEGATION_WINDOW) : match.start()]
        if not any(token in window for token in _NEGATIONS):
            return True
    return False


def claims_booking(text: str) -> bool:
    """True when the text asserts a reservation was taken."""
    return unnegated_match(_TOOK_IT_RE, text.casefold())


def denies_booking(text: str) -> bool:
    """True when the text says a reservation was not taken.

    Catches the failure the field checks cannot: `booked=true` whose own evidence
    quotes the restaurant refusing to book.
    """
    return _DENIES_RE.search(text.casefold()) is not None


@dataclass(frozen=True, slots=True)
class VerifyOk:
    """Parsed outcome that cleared the task's gate."""

    outcome: BaseModel


@dataclass(frozen=True, slots=True)
class VerifyFail:
    """Why the candidate is not a valid outcome."""

    reasons: tuple[str, ...]


type VerifyResult = VerifyOk | VerifyFail

#: Plain-English summary of the rules enforced by `policy_reasons`. Single-sourced
#: here, next to the enforcement, so the eval card cannot drift from the gate.
CHECKS_SUMMARY = (
    "JSON schema; restaurant and evidence non-empty; "
    "booked only on explicit staff confirmation; availability never booked; "
    "unknown never booked; confirmed reservation must set booked true; "
    "booked rejected when the evidence denies it was taken; "
    "status rejected when the evidence contradicts it; "
    "party_size rejected when the evidence names a different one"
)


def parse_and_verify(raw: str) -> VerifyResult:
    """Parse JSON into a RestaurantOutcome and check intent/booked rules."""
    return parse_and_gate(raw, model=RestaurantOutcome, gate=policy_reasons)


def parse_and_gate(
    raw: str,
    *,
    model: type[BaseModel],
    gate: Callable[[object], tuple[str, ...]],
) -> VerifyResult:
    """Parse into any task's schema, then apply that task's gate.

    The gate is a parameter rather than a hardcoded call, so a second task class
    gets the same parse-then-check loop without a second copy of it.
    """
    try:
        outcome = model.model_validate_json(raw)
    except ValidationError as exc:
        return VerifyFail(reasons=tuple(err["msg"] for err in exc.errors()))
    reasons = gate(outcome)
    if reasons:
        return VerifyFail(reasons=reasons)
    return VerifyOk(outcome=outcome)


def policy_reasons(outcome: RestaurantOutcome) -> tuple[str, ...]:
    """Every deterministic reason this outcome is unacceptable.

    Public because `tasks.RESTAURANT` carries it as the task's gate, so the same
    checks run whether a candidate is being filtered during search or a gold row
    is being audited.
    """
    reasons: list[str] = []
    if not outcome.restaurant.strip():
        reasons.append("restaurant is empty")
    if not outcome.evidence.strip():
        reasons.append("evidence is empty")
    if outcome.intent == "availability" and outcome.booked:
        reasons.append("availability cannot be booked")
    if outcome.status == "unknown" and outcome.booked:
        reasons.append("unknown cannot be booked")
    if outcome.booked and outcome.intent != "reservation":
        reasons.append("booked requires reservation intent")
    if outcome.booked and outcome.status != "confirmed":
        reasons.append("booked requires confirmed status")
    if outcome.booked and denies_booking(outcome.evidence):
        # The model can satisfy every field check and still contradict the call.
        reasons.append("booked is true but the evidence denies it was taken")
    reasons.extend(_status_evidence_reasons(outcome))
    reasons.extend(_party_size_reasons(outcome))
    if (
        outcome.intent == "reservation"
        and outcome.status == "confirmed"
        and not outcome.booked
    ):
        reasons.append("confirmed reservation must set booked true")
    return tuple(reasons)


def party_size_in(text: str) -> int | None:
    """The headcount a phrase names, if it names one."""
    token = text.strip().casefold()
    if token.isdigit():
        return int(token)
    for size, words in NUMBER_WORDS.items():
        if token in words:
            return size
    return None


def party_sizes_in(text: str) -> set[int]:
    """Every headcount named in the text, ignoring times and other numbers."""
    found: set[int] = set()
    for match in _PARTY_RE.finditer(text.casefold()):
        size = party_size_in(match.group(1))
        if size is not None:
            found.add(size)
    return found


def _party_size_reasons(outcome: RestaurantOutcome) -> tuple[str, ...]:
    """Reject a headcount the evidence contradicts.

    Only fires when the evidence names a headcount at all, and only when none of
    the headcounts it names matches. Evidence that never says a number is not a
    contradiction, it is silence.
    """
    if outcome.party_size is None:
        return ()
    named = party_sizes_in(outcome.evidence)
    if named and outcome.party_size not in named:
        return (
            (
                f"party_size is {outcome.party_size} but the evidence says "
                f"{sorted(named)}"
            ),
        )
    return ()


def _status_evidence_reasons(outcome: RestaurantOutcome) -> tuple[str, ...]:
    """Reject a status the evidence contradicts.

    The same hole as a hallucinated booking: `status` is checked against the other
    fields, so a model can report `confirmed` while its own evidence reads "they
    were fully booked". Only the text can catch it.
    """
    low = outcome.evidence.casefold()
    reasons: list[str] = []
    if outcome.status == "confirmed" and _NOTHING_AVAILABLE_RE.search(low):
        reasons.append("status is confirmed but the evidence says nothing was free")
    if outcome.status == "unavailable" and unnegated_match(_AVAILABILITY_RE, low):
        reasons.append("status is unavailable but the evidence offers a table")
    if outcome.status == "unknown" and claims_booking(low):
        reasons.append("status is unknown but the evidence says it was taken")
    return tuple(reasons)


def explain(result: VerifyResult) -> str:
    """Single-line reason for a retry prompt."""
    match result:
        case VerifyOk():
            return "ok"
        case VerifyFail(reasons=reasons):
            return "; ".join(reasons)
        case _ as unreachable:
            assert_never(unreachable)
