"""Holdout scoring.

Fields come in two kinds, and they are not scored the same way:

- **Closed fields** (`intent`, `status`, `party_size`, `booked`) compare exactly.
- **Free-text fields** (`restaurant`, `when`, `under_name`) compare on normalized
  tokens, so "9am Saturday" and "Saturday 9am" are the same answer. `when` goes
  further and ignores time formatting and filler, so "Sunday at 1:00 PM" and
  "Sunday at 1pm" agree. PLAN §16.3: free text is judged, never gated on exact
  match.

`quality` is the primary number (normalized). `strict_quality` keeps raw string
comparison so the difference stays visible instead of hidden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mekoy.outcome import RestaurantOutcome

__all__ = [
    "PHRASE_FIELDS",
    "SCORED_FIELDS",
    "ExampleScore",
    "fields_match",
    "score_outcome",
]

_FIELDS = 7

#: The seven scored fields. `evidence` is a quote, not a lookup, and is not scored.
SCORED_FIELDS = (
    "restaurant",
    "intent",
    "status",
    "party_size",
    "when",
    "under_name",
    "booked",
)

#: Fields whose wording is a time; compared without formatting.
TIME_FIELDS = frozenset({"when"})

#: Fields whose wording is the model's choice, not a lookup.
PHRASE_FIELDS = frozenset({"restaurant", "when", "under_name"})

#: Words that say nothing about which time was meant.
_FILLER = frozenset({"at", "on", "for", "the", "a"})
#: Meridiem markers. Compared only when both sides state one.
_MERIDIEM = frozenset({"am", "pm"})
#: A time glued to its meridiem: "1pm", "10:30am".
_GLUED = re.compile(r"^(\d+(?::\d+)?)(am|pm)$")
_TOKEN = re.compile(r"[a-z0-9:]+")


@dataclass(frozen=True, slots=True)
class ExampleScore:
    """Per-document extraction quality."""

    schema_ok: bool
    field_hits: int
    field_total: int
    line_f1: float
    strict_hits: int = 0

    @property
    def quality(self) -> float:
        """Field accuracy with free-text phrasing normalized."""
        if self.field_total == 0:
            return 0.0
        return self.field_hits / self.field_total

    @property
    def strict_quality(self) -> float:
        """Field accuracy with raw string equality. Reported, not selected on."""
        if self.field_total == 0:
            return 0.0
        return self.strict_hits / self.field_total


def score_outcome(*, gold: RestaurantOutcome, pred: RestaurantOutcome) -> ExampleScore:
    """Compare one restaurant prediction to gold, both ways."""
    return score_fields(
        gold=gold,
        pred=pred,
        scored_fields=SCORED_FIELDS,
        phrase_fields=PHRASE_FIELDS,
        time_fields=TIME_FIELDS,
    )


def _field(record: object, name: str) -> object:
    """One field of a record, whether it is an object or a plain dict.

    A job defined in data carries dict labels, because that is what JSON gives back
    and what a user's examples file contains. Reading only attributes would score
    every field of such a job as missing, so both shapes are read here.
    """
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def score_fields(
    *,
    gold: object,
    pred: object,
    scored_fields: tuple[str, ...],
    phrase_fields: frozenset[str],
    time_fields: frozenset[str],
) -> ExampleScore:
    """Score any pair of models on a named field list.

    Task-agnostic on purpose: `tasks.Task` supplies the field lists, so a second
    schema gets the same comparison rules without a second scorer.
    """
    hits = 0
    strict = 0
    for name in scored_fields:
        want, got = _field(gold, name), _field(pred, name)
        if _exact(want, got):
            strict += 1
        if _field_matches(name, want, got, phrase_fields, time_fields):
            hits += 1
    booked_same = _field(gold, "booked") == _field(pred, "booked")
    return ExampleScore(
        schema_ok=True,
        field_hits=hits,
        field_total=len(scored_fields),
        # booked is the safety-critical field where a schema has one; elsewhere
        # this secondary signal is 1.0 and unused.
        line_f1=1.0 if booked_same else 0.0,
        strict_hits=strict,
    )


def fields_match(field: str, gold: object, pred: object) -> bool:
    """Whether two field values agree under the restaurant field rules."""
    return _field_matches(field, gold, pred, PHRASE_FIELDS, TIME_FIELDS)


def _field_matches(
    field: str,
    gold: object,
    pred: object,
    phrase_fields: frozenset[str],
    time_fields: frozenset[str],
) -> bool:
    if field in time_fields:
        return _when_equal(gold, pred)
    if field in phrase_fields:
        return _phrase(gold) == _phrase(pred)
    return _exact(gold, pred)


def _exact(gold: object, pred: object) -> bool:
    if isinstance(gold, str) and isinstance(pred, str):
        return gold.strip().casefold() == pred.strip().casefold()
    return gold == pred


def _phrase(value: object) -> str:
    """Casefold and sort tokens: 'Saturday 9am' == '9am Saturday'."""
    if not isinstance(value, str):
        return "" if value is None else str(value)
    return " ".join(sorted(value.casefold().split()))


def _when_equal(gold: object, pred: object) -> bool:
    """Compare two time phrases without punishing formatting.

    A meridiem marker is compared only when both sides give one. If the gold says
    "10:30" and the model says "10:30 AM", the gold was under-specified, not
    contradicted. If both say AM or PM and they disagree, that is a real miss.
    """
    left, right = _when_tokens(gold), _when_tokens(pred)
    if left == right:
        return True
    if (left & _MERIDIEM) and (right & _MERIDIEM):
        return False
    return (left - _MERIDIEM) == (right - _MERIDIEM)


def _when_tokens(value: object) -> frozenset[str]:
    """Tokens that say which time was meant, ignoring formatting.

    A time is a loose phrase, so `"Sunday at 1:00 PM"`, `"Sunday at 1pm"`, and
    `"1pm Sunday"` are one answer. Trailing `:00` is dropped, filler words are
    dropped, and order does not matter. What survives distinguishes 1pm from 2pm
    and Saturday from Sunday, which is the only distinction worth scoring.
    """
    if not isinstance(value, str):
        return frozenset() if value is None else frozenset(_TOKEN.findall(str(value)))
    out: set[str] = set()
    for raw in _TOKEN.findall(value.casefold()):
        if raw in _FILLER:
            continue
        glued = _GLUED.match(raw)
        token = glued.group(1) if glued else raw
        if glued:
            out.add(glued.group(2))
        stripped = token.removesuffix(":00")
        if stripped:
            out.add(stripped)
    return frozenset(out)
