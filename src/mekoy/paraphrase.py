"""Paraphrased eval cases: gold by construction, surface text by a local model.

PLAN §41.19 asks for generated edge cases. The failure mode of model-generated
evals is that the model writes both the input and the label, so a mistake in the
label is invisible. This inverts that: the *scenario* (and therefore the gold) is
declared in code, and the model only writes the messy surface text.

That keeps the label trustworthy while removing the templating. Every rendered
row must still pass `accepts()`, which checks the transcript actually says the
things the gold claims and does not say the things the gold denies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from mekoy.errors import CompileError, ModelUnreachableError
from mekoy.evalgen import RESTAURANTS, Scenario
from mekoy.outcome import RestaurantOutcome
from mekoy.runtime import Completer
from mekoy.verify import NUMBER_WORDS, claims_booking, unnegated_match

__all__ = [
    "Rendered",
    "accepts",
    "claims_booking",
    "generate",
    "label_problems",
    "mentions_party_size",
    "render_prompt",
    "requests_booking",
    "write_corpus",
]

_SYSTEM = """You write realistic, messy phone-call notes for a family assistant.

You are given the FACTS of one restaurant call. Write the transcript exactly as a
voicemail-to-text or a hurried staff summary would render it: clipped, lowercase,
occasionally starting with the restaurant name, no headings, no JSON, no
quotation of these instructions.

Hard rules:
- Include ONLY the facts given. Invent nothing.
- If booked is false, you must NOT say a reservation was taken, confirmed, held,
  or put on the books.
- If status is unavailable, the staff did not talk to us.
- If status is unknown, the call started but the outcome is unclear.
- 1 to 3 sentences. Plain text only.

Reply with the transcript and nothing else."""

#: A booking request: a verb aimed at a reservation or a table.
_REQUEST_RE = re.compile(
    r"(?:"
    r"(?:make|made|making|book|books|booked|booking|reserve|reserves|reserved"
    r"|request|requested|hold|holding|secure|secured|get|got|getting"
    r"|trying to|try to|tried to|wanted to|want to|like to|hoping to)\s+"
    r"(?:a|the|our|your|us|me)?\s*(?:reservation|table|booking|spot)"
    r"|reservation\s+(?:for|under)\b"
    r")",
)

#: Phrases that mean staff never spoke to us.
_NO_CONTACT = (
    "did not answer",
    "didn't answer",
    "no answer",
    "nobody answered",
    "no one answered",
    "voicemail",
    "went to voicemail",
    "hung up",
    "unreachable",
)

#: Phrases that mean we could not tell the outcome.
_UNCLEAR = (
    "not sure",
    "unsure",
    "unclear",
    "not clear",
    "dropped",
    "cut off",
    "disconnected",
    "couldn't tell",
    "could not tell",
    "may have",
    "might have",
    "no confirmation",
    "never confirmed",
    "not confirm",
    "never confirm",
    "not explicitly confirmed",
)

#: "never/not (explicitly) confirmed ..." — a denial of confirmation, whatever
#: word order the writer chose. Found by auditing the hand-labeled fixture.
_NOT_CONFIRMED = re.compile(r"(?:never|not|didn't|did not)\s+(?:explicitly\s+)?confirm")

#: Phrases that mean nothing was available.
_NOTHING = (
    "fully booked",
    "completely booked",
    "no table",
    "nothing available",
    "not available",
    "no availability",
    "no room",
    "booked solid",
    "no openings",
)


@dataclass(frozen=True, slots=True)
class Rendered:
    """One accepted paraphrase and the scenario it was rendered from."""

    scenario: Scenario
    text: str

    def outcome(self) -> RestaurantOutcome:
        """The gold label, unchanged by rendering."""
        return self.scenario.outcome()


def render_prompt(scenario: Scenario) -> str:
    """The facts block handed to the model.

    Gold is implied by this block, not produced by the model.
    """
    lines = [
        f"restaurant: {scenario.restaurant}",
        f"intent: {scenario.intent}",
        f"status: {scenario.status}",
        f"party_size: {scenario.party_size}",
        f"when: {scenario.when}",
        f"under_name: {scenario.under_name}",
        f"booked: {str(scenario.booked).lower()}",
    ]
    return "\n".join(lines)


def _has(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def requests_booking(low: str) -> bool:
    """True when the transcript asks to book, rather than denying that we did."""
    return unnegated_match(_REQUEST_RE, low)


def mentions_party_size(low: str, size: int) -> bool:
    """Accept a digit or the spoken form: "a table for two" counts as 2."""
    tokens = NUMBER_WORDS.get(size, (str(size),))
    return any(re.search(rf"\b{token}\b", low) for token in tokens)


def accepts(scenario: Scenario, raw: str) -> tuple[str, ...]:
    """Complaints about a rendered transcript. Empty tuple means usable.

    The bar is **contradiction**, not completeness, and the asymmetry is
    deliberate. A label is corrupted when the transcript asserts the opposite of
    the gold: a booking that did not happen, a flat refusal rendered as a maybe,
    or a fact the gold names being absent entirely. A transcript that merely
    leaves a booking implicit is a *harder* row, not a broken one, so a missing
    positive claim is accepted and a false one is rejected.
    """
    return label_problems(scenario.outcome(), raw)


def label_problems(gold: RestaurantOutcome, raw: str) -> tuple[str, ...]:
    """Why a transcript fails to support its own label, if it does not.

    Takes a label rather than a scenario, so the same gate audits any corpus,
    including a hand-labeled one. If the transcript never says whether we were
    asking *if* a table exists or asking *to book* one, the intent label is
    unknowable and scoring anything against it is unfair.
    """
    text = " ".join(raw.split())
    if not text:
        return ("empty transcript",)
    low = text.casefold()
    return (
        _identity_problems(gold, low)
        + _booking_problems(gold, low)
        + _intent_problems(gold, low)
        + _status_problems(gold, low)
        + _field_problems(gold, low)
    )


def _identity_problems(gold: RestaurantOutcome, low: str) -> tuple[str, ...]:
    """The transcript must name the right restaurant and only that one.

    Matches on the brand token, not the whole name. Gold normalizes a short name
    to a full one — "Home Slice" is labeled `Home Slice Pizza`, and
    "Torchy's South Congress" is labeled `Torchy's Tacos` — so requiring the full
    string would flag correct hand-labeled rows. The brand token still catches a
    genuinely different restaurant.
    """
    problems: list[str] = []
    brand = gold.restaurant.split()[0].casefold().strip("'s")
    if brand and brand not in low:
        problems.append("restaurant name missing")
    for other in RESTAURANTS:
        if other == gold.restaurant:
            continue
        other_brand = other.split()[0].casefold().strip("'s")
        if other_brand and other_brand != brand and other_brand in low:
            problems.append(f"invents a second restaurant: {other}")
            break
    return tuple(problems)


def _booking_problems(gold: RestaurantOutcome, low: str) -> tuple[str, ...]:
    """Only one direction is fatal: claiming a booking that did not happen.

    A booked row rendered without an explicit claim is still a valid hard case, so
    it is allowed through. The reverse is a contradiction and is rejected.
    """
    if not gold.booked and claims_booking(low):
        return ("booked is false but the transcript claims it was taken",)
    return ()


def _intent_problems(gold: RestaurantOutcome, low: str) -> tuple[str, ...]:
    """Only the contradiction is fatal: an availability label on a booking call.

    A stricter rule was tried and thrown away. Requiring an explicit booking
    request for `reservation`, or explicit inquiry language for `availability`,
    rejected legitimate hand-labeled rows — "host said a table for two is free
    tonight at 8, they did not book it" carries an availability label with no
    inquiry verb, and "did not answer. Voicemail. Party of 6" carries a
    reservation label with no request verb. Real calls imply the request. The
    check that survives is the one that is actually a contradiction.
    """
    if gold.intent == "availability" and requests_booking(low):
        return ("availability label but the transcript asks to book one",)
    return ()


def _status_problems(gold: RestaurantOutcome, low: str) -> tuple[str, ...]:
    """unavailable, unknown, and confirmed must not be flattened into each other."""
    problems: list[str] = []
    if gold.status == "unavailable":
        if not _has(low, _NO_CONTACT) and not _has(low, _NOTHING):
            problems.append(
                "unavailable but no contact failure or full-booked language"
            )
        if _has(low, _UNCLEAR):
            problems.append("unavailable but the transcript sounds merely unclear")
    if gold.status == "unknown":
        if not _has(low, _UNCLEAR) and _NOT_CONFIRMED.search(low) is None:
            problems.append("unknown but the transcript does not sound unclear")
        if _has(low, _NO_CONTACT):
            problems.append("unknown but the transcript says nobody spoke")
    if gold.status == "confirmed" and not gold.booked:
        if _has(low, _NOTHING):
            problems.append("availability confirmed but the transcript says full")
        elif not _has(low, ("open", "available", "free")):
            problems.append("availability confirmed but no availability language")
    return tuple(problems)


def _field_problems(gold: RestaurantOutcome, low: str) -> tuple[str, ...]:
    """Facts the gold asserts must appear in the surface text."""
    problems: list[str] = []
    if gold.party_size is not None and not mentions_party_size(low, gold.party_size):
        problems.append(f"party_size {gold.party_size} missing from the transcript")
    if gold.when is not None:
        tokens = re.findall(r"[a-z0-9:]+", gold.when.casefold())
        if tokens and not any(token in low for token in tokens):
            problems.append(f"when {gold.when!r} missing from the transcript")
    if gold.under_name is not None and gold.under_name.casefold() not in low:
        problems.append(f"under_name {gold.under_name} missing")
    return tuple(problems)


def generate(
    completer: Completer,
    scenarios: tuple[Scenario, ...],
    *,
    attempts: int = 2,
) -> tuple[tuple[Rendered, ...], tuple[tuple[Scenario, tuple[str, ...]], ...]]:
    """Paraphrase every scenario. Returns (accepted, rejected-with-reasons).

    A model error costs one attempt, not the run: rendering a few hundred cases
    on a local server outlives a single request's timeout.
    """
    ok: list[Rendered] = []
    bad: list[tuple[Scenario, tuple[str, ...]]] = []
    for scenario in scenarios:
        prompt = render_prompt(scenario)
        complaints: tuple[str, ...] = ("not attempted",)
        for _ in range(max(1, attempts)):
            try:
                # Free text, not JSON: constrained decode would force the model
                # into json_object mode and it would answer with a schema.
                raw = completer.complete(
                    system=_SYSTEM,
                    user=prompt,
                    constrained=False,
                    temperature=0.9,
                )
            except (ModelUnreachableError, CompileError) as exc:
                complaints = (f"model error: {exc}",)
                continue
            complaints = accepts(scenario, raw)
            if not complaints:
                ok.append(Rendered(scenario=scenario, text=" ".join(raw.split())))
                break
        else:
            bad.append((scenario, complaints))
    return tuple(ok), tuple(bad)


def write_corpus(path: Path, rows: tuple[Rendered, ...]) -> Path:
    """Write accepted rows in the same shape as the hand-labeled fixture."""
    lines = [
        json.dumps(
            {"text": row.text, "outcome": row.outcome().model_dump()},
            ensure_ascii=False,
        )
        for row in rows
    ]
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
