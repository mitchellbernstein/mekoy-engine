"""Deterministic edge-case generator for the Bucko restaurant eval.

PLAN §16.4 asks for generated cases: malformed input, missing fields, adversarial
cases. This builds them from a scenario table where the gold is known *by
construction* — the transcript and its label are rendered from the same values,
so a generated row cannot silently mislabel itself.

Nothing here is a model's guess. Every row is auditable by reading the table.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mekoy.errors import CompileError
from mekoy.outcome import RestaurantOutcome

Intent = Literal["availability", "reservation"]
Status = Literal["confirmed", "unknown", "unavailable"]

RESTAURANTS = (
    "North Loop Bistro",
    "Franklin BBQ",
    "Home Slice Pizza",
    "Torchy's Tacos",
    "Uchi",
    "Kerbey Lane Cafe",
    "Matt's El Rancho",
    "Loro",
    "P. Terry's",
    "Snooze AM",
    "Ramen Tatsu-Ya",
    "Juan in a Million",
)

NAMES = ("Mitchell", "Maya", "Sam", "Bernstein", "Priya", "Dev", "Alex")
PARTIES = (2, 3, 4, 5, 6, 8)
TIMES = (
    "tonight at 8",
    "tomorrow at 6:30",
    "Friday at 7:00 PM",
    "Saturday brunch 10:30",
    "Sunday at 1pm",
    "next Friday at 7:00 PM",
)


@dataclass(frozen=True, slots=True)
class Scenario:
    """One transcript and the label it was rendered from."""

    text: str
    restaurant: str
    intent: Intent
    status: Status
    party_size: int | None
    when: str | None
    under_name: str | None
    booked: bool
    evidence: str

    def outcome(self) -> RestaurantOutcome:
        """The gold label. `booked` is asserted against the policy gate."""
        return RestaurantOutcome(
            restaurant=self.restaurant,
            intent=self.intent,
            status=self.status,
            party_size=self.party_size,
            when=self.when,
            under_name=self.under_name,
            evidence=self.evidence,
            booked=self.booked,
        )


def _confirmed_reservation(
    restaurant: str, party: int, when: str, name: str
) -> Scenario:
    text = f"{restaurant} host confirmed a table for {party} {when} under {name}."
    return Scenario(
        text=text,
        restaurant=restaurant,
        intent="reservation",
        status="confirmed",
        party_size=party,
        when=when,
        under_name=name,
        booked=True,
        evidence=f"Host confirmed a table for {party} {when} under {name}.",
    )


def _availability_open(restaurant: str, party: int, when: str) -> Scenario:
    text = (
        f"{restaurant}: staff said a table for {party} is open {when}, "
        "but they did not take a reservation."
    )
    return Scenario(
        text=text,
        restaurant=restaurant,
        intent="availability",
        status="confirmed",
        party_size=party,
        when=when,
        under_name=None,
        booked=False,
        evidence=f"A table for {party} is open {when}, but they did not take it.",
    )


def _availability_closed(restaurant: str, party: int, when: str) -> Scenario:
    text = f"{restaurant} is fully booked {when}; no table for {party}."
    return Scenario(
        text=text,
        restaurant=restaurant,
        intent="availability",
        status="unavailable",
        party_size=party,
        when=when,
        under_name=None,
        booked=False,
        evidence=f"Fully booked {when}.",
    )


def _reservation_unknown(restaurant: str) -> Scenario:
    text = (
        f"{restaurant}: the call dropped mid-sentence. They may have taken the "
        "reservation, we cannot tell."
    )
    return Scenario(
        text=text,
        restaurant=restaurant,
        intent="reservation",
        status="unknown",
        party_size=None,
        when=None,
        under_name=None,
        booked=False,
        evidence="The call dropped mid-sentence.",
    )


def _no_answer(restaurant: str, party: int, when: str) -> Scenario:
    text = (
        f"{restaurant} did not pick up after four rings. Voicemail. "
        f"Was trying for {party} {when}."
    )
    return Scenario(
        text=text,
        restaurant=restaurant,
        intent="reservation",
        status="unavailable",
        party_size=party,
        when=when,
        under_name=None,
        booked=False,
        evidence="Did not pick up after four rings. Voicemail.",
    )


def _time_trap(restaurant: str, when: str) -> Scenario:
    """The bare number is a time. A model that reads it as a party size is wrong."""
    text = (
        f"{restaurant}: connection broke. Not sure whether they wrote us down "
        f"for {when}."
    )
    return Scenario(
        text=text,
        restaurant=restaurant,
        intent="reservation",
        status="unknown",
        party_size=None,
        when=when,
        under_name=None,
        booked=False,
        evidence=f"Not sure whether they wrote us down for {when}.",
    )


def _hold_without_confirmation(restaurant: str, party: int, name: str) -> Scenario:
    """On a waitlist is not a reservation."""
    text = (
        f"{restaurant}: we are on the waitlist for {party} under {name}. "
        "They did not confirm a reservation."
    )
    return Scenario(
        text=text,
        restaurant=restaurant,
        intent="reservation",
        status="unknown",
        party_size=party,
        when=None,
        under_name=name,
        booked=False,
        evidence="We are on the waitlist. They did not confirm a reservation.",
    )


def _confirmed_without_name(restaurant: str, party: int, when: str) -> Scenario:
    text = f"{restaurant} took the reservation for {party} {when}. No name given."
    return Scenario(
        text=text,
        restaurant=restaurant,
        intent="reservation",
        status="confirmed",
        party_size=party,
        when=when,
        under_name=None,
        booked=True,
        evidence=f"Took the reservation for {party} {when}.",
    )


def build_scenarios() -> tuple[Scenario, ...]:
    """Every generated case, deterministic order. No randomness."""
    rows: list[Scenario] = []
    for i, restaurant in enumerate(RESTAURANTS):
        party = PARTIES[i % len(PARTIES)]
        when = TIMES[i % len(TIMES)]
        name = NAMES[i % len(NAMES)]
        rows.append(_confirmed_reservation(restaurant, party, when, name))
        rows.append(_availability_open(restaurant, party, when))
        rows.append(_no_answer(restaurant, party, when))
        rows.append(_confirmed_without_name(restaurant, party, when))
        rows.append(_reservation_unknown(restaurant))
        rows.append(_hold_without_confirmation(restaurant, party, name))
    # Adversarial time-vs-party-size traps, one per restaurant.
    rows.extend(_time_trap(r, TIMES[i % len(TIMES)]) for i, r in enumerate(RESTAURANTS))
    # Availability-closed cases, to keep `unavailable` from being reservation-only.
    rows.extend(
        _availability_closed(r, PARTIES[i % len(PARTIES)], TIMES[i % len(TIMES)])
        for i, r in enumerate(RESTAURANTS)
    )
    return tuple(rows)


def _family(family_index: int, i: int) -> Scenario:
    """One scenario for family `family_index`, with parameters rotated by `i`.

    Rotation matters: the same restaurant paired with the same party size across
    every family would let a model win by memorising the pairing.
    """
    restaurant = RESTAURANTS[i % len(RESTAURANTS)]
    party = PARTIES[(i * 3 + family_index) % len(PARTIES)]
    when = TIMES[(i * 5 + family_index) % len(TIMES)]
    name = NAMES[(i * 2 + family_index) % len(NAMES)]
    builders: tuple[Callable[[], Scenario], ...] = (
        lambda: _confirmed_reservation(restaurant, party, when, name),
        lambda: _availability_open(restaurant, party, when),
        lambda: _no_answer(restaurant, party, when),
        lambda: _confirmed_without_name(restaurant, party, when),
        lambda: _reservation_unknown(restaurant),
        lambda: _hold_without_confirmation(restaurant, party, name),
        lambda: _time_trap(restaurant, when),
        lambda: _availability_closed(restaurant, party, when),
    )
    return builders[family_index % len(builders)]()


def expand_scenarios(*, target: int = 240) -> tuple[Scenario, ...]:
    """A larger corpus by rotating parameters within each family.

    Deterministic, and every family keeps its share, so the label distribution of
    the hand-written table is preserved as the corpus grows. The audit still has
    to pass: growth must not introduce a booked-rule violation.
    """
    if target <= 0:
        msg = "target must be positive"
        raise CompileError(message=msg)
    rows = [_family(i % 8, i // 8) for i in range(target)]
    return tuple(rows)


def audit(rows: tuple[Scenario, ...]) -> tuple[str, ...]:
    """Re-check each row against the policy rules. Returns complaints, empty if clean.

    Generated data is only usable if its labels obey the same gate the model is
    held to. If this returns anything, the row set is not trustworthy.
    """
    problems: list[str] = []
    for i, row in enumerate(rows, start=1):
        gold = row.outcome()
        if gold.intent == "availability" and gold.booked:
            problems.append(f"row {i}: availability cannot be booked")
        if gold.status == "unknown" and gold.booked:
            problems.append(f"row {i}: unknown cannot be booked")
        if gold.booked and gold.status != "confirmed":
            problems.append(f"row {i}: booked requires confirmed")
        if gold.booked and gold.intent != "reservation":
            problems.append(f"row {i}: booked requires reservation intent")
        if not gold.evidence.strip():
            problems.append(f"row {i}: empty evidence")
        if gold.restaurant not in row.text:
            problems.append(f"row {i}: restaurant not present in the transcript")
    return tuple(problems)


def write_examples(path: Path, rows: tuple[Scenario, ...]) -> Path:
    """Write examples.jsonl in the same shape as the hand-labeled fixture."""
    lines = [
        json.dumps(
            {"text": row.text, "outcome": row.outcome().model_dump()},
            ensure_ascii=False,
        )
        for row in rows
    ]
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    """Regenerate the generated eval fixture."""
    rows = build_scenarios()
    problems = audit(rows)
    if problems:
        msg = "generated labels violate the gate:\n" + "\n".join(problems)
        raise SystemExit(msg)
    out = write_examples(Path("examples/bucko-restaurant/generated.jsonl"), rows)
    booked = sum(1 for r in rows if r.booked)
    print(f"wrote {out} ({len(rows)} rows, {booked} booked, audit clean)")


if __name__ == "__main__":
    main()
