"""Task classes: what makes one job different from another.

The compiler is a search over model + harness + eval + runtime. None of that is
specific to restaurant calls; what varies per job is the schema, the instructions,
the fields worth scoring, and the deterministic checks that constrain them.

`RESTAURANT` is the shipped default, so every existing call site keeps working
without naming a task. `RECEIPT` is the second class, and adding it is the test of
whether the abstraction is real.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from pydantic import BaseModel

from mekoy.banking77 import (
    Intent,
    banking_gate,
    banking_prompt,
    score_intent,
)
from mekoy.outcome import RestaurantOutcome
from mekoy.receipt import RECEIPT_PROMPT, Receipt, numeric_problems, score_receipt
from mekoy.score import score_fields
from mekoy.verify import policy_reasons

__all__ = [
    "ALL_TASKS",
    "BANKING77",
    "RECEIPT",
    "RESTAURANT",
    "Scored",
    "Task",
    "task_by_name",
    "task_for_path",
]

#: Restaurant instructions. The task carries its own prompt so the harness does
#: not have to know which job it is running.
RESTAURANT_PROMPT = """You extract a restaurant-call result for a family assistant.
Return ONLY a JSON object with keys:
restaurant (string),
intent (availability or reservation),
status (confirmed, unknown, or unavailable),
party_size (integer or null),
when (string or null),
under_name (string or null),
evidence (one sentence staff actually said),
booked (boolean).
Rules:
- booked is true only if staff explicitly confirmed a reservation.
- availability is never booked.
- unknown is never booked.
- confirmed reservation must set booked true.
"""


#: A terser instruction variant the search may try.
_TERSE = """Extract one restaurant-call result as JSON:
restaurant, intent (availability|reservation),
status (confirmed|unknown|unavailable),
party_size (int|null), when (str|null), under_name (str|null),
evidence (str), booked (bool).
booked=true ONLY when staff explicitly took a reservation. A number after "for" may be
a time, not a party size; put it in `when` if the sentence is about when. Never set
booked for availability or unknown. A confirmed reservation must set booked true."""

#: A stricter variant that names the confusions the gate actually catches.
_STRICT = (
    RESTAURANT_PROMPT
    + """
Work carefully:
- "nobody answered", "voicemail", "did not answer" -> status unavailable.
- a call that started but left the outcome unclear -> status unknown.
- These are different. Prefer unavailable when staff never spoke.
- party_size is the number of people asked for, and it counts even when the call
  was not answered. Use null only when no party size was ever mentioned.
- intent is availability when the question was whether a table exists, and
  reservation when the question was to book one. "They did not take a
  reservation" is still availability."""
)


class Scored(Protocol):
    """What the search needs from any per-document score."""

    @property
    def quality(self) -> float:
        """Primary accuracy, with free text normalized."""
        ...

    @property
    def strict_quality(self) -> float:
        """Raw-string accuracy, reported alongside."""
        ...

    @property
    def schema_ok(self) -> bool:
        """Whether the document cleared the gate."""
        ...


@dataclass(frozen=True, slots=True)
class Task:
    """One job: a schema, instructions, scored fields, and hard checks."""

    name: str
    model: type[BaseModel]
    prompt: str
    scored_fields: tuple[str, ...]
    #: Fields whose wording is the model's choice.
    phrase_fields: frozenset[str]
    #: Fields whose wording is a time; compared without formatting.
    time_fields: frozenset[str]
    #: Deterministic reasons an outcome is unacceptable. Empty means sound.
    gate: Callable[[object], tuple[str, ...]]
    #: Field-level scoring for this schema.
    score: Callable[..., Scored]
    #: Extra instruction variants the search may try, keyed by name.
    prompt_variants: dict[str, str] = field(default_factory=dict)
    #: Retry counts worth trying. An arithmetic gate usually needs more than one
    #: repair pass; a policy violation is normally fixed in one.
    retry_ladder: tuple[int, ...] = (0, 1)

    def score_pair(self, gold: object, pred: object) -> Scored:
        """Score one prediction with this task's own scorer."""
        return self.score(gold=gold, pred=pred)


def _score_restaurant(*, gold: object, pred: object) -> Scored:
    return score_fields(
        gold=gold,
        pred=pred,
        scored_fields=RESTAURANT.scored_fields,
        phrase_fields=RESTAURANT.phrase_fields,
        time_fields=RESTAURANT.time_fields,
    )


def _score_receipt(*, gold: object, pred: object) -> Scored:
    return score_receipt(gold=cast("Receipt", gold), pred=cast("Receipt", pred))


def _score_banking(*, gold: object, pred: object) -> Scored:
    return score_intent(gold=cast("Intent", gold), pred=cast("Intent", pred))


def _restaurant_gate(outcome: object) -> tuple[str, ...]:
    return policy_reasons(cast("RestaurantOutcome", outcome))


def _receipt_gate(receipt: object) -> tuple[str, ...]:
    return numeric_problems(cast("Receipt", receipt))


#: The shipped default. Existing call sites get this without naming it.
RESTAURANT = Task(
    name="restaurant",
    model=RestaurantOutcome,
    prompt=RESTAURANT_PROMPT,
    scored_fields=(
        "restaurant",
        "intent",
        "status",
        "party_size",
        "when",
        "under_name",
        "booked",
    ),
    phrase_fields=frozenset({"restaurant", "when", "under_name"}),
    time_fields=frozenset({"when"}),
    gate=_restaurant_gate,
    score=_score_restaurant,
    prompt_variants={
        "default": RESTAURANT_PROMPT,
        "terse": _TERSE,
        "strict": _STRICT,
    },
)

#: The second class. Arithmetic is the check, which is why it is the first proof
#: job: a model cannot bluff a total.
RECEIPT = Task(
    name="receipt",
    model=Receipt,
    prompt=RECEIPT_PROMPT,
    scored_fields=(
        "merchant",
        "date",
        "address",
        "currency",
        "subtotal",
        "tax",
        "total",
    ),
    phrase_fields=frozenset({"merchant", "address"}),
    time_fields=frozenset(),
    gate=_receipt_gate,
    score=_score_receipt,
    prompt_variants={"default": RECEIPT_PROMPT},
    retry_ladder=(0, 3),
)

#: Classification. One label from a closed set: no free text, no arithmetic, and
#: accuracy rather than field accuracy. The conservative retry ladder reflects
#: that a wrong label is not usually fixed by being told it was wrong.
BANKING77 = Task(
    name="banking77",
    model=Intent,
    prompt=banking_prompt(),
    scored_fields=("label",),
    phrase_fields=frozenset(),
    time_fields=frozenset(),
    gate=banking_gate,
    score=_score_banking,
    prompt_variants={"default": banking_prompt()},
    retry_ladder=(0, 1),
)


#: Every task class the compiler knows. Adding one here is what makes it
#: addressable by name from stored rows.
ALL_TASKS: tuple[Task, ...] = (RESTAURANT, RECEIPT, BANKING77)

_BY_NAME: dict[str, Task] = {task.name: task for task in ALL_TASKS}


def task_by_name(name: str) -> Task:
    """Resolve a stored task name, defaulting to the shipped one."""
    return _BY_NAME.get(name, RESTAURANT)


def task_for_row(row: dict[str, object]) -> Task:
    """Detect the task from a single row's label key.

    `outcome` is the restaurant label, `receipt` and `label` are the others. Kept
    beside `task_for_path` so the API and the file loader cannot disagree about
    which job a row describes.
    """
    if "receipt" in row:
        return RECEIPT
    if "label" in row:
        return BANKING77
    return RESTAURANT


def task_for_path(path: Path) -> Task:
    """Detect the task from the shape of the fixture's first row.

    Receipts carry a `receipt` key; restaurant rows carry `outcome`. Detection
    lives with the task definitions so the CLI, the loader, and `check-eval` all
    agree about which job a file describes.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                return RESTAURANT
            if not isinstance(row, dict):
                return RESTAURANT
            return task_for_row(row)
    return RESTAURANT
