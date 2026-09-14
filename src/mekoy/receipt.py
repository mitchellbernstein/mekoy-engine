"""Frozen receipt schema, numeric checks, and scoring.

The second task class. `PLAN.md` §43 makes receipt extraction the first prototype
and §16.1 puts deterministic numeric checks at the bottom of the eval stack:
`qty * unit_price ≈ line_total`, `Σ line_total ≈ subtotal`, and
`subtotal + tax ≈ total`. Those identities are why a receipt is the right first
proof job — a model cannot bluff arithmetic.

This module is self-contained on purpose. The restaurant path is the shipped one;
this is the second class, and it must be able to prove itself without changing
that path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mekoy.errors import CompileError

#: Money tolerance, in major units.
#:
#: Half a cent was too tight. SROIE is Malaysian and its receipts round to 5 sen,
#: so `subtotal + tax == total` misses by two or three hundredths on correct
#: receipts — observed three times in 21 test rows. Five hundredths clears the
#: rounding without hiding a real mistake: the genuine errors in the same run were
#: 0.90 and 0.10, and both still fail. It is a judgement call, and a currency that
#: rounds coarser than 0.05 would need a larger one.
EPSILON = 0.05

#: Currencies print as ISO codes ("USD") or as symbols ("RM", "$", "€").
#: Demanding three capitals rejected the Malaysian Ringgit, which is two letters.
_CURRENCY_RE = re.compile(r"^[A-Za-z$€£¥₩฿₹]{1,5}$")

#: Fields compared as text rather than numbers.
TEXT_FIELDS = ("merchant", "date", "currency")
#: Fields compared numerically.
MONEY_FIELDS = ("subtotal", "tax", "total")

RECEIPT_PROMPT = """You extract a receipt from already-text OCR output.
Return ONLY a JSON object with keys:
merchant (string),
date (ISO 8601, YYYY-MM-DD),
currency (three-letter code),
subtotal (number),
tax (number),
total (number),
items (array of {desc, qty, unit_price, line_total}).
Rules:
- Copy numbers exactly as printed. Never recompute a total.
- If a line is unreadable, still emit it with your best reading of the amount.
"""


class LineItem(BaseModel):
    """One purchased line."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    desc: str
    qty: float = Field(gt=0)
    unit_price: float
    line_total: float


class Receipt(BaseModel):
    """Header plus line items. Amounts are major currency units.

    Every money field but `total` is optional, because real receipts are not
    uniform. In the CORD test split, 42 of 100 printed no subtotal at all, and
    service charges and discounts appear on top of tax. A schema that demanded the
    full set could not represent those rows, and a gate that demanded them would
    reject correct extractions for being incomplete rather than wrong.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    #: Optional: CORD's test split prints no store info on any of its 100 rows.
    merchant: str = ""
    date: str | None = None
    #: SROIE annotates an address on every receipt; CORD rarely does.
    address: str | None = None
    currency: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    service: float | None = None
    discount: float | None = None
    total: float | None = None
    items: tuple[LineItem, ...] = ()


class ReceiptExample(BaseModel):
    """One labeled receipt document."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    text: str
    receipt: Receipt


def load_receipt_examples(path: Path) -> tuple[ReceiptExample, ...]:
    """Load a receipt JSONL fixture."""
    if not path.is_file():
        msg = f"examples not found: {path}"
        raise CompileError(message=msg)
    rows: list[ReceiptExample] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(ReceiptExample.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            msg = f"{path}:{line_no}: {exc}"
            raise CompileError(message=msg) from exc
    if not rows:
        msg = f"no receipts in {path}"
        raise CompileError(message=msg)
    return tuple(rows)


def numeric_problems(receipt: Receipt) -> tuple[str, ...]:
    """Every arithmetic identity this receipt breaks. Empty means it is sound.

    Only asserted numbers are checked. An absent subtotal is silence, not an
    error, and inventing a check for it would reject a correct reading of a
    receipt that never printed one. This is the same asymmetry the restaurant
    gate uses: contradict, reject; stay quiet, accept.
    """
    problems: list[str] = []
    for i, item in enumerate(receipt.items, start=1):
        expected = item.qty * item.unit_price
        if abs(expected - item.line_total) > EPSILON:
            problems.append(
                f"item {i}: qty*unit_price={expected:.2f} != line_total="
                f"{item.line_total:.2f}"
            )
        if item.unit_price < 0 or item.line_total < 0:
            problems.append(f"item {i}: negative amount")
    if receipt.total is None and not receipt.items:
        problems.append("no total and no line items: nothing was extracted")
    if receipt.items and receipt.subtotal is not None:
        summed = sum(item.line_total for item in receipt.items)
        if abs(summed - receipt.subtotal) > EPSILON:
            problems.append(
                f"sum(line_total)={summed:.2f} != subtotal={receipt.subtotal:.2f}"
            )
    if receipt.subtotal is not None and receipt.total is not None:
        charged = (
            receipt.subtotal
            + (receipt.tax or 0.0)
            + (receipt.service or 0.0)
            + (receipt.discount or 0.0)
        )
        if abs(charged - receipt.total) > EPSILON:
            problems.append(
                f"subtotal+tax+service+discount={charged:.2f} != "
                f"total={receipt.total:.2f}"
            )
    problems.extend(_header_problems(receipt))
    return tuple(problems)


def _header_problems(receipt: Receipt) -> tuple[str, ...]:
    """Check the fields that are present; say nothing about the ones that are not."""
    problems: list[str] = []
    if receipt.currency is not None and not _CURRENCY_RE.match(
        receipt.currency.strip()
    ):
        problems.append(f"currency {receipt.currency!r} is not a code or symbol")
    if receipt.date is not None:
        try:
            _date.fromisoformat(receipt.date)
        except ValueError:
            problems.append(f"date {receipt.date!r} is not ISO 8601")
    return tuple(problems)


@dataclass(frozen=True, slots=True)
class ReceiptScore:
    """Per-document receipt accuracy."""

    header_hits: int
    header_total: int
    item_matches: int
    item_total: int
    #: Raw-string agreement, reported alongside the normalized number.
    strict_hits: int = 0
    #: Whether the document cleared the gate. Set by the caller.
    schema_ok: bool = True

    @property
    def quality(self) -> float:
        """Header fields and line items weighted equally, by count."""
        total = self.header_total + self.item_total
        if total == 0:
            return 0.0
        return (self.header_hits + self.item_matches) / total

    @property
    def strict_quality(self) -> float:
        """Accuracy with descriptions compared exactly rather than fuzzily."""
        total = self.header_total + self.item_total
        if total == 0:
            return 0.0
        return (self.strict_hits + self.strict_item_matches) / total

    #: Exact-description item matches, filled by `score_receipt`.
    strict_item_matches: int = 0


def score_receipt(*, gold: Receipt, pred: Receipt) -> ReceiptScore:
    """Compare one predicted receipt to gold.

    Header amounts compare with `EPSILON`, text fields casefold, and line items are
    matched one-to-one on amount with a fuzzy description, so reordering the lines
    is not punished (PLAN §16.2).
    """
    header_hits = 0
    for field in TEXT_FIELDS:
        if _text(getattr(gold, field)) == _text(getattr(pred, field)):
            header_hits += 1
    for field in MONEY_FIELDS:
        want, got = getattr(gold, field), getattr(pred, field)
        if want is None and got is None:
            header_hits += 1  # correctly reported as absent
        elif want is not None and got is not None and abs(want - got) <= EPSILON:
            header_hits += 1
    # Strict means raw equality: no casefolding, no epsilon.
    strict_header = sum(
        1
        for name in (*TEXT_FIELDS, *MONEY_FIELDS)
        if getattr(gold, name) == getattr(pred, name)
    )
    return ReceiptScore(
        header_hits=header_hits,
        header_total=len(TEXT_FIELDS) + len(MONEY_FIELDS),
        item_matches=_match_items(gold.items, pred.items),
        item_total=max(len(gold.items), len(pred.items)),
        strict_hits=strict_header,
        strict_item_matches=_match_items(gold.items, pred.items, exact=True),
    )


def _text(value: str | None) -> str:
    """Normalized text, where absent and empty are the same answer."""
    if value is None:
        return ""
    return " ".join(value.casefold().split())


def _match_items(
    gold: tuple[LineItem, ...], pred: tuple[LineItem, ...], *, exact: bool = False
) -> int:
    """Greedy one-to-one match on amount, with description agreement as a tiebreak."""
    remaining = list(pred)
    matched = 0
    for want in gold:
        best: int | None = None
        best_key: tuple[int, float] | None = None
        for i, got in enumerate(remaining):
            same_amount = (
                abs(want.line_total - got.line_total) <= EPSILON
                and abs(want.qty - got.qty) <= EPSILON
            )
            if not same_amount:
                continue
            if exact and _text(want.desc) != _text(got.desc):
                continue
            key = (
                _desc_similarity(want.desc, got.desc),
                -abs(want.line_total - got.line_total),
            )
            if best_key is None or key > best_key:
                best, best_key = i, key
        if best is not None:
            _ = remaining.pop(best)
            matched += 1
    return matched


def _desc_similarity(left: str, right: str) -> int:
    """Count shared tokens. OCR mangles descriptions, so this is a soft signal."""
    return len(set(_text(left).split()) & set(_text(right).split()))
