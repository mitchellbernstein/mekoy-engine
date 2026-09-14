"""BANKING77: a classification task, not an extraction task.

The third task class, and the one that tests whether `tasks.Task` is a real
abstraction or extraction wearing a nicer name. Everything here differs from the
receipt and restaurant jobs: the output is one label from a closed set, there is
nothing to quote, no arithmetic to check, and the metric is accuracy rather than
field accuracy.

The label set is embedded rather than fetched, because BANKING77's categories are
fixed. `load_banking77` pulls the utterances from the dataset's public repository.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from mekoy.errors import CompileError

__all__ = [
    "CATEGORIES",
    "BankingScore",
    "Intent",
    "banking_gate",
    "banking_prompt",
    "load_banking77",
    "score_intent",
]

RAW = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets"
    "/master/banking_data"
)

#: Every BANKING77 intent. Sorted, so the prompt and the set are reproducible.
CATEGORIES: tuple[str, ...] = (
    "Refund_not_showing_up",
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
)

_FROZEN: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class Intent(BaseModel):
    """One utterance and the intent it expresses."""

    model_config = _FROZEN

    label: str


def banking_prompt() -> str:
    """Instructions with the closed label set inline.

    A classifier cannot choose from a set it has not been shown, so the labels are
    part of the task's instructions rather than something the model must guess.
    """
    labels = "\n".join(f"- {c}" for c in CATEGORIES)
    return (
        "You classify a customer's banking support message into exactly one "
        "intent.\nReturn ONLY a JSON object with one key:\n"
        "label (one of the intents below, copied exactly)\n\n"
        f"Intents:\n{labels}\n"
    )


def banking_gate(intent: Intent) -> tuple[str, ...]:
    """The deterministic check available: is the label one we know?"""
    if intent.label not in CATEGORIES:
        return (f"label {intent.label!r} is not one of the 77 intents",)
    return ()


@dataclass(frozen=True, slots=True)
class BankingScore:
    """Accuracy on a single utterance."""

    correct: bool
    schema_ok: bool = True

    @property
    def quality(self) -> float:
        """1.0 for the right intent, 0.0 otherwise."""
        return 1.0 if self.correct else 0.0

    @property
    def strict_quality(self) -> float:
        """Classification has no fuzzy form, so this matches `quality`."""
        return self.quality


def score_intent(*, gold: Intent, pred: Intent) -> BankingScore:
    """Exact intent match. No partial credit, which is the standard metric."""
    return BankingScore(correct=gold.label.strip() == pred.label.strip())


def _rows(csv_text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for row in csv.DictReader(io.StringIO(csv_text)):
        text = (row.get("text") or "").strip()
        label = (row.get("category") or "").strip()
        if text and label:
            out.append((text, label))
    return out


def _fetch(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=90) as response:  # noqa: S310
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        msg = f"BANKING77 fetch failed: {url}: {exc}"
        raise CompileError(message=msg) from exc


def _interleave(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Deal rows round-robin across categories.

    Both CSVs are sorted by category, so taking the first N rows yields a handful
    of intents rather than a slice of the problem: 600 rows of a 77-class task
    covered five classes. Interleaving makes any prefix cover every class.
    """
    groups: dict[str, list[str]] = {}
    for text, label in rows:
        groups.setdefault(label, []).append(text)
    out: list[tuple[str, str]] = []
    depth = max(len(v) for v in groups.values())
    for i in range(depth):
        out.extend(
            (groups[label][i], label)
            for label in sorted(groups)
            if i < len(groups[label])
        )
    return out


def load_banking77(path: Path, *, limit: int = 600) -> tuple[Path, int, int]:
    """Fetch utterances, write them, and report how many were dropped.

    A row whose category is outside the published set is dropped rather than kept
    with a label nothing can be scored against.
    """
    collected: list[tuple[str, str]] = []
    skipped = 0
    for split in ("train", "test"):
        for text, label in _rows(_fetch(f"{RAW}/{split}.csv")):
            if label not in CATEGORIES:
                skipped += 1
                continue
            collected.append((text, label))
    selected = _interleave(collected)[: max(1, limit)]
    pairs: list[tuple[str, Intent]] = []
    pairs.extend((text, Intent(label=label)) for text, label in selected)
    if not pairs:
        msg = "no usable BANKING77 rows"
        raise CompileError(message=msg)
    lines = [
        json.dumps({"text": text, "label": intent.label}, ensure_ascii=False)
        for text, intent in pairs
    ]
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, len(pairs), skipped
