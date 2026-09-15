from pathlib import Path

import pytest

from mekoy.receipt import (
    LineItem,
    Receipt,
    load_receipt_examples,
    numeric_problems,
    score_receipt,
)

_HEAD = {
    "merchant": "Rodeo Austin",
    "date": "2024-03-14",
    "currency": "USD",
    "subtotal": 16.5,
    "tax": 1.36,
    "total": 17.86,
}
_ITEMS = (
    LineItem(desc="Brisket sandwich", qty=1, unit_price=14.0, line_total=14.0),
    LineItem(desc="Pickle", qty=1, unit_price=2.5, line_total=2.5),
)


def _receipt(**over: object) -> Receipt:
    fields = dict(_HEAD)
    fields.update(over)
    fields.setdefault("items", _ITEMS)
    return Receipt.model_validate(fields)


def test_a_sound_receipt_has_no_problems() -> None:
    assert numeric_problems(_receipt()) == ()


def test_line_total_must_match_qty_times_unit_price() -> None:
    bad = (
        LineItem(desc="Brisket sandwich", qty=2, unit_price=14.0, line_total=14.0),
        LineItem(desc="Pickle", qty=1, unit_price=2.5, line_total=2.5),
    )
    problems = numeric_problems(_receipt(items=bad, subtotal=16.5, total=17.86))
    assert any("qty*unit_price" in p for p in problems), problems


def test_lines_must_sum_to_subtotal() -> None:
    problems = numeric_problems(_receipt(subtotal=99.0, total=100.36))
    assert any("!= subtotal" in p for p in problems), problems


def test_subtotal_plus_tax_must_equal_total() -> None:
    problems = numeric_problems(_receipt(total=99.99))
    assert any("!= total" in p for p in problems), problems


def test_currency_accepts_codes_and_symbols() -> None:
    """SROIE prints "RM" for Ringgit; demanding three capitals rejected it."""
    assert numeric_problems(_receipt(currency=None)) == ()
    for good in ("USD", "RM", "$", "€", "Rp"):
        assert numeric_problems(_receipt(currency=good)) == (), good
    assert any("currency" in p for p in numeric_problems(_receipt(currency="Dollars")))


def test_date_must_be_iso() -> None:
    problems = numeric_problems(_receipt(date="14 MAR 24"))
    assert any("ISO 8601" in p for p in problems), problems


def test_an_empty_extraction_is_unsound() -> None:
    """No total and no items is nothing extracted, not a sparse receipt."""
    problems = numeric_problems(_receipt(items=(), subtotal=None, tax=None, total=None))
    assert any("nothing was extracted" in p for p in problems), problems


def test_a_receipt_without_a_subtotal_is_fine() -> None:
    """42 of 100 CORD receipts print no subtotal; requiring one is unrealistic."""
    problems = numeric_problems(_receipt(items=(), subtotal=None, tax=None, total=1.36))
    assert problems == ()


def test_cent_rounding_is_not_a_finding() -> None:
    """Half a cent of rounding must not fail a correct receipt."""
    items = (LineItem(desc="Coffee", qty=3, unit_price=1.111, line_total=3.33),)
    receipt = _receipt(items=items, subtotal=3.33, tax=0.27, total=3.60)
    assert numeric_problems(receipt) == ()


def test_the_shipped_receipt_fixtures_are_sound() -> None:
    """An eval whose gold breaks its own arithmetic cannot score anything."""
    for name in ("examples", "hard"):
        path = Path(f"examples/cord-receipt/{name}.jsonl")
        rows = load_receipt_examples(path)
        assert rows, name
        for row in rows:
            assert numeric_problems(row.receipt) == (), (name, row.receipt.merchant)
        assert rows


def test_score_is_perfect_for_an_identical_receipt() -> None:
    gold = _receipt()
    assert score_receipt(gold=gold, pred=gold).quality == pytest.approx(1.0)


def test_reordered_line_items_still_score() -> None:
    """Line items are matched, not compared positionally."""
    gold = _receipt()
    pred = _receipt(items=tuple(reversed(_ITEMS)))
    assert score_receipt(gold=gold, pred=pred).quality == pytest.approx(1.0)


def test_a_wrong_total_costs_a_header_field() -> None:
    gold = _receipt()
    pred = _receipt(total=18.86)
    score = score_receipt(gold=gold, pred=pred)
    assert score.header_hits == score.header_total - 1
    assert score.quality < 1.0


def test_a_missing_line_item_costs_an_item_credit() -> None:
    gold = _receipt()
    pred = _receipt(items=_ITEMS[:1], subtotal=14.0, total=15.36)
    score = score_receipt(gold=gold, pred=pred)
    assert score.item_matches == 1
    assert score.item_total == 2
