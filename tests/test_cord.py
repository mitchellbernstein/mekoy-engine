"""CORD conversion. No network: fixtures mirror the real payload shape."""

import json
from pathlib import Path

import pytest

from mekoy.cord import convert_gt, convert_rows, render_text, write_examples
from mekoy.dataset import load_task_examples, split_examples
from mekoy.receipt import numeric_problems
from mekoy.tasks import RECEIPT

# Mirrors a real CORD row: a service charge on top of tax, a list menu, and OCR
# lines to render as text.
_WITH_SERVICE = {
    "gt_parse": {
        "menu": [
            {"nm": "Brisket", "cnt": "1", "price": "14.000", "unitprice": "14.000"},
            {"nm": "Pickle", "cnt": "1", "price": "2.500", "unitprice": "2.500"},
        ],
        "sub_total": {
            "subtotal_price": "16.500",
            "tax_price": "1.360",
            "service_price": "1.000",
        },
        "total": {"total_price": "18.860"},
    },
    "valid_line": [
        {"words": [{"text": "Rodeo"}, {"text": "Austin"}]},
        {"words": [{"text": "Brisket"}, {"text": "14.000"}]},
    ],
}

# 42 of 100 real CORD rows print no subtotal at all.
_NO_SUBTOTAL = {
    "gt_parse": {"menu": None, "total": {"total_price": "46.000"}},
    "valid_line": [{"words": [{"text": "TOTAL"}, {"text": "46.000"}]}],
}

_NO_TOTAL_NO_ITEMS = {
    "gt_parse": {"menu": None, "sub_total": {"subtotal_price": "10.000"}},
    "valid_line": [{"words": [{"text": "nothing"}]}],
}


def test_service_charge_is_modelled() -> None:
    """The identity that actually holds is subtotal+tax+service+discount."""
    receipt = convert_gt(_WITH_SERVICE)
    assert receipt is not None
    assert receipt.service == pytest.approx(1.0)
    assert numeric_problems(receipt) == ()


def test_a_receipt_with_no_subtotal_is_fine() -> None:
    receipt = convert_gt(_NO_SUBTOTAL)
    assert receipt is not None
    assert receipt.subtotal is None
    assert numeric_problems(receipt) == ()


def test_an_empty_annotation_is_refused() -> None:
    assert convert_gt(_NO_TOTAL_NO_ITEMS) is None


def test_menu_list_and_single_dict_both_convert() -> None:
    single = {
        "gt_parse": {
            "menu": {"nm": "Taco", "cnt": "2", "price": "5.000"},
            "total": {"total_price": "10.000"},
        }
    }
    receipt = convert_gt(single)
    assert receipt is not None
    assert len(receipt.items) == 1
    assert receipt.items[0].line_total == pytest.approx(5.0)
    assert receipt.items[0].qty == pytest.approx(2.0)


def test_unit_price_is_derived_when_not_printed() -> None:
    """Derived, so the item identity is a tautology for these rows."""
    no_price = {
        "gt_parse": {
            "menu": {"nm": "Taco", "cnt": "2", "price": "10.000"},
            "total": {"total_price": "10.000"},
        }
    }
    receipt = convert_gt(no_price)
    assert receipt is not None
    assert receipt.items[0].unit_price == pytest.approx(5.0)
    assert receipt.items[0].line_total == pytest.approx(10.0)


def test_ocr_lines_render_in_order() -> None:
    assert render_text(_WITH_SERVICE) == "Rodeo Austin\nBrisket 14.000"


def test_convert_rows_skips_unusable_and_keeps_the_rest() -> None:
    rows = [
        {"ground_truth": _WITH_SERVICE},
        {"ground_truth": _NO_TOTAL_NO_ITEMS},
        {"ground_truth": _NO_SUBTOTAL},
        {"ground_truth": "not json at all"},
    ]
    pairs = convert_rows(rows)
    assert len(pairs) == 2
    assert all(isinstance(receipt, object) for _, receipt in pairs)


def test_the_written_corpus_loads_through_the_task_loader(tmp_path: Path) -> None:
    third = {
        "gt_parse": {
            "menu": {"nm": "A", "cnt": "1", "price": "1.000"},
            "total": {"total_price": "1.000"},
        },
        "valid_line": [{"words": [{"text": "A"}]}],
    }
    pairs = convert_rows(
        [
            {"ground_truth": _WITH_SERVICE},
            {"ground_truth": _NO_SUBTOTAL},
            {"ground_truth": third},
        ]
    )
    path = write_examples(tmp_path / "cord.jsonl", pairs)
    rows = load_task_examples(path, RECEIPT)
    assert len(rows) == 3
    assert split_examples(rows).test
    raw = json.loads(path.read_text().splitlines()[0])
    assert "receipt" in raw
    assert "text" in raw
