"""SROIE conversion. Fixtures mirror the real file shapes; no network."""

import json
from pathlib import Path

import pytest

from mekoy.cord import write_examples
from mekoy.dataset import load_task_examples
from mekoy.receipt import numeric_problems
from mekoy.sroie import boxes_to_text, convert_key
from mekoy.tasks import RECEIPT

# Real box rows: 8 coordinates then text, text may contain commas, and the CSV
# order is an annotation artefact rather than the printed order.
_CSV = (
    "72,25,326,25,326,64,72,64,TAN WOON YANN\n"
    "50,82,440,82,440,121,50,121,BOOK TA .K(TAMAN DAYA) SDN BND\n"
    "205,121,285,121,285,139,205,139,789417-W\n"
    # Deliberately last in the file, but first on the page by y.
    "50,5,300,5,300,20,50,20,TOP LINE\n"
    # Commas inside the text must survive.
    "110,144,383,144,383,163,110,163,NO.53 55,57 & 59, JALAN SAGU 18,\n"
    # Two words on one visual line, printed right to left in the file.
    "400,600,500,600,500,615,400,615,RIGHT\n"
    "100,600,200,600,200,615,100,615,LEFT"
)


def test_text_is_rebuilt_in_printed_order() -> None:
    lines = boxes_to_text(_CSV).splitlines()
    assert lines[0] == "TOP LINE"
    assert lines[1] == "TAN WOON YANN"
    assert lines[2].startswith("BOOK TA")


def test_commas_inside_a_box_survive() -> None:
    text = boxes_to_text(_CSV)
    assert "NO.53 55,57 & 59, JALAN SAGU 18," in text


def test_boxes_on_one_line_are_ordered_left_to_right() -> None:
    text = boxes_to_text(_CSV)
    assert "LEFT RIGHT" in text


def test_short_rows_are_ignored() -> None:
    """A row without eight coordinates is not a box."""
    assert boxes_to_text("1,2,3,text only\n72,25,326,25,326,64,72,64,REAL") == "REAL"


def test_non_numeric_coordinates_are_skipped_not_fatal() -> None:
    mixed = "x,y,a,b,c,d,e,f,BAD\n72,25,326,25,326,64,72,64,GOOD"
    assert boxes_to_text(mixed) == "GOOD"


def test_dates_are_normalised_to_iso() -> None:
    receipt = convert_key(
        {
            "company": "BOOK TA",
            "date": "25/12/2018",
            "address": "JOHOR",
            "total": "9.00",
        }
    )
    assert receipt is not None
    assert receipt.date == "2018-12-25"
    assert receipt.total == pytest.approx(9.0)
    assert numeric_problems(receipt) == ()


def test_two_digit_years_and_other_separators() -> None:
    assert convert_key({"company": "X", "date": "01/02/99"}).date == "1999-02-01"
    assert convert_key({"company": "X", "date": "31-12-2020"}).date == "2020-12-31"


def test_an_unparseable_date_becomes_absent_not_wrong() -> None:
    """Silence beats a guess: the gate checks a date only when one is stated."""
    receipt = convert_key({"company": "X", "date": "not a date", "total": "1.00"})
    assert receipt is not None
    assert receipt.date is None
    assert numeric_problems(receipt) == ()


def test_an_empty_annotation_is_refused() -> None:
    assert convert_key({"company": "", "total": ""}) is None


def test_sroie_rows_have_no_items_and_still_pass_the_gate() -> None:
    """SROIE annotates four fields, so no arithmetic identity applies."""
    receipt = convert_key(
        {
            "company": "BOOK TA",
            "date": "25/12/2018",
            "address": "JOHOR",
            "total": "9.00",
        }
    )
    assert receipt is not None
    assert receipt.items == ()
    assert receipt.subtotal is None
    assert numeric_problems(receipt) == ()


def test_a_converted_corpus_loads_through_the_task_loader(tmp_path: Path) -> None:

    receipts = [
        convert_key({"company": "A", "date": "01/01/2020", "total": "1.00"}),
        convert_key({"company": "B", "date": "02/01/2020", "total": "2.00"}),
        convert_key({"company": "C", "date": "03/01/2020", "total": "3.00"}),
    ]
    pairs = tuple(("line one\nline two", r) for r in receipts if r is not None)
    path = write_examples(tmp_path / "sroie.jsonl", pairs)
    rows = load_task_examples(path, RECEIPT)
    assert len(rows) == 3
    assert "receipt" in json.loads(path.read_text().splitlines()[0])
