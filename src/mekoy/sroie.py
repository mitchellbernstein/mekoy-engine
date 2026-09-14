"""Load SROIE receipts as text-only extraction examples.

SROIE (ICDAR 2019) ships 626 receipts across two files each:

- `data/box/NNN.csv` — OCR words as `x1,y1,...,x4,y4,text`, no reading order.
- `data/key/NNN.json` — `{company, date, address, total}`, dates as `DD/MM/YYYY`.

Two things about it differ from CORD, and both matter:

1. **The annotation is four fields.** No line items, no subtotal, no tax. So the
   arithmetic gate has almost nothing to check here: with no items and no
   subtotal, the identities that catch a hallucinated total simply do not apply.
   A row passing the gate means the schema validated, not that the numbers were
   verified. Saying so is more useful than implying the gate did work it did not.
2. **Dates are `DD/MM/YYYY`.** The receipt schema stores ISO 8601, so the loader
   converts rather than loosening the schema to accept any string.

Reading order is recovered from the box coordinates: lines are grouped by
vertical overlap and ordered left to right within a line. The raw CSV order is
not the printed order.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from mekoy.cord import write_examples
from mekoy.errors import CompileError
from mekoy.receipt import Receipt, numeric_problems

__all__ = [
    "SROIE_ROWS",
    "SroieFetch",
    "boxes_to_text",
    "convert_key",
    "fetch_receipt",
    "load_sroie",
]

#: How many receipts the published test split holds.
SROIE_ROWS = 626
#: A `DD/MM/YYYY` year below this is two-digit and needs a century.
_SHORT_YEAR = 100
#: Pivot for a two-digit year: below this is the 2000s, at or above is the 1900s.
#: Without it, "99" becomes 2099 and every dated receipt is wrong.
_YEAR_PIVOT = 70
#: Calendar bounds for a valid day and month.
_MAX_DAY = 31
_MAX_MONTH = 12
#: Parts in a `DD/MM/YYYY` date.
_DATE_PARTS = 3

RAW = "https://raw.githubusercontent.com/zzzDavid/ICDAR-2019-SROIE/master/data"
#: Text starts at the ninth field; descriptions contain commas.
_COORD_FIELDS = 8
#: Boxes whose vertical spans overlap by more than this fraction are one line.
_LINE_OVERLAP = 0.5


def _boxes(csv_text: str) -> list[tuple[float, float, float, float, str]]:
    """Parse the box CSV into (x0, y0, x1, y1, text) tuples."""
    out: list[tuple[float, float, float, float, str]] = []
    for row in csv.reader(io.StringIO(csv_text)):
        if len(row) <= _COORD_FIELDS:
            continue
        try:
            # Slice before converting. Stepping over the whole row would call
            # float() on the trailing text field and drop every ordinary row.
            xs = [float(v) for v in row[:_COORD_FIELDS:2]]
            ys = [float(v) for v in row[1:_COORD_FIELDS:2]]
        except ValueError:
            continue
        text = ",".join(row[_COORD_FIELDS:]).strip()
        if not text:
            continue
        out.append((min(xs), min(ys), max(xs), max(ys), text))
    return out


def boxes_to_text(csv_text: str) -> str:
    """Rebuild the printed layout from box coordinates.

    Sort by y to get lines, merge boxes whose vertical spans overlap, then sort
    each line by x. The CSV's own order is an annotation artefact, not the page.
    """
    boxes = sorted(_boxes(csv_text), key=lambda b: (b[1], b[0]))
    lines: list[list[tuple[float, float, float, float, str]]] = []
    for box in boxes:
        placed = False
        for line in lines:
            top = min(b[1] for b in line)
            bottom = max(b[3] for b in line)
            span = min(bottom, box[3]) - max(top, box[1])
            if span > 0 and span >= _LINE_OVERLAP * (box[3] - box[1]):
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])
    return "\n".join(
        " ".join(b[4] for b in sorted(line, key=lambda b: b[0])) for line in lines
    )


def _iso_date(raw: object) -> str | None:
    """SROIE prints DD/MM/YYYY. Return ISO 8601, or None if unparseable."""
    text = str(raw or "").strip()
    for sep in ("/", "-", "."):
        parts = text.split(sep)
        if len(parts) == _DATE_PARTS and all(p.isdigit() for p in parts):
            day, month, year = (int(p) for p in parts)
            if year < _SHORT_YEAR:
                year += 2000 if year < _YEAR_PIVOT else 1900
            if 1 <= day <= _MAX_DAY and 1 <= month <= _MAX_MONTH:
                return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _as_float(value: object) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def convert_key(key: dict[str, object]) -> Receipt | None:
    """Map one SROIE annotation onto a Receipt, or None if unusable."""
    total = _as_float(key.get("total"))
    company = str(key.get("company") or "").strip()
    if total is None and not company:
        return None
    address = str(key.get("address") or "").strip() or None
    return Receipt(
        merchant=company,
        date=_iso_date(key.get("date")),
        address=address,
        total=total,
    )


def fetch_receipt(index: int) -> tuple[str, Receipt] | None:
    """Fetch and convert one receipt by index. None if either half is missing."""
    stem = f"{index:03d}"
    try:
        with urllib.request.urlopen(f"{RAW}/key/{stem}.json", timeout=60) as r:  # noqa: S310
            key = json.loads(r.read())
        with urllib.request.urlopen(f"{RAW}/box/{stem}.csv", timeout=60) as r:  # noqa: S310
            csv_text = r.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(key, dict):
        return None
    receipt = convert_key(key)
    text = boxes_to_text(csv_text)
    if receipt is None or not text.strip():
        return None
    return text, receipt


@dataclass(frozen=True, slots=True)
class SroieFetch:
    """What a fetch produced, and what could not be used."""

    pairs: tuple[tuple[str, Receipt], ...]
    attempted: int
    skipped: int


def load_sroie(
    path: Path,
    *,
    limit: int = 120,
    workers: int = 8,
    sound_only: bool = True,
) -> tuple[Path, int, int, int]:
    """Fetch, convert, optionally filter, and write the corpus.

    Returns (path, kept, dropped_unsound, unreachable). Concurrent because 626
    receipts is 1252 small HTTP requests, which is slow one at a time.
    """
    attempted = max(1, min(limit, SROIE_ROWS))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        fetched = list(pool.map(fetch_receipt, range(attempted)))
    pairs = tuple(item for item in fetched if item is not None)
    unreachable = attempted - len(pairs)
    if sound_only:
        sound = tuple(p for p in pairs if not numeric_problems(p[1]))
        dropped = len(pairs) - len(sound)
        pairs = sound
    else:
        dropped = 0
    if not pairs:
        msg = "no usable SROIE rows; the raw host may be unreachable"
        raise CompileError(message=msg)
    return write_examples(path, pairs), len(pairs), dropped, unreachable
