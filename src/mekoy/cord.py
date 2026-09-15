"""Load CORD receipts as text-only extraction examples.

CORD (Consolidated Receipt Dataset) ships receipt photos plus per-receipt JSON:
`gt_parse` holds the annotation and `valid_line` holds the OCR lines.
asks for text-only loaders, so this module renders the OCR lines into a plain-text
receipt and maps the annotation onto our `Receipt` schema. No pixels are read.

Three things about the real data shaped this loader, and all three contradicted the
hand-written fixture:

1. **42 of 100 test receipts print no subtotal.** A schema that required one could
   not represent them.
2. **Service charges and discounts exist on top of tax.** The identity that holds is
   `subtotal + tax + service + discount == total`, not `subtotal + tax == total`.
3. **No row carries store info**, so a required `merchant` field would reject the
   whole split.

Mapping caveat worth stating plainly: when a line prints no unit price, the
converter derives it from the printed line total and count. For those rows the
per-item identity holds by construction and proves nothing. For the rows that do
print a unit price, it is a real check.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mekoy.errors import CompileError
from mekoy.receipt import LineItem, Receipt, numeric_problems

__all__ = [
    "CORD_TEST_ROWS",
    "CordFetch",
    "convert_gt",
    "convert_rows",
    "fetch_rows",
    "load_cord",
    "render_text",
    "write_examples",
]

#: HuggingFace rows endpoint. No parquet dependency; it returns JSON.
CORD_ROWS_API = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=naver-clova-ix%2Fcord-v2&config=default&split={split}"
    "&offset={offset}&length={length}"
)
CORD_TEST_ROWS = 100
_PAGE = 50


def _as_float(value: object) -> float | None:
    """CORD prints numbers like '60.000', sometimes with a comma decimal."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ".").strip())
    except ValueError:
        return None


def _as_int(value: object, default: int = 1) -> int:
    parsed = _as_float(value)
    if parsed is None or parsed <= 0:
        return default
    return max(1, round(parsed))


def _items(menu: object) -> tuple[LineItem, ...]:
    """Map CORD's menu (one dict or a list) into line items."""
    if isinstance(menu, dict):
        raw_items: list[dict[str, Any]] = [menu]
    elif isinstance(menu, list):
        raw_items = [m for m in menu if isinstance(m, dict)]
    else:
        return ()
    out: list[LineItem] = []
    for raw in raw_items:
        desc = str(raw.get("nm", "")).strip()
        line_total = _as_float(raw.get("itemsubtotal"))
        if line_total is None:
            line_total = _as_float(raw.get("price"))
        if line_total is None or not desc:
            continue
        qty = _as_int(raw.get("cnt"))
        unit_price = _as_float(raw.get("unitprice"))
        if unit_price is None:
            # Derived, not printed. See the module docstring.
            unit_price = line_total / qty
        out.append(
            LineItem(
                desc=desc,
                qty=float(qty),
                unit_price=unit_price,
                line_total=line_total,
            )
        )
    return tuple(out)


def convert_gt(gt: dict[str, Any]) -> Receipt | None:
    """Map one CORD `ground_truth` object onto a Receipt, or None if unusable."""
    parsed = gt.get("gt_parse")
    if not isinstance(parsed, dict):
        return None
    sub = parsed.get("sub_total") or {}
    total = parsed.get("total") or {}
    receipt = Receipt(
        merchant="",
        date=None,
        currency=None,
        subtotal=_as_float(sub.get("subtotal_price")),
        tax=_as_float(sub.get("tax_price")),
        service=_as_float(sub.get("service_price")),
        discount=_as_float(sub.get("discount_price")),
        total=_as_float(total.get("total_price")),
        items=_items(parsed.get("menu")),
    )
    if receipt.total is None and not receipt.items:
        return None
    return receipt


def render_text(gt: dict[str, Any]) -> str:
    """Render CORD's OCR lines as the plain-text receipt the model reads.

    Reading order is the order CORD stores, which is the order the receipt was
    annotated in.
    """
    lines: list[str] = []
    for line in gt.get("valid_line") or []:
        words = line.get("words") or []
        text = " ".join(str(w.get("text", "")).strip() for w in words).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def convert_rows(rows: list[dict[str, Any]]) -> tuple[tuple[str, Receipt], ...]:
    """Convert raw API rows into (text, receipt) pairs, skipping unusable ones."""
    out: list[tuple[str, Receipt]] = []
    for row in rows:
        gt = row.get("ground_truth")
        if isinstance(gt, str):
            try:
                gt = json.loads(gt)
            except json.JSONDecodeError:
                continue
        if not isinstance(gt, dict):
            continue
        receipt = convert_gt(gt)
        text = render_text(gt)
        if receipt is None or not text.strip():
            continue
        out.append((text, receipt))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class CordFetch:
    """What a fetch produced, and what could not be used."""

    pairs: tuple[tuple[str, Receipt], ...]
    fetched: int
    skipped: int


def fetch_rows(
    split: str = "test", limit: int = CORD_TEST_ROWS
) -> list[dict[str, Any]]:
    """Page the HuggingFace rows endpoint. No dataset library required."""
    rows: list[dict[str, Any]] = []
    offset = 0
    while offset < limit:
        length = min(_PAGE, limit - offset)
        url = CORD_ROWS_API.format(split=split, offset=offset, length=length)
        try:
            with urllib.request.urlopen(url, timeout=90) as response:  # noqa: S310
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            msg = f"CORD fetch failed at offset {offset}: {exc}"
            raise CompileError(message=msg) from exc
        page = payload.get("rows") or []
        if not page:
            break
        rows.extend(item["row"] for item in page)
        offset += len(page)
    return rows


def load_cord(
    path: Path,
    *,
    split: str = "test",
    limit: int = CORD_TEST_ROWS,
    sound_only: bool = True,
) -> tuple[Path, int, int]:
    """Fetch, convert, optionally drop unsound gold, and write the corpus.

    `sound_only` matters more than it sounds. 17 of CORD's 100 test annotations
    break a deterministic arithmetic identity, so scoring a model against them
    would penalise it for disagreeing with gold that does not add up. Returns
    (path, kept, dropped).
    """
    rows = fetch_rows(split, limit)
    pairs = convert_rows(rows)
    if sound_only:
        sound = tuple(p for p in pairs if not numeric_problems(p[1]))
        dropped = len(pairs) - len(sound)
        pairs = sound
    else:
        dropped = 0
    if not pairs:
        msg = f"no usable CORD rows from split {split!r}"
        raise CompileError(message=msg)
    return write_examples(path, pairs), len(pairs), dropped


def write_examples(path: Path, pairs: tuple[tuple[str, Receipt], ...]) -> Path:
    """Write the converted corpus in the repository's JSONL row shape."""
    lines = [
        json.dumps(
            {"text": text, "receipt": receipt.model_dump()},
            ensure_ascii=False,
        )
        for text, receipt in pairs
    ]
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
