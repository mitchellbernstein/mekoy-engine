"""Scoring several rows at once, and naming the server as the bottleneck.

Concurrency here is a throughput knob, not an accuracy knob, so the test that matters is
that a concurrent run produces exactly the same scores as a serial one, in the same
order. The rest cover the check that reports a single-slot server, because a bottleneck
the user cannot see is a bottleneck they cannot fix.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from mekoy.dataset import load_examples, split_examples
from mekoy.doctor import Finding, ModelShape, _shape_from_info, check, render
from mekoy.outcome import RestaurantOutcome
from mekoy.search import HarnessConfig, _concurrency, evaluate
from mekoy.tasks import RESTAURANT

if TYPE_CHECKING:
    import pytest


def _outcome(party: int) -> RestaurantOutcome:
    return RestaurantOutcome(
        restaurant="Uchi",
        intent="availability",
        status="confirmed",
        party_size=party,
        when="Friday",
        under_name=None,
        evidence="A table for two is open Friday.",
        booked=False,
    )


class _Stub:
    """Answers with a party size taken from the document, so row order is visible."""

    local: bool = True

    def __init__(self) -> None:
        self.seen: list[str] = []

    def complete(self, *, system: str, user: str, **_rest: object) -> str:
        del system
        self.seen.append(user)
        return _outcome(party=len(self.seen)).model_dump_json()


def test_concurrency_is_a_throughput_knob_only() -> None:
    """A concurrent run must be byte-identical to a serial one."""
    rows = load_examples(Path("examples/bucko-restaurant/examples.jsonl"))
    records = split_examples(rows).dev[:6]
    config = HarnessConfig(k_shot=0, retries=0, constrained=True)

    previous = os.environ.get("MEKOY_CONCURRENCY")
    try:
        os.environ["MEKOY_CONCURRENCY"] = "1"
        serial = evaluate(_Stub(), records, config, shots=(), task=RESTAURANT)
        os.environ["MEKOY_CONCURRENCY"] = "4"
        concurrent = evaluate(_Stub(), records, config, shots=(), task=RESTAURANT)
    finally:
        if previous is None:
            os.environ.pop("MEKOY_CONCURRENCY", None)
        else:
            os.environ["MEKOY_CONCURRENCY"] = previous

    assert [s.quality for s in serial.scores] == [s.quality for s in concurrent.scores]
    assert serial.quality == concurrent.quality


def test_concurrency_width_is_clamped_and_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silly value must not become a silly number of threads."""
    monkeypatch.setenv("MEKOY_CONCURRENCY", "0")
    assert _concurrency() == 1
    monkeypatch.setenv("MEKOY_CONCURRENCY", "9999")
    assert _concurrency() == 64
    monkeypatch.setenv("MEKOY_CONCURRENCY", "not a number")
    assert _concurrency() == 4
    monkeypatch.delenv("MEKOY_CONCURRENCY")
    assert _concurrency() == 4


def test_doctor_names_a_single_slot_server_as_the_bottleneck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The complaint has to name the cause and the fix, or it is just noise."""
    monkeypatch.setattr("mekoy.doctor.server_slots", lambda: 1)
    monkeypatch.setenv("MEKOY_CONCURRENCY", "4")

    findings = check()
    rendered = render(findings)

    slots = next(f for f in findings if f.name == "server serves at once")
    assert slots.is_problem
    assert "queue" in (slots.advice or "")
    assert "OLLAMA_NUM_PARALLEL" in rendered
    assert "2.8x" in rendered


def test_doctor_is_quiet_when_the_server_keeps_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No problem must read as no problem, not as advice."""
    monkeypatch.setattr("mekoy.doctor.server_slots", lambda: 8)
    monkeypatch.setenv("MEKOY_CONCURRENCY", "4")
    assert not [f for f in check() if f.is_problem]
    assert "nothing is slowing" in render(check())


def test_kv_cache_memory_follows_the_formula() -> None:
    """`layers x KV heads x head dim x 2 x bytes`, checked against a known model.

    A wrong number here silently decides how many requests the machine runs at once, so
    it is pinned: qwen2.5-7b is 28 layers, 4 KV heads, head dim 128, two bytes per
    value, which is 28 x 4 x 128 x 2 x 2 = 57,344 bytes, or 56 KB per token.
    """
    shape = ModelShape(
        name="qwen2.5:7b",
        layers=28,
        kv_heads=4,
        head_dim=128,
        context=32768,
    )
    assert shape.kv_bytes_per_token == 57_344
    assert shape.kv_bytes(tokens=1024, width=1) == 57_344 * 1024
    assert shape.kv_bytes(tokens=1024, width=4) == 57_344 * 1024 * 4


def test_a_full_context_costs_more_than_the_engine_needs() -> None:
    """Reserving 32K when the engine uses a few thousand is the trap this guards.

    The same model at the same width costs four times the KV memory at 32K context as at
    8K, which is the difference between fitting on a laptop and not.
    """
    shape = ModelShape(name="m", layers=28, kv_heads=4, head_dim=128, context=32768)
    small = shape.kv_bytes(tokens=8192, width=4)
    large = shape.kv_bytes(tokens=32768, width=4)
    assert large == small * 4
    assert small / 1e9 < 2.0
    assert large / 1e9 > 7.0


def test_head_dim_comes_from_embedding_over_heads() -> None:
    """The server reports embedding length and head count, not head dimension."""
    shape = _shape_from_info(
        "m",
        {
            "qwen2.block_count": 28,
            "qwen2.attention.head_count": 28,
            "qwen2.attention.head_count_kv": 4,
            "qwen2.embedding_length": 3584,
            "qwen2.context_length": 32768,
        },
    )
    assert shape is not None
    assert shape.head_dim == 3584 // 28 == 128
    assert shape.layers == 28
    assert shape.kv_heads == 4


def test_an_unreadable_model_shape_is_reported_not_guessed() -> None:
    """A made-up shape would make the memory plan a fiction."""
    assert _shape_from_info("m", {}) is None
    assert _shape_from_info("m", {"qwen2.block_count": 28}) is None


def test_a_finding_without_advice_is_not_a_problem() -> None:
    assert not Finding(name="rows scored at once", detail="4").is_problem
    assert Finding(name="x", detail="y", advice="z").is_problem
