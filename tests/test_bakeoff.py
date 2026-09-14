from pathlib import Path
from typing import Self

import pytest

from mekoy import openai_runtime
from mekoy.bakeoff import LaneSpec, format_bakeoff, run_bakeoff
from mekoy.compile import SearchSpace
from mekoy.dataset import ExampleRecord, load_examples, split_examples
from mekoy.openai_runtime import OpenAICompleter, TokenMeter
from mekoy.verify import VerifyOk, parse_and_verify


class _GoldEcho:
    def __init__(self, rows: tuple[ExampleRecord, ...]) -> None:
        self._by_text: dict[str, str] = {
            row.text: row.outcome.model_dump_json() for row in rows
        }

    local: bool = True

    def complete(
        self,
        *,
        system: str,
        user: str,
        constrained: bool = True,
        temperature: float = 0.0,
        schema: dict[str, object] | None = None,
    ) -> str:
        del system, constrained, temperature, schema
        # Shots carry their own labels, so read only the final target document.
        tail = user.rsplit("Text:\n", 1)[-1]
        target = tail.rsplit("\nJSON:", 1)[0]
        return self._by_text.get(target, "{}")


class _AlwaysBad:
    local: bool = True

    def complete(
        self,
        *,
        system: str,
        user: str,
        constrained: bool = True,
        temperature: float = 0.0,
        schema: dict[str, object] | None = None,
    ) -> str:
        del system, user, constrained, temperature, schema
        return "{}"


class _FakeResponse:
    def raise_for_status(self) -> None:
        """Pretend the call succeeded."""

    def json(self) -> dict[str, object]:
        """Canned chat-completion payload carrying usage."""
        return {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1_000, "completion_tokens": 500},
        }


class _FakeClient:
    def __enter__(self) -> Self:
        """Enter the fake client context."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the fake client context."""

    def post(self, url: str, *, json: dict[str, object]) -> _FakeResponse:
        """Return a canned response for any request."""
        del url, json
        return _FakeResponse()


def _fake_create_client(**kwargs: object) -> _FakeClient:
    del kwargs
    return _FakeClient()


def test_hard_gold_passes_verifier() -> None:
    rows = load_examples(Path("examples/bucko-restaurant/examples.jsonl"))
    assert len(rows) >= 12
    for row in rows:
        result = parse_and_verify(row.outcome.model_dump_json())
        assert isinstance(result, VerifyOk)


def test_token_meter_prices_astra_list_rate() -> None:
    meter = TokenMeter(input_tokens=1_000_000, output_tokens=1_000_000)
    assert meter.usd == pytest.approx(60.0)


def test_openai_completer_records_token_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_runtime, "create_client", _fake_create_client)
    completer = OpenAICompleter(api_key="test")
    assert completer.complete(system="s", user="u") == "{}"
    assert completer.meter.input_tokens == 1_000
    assert completer.meter.output_tokens == 500
    assert completer.meter.usd == pytest.approx(0.035)


def test_bakeoff_challenger_beats_bad_baseline() -> None:
    rows = load_examples(Path("examples/bucko-restaurant/examples.jsonl"))
    split = split_examples(rows)
    result = run_bakeoff(
        split=split,
        baseline=LaneSpec(name="gpt-6-astra", completer=_AlwaysBad()),
        challenger=LaneSpec(name="qwen2.5:7b", completer=_GoldEcho(rows)),
        space=SearchSpace.single(),
    )
    text = format_bakeoff(result)
    assert result.challenger.quality > result.baseline.quality
    assert "CHALLENGER WINS" in text
    assert "training: skipped" in text
    assert result.quality_win is True


def test_bakeoff_reports_only_a_quality_win_when_cost_is_higher() -> None:
    """A tie on quality with a paid baseline is still a win; behind is not."""
    rows = load_examples(Path("examples/bucko-restaurant/examples.jsonl"))
    split = split_examples(rows)
    result = run_bakeoff(
        split=split,
        baseline=LaneSpec(name="gpt-6-astra", completer=_AlwaysBad()),
        challenger=LaneSpec(name="qwen2.5:7b", completer=_AlwaysBad()),
        space=SearchSpace.single(),
    )
    assert result.quality_win is True  # 0.0 >= 0.0
    assert result.cheaper is True  # both free
    assert result.verdict.startswith("CHALLENGER WINS")


def test_bakeoff_few_shot_search_uses_train_shots() -> None:
    rows = load_examples(Path("examples/bucko-restaurant/examples.jsonl"))
    split = split_examples(rows)
    result = run_bakeoff(
        split=split,
        baseline=LaneSpec(name="gpt-6-astra", completer=_AlwaysBad()),
        challenger=LaneSpec(name="qwen2.5:7b", completer=_GoldEcho(rows)),
        space=SearchSpace(k_shots=(4,), retries=(0,), constrained=(True,)),
    )
    assert result.report.winner.config.k_shot == 4
    assert result.challenger.quality == pytest.approx(1.0)
    assert result.challenger.quality > result.baseline.quality
    assert "scored once" in format_bakeoff(result)
