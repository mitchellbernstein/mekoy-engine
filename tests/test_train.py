"""Training is gated by the eval, and only one backend is verified here."""

import json
from pathlib import Path

import pytest

from mekoy.compile import Budget, CompileReport, SearchSpace, compile_system
from mekoy.dataset import ExampleRecord, load_examples, split_examples
from mekoy.errors import CompileError
from mekoy.spec import Slos
from mekoy.tasks import RESTAURANT
from mekoy.train import (
    FIREWORKS_KEY_ENV,
    SCRATCH_SIZES,
    LoRaBudget,
    fireworks_payload,
    mlx_command,
    plan_training,
    require_fireworks_key,
    require_scratch_approval,
    scratch_plan,
    should_train,
    write_mlx_dataset,
)

_FIXTURE = Path("examples/bucko-restaurant/generated.jsonl")


class _Echo:
    local: bool = True

    def __init__(self, rows: tuple[ExampleRecord, ...]) -> None:
        self._by_text = {r.text: r.outcome.model_dump_json() for r in rows}

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
        tail = user.rsplit("Text:\n", 1)[-1].rsplit("\nJSON:", 1)[0]
        return self._by_text.get(tail, "{}")


def _report(rows: tuple[ExampleRecord, ...]) -> CompileReport:
    return compile_system(
        _Echo(rows), split_examples(rows), SearchSpace.single(), Budget(trials=1)
    )


def test_the_rank_band_is_enforced_not_clamped() -> None:
    """The rank band is 8-16. A silent clamp would hide a bad config."""
    with pytest.raises(CompileError, match="rank 8-16"):
        _ = LoRaBudget(rank=32)


def test_an_unknown_training_backend_is_refused() -> None:
    with pytest.raises(CompileError, match="unknown training backend"):
        _ = LoRaBudget(backend="together")


def test_training_is_skipped_when_the_gate_is_already_clear() -> None:
    """Training runs only when below the gate."""
    rows = load_examples(_FIXTURE)
    report = _report(rows)
    slos = Slos(quality=0.5, cost_per_doc=1.0, latency_ms=60_000)
    assert should_train(report, slos=slos) is False
    plan = plan_training(report, slos=slos)
    assert plan.train is False
    assert "already clears" in plan.reason
    assert plan.report_line == "training: skipped"


class _Partial(_Echo):
    """Knows the training rows only, so the held-out rows fail the gate."""

    def __init__(self, rows: tuple[ExampleRecord, ...], known: frozenset[str]) -> None:
        super().__init__(rows)
        self._known = known

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
        tail = user.rsplit("Text:\n", 1)[-1].rsplit("\nJSON:", 1)[0]
        return self._by_text.get(tail, "{}") if tail in self._known else "{}"


def test_training_is_warranted_below_the_gate() -> None:
    rows = load_examples(_FIXTURE)
    split = split_examples(rows)
    known = frozenset(r.text for r in split.train)
    report = compile_system(
        _Partial(rows, known),
        split,
        SearchSpace.single(),
        Budget(trials=1),
    )
    assert report.test.quality < 1.0
    slos = Slos(quality=0.9, cost_per_doc=1.0, latency_ms=60_000)
    plan = plan_training(report, slos=slos)
    assert plan.train is True
    assert "below gate" in plan.reason
    assert plan.report_line.startswith("training: mlx")


def test_a_perfect_score_skips_training_without_a_declared_gate() -> None:
    rows = load_examples(_FIXTURE)
    report = _report(rows)
    if report.test.quality >= 1.0:
        assert should_train(report) is False


def test_the_mlx_dataset_uses_the_systems_own_prompt(tmp_path: Path) -> None:
    """The adapter must learn the harness that was measured."""
    rows = load_examples(_FIXTURE)
    split = split_examples(rows)
    pairs = tuple((r.text, r.outcome) for r in split.train)
    valid = tuple((r.text, r.outcome) for r in split.dev)
    out = write_mlx_dataset(tmp_path / "d", train=pairs, valid=valid)
    for name in ("train.jsonl", "valid.jsonl"):
        assert (out / name).is_file()
    row = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert row["messages"][0]["content"] == RESTAURANT.prompt_variants["default"]
    assert json.loads(row["messages"][2]["content"])["restaurant"]


def test_the_local_command_is_reproducible() -> None:
    cmd = mlx_command(
        base_model="m", data=Path("d"), adapter_path=Path("a"), budget=LoRaBudget()
    )
    assert cmd[0] == "mlx_lm.lora"
    assert "--train" in cmd
    assert "--mask-prompt" in cmd
    assert cmd[cmd.index("--num-layers") + 1] == "8"


def test_the_fireworks_adapter_refuses_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training is gated. There is no credential on this machine, so this is
    the only behaviour of it that can be verified here."""
    monkeypatch.delenv(FIREWORKS_KEY_ENV, raising=False)
    with pytest.raises(CompileError, match="cannot submit a job"):
        require_fireworks_key()


def test_the_fireworks_payload_shape_is_documented_not_tested() -> None:
    body = fireworks_payload(
        base_model="qwen2.5-7b",
        dataset_id="ds_1",
        budget=LoRaBudget(backend="fireworks"),
    )
    assert body["base_model"] == "qwen2.5-7b"
    assert body["lora_rank"] == 8
    assert set(body) >= {"base_model", "dataset", "lora_rank"}


def test_the_scratch_dial_offers_only_real_sizes() -> None:
    """nanochat's lesson: one dial, and it must be a dial that exists."""
    assert [s.depth for s in SCRATCH_SIZES] == [12, 20, 26]
    assert scratch_plan(26).size.label == "gpt2"
    with pytest.raises(CompileError, match="not on the dial"):
        scratch_plan(99)


def test_the_published_cost_is_stated_up_front() -> None:
    """Renting GPUs is a spending decision, so the number comes with the plan."""
    line = scratch_plan(26).cost_line
    assert "48" in line
    assert "15" in line
    assert "GPU-hours" in line


def test_the_stages_are_ordered_and_complete() -> None:
    """Each stage consumes the previous artifact, so none is optional."""
    plan = scratch_plan(20)
    assert plan.stages == ("tokenizer", "pretrain", "evaluate", "supervise", "chat")
    assert len(plan.commands) == len(plan.stages)
    assert "--depth=20" in plan.commands[1]


def test_scratch_training_is_gated() -> None:
    """The LoRA path is minutes and free. This one is neither."""
    with pytest.raises(CompileError, match="explicit approval"):
        require_scratch_approval(approved=False, provider="rented-8xh100")
    assert require_scratch_approval(approved=True, provider="x") is None
