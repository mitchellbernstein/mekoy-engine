from pathlib import Path

from mekoy.compile import Budget, CompileReport, SearchSpace, compile_system
from mekoy.dataset import ExampleRecord, load_examples, split_examples
from mekoy.report import render_markdown

_FIXTURE = Path("examples/bucko-restaurant/examples.jsonl")


class _GoldEcho:
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
        tail = user.rsplit("Text:\n", 1)[-1]
        return self._by_text.get(tail.rsplit("\nJSON:", 1)[0], "{}")


def _report() -> CompileReport:
    rows = load_examples(_FIXTURE)
    split = split_examples(rows)
    return compile_system(
        _GoldEcho(rows),
        split,
        SearchSpace.local(train_n=len(split.train)),
        Budget(trials=2),
    )


def test_markdown_report_has_every_section() -> None:
    text = render_markdown(_report(), task="unit test task")
    for section in (
        "# Compile report",
        "unit test task",
        "## Winner",
        "## Selection provenance",
        "## Pareto front",
        "## Every arm measured",
        "## Notes",
        "training: skipped",
    ):
        assert section in text, section


def test_markdown_report_states_test_and_dev_separately() -> None:
    text = render_markdown(_report())
    assert "dev quality:" in text
    assert "test quality:" in text
    assert "strict test quality:" in text


def test_markdown_report_lists_pruned_arms() -> None:
    text = render_markdown(_report())
    assert "pruned at the minibatch rung:" in text
