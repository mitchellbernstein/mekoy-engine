from pathlib import Path

from mekoy.compile import Budget, SearchSpace, compile_system, format_report
from mekoy.dataset import ExampleRecord, Split, load_examples, split_examples
from mekoy.search import Slos, search

_FIXTURE = Path("examples/bucko-restaurant/examples.jsonl")
_GENERATED = Path("examples/bucko-restaurant/generated.jsonl")


class _GoldEcho:
    """Returns gold JSON for whichever labelled text appears in the prompt."""

    def __init__(self, rows: tuple[ExampleRecord, ...]) -> None:
        self._by_text: dict[str, str] = {
            row.text: row.outcome.model_dump_json() for row in rows
        }
        self.prompts: list[str] = []

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
        self.prompts.append(user)
        # Shots carry their own labels, so read only the final target document.
        tail = user.rsplit("Text:\n", 1)[-1]
        target = tail.rsplit("\nJSON:", 1)[0]
        return self._by_text.get(target, "{}")


def _split() -> Split:
    return split_examples(load_examples(_FIXTURE))


def test_split_is_three_way() -> None:
    split = _split()
    assert len(split.train) == 8
    assert len(split.dev) == 2
    assert len(split.test) == 2
    # Nothing appears in two places.
    texts = [r.text for r in (*split.train, *split.dev, *split.test)]
    assert len(texts) == len(set(texts))


def test_search_never_reads_the_test_split() -> None:
    """PLAN §16.5: the search must not see the examples it is judged on."""
    split = _split()
    completer = _GoldEcho((*split.train, *split.dev, *split.test))
    _winner, _tried, _stopped = search(
        completer,
        shots=split.train,
        dev=split.dev,
        space=SearchSpace.local(train_n=len(split.train)),
        budget=Budget(trials=3),
    )
    asked = "\n".join(completer.prompts)
    for row in split.test:
        assert row.text not in asked, "search read a test example"


def test_compile_selects_on_dev_and_reports_test_separately() -> None:
    split = _split()
    report = compile_system(
        _GoldEcho((*split.train, *split.dev, *split.test)), split, SearchSpace.single()
    )
    assert report.winner.quality == 1.0
    assert len(report.winner.scores) == len(split.dev)
    assert report.test.quality == 1.0
    assert len(report.test.scores) == len(split.test)
    assert "selected on dev" in report.describe_selection()
    assert "training: skipped" in format_report(report)


def test_budget_caps_the_number_of_candidates() -> None:
    split = _split()
    space = SearchSpace.local(train_n=len(split.train))
    assert len(space.candidates()) > 5
    report = compile_system(
        _GoldEcho((*split.train, *split.dev, *split.test)),
        split,
        space,
        Budget(trials=5),
    )
    assert report.tried == 5


def test_single_space_is_one_candidate() -> None:
    split = _split()
    report = compile_system(
        _GoldEcho((*split.train, *split.dev, *split.test)),
        split,
        SearchSpace.single(),
    )
    assert report.tried == 1
    assert report.winner.config.k_shot == 0


def test_seeded_search_is_reproducible() -> None:
    split = _split()
    space = SearchSpace.local(train_n=len(split.train))
    completer = _GoldEcho((*split.train, *split.dev, *split.test))
    first = search(
        completer,
        shots=split.train,
        dev=split.dev,
        space=space,
        budget=Budget(trials=4, seed=7),
    )[0]
    second = search(
        completer,
        shots=split.train,
        dev=split.dev,
        space=space,
        budget=Budget(trials=4, seed=7),
    )[0]
    assert first.config == second.config


def test_staging_prunes_candidates_before_a_full_dev_pass() -> None:
    """ASHA rung: survivors alone earn the full dev evaluation.

    Uses the 96-row generated corpus: staging only engages on a dev slice big
    enough to afford a minibatch.
    """
    split = split_examples(load_examples(_GENERATED))
    space = SearchSpace.local(train_n=len(split.train))
    report = compile_system(
        _GoldEcho((*split.train, *split.dev, *split.test)),
        split,
        space,
        Budget(trials=6),
    )
    assert report.tried == 6
    assert len(split.dev) > 12
    assert report.pruned > 0
    assert report.pruned < report.tried
    # Polled candidates were measured on fewer rows than the winner was.
    rung = [t for t in report.trials if len(t.scores) < len(report.winner.scores)]
    assert rung, "no candidate was priced on a minibatch"


def test_unstaged_search_evaluates_every_candidate_fully() -> None:
    split = _split()
    space = SearchSpace.local(train_n=len(split.train))
    report = compile_system(
        _GoldEcho((*split.train, *split.dev, *split.test)),
        split,
        space,
        Budget(trials=3, staged=False),
    )
    assert report.pruned == 0


def test_slo_clearing_candidate_stops_the_search() -> None:
    """PLAN §15.12: stop on the SLO, do not spend the rest of the budget."""
    split = _split()
    space = SearchSpace.local(train_n=len(split.train))
    report = compile_system(
        _GoldEcho((*split.train, *split.dev, *split.test)),
        split,
        space,
        Budget(trials=8, slos=Slos(quality=0.5, cost_per_doc=1.0, latency_ms=60_000)),
    )
    assert report.stopped_early is True
    assert report.tried < 8
