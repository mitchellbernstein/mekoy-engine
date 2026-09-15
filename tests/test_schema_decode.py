"""Schema-constrained decoding: the wire shape, and how servers differ."""

from pathlib import Path

import pytest

from mekoy.dataset import load_examples, split_examples
from mekoy.errors import CompileError, ModelUnreachableError
from mekoy.harness import Decode, extract
from mekoy.outcome import RestaurantOutcome
from mekoy.runtime import OllamaCompleter
from mekoy.search import HarnessConfig, SearchSpace, evaluate
from mekoy.tasks import BANKING77, RECEIPT, RESTAURANT

_ROW = RestaurantOutcome(
    restaurant="Uchi",
    intent="availability",
    status="confirmed",
    party_size=2,
    when="Friday",
    under_name=None,
    evidence="A table for two is open Friday.",
    booked=False,
)


class _Recorded:
    """Captures what the harness actually asked the server for."""

    local: bool = True

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        *,
        system: str,
        user: str,
        constrained: bool = True,
        temperature: float = 0.0,
        schema: dict[str, object] | None = None,
    ) -> str:
        del system, user, temperature
        self.calls.append({"constrained": constrained, "schema": schema is not None})
        return _ROW.model_dump_json()


def test_the_schema_axis_is_a_b_a_ed() -> None:
    """Always A/B unconstrained. That applies to schemas too."""
    space = SearchSpace.for_task(RESTAURANT, train_n=20)
    assert space.schema == (False, True)
    labels = {c.label.split()[2] for c in space.candidates()}
    assert labels == {"grammar", "free", "schema"}


def test_a_smoke_space_stays_one_candidate() -> None:
    assert len(SearchSpace.single().candidates()) == 1


def test_a_schema_label_says_so() -> None:
    assert HarnessConfig(k_shot=4, schema=True).label.split()[2] == "schema"
    assert HarnessConfig(k_shot=4, constrained=True).label.split()[2] == "grammar"
    assert HarnessConfig(k_shot=4, constrained=False).label.split()[2] == "free"


def test_the_task_schema_is_sent_when_asked_for() -> None:
    completer = _Recorded()
    _ = extract(
        completer,
        text="x",
        retries=0,
        decode=Decode(schema=RESTAURANT.model.model_json_schema()),
    )
    assert completer.calls[0]["schema"] is True


def test_no_schema_is_sent_without_one() -> None:
    completer = _Recorded()
    _ = extract(completer, text="x", retries=0, decode=Decode())
    assert completer.calls[0]["constrained"] is True
    assert completer.calls[0]["schema"] is False


def test_a_schema_is_only_sent_when_decoding_is_constrained() -> None:
    """Unconstrained means unconstrained: a schema would re-constrain it."""
    rows = load_examples(Path("examples/bucko-restaurant/examples.jsonl"))
    split = split_examples(rows)
    completer = _Recorded()
    cfg = HarnessConfig(k_shot=0, retries=0, constrained=False, schema=True)
    _ = evaluate(completer, split.dev, cfg, task=RESTAURANT)
    assert all(call["schema"] is False for call in completer.calls)


def test_every_task_exposes_a_json_schema() -> None:
    """Schema decoding is only available if the task can describe itself."""
    for task in (RESTAURANT, RECEIPT, BANKING77):
        schema = task.model.model_json_schema()
        assert schema.get("type") == "object", task.name
        assert schema.get("properties"), task.name


def test_the_wire_shape_matches_what_servers_expect() -> None:
    """llama.cpp, vLLM, and OpenAI all take this shape; Ollama accepts and may
    ignore it, which is why the search decides rather than a default."""
    schema = RESTAURANT.model.model_json_schema()
    body: dict[str, object] = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "outcome", "schema": schema},
        }
    }
    fmt = body["response_format"]
    assert isinstance(fmt, dict)
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "outcome"


def test_a_completer_reports_a_server_error_rather_than_masking_it() -> None:
    """A server that rejects the schema must fail loudly, not silently degrade."""
    completer = OllamaCompleter(base_url="http://127.0.0.1:9/v1", model="m")
    with pytest.raises(ModelUnreachableError):
        _ = extract(
            completer,
            text="x",
            retries=0,
            decode=Decode(schema={"type": "object"}),
        )


def test_the_model_is_a_search_axis() -> None:
    """The search samples over several open models, and model comes first."""
    space = SearchSpace.for_task(RESTAURANT, train_n=10)
    assert space.models == ("",), "one model must behave exactly as before"
    with_models = SearchSpace(
        k_shots=(0,),
        retries=(0,),
        constrained=(True,),
        schema=(False,),
        models=("", "big"),
    )
    labels = [c.label for c in with_models.candidates()]
    assert labels == ["k=0 r=0 grammar default", "big k=0 r=0 grammar default"]


def test_a_candidate_without_a_registered_completer_is_refused() -> None:
    """Silently scoring another model's output would misattribute the numbers."""
    rows = load_examples(Path("examples/bucko-restaurant/generated.jsonl"))
    split = split_examples(rows)
    cfg = HarnessConfig(k_shot=0, retries=0, model="not-registered")
    with pytest.raises(CompileError, match="no completer was registered"):
        evaluate(
            _Recorded(), split.dev, cfg, task=RESTAURANT, models={"other": _Recorded()}
        )


def test_the_registered_completer_is_used_for_its_model() -> None:
    rows = load_examples(Path("examples/bucko-restaurant/generated.jsonl"))
    split = split_examples(rows)
    chosen = _Recorded()
    cfg = HarnessConfig(k_shot=0, retries=0, model="big")
    _ = evaluate(
        _Recorded(),
        split.dev,
        cfg,
        task=RESTAURANT,
        models={"big": chosen},
    )
    assert chosen.calls, "the registered completer for this model was not used"
