from pathlib import Path

import pytest

from mekoy.compile import SearchSpace, compile_system
from mekoy.dataset import load_examples, split_examples
from mekoy.openai_runtime import OpenAICompleter
from mekoy.privacy import ClosedApiError, assert_local, is_local
from mekoy.runtime import OllamaCompleter


class _Fake:
    """An unmarked completer: must be treated as closed."""

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


def test_local_runtime_is_local() -> None:
    completer = OllamaCompleter(base_url="http://127.0.0.1:11434/v1", model="m")
    assert is_local(completer) is True
    assert_local(completer)


def test_closed_runtime_is_not_local() -> None:
    completer = OpenAICompleter(api_key="k", model="gpt")
    assert is_local(completer) is False
    with pytest.raises(ClosedApiError):
        assert_local(completer)


def test_unmarked_completer_fails_closed() -> None:
    assert is_local(_Fake()) is False
    with pytest.raises(ClosedApiError):
        assert_local(_Fake())


def test_explicit_opt_in_allows_a_closed_runtime() -> None:
    assert_local(OpenAICompleter(api_key="k", model="gpt"), allow_closed=True)


def test_compile_refuses_a_closed_completer() -> None:
    rows = load_examples(Path("examples/bucko-restaurant/examples.jsonl"))
    with pytest.raises(ClosedApiError):
        compile_system(_Fake(), split_examples(rows), SearchSpace.single())
