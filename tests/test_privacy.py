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


def test_a_closed_endpoint_cannot_hide_behind_a_local_class() -> None:
    """The hole this test exists for: every OllamaCompleter declares itself local.

    The gate read that declaration and nothing else, so pointing the engine at a closed
    API - `--base-url https://api.openai.com/v1` - passed the check and that model's
    output became compile data. It was enforced by which class was constructed rather
    than by where it connects, on the one rule this project calls non-negotiable.
    """
    for remote in (
        "https://api.openai.com/v1",
        "https://api.anthropic.com/v1",
        "https://generativelanguage.googleapis.com/v1",
        "https://someones-box.example.com/v1",
    ):
        completer = OllamaCompleter(base_url=remote, model="m")
        assert not is_local(completer), remote
        with pytest.raises(ClosedApiError):
            assert_local(completer)


def test_a_genuinely_local_endpoint_is_allowed() -> None:
    """The fix must not block the runtime the engine ships with."""
    for local in (
        "http://127.0.0.1:11434/v1",
        "http://localhost:11434/v1",
        "http://[::1]:11434/v1",
        "http://192.168.1.50:11434/v1",
        "http://workstation.local:11434/v1",
    ):
        assert is_local(OllamaCompleter(base_url=local, model="m")), local
    assert_local(OllamaCompleter(base_url="http://127.0.0.1:11434/v1", model="m"))


def test_an_unmarked_completer_still_fails_closed() -> None:
    """A new runtime that forgets to declare itself must not become compile data."""

    class Unmarked:
        def __init__(self) -> None:
            self.base_url = "http://127.0.0.1:11434/v1"

    assert not is_local(Unmarked())
    with pytest.raises(ClosedApiError):
        assert_local(Unmarked())


def test_allow_closed_is_still_an_explicit_opt_in() -> None:
    """A baseline score is the one legal use, and it stays possible."""
    assert_local(
        OllamaCompleter(base_url="https://api.openai.com/v1", model="m"),
        allow_closed=True,
    )
