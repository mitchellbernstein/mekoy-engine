"""Privacy gate: no closed-API output may enter a compile.

`AGENTS.md` bans closed-model distillation outright — no GPT / Claude / Gemini
output in SFT, DPO, RL, or LoRA data. PLAN §41.24 asks for the same rule as a
flag. This module makes it a runtime check rather than a paragraph.

A closed API stays legal in exactly one place: a bake-off score baseline.
"""

from __future__ import annotations

from mekoy.errors import CompileError

__all__ = ["ClosedApiError", "assert_compile_safe", "assert_local", "is_local"]


class ClosedApiError(CompileError):
    """A compile was pointed at a closed API without an explicit opt-in."""


def is_local(completer: object) -> bool:
    """Report whether a completer runs on hardware the customer controls.

    Unmarked completers count as closed. Failing closed matters: a new runtime
    that forgets to declare itself must not silently become allowed compile data.
    """
    return getattr(completer, "local", None) is True


def assert_local(completer: object, *, allow_closed: bool = False) -> None:
    """Refuse closed APIs in a compile unless the caller opted in explicitly."""
    if is_local(completer) or allow_closed:
        return
    msg = (
        f"{type(completer).__name__} is not marked local, so its output cannot "
        "be compile data (AGENTS.md: no closed-model distillation). Pass "
        "allow_closed=True only if you accept that, or point at a local runtime."
    )
    raise ClosedApiError(message=msg)


def assert_compile_safe(completer: object, *, allow_closed: bool = False) -> None:
    """Alias used at the compile boundary."""
    assert_local(completer, allow_closed=allow_closed)
