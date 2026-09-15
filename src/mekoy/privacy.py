"""Privacy gate: no closed-API output may enter a compile.

`AGENTS.md` bans closed-model distillation outright — no GPT / Claude / Gemini
output in SFT, DPO, RL, or LoRA data. This module makes it a runtime check rather
than a paragraph.

A closed API stays legal in exactly one place: a bake-off score baseline.
"""

from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlparse

from mekoy.errors import CompileError

__all__ = ["ClosedApiError", "assert_compile_safe", "assert_local", "is_local"]

#: Hostnames that mean "this machine" without needing a lookup.
_LOCAL_NAMES = frozenset({"localhost", "127.0.0.1", "::1", "host.docker.internal"})


class ClosedApiError(CompileError):
    """A compile was pointed at a closed API without an explicit opt-in."""


#: Attribute names a completer may store its endpoint under. Public and private both,
#: because `OllamaCompleter` keeps `_base_url` and reading only the public name made
#: every remote endpoint look like an in-process runtime.
_URL_ATTRS = ("base_url", "_base_url", "url", "_url")


def _host_of(completer: object) -> str | None:
    """The host a completer will actually connect to, if it will say."""
    raw: object = None
    for name in _URL_ATTRS:
        raw = getattr(completer, name, None)
        if isinstance(raw, str) and raw.strip():
            break
        raw = None
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return urlparse(raw).hostname or None
    except ValueError:
        return None


def _names_an_endpoint(completer: object) -> bool:
    """True when the completer carries a URL at all, even an unreadable one."""
    return any(hasattr(completer, name) for name in _URL_ATTRS)


def _host_is_local(host: str) -> bool:
    """True for loopback, and for the private ranges a home or office network uses."""
    if host in _LOCAL_NAMES or host.endswith((".local", ".internal")):
        return True
    try:
        address = ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def is_local(completer: object) -> bool:
    """Report whether a completer runs on hardware the customer controls.

    Two things must hold: the completer declares itself local, **and** its base URL
    points somewhere local. Checking only the declaration was a real hole — every
    `OllamaCompleter` sets `local = True` regardless of what it is pointed at, so
    `--base-url https://api.openai.com/v1` passed this gate and the closed model's
    output became compile data. That is the one rule this project calls
    non-negotiable, and it was enforced by which class you constructed rather than by
    where it connects.

    A completer with no `base_url` attribute at all is an in-process runtime with no
    destination to inspect, so its declaration is taken at face value. Anything that
    names a remote host counts as closed, however it is labelled.
    """
    if getattr(completer, "local", None) is not True:
        return False
    host = _host_of(completer)
    if host is None:
        return not _names_an_endpoint(completer)
    return _host_is_local(host)


def assert_local(completer: object, *, allow_closed: bool = False) -> None:
    """Refuse closed APIs in a compile unless the caller opted in explicitly."""
    if is_local(completer) or allow_closed:
        return
    host = _host_of(completer)
    where = (
        f"points at {host!r}, which is not this machine"
        if host is not None
        else "is not marked local"
    )
    msg = (
        f"{type(completer).__name__} {where}, so its output cannot be compile data "
        "(AGENTS.md: no closed-model distillation). Pass allow_closed=True only if "
        "you accept that, or point at a local runtime."
    )
    raise ClosedApiError(message=msg)


def assert_compile_safe(completer: object, *, allow_closed: bool = False) -> None:
    """Alias used at the compile boundary."""
    assert_local(completer, allow_closed=allow_closed)
