import json
import pathlib
from pathlib import Path

import pytest

from mekoy.dataset import ExampleRecord, load_examples
from mekoy.errors import CompileError
from mekoy.mcp_server import TOOLS, Server, _path, _Record
from mekoy.outcome import RestaurantOutcome
from mekoy.search import HarnessConfig
from mekoy.verify import CHECKS_SUMMARY

_FIXTURE = Path("examples/bucko-restaurant/examples.jsonl")
_NAMES = (
    "inspect_task",
    "propose_eval",
    "compile_system",
    "get_compile_status",
    "get_report",
    "deploy_system",
    "invoke_system",
    "list_systems",
)


class _Boom:
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
        raise AssertionError


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
        for text, raw in self._by_text.items():
            if text in user:
                return raw
        return "{}"


def _copy(tmp_path: Path) -> Path:
    dest = tmp_path / "examples.jsonl"
    _ = dest.write_text(_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def _call(server: Server, name: str, arguments: dict[str, object]) -> str:
    reply = server.handle_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
    )
    assert reply is not None
    return reply


def test_tool_schemas() -> None:
    assert tuple(TOOLS) == _NAMES
    blob = json.dumps(TOOLS).casefold()
    for word in ("train_model", "lora", "gepa", "api_key"):
        assert word not in blob
    for name, schema in TOOLS.items():
        assert schema.get("type") == "object"
        assert isinstance(schema.get("properties"), dict)
        del name
    deploy = json.dumps(TOOLS["deploy_system"])
    assert "hosted" in deploy
    assert "self_host" in deploy
    assert "download" in deploy
    assert TOOLS["inspect_task"].get("required") == ["job", "examples"]
    assert TOOLS["compile_system"].get("required") == ["examples"]
    assert TOOLS["invoke_system"].get("required") == ["text"]


def test_tools_list_matches_registry() -> None:
    reply = Server().handle_line(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    )
    assert reply is not None
    for name in _NAMES:
        assert name in reply
    assert "train_model" not in reply
    assert "inputSchema" in reply


def test_compile_refused_without_approve(tmp_path: Path) -> None:
    examples = _copy(tmp_path)
    reply = _call(Server(_Boom()), "compile_system", {"examples": str(examples)})
    assert '"isError": true' in reply
    assert "not approved" in reply.casefold()
    assert not (tmp_path / ".eval-approved").is_file()


def test_propose_eval_without_approve_keeps_compile_locked(tmp_path: Path) -> None:
    examples = _copy(tmp_path)
    server = Server(_Boom())
    proposed = _call(server, "propose_eval", {"examples": str(examples)})
    assert "not approved" in proposed.casefold()
    assert "approve=true" in proposed
    assert not (tmp_path / ".eval-approved").is_file()
    compile_reply = _call(server, "compile_system", {"examples": str(examples)})
    assert '"isError": true' in compile_reply
    assert "not approved" in compile_reply.casefold()


def test_propose_eval_approve_unlocks_compile(tmp_path: Path) -> None:
    examples = _copy(tmp_path)
    rows = load_examples(examples)
    server = Server(_GoldEcho(rows))
    approved = _call(
        server, "propose_eval", {"examples": str(examples), "approve": True}
    )
    assert "Eval approved" in approved
    assert (tmp_path / ".eval-approved").is_file()
    compiled = _call(
        server, "compile_system", {"examples": str(examples), "quick": True}
    )
    assert '"isError": true' not in compiled
    assert "winner" in compiled
    assert "training: skipped" in compiled


def test_eval_card_checks_match_the_enforced_rules(tmp_path: Path) -> None:
    """The card must quote the live gate, not a stale receipt-era string."""
    examples = _copy(tmp_path)
    proposed = _call(Server(_Boom()), "propose_eval", {"examples": str(examples)})
    assert CHECKS_SUMMARY in proposed
    for receipt_word in ("qty", "subtotal", "unit_price"):
        assert receipt_word not in proposed.casefold()


def test_inspect_schema_line_is_the_outcome_fields(tmp_path: Path) -> None:
    examples = _copy(tmp_path)
    inspected = _call(
        Server(),
        "inspect_task",
        {"job": "restaurant call", "examples": str(examples)},
    )
    assert f"schema: {' '.join(RestaurantOutcome.model_fields)}" in inspected


def test_deploy_hosted_is_not_hosted(tmp_path: Path) -> None:
    examples = _copy(tmp_path)
    reply = _call(
        Server(), "deploy_system", {"examples": str(examples), "mode": "hosted"}
    )
    assert "not hosted; use download/invoke" in reply
    self_host = _call(
        Server(), "deploy_system", {"examples": str(examples), "mode": "self_host"}
    )
    assert "not hosted; use download/invoke" in self_host


def test_inspect_and_list(tmp_path: Path) -> None:
    examples = _copy(tmp_path)
    server = Server()
    inspected = _call(
        server,
        "inspect_task",
        {"job": "extract restaurant call outcomes", "examples": str(examples)},
    )
    assert "extract restaurant call outcomes" in inspected
    assert "propose_eval" in inspected
    listed = _call(server, "list_systems", {})
    assert str(examples.resolve()) in listed


def _three_rows(tmp_path: Path) -> Path:
    """A three-row example file, which is the minimum a compile accepts."""
    rows = []
    for i, name in enumerate(["Uchi", "Loro", "Sway"]):
        rows.append(
            {
                "text": f"{name}, table for {i + 2} Friday 7pm under Maya.",
                "outcome": {
                    "restaurant": name,
                    "intent": "reservation",
                    "status": "confirmed",
                    "party_size": i + 2,
                    "when": "Friday 7pm",
                    "under_name": "Maya",
                    "evidence": "table for two",
                    "booked": True,
                },
            }
        )
    path = tmp_path / "examples.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    pathlib.Path(str(path) + ".eval-approved").write_text("")
    return path


class _CountingCompleter:
    """Records the prompt it was handed, so the harness it ran can be inspected."""

    local = True

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, *, system: str, user: str, **_rest: object) -> str:
        del system
        self.prompts.append(user)
        return RestaurantOutcome(
            restaurant="Uchi",
            intent="reservation",
            status="confirmed",
            party_size=2,
            when="Friday 7pm",
            under_name="Maya",
            evidence="table for two",
            booked=True,
        ).model_dump_json()


def test_invoke_replays_the_winner_shot_count(tmp_path: Path) -> None:
    """A System is the harness plus the model, so invoke must run the harness.

    The shot count is the observable: if invoke replays the winner, the prompt carries
    exactly as many examples as the winner asked for. It must scale with the winner, so
    a constant would fail this.
    """
    path = _three_rows(tmp_path)
    key = str(_path({"examples": str(path)}))

    for wanted in (1, 3):
        completer = _CountingCompleter()
        server = Server(completer=completer)
        record = server._systems.setdefault(key, _Record(pathlib.Path(key), "job"))
        record.config = HarnessConfig(k_shot=wanted, retries=0, constrained=True)
        _ = server._run(
            "invoke_system", {"examples": str(path), "text": "Loro, table for 4."}
        )
        shown = completer.prompts[0].count('"restaurant"')
        assert shown == wanted, f"winner asked for {wanted}, prompt carried {shown}"


def test_invoke_refuses_rather_than_inventing_a_harness(tmp_path: Path) -> None:
    """Silently running a default harness returns something that is not the System.

    That failure mode is worse than an error, because the answer looks fine. A caller
    who wants a one-off extraction can still say so by passing the settings explicitly.
    """
    path = _three_rows(tmp_path)

    with pytest.raises(CompileError):
        Server(completer=_CountingCompleter())._run(
            "invoke_system", {"examples": str(path), "text": "Loro, table for 4."}
        )

    completer = _CountingCompleter()
    out = Server(completer=completer)._run(
        "invoke_system",
        {
            "examples": str(path),
            "text": "Loro, table for 4.",
            "k_shot": 1,
            "retries": 0,
        },
    )
    assert "Uchi" in out
