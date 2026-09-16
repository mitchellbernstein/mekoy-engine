import json
import pathlib
from pathlib import Path

import pytest

from mekoy.api.store import Store, phase_of
from mekoy.dataset import ExampleRecord, load_examples, split_examples
from mekoy.errors import CompileError
from mekoy.mcp_server import TOOLS, Server, _path, _Record
from mekoy.outcome import RestaurantOutcome
from mekoy.search import HarnessConfig
from mekoy.verify import CHECKS_SUMMARY

_FIXTURE = Path("examples/bucko-restaurant/examples.jsonl")
_NAMES = (
    "inspect_task",
    "ask_questions",
    "record_answers",
    "propose_eval",
    "compile_system",
    "get_compile_status",
    "get_report",
    "compare_models",
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
    # The ask surface must not act, so neither ask tool requires a file to exist and
    # record_answers needs only the two answers a compile cannot proceed without.
    assert "required" not in TOOLS["ask_questions"]
    assert TOOLS["record_answers"].get("required") == ["dangerous", "good_enough"]
    assert TOOLS["compare_models"].get("required") == ["other_model", "examples"]


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


def test_a_system_compiled_through_the_connector_reaches_the_shared_store(
    tmp_path: Path,
) -> None:
    """Both doors have to describe one world.

    A System compiled through the connector used to live only in the MCP process's own
    dict, so the HTTP API could not see it and neither surface knew about the other's
    work. With a sink attached, the compile is published where the API can find it.
    """
    store = Store()
    server = Server(completer=_CountingCompleter(), sink=store)
    path = _three_rows(tmp_path)

    _ = server._run("propose_eval", {"examples": str(path), "approve": True})
    out = server._run("compile_system", {"examples": str(path), "quick": True})
    assert "quality" in out

    key = str(_path({"examples": str(path)}))
    system_id = server._systems[key].system_id
    assert system_id is not None, "the compile was not published"

    record = store.get_system(system_id)
    assert str(phase_of(record)) == "compiled"
    assert len(record.examples) == 3, "the rows travelled with it"
    assert record.winner is not None, "the winner travelled with it"


def test_the_connector_works_without_a_store(tmp_path: Path) -> None:
    """Standalone stdio use must not require a control plane to exist."""
    server = Server(completer=_CountingCompleter())
    path = _three_rows(tmp_path)
    _ = server._run("propose_eval", {"examples": str(path), "approve": True})
    out = server._run("compile_system", {"examples": str(path), "quick": True})
    assert "quality" in out
    assert server._systems[str(_path({"examples": str(path)}))].system_id is None


# --- the ask surface: six questions an agent puts to its user ---------------------


def test_ask_questions_asks_and_does_not_act(tmp_path: Path) -> None:
    """A tool that acts when it is supposed to ask compiles on a vibe.

    The ask surface must leave the eval stamp absent and write no report, because the
    whole point is that a compile waits for the answers.
    """
    examples = _copy(tmp_path)
    asked = _call(Server(_Boom()), "ask_questions", {"examples": str(examples)})
    for qid in ("job", "document", "output", "dangerous", "good_enough", "compare"):
        assert qid in asked, f"question {qid} was not asked"
    assert not (tmp_path / ".eval-approved").is_file()
    assert not (tmp_path / "compile-report.txt").is_file()


def test_ask_questions_marks_what_the_job_already_answered(tmp_path: Path) -> None:
    examples = _copy(tmp_path)
    asked = _call(
        Server(_Boom()),
        "ask_questions",
        {"job": "restaurant call outcome", "examples": str(examples)},
    )
    assert "[answered] job" in asked
    assert "[answered] document" in asked
    # The two the search cannot infer stay open no matter what arrived.
    assert "[open] dangerous" in asked
    assert "[open] good_enough" in asked


def test_ask_questions_names_the_comparison_cost(tmp_path: Path) -> None:
    """The preference has to be described honestly, or it is a hidden upsell."""
    examples = _copy(tmp_path)
    asked = _call(Server(_Boom()), "ask_questions", {"examples": str(examples)})
    assert "comparison, honestly" in asked
    assert "time" in asked.casefold()


def test_record_answers_refuses_without_the_gate_and_the_stop_rule(
    tmp_path: Path,
) -> None:
    """These two are the question set's whole point and cannot be defaulted.

    A missing dangerous answer means the gate cannot tell wrong from unacceptable; a
    missing stopping rule means the search never stops. Both are refused by name.
    """
    examples = _copy(tmp_path)
    reply = _call(
        Server(_Boom()),
        "record_answers",
        {
            "examples": str(examples),
            "job": "restaurant calls",
            "document": "transcripts",
            "output": "restaurant, intent, booked",
        },
    )
    assert '"isError": true' in reply
    assert "dangerous" in reply
    assert "good_enough" in reply
    assert not (tmp_path / ".intake.json").is_file()


def test_record_answers_persists_and_carries_the_preference(tmp_path: Path) -> None:
    examples = _copy(tmp_path)
    server = Server(_Boom())
    recorded = _call(
        server,
        "record_answers",
        {
            "examples": str(examples),
            "job": "restaurant calls",
            "dangerous": "saying booked when nobody took a reservation",
            "good_enough": "beat the 0.905 we get today",
            "compare": "qwen2.5:14b",
            "run_comparison": True,
        },
    )
    assert "recorded" in recorded
    assert "dangerous -> the gate" in recorded
    assert "run_comparison=true" in recorded
    assert (tmp_path / ".intake.json").is_file()
    rec = server._systems[str(_path({"examples": str(examples)}))]
    assert rec.run_comparison is True
    assert rec.answers["dangerous"].startswith("saying booked")


def test_the_dangerous_answer_reaches_the_proposed_eval(tmp_path: Path) -> None:
    """An eval that ignores the answer the user gave is not the eval they approved."""
    examples = _copy(tmp_path)
    server = Server(_Boom())
    danger = "claiming a booking when staff never took one"
    _ = _call(
        server,
        "record_answers",
        {"examples": str(examples), "dangerous": danger, "good_enough": "0.9"},
    )
    proposed = _call(server, "propose_eval", {"examples": str(examples)})
    assert danger in proposed


def test_answers_read_back_from_disk_for_a_new_session(tmp_path: Path) -> None:
    """A connector process dies between turns; the answers must outlive it."""
    examples = _copy(tmp_path)
    _ = _call(
        Server(_Boom()),
        "record_answers",
        {
            "examples": str(examples),
            "dangerous": "booked when nobody booked",
            "good_enough": "0.9",
        },
    )
    asked = _call(Server(_Boom()), "ask_questions", {"examples": str(examples)})
    assert "[answered] dangerous" in asked
    assert "[answered] good_enough" in asked


def test_compare_models_refuses_without_a_compile(tmp_path: Path) -> None:
    examples = _copy(tmp_path)
    reply = _call(
        Server(_Boom()),
        "compare_models",
        {"examples": str(examples), "other_model": "qwen2.5:14b"},
    )
    assert '"isError": true' in reply
    assert "no compile yet" in reply


def test_compare_models_scores_both_lanes_on_the_same_rows(tmp_path: Path) -> None:
    """The comparison is only worth showing when both lanes saw the same held-out rows.

    A comparison that measured one lane would be a claim with nothing behind it, so the
    card's row count must match the test split and both lanes must have run. Model calls
    are counted: the search scores dev only, so the extra calls are the two lanes.
    """
    path = _three_rows(tmp_path)
    server = Server(completer=_CountingCompleter())
    _ = server._run("propose_eval", {"examples": str(path), "approve": True})
    _ = server._run("compile_system", {"examples": str(path), "quick": True})

    n_test = len(split_examples(load_examples(path)).test)
    counter = _CountingCompleter()
    server.completer = counter
    card = server._run(
        "compare_models", {"examples": str(path), "other_model": "qwen2.5:14b"}
    )
    assert "only the model changes" in card
    assert "qwen2.5:14b" in card
    assert f"held-out rows: {n_test} (test, scored once)" in card
    # Two lanes, one document each on a one-row test slice.
    assert len(counter.prompts) == 2 * n_test, counter.prompts
    assert (tmp_path / "model-comparison.txt").is_file()
