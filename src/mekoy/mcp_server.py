"""Local MCP connector over stdio JSON-RPC."""

# ruff: noqa: E501
# fmt: off

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn, assert_never

from pydantic import TypeAdapter, ValidationError

from mekoy.bundle import spec_for, write_bundle
from mekoy.compile import (
    Budget,
    CompileReport,
    HarnessConfig,
    SearchSpace,
    compile_system,
    format_report,
)
from mekoy.dataset import load_examples, split_examples
from mekoy.errors import CompileError, ModelUnreachableError
from mekoy.harness import DEFAULT_PROMPT, PROMPTS, Decode, extract
from mekoy.outcome import RestaurantOutcome
from mekoy.report import render_markdown
from mekoy.runtime import Completer, OllamaCompleter
from mekoy.search import Trial
from mekoy.verify import CHECKS_SUMMARY, VerifyFail, VerifyOk, explain

type Json = dict[str, object]
_PROTOCOL, _STAMP, _URL, _MODEL = "2024-11-05", ".eval-approved", "http://127.0.0.1:11434/v1", "qwen2.5:7b"
_LOCK = "eval is not approved; call propose_eval with approve=true"
_SCHEMA_FIELDS = " ".join(RestaurantOutcome.model_fields)
#: PLAN §155: ask only for what the examples and job did not already say.
_INTAKE = (
    "intake (skip anything you already told me):",
    "  1. what counts as unacceptable, versus merely wrong?",
    "  2. constraints: local-only? cost ceiling? latency ceiling?",
    "  3. baseline to beat: which model, in which harness?",
    "  4. budget: how many candidates may I try on dev? (default 5)",
)
_TYPES = {"str": "string", "int": "integer", "bool": "boolean"}


def _die(message: str) -> NoReturn:
    raise CompileError(message=message)


def _schema(desc: str, fields: str, req: str) -> Json:
    props: Json = {}
    for part in fields.split(",") if fields else []:
        name, typ = part.split(":")
        spec: Json = {"type": _TYPES.get(typ, typ)}
        if name == "mode":
            spec["enum"] = ["hosted", "self_host", "download"]
        props[name] = spec
    out: Json = {"type": "object", "description": desc, "properties": props}
    if req:
        out["required"] = req.split(",")
    return out


_SPEC = (
    ("inspect_task", "Draft a spec from a job and examples.", "job:str,examples:str", "job,examples"),
    ("propose_eval", "Propose checks. Locked until approve=true.", "examples:str,approve:bool", "examples"),
    ("compile_system", "Search on dev, report on test. Needs an approved eval.", "examples:str,model:str,base_url:str,quick:bool,trials:int", "examples"),
    ("get_compile_status", "Poll compile status.", "examples:str", "examples"),
    ("get_report", "Winning config. Training reported as skipped.", "examples:str", "examples"),
    ("deploy_system", "hosted, self_host, or download. Not hosted here.", "examples:str,mode:str", "examples,mode"),
    ("invoke_system", "Extract one document.", "text:str,examples:str,k_shot:int,retries:int,model:str,base_url:str", "text"),
    ("list_systems", "List Systems seen in this session.", "", ""),
)
TOOLS = {n: _schema(d, f, r) for n, d, f, r in _SPEC}
_JSON_OBJ: TypeAdapter[Json] = TypeAdapter(dict[str, object])


def _ok(req_id: object, result: object) -> Json:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: object, code: int, message: str) -> Json:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _as_obj(value: object) -> Json:
    if not isinstance(value, dict):
        return {}
    return _JSON_OBJ.validate_python(value)


def _counts(split: object) -> str:
    """Show train / dev / test counts so every card restates where selection happens."""
    s = split
    return (
        f"{len(s.train)} train / {len(s.dev)} dev / {len(s.test)} test"  # type: ignore[attr-defined]
    )


def _arg_str(args: Json, key: str, default: str | None = None) -> str:
    val = args.get(key, default)
    if isinstance(val, str) and val != "":
        return val
    if default is not None and key not in args:
        return default
    _die(f"{key} is required")


def _arg_int(args: Json, key: str, default: int) -> int:
    val = args.get(key, default)
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    _die(f"{key} must be an integer")


def _path(args: Json) -> Path:
    return Path(_arg_str(args, "examples")).expanduser().resolve()


def _stamp(examples: Path) -> Path:
    return examples.parent / _STAMP


@dataclass(slots=True)
class _Record:
    examples: Path
    job: str
    report: str | None = None
    config: HarnessConfig | None = None
    trial: Trial | None = None


@dataclass(slots=True)
class Server:
    """In-process MCP session. Inject a completer so tests skip Ollama."""

    completer: Completer | None = None
    _systems: dict[str, _Record] = field(default_factory=dict)

    def _lm(self, args: Json) -> Completer:
        if self.completer is not None:
            return self.completer
        return OllamaCompleter(base_url=_arg_str(args, "base_url", _URL), model=_arg_str(args, "model", _MODEL))

    def handle_line(self, line: str) -> str | None:
        """Parse one stdio line and encode the JSON-RPC response."""
        try:
            got = self.handle(_JSON_OBJ.validate_python(json.loads(line)))
        except (json.JSONDecodeError, ValidationError):
            return json.dumps(_err(None, -32700, "Parse error"))
        return json.dumps(got) if got is not None else None

    def handle(self, message: Json) -> Json | None:
        """Return a JSON-RPC response, or None for a notification."""
        method = message.get("method")
        if not isinstance(method, str):
            return _err(message.get("id"), -32600, "Invalid Request")
        if "id" not in message:
            return None
        req_id, params = message.get("id"), _as_obj(message.get("params", {}))
        if method not in {"initialize", "tools/list", "tools/call"}:
            return _err(req_id, -32601, f"Method not found: {method}")
        try:
            result = self._dispatch(method, params)
        except (CompileError, ModelUnreachableError) as exc:
            if method == "tools/call":
                return _ok(req_id, {"content": [{"type": "text", "text": str(exc)}], "isError": True})
            return _err(req_id, -32603, str(exc))
        return _ok(req_id, result)

    def _dispatch(self, method: str, params: Json) -> object:
        if method == "initialize":
            return {"protocolVersion": _PROTOCOL, "capabilities": {"tools": {}}, "serverInfo": {"name": "mekoy", "version": "0.1.0"}}
        if method == "tools/list":
            return {"tools": [{"name": n, "description": s["description"], "inputSchema": s} for n, s in TOOLS.items()]}
        name = params.get("name")
        if not isinstance(name, str) or name not in TOOLS:
            _die(f"unknown tool: {name}")
        return {"content": [{"type": "text", "text": self._run(name, _as_obj(params.get("arguments", {})))}]}

    def _run(self, name: str, args: Json) -> str:
        table = {
            "inspect_task": self._inspect, "propose_eval": self._propose,
            "compile_system": self._compile, "get_compile_status": self._status,
            "get_report": self._report, "deploy_system": self._deploy,
            "invoke_system": self._invoke, "list_systems": self._list,
        }
        return table[name](args)

    def _inspect(self, args: Json) -> str:
        job, examples = _arg_str(args, "job"), _path(args)
        rows = load_examples(examples)
        split = split_examples(rows)
        self._systems[str(examples)] = _Record(examples, job)
        return "\n".join(
            (
                f"job={job}",
                f"examples={examples}",
                f"{len(rows)} examples -> {_counts(split)}",
                f"schema: {_SCHEMA_FIELDS}",
                *_INTAKE,
                "next: propose_eval. Compile stays locked until approve=true.",
            )
        )

    def _propose(self, args: Json) -> str:
        examples = _path(args)
        split = split_examples(load_examples(examples))
        _ = self._systems.setdefault(str(examples), _Record(examples, ""))
        if args.get("approve") is True:
            _ = _stamp(examples).write_text("approved\n", encoding="utf-8")
        nxt = "Eval approved. You may call compile_system." if _stamp(examples).is_file() else _LOCK
        return (
            f"{_counts(split)}\nchecks: {CHECKS_SUMMARY}\n"
            "selection: dev; test is scored once and never read by the search\n" + nxt
        )

    def _compile(self, args: Json) -> str:
        examples = _path(args)
        if not _stamp(examples).is_file():
            _die(_LOCK)
        split = split_examples(load_examples(examples))
        space = (
            SearchSpace.single()
            if args.get("quick") is True
            else SearchSpace.local(train_n=len(split.train))
        )
        rec = self._systems.setdefault(str(examples), _Record(examples, ""))
        compiled = compile_system(
            self._lm(args),
            split,
            space,
            Budget(trials=_arg_int(args, "trials", 5)),
        )
        rec.report = format_report(compiled)
        rec.config = compiled.winner.config
        rec.trial = compiled.test
        _ = (examples.parent / "compile-report.txt").write_text(
            rec.report + "\n", encoding="utf-8"
        )
        _ = (examples.parent / "compile-report.md").write_text(
            render_markdown(compiled, task=rec.job), encoding="utf-8"
        )
        return rec.report

    def _status(self, args: Json) -> str:
        examples = _path(args)
        rec, ok = self._systems.get(str(examples)), _stamp(examples).is_file()
        if rec is not None and rec.report:
            return f"status=done approved={ok} {rec.report.splitlines()[0]}"
        if ok:
            return "status=not_started approved=True eval approved"
        return "status=blocked approved=False eval is not approved"

    def _report(self, args: Json) -> str:
        examples = _path(args)
        rec = self._systems.get(str(examples))
        if rec is not None and rec.report:
            return rec.report
        path = examples.parent / "compile-report.txt"
        return path.read_text(encoding="utf-8") if path.is_file() else _die("no report; call compile_system after eval approval")

    def _deploy(self, args: Json) -> str:
        mode = _arg_str(args, "mode")
        if mode in {"hosted", "self_host"}:
            return "not hosted; use download/invoke"
        if mode != "download":
            _die("mode must be hosted, self_host, or download")
        examples = _path(args)
        rec = self._systems.get(str(examples))
        if rec is None or rec.report is None or rec.trial is None:
            _die("no compile yet; call compile_system first")
        compiled = CompileReport(
            winner=rec.trial,
            trials=(rec.trial,),
            test=rec.trial,
            stopped_early=False,
        )
        spec = spec_for(
            compiled,
            task=rec.job or "restaurant call extraction",
            model_id=_arg_str(args, "model", _MODEL),
        )
        out = write_bundle(examples.parent / "bundle", spec, rec.report)
        return (
            f"bundle written: {out} "
            "(spec.json, report.txt, README.md, docker-compose.yml)"
        )

    def _invoke(self, args: Json) -> str:
        """Extract one document, replaying the compiled winner's harness."""
        winner = (self._systems.get(str(_path(args))) or _Record(_path(args), "")).config
        shots: tuple[tuple[str, RestaurantOutcome], ...] = ()
        k_shot = _arg_int(args, "k_shot", -1)
        if k_shot < 0 and winner is not None:
            k_shot = winner.k_shot
        retries = args.get("retries")
        if not isinstance(retries, int) or isinstance(retries, bool):
            retries = winner.retries if winner is not None else 1
        if k_shot > 0:
            rows = load_examples(_path(args))
            shots = tuple((row.text, row.outcome) for row in rows[:k_shot])
        decode = Decode(system=PROMPTS[DEFAULT_PROMPT], constrained=True)
        if winner is not None:
            decode = Decode(
                system=PROMPTS.get(winner.prompt, PROMPTS[DEFAULT_PROMPT]),
                constrained=winner.constrained,
            )
        result = extract(
            self._lm(args),
            text=_arg_str(args, "text"),
            shots=shots,
            retries=retries,
            decode=decode,
        )
        match result:
            case VerifyOk(outcome=outcome):
                return outcome.model_dump_json()
            case VerifyFail():
                _die(f"verify failed: {explain(result)}")
            case _ as unreachable:
                assert_never(unreachable)

    def _list(self, args: Json) -> str:
        del args
        rows = [
            f"{r.examples} approved={_stamp(r.examples).is_file()} report={r.report is not None} {r.job}"
            for r in self._systems.values()
        ]
        return "\n".join(rows) if rows else "(none)"


def main() -> None:
    """Read JSON-RPC lines from stdin and write responses to stdout."""
    server = Server()
    for raw in sys.stdin:
        if (line := raw.strip()) and (reply := server.handle_line(line)):
            _ = sys.stdout.write(reply + "\n")
            _ = sys.stdout.flush()


if __name__ == "__main__":
    main()
