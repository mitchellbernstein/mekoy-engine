"""Local MCP connector over stdio JSON-RPC."""

# ruff: noqa: E501
# fmt: off

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn, Protocol, assert_never

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
from mekoy.dataset import TaskExample, load_examples, split_examples
from mekoy.errors import CompileError, ModelUnreachableError
from mekoy.harness import DEFAULT_PROMPT, PROMPTS, Decode, extract
from mekoy.outcome import RestaurantOutcome
from mekoy.repoint import format_model_comparison, measure
from mekoy.report import render_markdown
from mekoy.runtime import Completer, OllamaCompleter
from mekoy.search import Trial
from mekoy.tasks import task_by_name, task_for_path
from mekoy.verify import CHECKS_SUMMARY, VerifyFail, VerifyOk, explain

type Json = dict[str, object]
_PROTOCOL, _STAMP, _URL, _MODEL = "2024-11-05", ".eval-approved", "http://127.0.0.1:11434/v1", "qwen2.5:7b"
_LOCK = "eval is not approved; call propose_eval with approve=true"
_SCHEMA_FIELDS = " ".join(RestaurantOutcome.model_fields)
_ANSWERS = ".intake.json"
#: Ask only for what the examples and job did not already say.
_INTAKE = (
    "intake (skip anything you already told me):",
    "  1. what counts as unacceptable, versus merely wrong?",
    "  2. constraints: local-only? cost ceiling? latency ceiling?",
    "  3. baseline to beat: which model, in which harness?",
    "  4. budget: how many candidates may I try on dev? (default 5)",
)
_TYPES = {"str": "string", "int": "integer", "bool": "boolean"}

#: The six questions the portal's guided flow asks as cards, in the same order. An agent
#: has no cards, so it asks these in prose and passes the answers back. The `why` line is
#: what makes an agent ask rather than guess, and `good` is what a usable answer looks
#: like - a question an agent cannot tell it got wrong is a question it will skip.
#:
#: A question already answered by the job or the examples is marked `answered` by
#: `ask_questions` rather than dropped, because an agent that sees the answer confirmed
#: stops asking, and one that sees it missing starts.
_QUESTIONS: tuple[dict[str, str], ...] = (
    {
        "id": "job",
        "question": "What is the job, in one sentence?",
        "why": "The whole compile hangs off this. A job stated as a want cannot be scored.",
        "good": "Extract a restaurant-call result from a transcript is a job; make my calls better is not.",
    },
    {
        "id": "document",
        "question": "What does one document look like, and where do examples come from?",
        "why": "The examples are the eval. Without them there is nothing to score and no compile.",
        "good": "20-200 labeled rows, or a path to a JSONL file with text plus the label.",
    },
    {
        "id": "output",
        "question": "What should come out? Name the fields.",
        "why": "The output shape is the schema the decode is constrained to and the fields that get scored.",
        "good": "A field list with types. Restaurant: restaurant, intent, status, booked.",
    },
    {
        "id": "dangerous",
        "question": "What is a dangerous answer - the one that must never happen?",
        "why": (
            "This is the question that separates a wrong answer from an unacceptable one, "
            "and it becomes the hard gate. A compile that does not know it cannot refuse "
            "the outcome a user actually fears."
        ),
        "good": (
            "Saying a table was booked when nobody took a reservation. Cheap to check, "
            "and getting it wrong is worse than answering unknown."
        ),
    },
    {
        "id": "good_enough",
        "question": "How good is good enough to stop?",
        "why": "The search stops on this, and the report is read against it. Unstated means the search has no stop condition.",
        "good": "A quality floor, or a cost or latency ceiling, or 'beat what we run today'.",
    },
    {
        "id": "compare",
        "question": "Which models should I compare this against, and should I run them first?",
        "why": (
            "One number with nothing beside it is a claim nobody can check. Running the "
            "other models on the same held-out rows costs time and proves the choice."
        ),
        "good": (
            "e.g. qwen2.5:14b alongside qwen2.5:7b. Running the comparison first costs "
            "roughly a full pass per extra model before the compile, so it is a real "
            "time cost, stated rather than hidden."
        ),
    },
)

#: Which ids are required before a compile is worth starting. The dangerous answer and
#: the stopping rule are the two the search cannot infer, so they are the two it refuses.
_REQUIRED_ANSWERS = ("dangerous", "good_enough")

#: The honest cost line for question six, so an agent repeats a number rather than
#: inventing a reassurance.
_COMPARE_COST = (
    "running a comparison measures each extra model over the same held-out rows: "
    "about one full pass per model, minutes to tens of minutes on local hardware. "
    "Skipping it is the default and the compile still runs; the comparison can be "
    "asked for afterwards with compare_models."
)


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
    (
        "ask_questions",
        "Ask the six questions a compile needs, before act. Nothing is changed.",
        "job:str,examples:str",
        "",
    ),
    (
        "record_answers",
        "Record the user's answers to ask_questions so they steer the eval and gate.",
        "job:str,document:str,output:str,dangerous:str,good_enough:str,compare:str,run_comparison:bool",
        "dangerous,good_enough",
    ),
    ("propose_eval", "Propose checks. Locked until approve=true.", "examples:str,approve:bool", "examples"),
    ("compile_system", "Search on dev, report on test. Needs an approved eval.", "examples:str,model:str,base_url:str,quick:bool,trials:int", "examples"),
    ("get_compile_status", "Poll compile status.", "examples:str", "examples"),
    ("get_report", "Winning config. Training reported as skipped.", "examples:str", "examples"),
    (
        "compare_models",
        "Score this System and another model on the same held-out rows, harness held fixed.",
        "other_model:str,examples:str,base_url:str",
        "other_model,examples",
    ),
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


def _arg_optional(args: Json, key: str) -> str:
    """A string argument that may be absent or empty, unlike `_arg_str`'s required one."""
    val = args.get(key, "")
    return val if isinstance(val, str) else ""


def _path(args: Json) -> Path:
    return Path(_arg_str(args, "examples")).expanduser().resolve()


def _stamp(examples: Path) -> Path:
    return examples.parent / _STAMP


def _answers_file(examples: Path) -> Path:
    """Where a session's answers live: beside the eval stamp, same idea."""
    return examples.parent / _ANSWERS


def _read_answers(examples: Path) -> dict[str, str]:
    """Answers written earlier for this examples file, or empty if none.

    A malformed file reads as absent rather than raising: the answers steer the eval,
    and losing them is a prompt to ask again, not a reason to fail the session.
    """
    path = _answers_file(examples)
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(k): str(v) for k, v in loaded.items()}


@dataclass(slots=True)
class _Record:
    examples: Path
    job: str
    #: Which task class the compile ran. The schema, gate, and scorer all hang off it,
    #: so a reloaded System has to carry it or it is not the same System.
    task_name: str = "restaurant"
    #: The id this System has in the shared store, when there is one. It is what makes a
    #: System built through the connector reachable through the HTTP API.
    system_id: str | None = None
    report: str | None = None
    config: HarnessConfig | None = None
    trial: Trial | None = None
    #: What the user answered to `ask_questions`, by question id. The dangerous answer
    #: and the stopping rule steer the eval and the gate, so they outlive the call that
    #: collected them.
    answers: dict[str, str] = field(default_factory=dict)
    #: Whether the user asked for the other models to be run before the compile.
    run_comparison: bool = False


class SystemSink(Protocol):
    """What the MCP surface needs from a System store.

    Structural rather than a concrete import, so this module does not depend on the
    control plane's package: the connector has to work standalone over stdio, and a real
    import would also cycle, since the API mounts this router.
    """

    def create(
        self, *, task: str, examples: tuple[TaskExample, ...], task_name: str = ...
    ) -> object:
        """Insert a draft System and return it."""
        ...

    def approve(self, system_id: str) -> object:
        """Record eval approval."""
        ...

    def create_run(self, system_id: str, *, quick: bool, model: str = ...) -> object:
        """Open a compile run."""
        ...

    def succeed_run(self, run_id: str, report: CompileReport) -> object:
        """Attach the winner to the run and the System."""
        ...


@dataclass(slots=True)
class Server:
    """In-process MCP session. Inject a completer so tests skip Ollama."""

    completer: Completer | None = None
    #: A System store to publish what this session compiles. Without one, a System
    #: compiled through the connector is invisible to the HTTP API, and the two doors
    #: describe different worlds.
    sink: SystemSink | None = None
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
            "inspect_task": self._inspect, "ask_questions": self._ask,
            "record_answers": self._record_answers, "propose_eval": self._propose,
            "compile_system": self._compile, "get_compile_status": self._status,
            "get_report": self._report, "compare_models": self._compare,
            "deploy_system": self._deploy, "invoke_system": self._invoke,
            "list_systems": self._list,
        }
        return table[name](args)

    def _ask(self, args: Json) -> str:
        """Put the six questions to the calling agent. Changes nothing.

        The agent has no cards, so this returns the questions as text: one block per
        question, with the reason it is asked and what a usable answer looks like. The
        two that already have an answer because a job or examples arrived are marked
        answered, so an agent does not re-ask what it already knows and does not skip
        what it does not.
        """
        job = _arg_optional(args, "job")
        examples = _arg_optional(args, "examples")
        # The file is the source of truth, not this process's dict: a connector is a
        # new process per turn, so an answer recorded last turn lives on disk.
        known = set(_read_answers(_path(args))) if examples else set()
        known.discard("run_comparison")
        if job:
            known.add("job")
        if examples:
            known.add("document")
        lines = ["ask your user these, then call record_answers with what they said:"]
        for q in _QUESTIONS:
            mark = "answered" if q["id"] in known else "open"
            lines.extend(
                (
                    f"[{mark}] {q['id']}: {q['question']}",
                    f"    why: {q['why']}",
                    f"    good: {q['good']}",
                )
            )
        lines.extend(
            (
                f"comparison, honestly: {_COMPARE_COST}",
                (
                    "record_answers refuses without dangerous and good_enough: the "
                    "search has no gate and no stop condition until it has both."
                ),
            )
        )
        return "\n".join(lines)

    def _record_answers(self, args: Json) -> str:
        """Store the user's answers, refusing the ones a compile cannot proceed without.

        A missing dangerous answer means the gate cannot distinguish wrong from
        unacceptable, and a missing stopping rule means the search never stops. Rather
        than compile on a default the user never chose, this refuses and names the gap.
        """
        examples = _path(args)
        answers = {q["id"]: _arg_optional(args, q["id"]) for q in _QUESTIONS}
        missing = [q for q in _REQUIRED_ANSWERS if not answers[q]]
        if missing:
            _die(
                "answers are missing for: " + ", ".join(missing) +
                ". Both are required before a compile: one becomes the gate, the "
                "other the stop condition."
            )
        run_comparison = args.get("run_comparison") is True
        rec = self._systems.setdefault(str(examples), _Record(examples, answers["job"]))
        rec.answers = {k: v for k, v in answers.items() if v}
        rec.run_comparison = run_comparison
        _ = _answers_file(examples).write_text(
            json.dumps({**rec.answers, "run_comparison": run_comparison}, indent=2) + "\n",
            encoding="utf-8",
        )
        said = [f"recorded {len(rec.answers)}/6 answers"]
        said.append(
            "dangerous -> the gate: " + answers["dangerous"]
        )
        said.append(
            "good_enough -> the stop condition: " + answers["good_enough"]
        )
        said.append(
            "compare: " + answers["compare"] if answers["compare"] else "compare: none named; specify one"
        )
        if run_comparison:
            said.append(
                "run_comparison=true: before the compile, measure the named model on "
                "the same held-out rows. " + _COMPARE_COST
            )
        else:
            said.append("run_comparison=false: the compile runs first; compare_models can follow.")
        said.append("next: propose_eval, then compile_system.")
        return "\n".join(said)

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
        rec = self._systems.setdefault(str(examples), _Record(examples, ""))
        if args.get("approve") is True:
            _ = _stamp(examples).write_text("approved\n", encoding="utf-8")
        nxt = "Eval approved. You may call compile_system." if _stamp(examples).is_file() else _LOCK
        # The dangerous answer is the gate, so the checks line names it when the user
        # gave one. A proposed eval that ignores the answer the user just gave is the
        # eval nobody approved.
        gate = f"\ngate from the dangerous answer: {rec.answers['dangerous']}" if rec.answers.get("dangerous") else ""
        return (
            f"{_counts(split)}\nchecks: {CHECKS_SUMMARY}{gate}\n"
            "selection: dev; test is scored once and never read by the search\n" + nxt
        )

    def _compare(self, args: Json) -> str:
        """Measure this System and another model on the same held-out rows.

        The harness comes from the compile's own winner, because a harness that was not
        the one measured proves nothing about the model swap. Both lanes run the test
        split - the rows neither selection nor the other model saw.
        """
        examples = _path(args)
        rec = self._systems.get(str(examples))
        if rec is None or rec.report is None or rec.trial is None:
            _die("no compile yet; call compile_system first, then compare models")
        other = _arg_str(args, "other_model")
        split = split_examples(load_examples(examples))
        task = task_by_name(rec.task_name)
        # The winner's harness, as a spec so both lanes read the same axes. The bundle
        # spec_for already carries them; only the model id changes per lane.
        spec = spec_for(
            CompileReport(
                winner=rec.trial, trials=(rec.trial,), test=rec.trial, stopped_early=False
            ),
            task=rec.job or "restaurant call extraction",
            model_id=rec.trial.config.model or _MODEL,
        )
        rows, train = tuple(split.test), tuple(split.train)
        source = measure(
            spec,
            completer=self._lm_for(args, spec.model_id),
            model=spec.model_id,
            rows=rows,
            train=train,
            task=task,
        )
        target = measure(
            spec,
            completer=self._lm_for(args, other),
            model=other,
            rows=rows,
            train=train,
            task=task,
        )
        card = format_model_comparison(
            spec=spec,
            source=source,
            target=target,
            source_label=spec.model_id,
            target_label=other,
        )
        _ = (examples.parent / "model-comparison.txt").write_text(
            card + "\n", encoding="utf-8"
        )
        return card

    def _lm_for(self, args: Json, model: str) -> Completer:
        """A completer for one named model, which is what a comparison needs.

        The single-lane tools take `self._lm`, but a comparison runs two models, so it
        needs one completer per model rather than one per request. An injected completer
        is reused for both lanes so a test can watch what each lane asked.
        """
        if self.completer is not None:
            return self.completer
        return OllamaCompleter(
            base_url=_arg_str(args, "base_url", _URL), model=model
        )

    def _compile(self, args: Json) -> str:
        examples = _path(args)
        if not _stamp(examples).is_file():
            _die(_LOCK)
        # The task decides the schema, the gate, and the scorer. Compiling without it
        # ran the connector against the default task, so the setup behind the measured
        # result was not the one the connector exercised.
        task = task_for_path(examples)
        split = split_examples(load_examples(examples))
        space = (
            SearchSpace.single()
            if args.get("quick") is True
            else SearchSpace.for_task(task, train_n=len(split.train))
        )
        rec = self._systems.setdefault(str(examples), _Record(examples, "", task.name))
        compiled = compile_system(
            self._lm(args),
            split,
            space,
            Budget(trials=_arg_int(args, "trials", 5)),
            task=task,
        )
        rec.report = format_report(compiled)
        rec.config = compiled.winner.config
        rec.trial = compiled.test
        rec.system_id = self._publish(examples, split, task, compiled, rec)
        _ = (examples.parent / "compile-report.txt").write_text(
            rec.report + "\n", encoding="utf-8"
        )
        _ = (examples.parent / "compile-report.md").write_text(
            render_markdown(compiled, task=rec.job), encoding="utf-8"
        )
        return rec.report

    def _publish(
        self,
        examples: Path,
        split: object,
        task: object,
        compiled: CompileReport,
        rec: _Record,
    ) -> str | None:
        """Record this compile in the shared store, when one is attached.

        A System built through the connector used to live only in this process, so the
        HTTP API could not see it and neither door knew about the other's work. Publishing
        it here is what makes the tool surface and the API describe one world.

        Returns the System id, or None when no store is attached.
        """
        if self.sink is None:
            return None
        rows = tuple(load_examples(examples))
        name = getattr(task, "name", rec.task_name)
        draft = self.sink.create(
            task=getattr(rec, "job", "") or f"compiled from {examples.name}",
            examples=rows,
            task_name=str(name),
        )
        system_id = str(draft.id)
        _ = self.sink.approve(system_id)
        run = self.sink.create_run(system_id, quick=False, model=str(getattr(rec, "model", "")))
        _ = self.sink.succeed_run(str(run.id), compiled)
        del split
        return system_id

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
        """Extract one document, replaying the compiled winner's harness.

        The harness is the System. Running a document through a default harness and
        returning the answer silently gives the caller something that is not the System
        they compiled, and it looks like it worked - so a missing winner is refused
        rather than substituted. A caller who genuinely wants a one-off extraction can
        say so by passing `k_shot` and `retries` themselves.
        """
        record = self._systems.get(str(_path(args)))
        winner = record.config if record is not None else None
        stated = _arg_int(args, "k_shot", -1) >= 0 or isinstance(args.get("retries"), int)
        if winner is None and not stated:
            _die(
                f"no compiled System for {_path(args)} in this session, and the "
                "harness is part of the System. Call compile_system first, or pass "
                "k_shot and retries yourself to extract with an explicit harness."
            )
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
            f"{r.examples} approved={_stamp(r.examples).is_file()} "
            f"report={r.report is not None} answers={len(r.answers)}/6 compare={r.run_comparison} {r.job}"
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
