"""HTTP control plane. Eval gate, compile, invoke."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter, sleep
from typing import Annotated, assert_never

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from mekoy.api import mcp_http
from mekoy.api.auth import (
    API_KEY_ENV,
    AuthMiddleware,
    Principal,
    resolve_principal,
    settings_from_env,
)
from mekoy.api.events import TERMINAL, EventKind, EventLog, RunEvent
from mekoy.api.limits import Quota, QuotaRefusedError, UsageLog, quota_from_env
from mekoy.api.mcp_http import router as mcp_router
from mekoy.api.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ComparisonOut,
    CompileRequest,
    CreateSystemRequest,
    DeployRequest,
    DeployResponse,
    EvalRequest,
    EvalResponse,
    ExampleRow,
    FailureOut,
    HealthResponse,
    InvokeFail,
    InvokeOk,
    InvokeRequest,
    MetricsResponse,
    ReportResponse,
    RunResponse,
    SystemCreated,
    SystemDetail,
    SystemListResponse,
    run_out,
    system_out,
)
from mekoy.api.observe import CompileMetering, Metrics
from mekoy.api.store import (
    DraftSystem,
    NotFoundError,
    PhaseError,
    Store,
    phase_of,
    task_for,
)
from mekoy.artifacts import ArtifactStore, store_from_env, write_bundle_to
from mekoy.bundle import spec_for
from mekoy.catalog import Catalog
from mekoy.compare import compare_cards
from mekoy.compile import Budget, CompileReport, SearchSpace, compile_system
from mekoy.dataset import Split, TaskExample, coerce_label, split_examples
from mekoy.errors import CompileError, ModelUnreachableError
from mekoy.harness import Decode, extract
from mekoy.jobs import JobDefinition, validate_job
from mekoy.runtime import Completer, OllamaCompleter
from mekoy.spec import Slos
from mekoy.tasks import Task, task_for_row, task_from_definition
from mekoy.verify import VerifyFail, VerifyOk, explain

_router = APIRouter()

_DEFAULT_URL = "http://127.0.0.1:11434/v1"
_DEFAULT_MODEL = "qwen2.5:7b"
_logger = logging.getLogger(__name__)


def _warn_open_api() -> None:
    """Say out loud that /v1 is reachable without a key.

    Logged rather than warned: this is an operational notice about the running
    server, and a Python warning here fires at import and trips a strict test run.
    """
    _logger.warning("/v1 is open: set %s to require a bearer key", API_KEY_ENV)


@dataclass(frozen=True, slots=True)
class AppContext:
    """Per-app store, optional injected completer, artifact destination, catalog."""

    store: Store
    completer: Completer | None
    #: Where a compile's progress lands. Beside the store because both live in the same
    #: SQLite file, and the log outlives the process that wrote it so a client can
    #: attach late and still see the run.
    events: EventLog = field(default_factory=EventLog)
    artifacts: ArtifactStore = field(default_factory=store_from_env)
    #: Published Systems. Empty by default, which is the honest state: nothing can be
    #: listed until its score is recomputable.
    catalog: Catalog = field(default_factory=Catalog)
    #: Documents consumed per principal per day. Same file as everything else, so a
    #: self-hoster still has one thing to back up.
    usage: UsageLog = field(default_factory=UsageLog)
    #: Metering rows, counters, and failures. Read by `GET /v1/metrics`.
    metrics: Metrics = field(default_factory=Metrics)
    quota: Quota | None = None

    @property
    def limits(self) -> Quota:
        """The configured quota, read from the environment once per process."""
        return self.quota or quota_from_env()


def get_ctx() -> AppContext:
    """Overridden per create_app() instance."""
    msg = "app context is not configured"
    raise CompileError(message=msg)


def get_principal(request: Request) -> Principal:
    """The one place a route asks who is calling.

    Reads what the middleware resolved rather than re-deriving it, so the HTTP
    routes and anything else holding the request agree about the caller.
    """
    return resolve_principal(request)


def owner_scope(principal: Principal) -> str | None:
    """The store's ownership filter for a principal. `None` means unfiltered.

    Local mode has no principals, so nothing is hidden and a System written before
    ownership existed stays reachable. Hosted mode names the principal, and the store
    then answers another principal's id with a 404.
    """
    return None if principal.is_local else principal.id


def _completer(ctx: AppContext, *, model: str, base_url: str) -> Completer:
    """Build a completer, honouring the deployment's model host.

    A container's `127.0.0.1` is the container, so a deployed control plane with no
    model beside it cannot compile. `MEKOY_MODEL_BASE_URL` is how a deployment points
    at a model host, and a caller can still override it per request.
    """
    if ctx.completer is not None:
        return ctx.completer
    base = base_url or os.environ.get("MEKOY_MODEL_BASE_URL", _DEFAULT_URL)
    return OllamaCompleter(base_url=base, model=model or _DEFAULT_MODEL)


@_router.get("/health", tags=["health"])
def health() -> HealthResponse:
    """Liveness. Does not talk to a model server."""
    return HealthResponse()


@_router.get("/v1/metrics", tags=["metrics"])
def get_metrics(
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> MetricsResponse:
    """Operational counters and recent compile failures, for a host.

    A failed compile is otherwise a run row whose status is `failed`, which nothing
    queries and nothing alerts on. This is where a host sees it. Scoped to the caller
    in hosted mode and to everything in local mode, because a self-hoster is the only
    user and wants every number, while a tenant should not read another's error text.
    """
    snapshot = ctx.metrics.snapshot(principal_id=owner_scope(principal))
    return MetricsResponse(
        counters=snapshot.counters,
        failures=tuple(
            FailureOut(run_id=f.run_id, system_id=f.system_id, error=f.error, at=f.at)
            for f in snapshot.failures
        ),
        metering={
            "compiles": snapshot.compiles_metered,
            "documents_scored": snapshot.documents_scored,
        },
        scope=principal.id,
    )


@_router.post("/v1/systems", tags=["systems"])
def create_system(
    body: CreateSystemRequest,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> SystemCreated:
    """Store a job and labeled examples as a draft System, owned by the caller."""
    task, examples = _task_examples(body.examples, definition=body.definition)
    record = ctx.store.create(
        task=body.task,
        examples=examples,
        task_name=task.name,
        principal_id=principal.id,
        definition=(
            body.definition.model_dump(mode="json")
            if body.definition is not None
            else None
        ),
    )
    return system_out(record)


def _task_examples(
    rows: tuple[ExampleRow, ...],
    *,
    definition: JobDefinition | None = None,
) -> tuple[Task, tuple[TaskExample, ...]]:
    """Detect the task from the row keys and validate every label against it.

    A caller-defined job wins over detection: its fields are the schema the rows are
    checked against, so a row shaped for the user's own job is not read as a shipped
    class and rejected. Without one, the row keys name a shipped class exactly as
    before.
    """
    task = _task_for_request(definition) or task_for_row(_labels_of(rows[0]))
    out: list[TaskExample] = []
    for index, row in enumerate(rows):
        payload = _labels_of(row)
        if not payload:
            msg = f"row {index}: no label key (outcome, receipt, or label)"
            raise CompileError(message=msg)
        try:
            outcome = task.model.model_validate(
                coerce_label(task.model, next(iter(payload.values())))
            )
        except ValueError as exc:
            msg = f"row {index} is not a valid {task.name} label: {exc}"
            raise CompileError(message=msg) from exc
        out.append(TaskExample(text=row.text, outcome=outcome))
    return task, tuple(out)


def _labels_of(row: ExampleRow) -> dict[str, object]:
    """The label keys a row actually carries, ignoring the unset ones."""
    return {
        key: value
        for key, value in row.model_dump().items()
        if value is not None and key in {"outcome", "receipt", "label"}
    }


def _task_for_request(definition: JobDefinition | None) -> Task | None:
    """The task a caller's job definition describes, or None when there is none.

    Validated here rather than left to fail at compile time: a definition naming a
    field that does not exist, or with no required check, is a typo the caller can fix
    now, and a run that dies halfway through with no explanation is the alternative.
    """
    if definition is None:
        return None
    problems = validate_job(definition)
    if problems:
        msg = "; ".join(str(problem) for problem in problems)
        raise CompileError(message=msg)
    return task_from_definition(definition)


@_router.post("/v1/systems/{system_id}/evals", tags=["systems"])
def evaluate_system(
    system_id: str,
    body: EvalRequest,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> EvalResponse:
    """Split examples and optionally unlock compile."""
    owner = owner_scope(principal)
    record = ctx.store.get_system(system_id, owner=owner)
    if body.approve:
        record = ctx.store.approve(system_id, owner=owner)
    split = split_examples(record.examples)
    return EvalResponse(
        id=record.id,
        approved=not isinstance(record, DraftSystem),
        n_examples=len(record.examples),
        n_train=len(split.train),
        n_dev=len(split.dev),
        n_test=len(split.test),
        n_holdout=len(split.test),
        phase=phase_of(record),
    )


@_router.post("/v1/systems/{system_id}/compile", tags=["systems"])
def compile_endpoint(
    system_id: str,
    body: CompileRequest,
    background: BackgroundTasks,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> RunResponse:
    """Start a compile and return at once with the run id.

    A compile is minutes of model calls. Running it inside the request meant every other
    caller queued behind it and the client's connection had to stay open for the whole
    thing, so closing a laptop tab lost the run and nothing else could be served. The
    work is handed to the background and the caller follows the run id.

    Everything that fails cheaply - the approval gate, the quota, the split, the task,
    the completer - is still resolved here, so a request that cannot compile says so now
    rather than leaving a run to die later.
    """
    owner = owner_scope(principal)
    _check_compile_quota(ctx, principal)
    run = ctx.store.create_run(
        system_id,
        quick=body.quick,
        model=body.model,
        principal_id=principal.id,
        owner=owner,
    )
    record = ctx.store.get_system(system_id, owner=owner)
    split = split_examples(record.examples)
    task = task_for(record)
    # Both paths carry the model axis. `single()` used to drop it, which would have made
    # a quick compile silently ignore the models the caller asked to compare.
    models = _models_for_search(body.models, ctx=ctx, base_url=body.base_url)
    space = replace(
        SearchSpace.single()
        if body.quick
        else SearchSpace.for_task(task, train_n=len(split.train)),
        models=tuple(body.models) or ("",),
    )
    completer = _completer(ctx, model=body.model, base_url=body.base_url)
    # Every exit from here has to resolve the run. A run left in `running` is worse than
    # a failed one: the caller cannot tell a slow compile from a dead one, and nothing
    # will ever move it. `ModelUnreachableError` is a `CompileError`, so a dead model
    # server was already recorded; what was not was anything else - a bug, a bad value,
    # an unexpected exception - which left the run open forever.
    background.add_task(
        _run_compile,
        ctx=ctx,
        run_id=run.id,
        principal=principal,
        completer=completer,
        split=split,
        space=space,
        task=task,
        slos=body.slos,
        models=models,
    )
    return run_out(run)


def _models_for_search(
    names: tuple[str, ...],
    *,
    ctx: AppContext,
    base_url: str,
) -> dict[str, Completer] | None:
    """One completer per model the caller asked to compare, or None for the default.

    The caller names base models, not base URLs, and every one of them is served by the
    same OpenAI-compatible endpoint the request already points at. An empty list means
    the single `model` the request names, which is what a compile before this field
    existed does. Duplicates are collapsed in order: the same model twice would price
    one completer as two candidates and report a comparison against itself.
    """
    if not names:
        return None
    return {
        name: _completer(ctx, model=name, base_url=base_url)
        for name in dict.fromkeys(names)
    }


def _check_compile_quota(ctx: AppContext, principal: Principal) -> None:
    """Refuse a compile that would cross the concurrent-compile limit.

    Checked before the run is opened, so a refused request leaves no run behind. A
    run that exists only to be refused is a leaked row a caller cannot see or clean
    up.
    """
    limit = ctx.limits.concurrent_compiles
    if limit <= 0:
        return
    running = ctx.store.count_running(owner=owner_scope(principal))
    if running >= limit:
        raise QuotaRefusedError(message=ctx.limits.explain_concurrent(limit))


def _charge_documents(ctx: AppContext, principal: Principal, *, asked: int) -> None:
    """Spend `asked` documents against the daily quota, refusing first.

    The check runs before the charge so a refused request is not counted: a limit
    that consumes the thing it refuses is a limit that punishes the caller twice.
    """
    ctx.usage.check(principal.id, ctx.limits, asked=asked)
    _ = ctx.usage.add(principal.id, asked)


def _run_compile(  # noqa: PLR0913 - the task carries what the request already resolved
    *,
    ctx: AppContext,
    run_id: str,
    principal: Principal,
    completer: Completer,
    split: Split,
    space: SearchSpace,
    task: Task,
    slos: Slos | None = None,
    models: Mapping[str, Completer] | None = None,
) -> None:
    """Run one compile and resolve the run whichever way it ends.

    Every exit has to resolve it, because a run left in `running` is worse than a failed
    one: the caller cannot tell a slow compile from a dead one, and nothing will ever
    move it. Nothing is raised past this point - the caller is gone by now, and a
    traceback would only reach the log while the run stayed open.

    Both exits also land in the metrics: a success writes the metering row a bill is
    built from, and a failure writes the row a host pages on. Neither is derivable
    later from the run alone, so both are written here.
    """

    def note(kind: str, data: dict[str, object]) -> None:
        """Record one step of the compile, for whoever is watching.

        Written to the log rather than pushed to a socket, so a client that attaches
        after this ran still sees the whole history.
        """
        _ = ctx.events.append(run_id, RunEvent(kind=EventKind(kind), data=data))

    def failed(reason: str) -> None:
        """Resolve the run as failed and make it visible to a host."""
        run = ctx.store.fail_run(run_id, reason)
        _ = ctx.metrics.record_failure(
            run_id=run_id,
            principal_id=principal.id,
            system_id=run.system_id,
            error=reason,
        )
        note("run_finished", {"status": "failed", "error": reason})

    try:
        report = compile_system(
            completer,
            split,
            space,
            Budget(slos=slos),
            task=task,
            models=models,
            on_event=note,
        )
    except CompileError as exc:
        failed(exc.message)
        return
    except BaseException as exc:  # noqa: BLE001 - recorded, never re-raised
        failed(f"{type(exc).__name__}: {exc}")
        return
    _ = ctx.store.succeed_run(run_id, report)
    # The metering row, written where the numbers are in hand rather than
    # reconstructed from the report later.
    _ = ctx.metrics.record_compile(
        CompileMetering(
            run_id=run_id,
            principal_id=principal.id,
            system_id=ctx.store.get_run(run_id).system_id,
            seconds=report.wall_seconds,
            model_calls=report.model_calls,
            documents=len(report.test.scores),
            model=report.winner.config.model,
        )
    )
    _ = ctx.metrics.bump("compile_succeeded", principal.id)
    note(
        "run_finished",
        {
            "status": "succeeded",
            "config": report.winner.config.label,
            "quality": round(report.test.quality, 6),
        },
    )


@_router.get("/v1/runs/{run_id}", tags=["runs"])
def get_run(
    run_id: str,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> RunResponse:
    """Poll a compile run."""
    return run_out(ctx.store.get_run(run_id, owner=owner_scope(principal)))


#: How long to wait between checks for new events once the log is caught up. Short
#: enough that a pane feels live, long enough not to spin a core.
_EVENT_POLL_S = 0.25

#: Give up following a run that never finishes. A compile has a budget, so a stream that
#: outlives this is a bug rather than patience.
_EVENT_TIMEOUT_S = 3600.0


@_router.get("/v1/runs/{run_id}/events", tags=["runs"])
def run_events(
    run_id: str,
    request: Request,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> StreamingResponse:
    """Stream a run's progress as server-sent events.

    Replays what already happened, then follows along. SSE rather than a websocket
    because progress only travels one way, and because the `id` on each frame gives
    reconnection for free: a client that drops sends `Last-Event-ID` and resumes after
    the last event it received, with no bookkeeping on either side.

    The run is checked first, so a client asking about a run that does not exist gets a
    404 rather than a stream that never says anything.
    """
    _ = ctx.store.get_run(run_id, owner=owner_scope(principal))
    after = _last_event_id(request)

    def frames() -> Iterator[str]:
        """Yield events until the run ends or the client gives up."""
        cursor = after
        deadline = perf_counter() + _EVENT_TIMEOUT_S
        while True:
            for event in ctx.events.since(run_id, after=cursor):
                cursor = event.seq
                yield event.to_sse()
                if event.kind in TERMINAL:
                    return
            if perf_counter() > deadline:
                return
            sleep(_EVENT_POLL_S)

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this a proxy will happily buffer the whole stream and deliver it
            # at the end, which is exactly the behaviour the pane exists to avoid.
            "X-Accel-Buffering": "no",
        },
    )


def _last_event_id(request: Request) -> int:
    """Where to resume from, if the client says it dropped mid-stream."""
    raw = request.headers.get("last-event-id", "").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


@_router.post("/v1/systems/{system_id}/invoke", tags=["systems"])
def invoke_system(
    system_id: str,
    body: InvokeRequest,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> InvokeOk | InvokeFail:
    """Extract one document with the compiled winner's harness."""
    owner = owner_scope(principal)
    record = ctx.store.require_compiled(system_id, owner=owner)
    # Charged after the ownership check, so a request against someone else's id is a
    # 404 that costs nothing rather than a quota unit the caller never used.
    _charge_documents(ctx, principal, asked=1)
    split = split_examples(record.examples)
    shots = tuple(
        (row.text, row.outcome) for row in split.train[: record.winner.config.k_shot]
    )
    latest = ctx.store.latest_run(system_id, owner=owner)
    # Replay the model and the harness the compile measured, not defaults.
    model = body.model or (latest.model if latest is not None else "") or _DEFAULT_MODEL
    completer = _completer(ctx, model=model, base_url=body.base_url)
    task = task_for(record)
    decode = Decode(
        system=task.prompt_variants.get(record.winner.config.prompt, task.prompt),
        constrained=record.winner.config.constrained,
    )
    result = extract(
        completer,
        text=body.text,
        shots=shots,
        retries=record.winner.config.retries,
        decode=decode,
        task=task,
    )
    match result:
        case VerifyOk(outcome=outcome):
            return InvokeOk(outcome=outcome)
        case VerifyFail():
            return InvokeFail(error=explain(result))
        case _ as unreachable:
            assert_never(unreachable)


@_router.get("/v1/systems", tags=["systems"])
def list_systems(
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> SystemListResponse:
    """Every System this control plane knows about.

    A catalog cannot be built on an API that cannot enumerate its own Systems, and a
    user cannot find what they built yesterday without it.
    """
    owner = owner_scope(principal)
    records = ctx.store.list_systems(owner=owner)
    return SystemListResponse(
        systems=tuple(
            SystemDetail(
                id=record.id,
                task=record.task,
                task_name=record.task_name,
                phase=phase_of(record),
                n_examples=len(record.examples),
                run=run_out(latest) if latest is not None else None,
            )
            for record in records
            for latest in (ctx.store.latest_run(record.id, owner=owner),)
        ),
        n=len(records),
    )


@_router.get("/v1/systems/{system_id}", tags=["systems"])
def get_system(
    system_id: str,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> SystemDetail:
    """A System's phase and, once compiled, its latest run.

    Another principal's id is a 404, the same answer an unknown id gets. A 403 would
    confirm the id exists, which is the enumeration oracle this replaced.
    """
    owner = owner_scope(principal)
    record = ctx.store.get_system(system_id, owner=owner)
    latest = ctx.store.latest_run(system_id, owner=owner)
    return SystemDetail(
        id=record.id,
        task=record.task,
        task_name=record.task_name,
        phase=phase_of(record),
        n_examples=len(record.examples),
        run=run_out(latest) if latest is not None else None,
    )


@_router.get("/v1/systems/{system_id}/report", tags=["systems"])
def get_report(
    system_id: str,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> ReportResponse:
    """The compile card for the latest run."""
    owner = owner_scope(principal)
    _ = ctx.store.require_compiled(system_id, owner=owner)
    latest = ctx.store.latest_run(system_id, owner=owner)
    if latest is None or not latest.report:
        msg = f"system {system_id!r} has no compile report yet"
        raise NotFoundError(message=msg)
    return ReportResponse(system_id=system_id, run_id=latest.id, report=latest.report)


@_router.post("/v1/systems/{system_id}/deploy", tags=["systems"])
def deploy_system(
    system_id: str,
    body: DeployRequest,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> DeployResponse:
    """Write a downloadable bundle, or refuse hosting.

    Hosted, self_host, and download are the three modes a caller chooses between.
    Hosting is not part of the local MVP, so those two modes say so rather than
    pretending. `download` writes
    spec.json, report.txt, README.md, and docker-compose.yml next to the run.
    """
    owner = owner_scope(principal)
    record = ctx.store.require_compiled(system_id, owner=owner)
    if body.mode != "download":
        return DeployResponse(
            mode=body.mode,
            detail="not hosted; use download or invoke",
        )
    latest = ctx.store.latest_run(system_id, owner=owner)
    if latest is None or not latest.report or latest.winner is None:
        msg = f"system {system_id!r} has no compile report to bundle"
        raise NotFoundError(message=msg)
    compiled = CompileReport(
        winner=latest.winner,
        trials=(latest.winner,),
        test=latest.winner,
        stopped_early=False,
    )
    spec = spec_for(compiled, task=record.task, model_id=body.model)
    # The held-out rows travel so the advertised score is checkable rather than
    # asserted, and the environment travels so a recomputation can match it.
    split = split_examples(record.examples)
    out = write_bundle_to(
        ctx.artifacts,
        system_id,
        spec,
        latest.report,
        examples=record.examples,
        holdout=split.test,
        environment={
            "model": latest.model or body.model,
            "base_url": os.environ.get("MEKOY_MODEL_BASE_URL", ""),
            "k_shot": latest.winner.config.k_shot,
            "retries": latest.winner.config.retries,
            "constrained": latest.winner.config.constrained,
            "task_name": record.task_name,
        },
    )
    return DeployResponse(
        mode=body.mode,
        detail="bundle written: spec.json, report.txt, README.md, docker-compose.yml",
        path=str(out),
    )


@_router.get("/v1/systems/{system_id}/compare", tags=["systems"])
def compare_systems(
    system_id: str,
    other: str,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> ComparisonOut:
    """Compare two compiled Systems on their measured test numbers."""
    owner = owner_scope(principal)
    left = _report_text(ctx, system_id, owner=owner)
    right = _report_text(ctx, other, owner=owner)
    try:
        cmp = compare_cards(system_id, left, other, right)
    except ValueError as exc:
        msg = str(exc)
        raise CompileError(message=msg) from exc
    return ComparisonOut(
        left=cmp.left,
        right=cmp.right,
        left_quality=cmp.left_quality,
        right_quality=cmp.right_quality,
        quality_winner=cmp.quality_winner,
        faster=cmp.faster,
        n_test=cmp.n_test,
        verdict=cmp.verdict,
    )


def _report_text(ctx: AppContext, system_id: str, *, owner: str | None = None) -> str:
    _ = ctx.store.require_compiled(system_id, owner=owner)
    latest = ctx.store.latest_run(system_id, owner=owner)
    if latest is None or not latest.report:
        msg = f"system {system_id!r} has no compile report yet"
        raise NotFoundError(message=msg)
    return latest.report


@_router.post("/v1/chat/completions", tags=["invoke"])
def chat_completions(
    body: ChatCompletionRequest,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> ChatCompletionResponse:
    """OpenAI-compatible invoke. `model` names the System, not a base model.

    Invoke is reachable this way so an existing OpenAI client can point at a System
    without new code. The last user message is the document.
    """
    owner = owner_scope(principal)
    record = ctx.store.require_compiled(body.model, owner=owner)
    _charge_documents(ctx, principal, asked=1)
    document = next(
        (m.content for m in reversed(body.messages) if m.role == "user"), ""
    )
    split = split_examples(record.examples)
    shots = tuple(
        (row.text, row.outcome) for row in split.train[: record.winner.config.k_shot]
    )
    latest = ctx.store.latest_run(body.model, owner=owner)
    # `model` on this route names the System, so the base model comes from the run.
    base_model = (latest.model if latest is not None else "") or _DEFAULT_MODEL
    task = task_for(record)
    completer = _completer(ctx, model=base_model, base_url=body.base_url)
    result = extract(
        completer,
        text=document,
        shots=shots,
        retries=record.winner.config.retries,
        decode=Decode(
            system=task.prompt_variants.get(record.winner.config.prompt, task.prompt),
            constrained=record.winner.config.constrained,
        ),
        task=task,
    )
    match result:
        case VerifyOk(outcome=outcome):
            content = outcome.model_dump_json()
        case VerifyFail():
            msg = f"verify failed: {explain(result)}"
            raise CompileError(message=msg)
        case _ as unreachable:
            assert_never(unreachable)
    return ChatCompletionResponse(
        id=f"chatcmpl-{record.id}",
        model=body.model,
        choices=(
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=content)
            ),
        ),
    )


def _register_errors(application: FastAPI) -> None:
    @application.exception_handler(NotFoundError)
    def _not_found(_request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse({"detail": exc.message}, status_code=404)

    @application.exception_handler(PhaseError)
    def _phase(_request: Request, exc: PhaseError) -> JSONResponse:
        return JSONResponse({"detail": exc.message}, status_code=409)

    @application.exception_handler(ModelUnreachableError)
    def _unreachable(_request: Request, exc: ModelUnreachableError) -> JSONResponse:
        return JSONResponse({"detail": exc.message}, status_code=503)

    @application.exception_handler(CompileError)
    def _compile_error(_request: Request, exc: CompileError) -> JSONResponse:
        return JSONResponse({"detail": exc.message}, status_code=400)

    # Registered after CompileError so this narrower type wins: both handlers match a
    # QuotaRefusedError, and FastAPI picks by exact class, but ordering the registration
    # the way a reader expects costs nothing.
    @application.exception_handler(QuotaRefusedError)
    def _quota(_request: Request, exc: QuotaRefusedError) -> JSONResponse:
        return JSONResponse({"detail": exc.message}, status_code=429)


def _default_db_path() -> Path | None:
    """Where to keep the index, or None to stay in memory.

    In-memory is the honest default for a test harness and for a one-shot run. A server
    that is meant to keep what it built sets `MEKOY_DB`, and then a restart is a
    nothing event rather than a wipe.
    """
    raw = os.environ.get("MEKOY_DB", "").strip()
    return Path(raw).expanduser() if raw else None


def create_app(*, completer: Completer | None = None) -> FastAPI:
    """Build an app with its own store, persisted when MEKOY_DB names a file."""
    _db = _default_db_path()
    ctx = AppContext(
        store=Store(db_path=_db),
        events=EventLog(db_path=_db),
        completer=completer,
        # Same file as the store and the event log, so a self-hoster is still backing
        # up one thing and reading one file.
        usage=UsageLog(db_path=_db),
        metrics=Metrics(db_path=_db),
    )
    # One store for both doors: a System compiled through the connector has to be the
    # System the HTTP API can find, or the two surfaces describe different worlds.
    mcp_http.bind_context(lambda: ctx)
    # A previous process may have died mid-compile, and its index may be gone while its
    # Systems are still on disk. Both are silent, so they are fixed here, at startup.
    _ = ctx.store.recover(Path(ctx.artifacts.root()))
    # Auth is mounted before CORS so CORS ends up outermost: a 401 still needs
    # CORS headers or a browser reports it as a network failure.
    application = FastAPI(
        title="Mekoy",
        description=(
            "Local control plane for owned Systems. "
            "Eval must be approved before compile. Training is not run."
        ),
        version="0.1.0",
    )
    _register_errors(application)
    settings = settings_from_env()
    if not settings.enabled:
        _warn_open_api()
    application.add_middleware(AuthMiddleware, settings=settings)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id"],
    )
    application.include_router(_router)
    application.include_router(mcp_router)
    # Imported here rather than at module scope: the catalog routes read AppContext and
    # get_ctx from this module, so a top-level import would be circular.
    from mekoy.api.catalog_routes import _router as catalog_router  # noqa: PLC0415

    application.include_router(catalog_router)
    from mekoy.api.safety_routes import _router as safety_router  # noqa: PLC0415

    application.include_router(safety_router)

    def _provide() -> AppContext:
        return ctx

    application.dependency_overrides[get_ctx] = _provide
    return application


app = create_app()
