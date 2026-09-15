"""HTTP control plane. Eval gate, compile, invoke."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, assert_never

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mekoy.api import mcp_http
from mekoy.api.auth import API_KEY_ENV, AuthMiddleware, settings_from_env
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
    HealthResponse,
    InvokeFail,
    InvokeOk,
    InvokeRequest,
    ReportResponse,
    RunResponse,
    SystemCreated,
    SystemDetail,
    SystemListResponse,
    run_out,
    system_out,
)
from mekoy.api.store import (
    DraftSystem,
    NotFoundError,
    PhaseError,
    Store,
    phase_of,
)
from mekoy.artifacts import ArtifactStore, store_from_env, write_bundle_to
from mekoy.bundle import spec_for
from mekoy.compare import compare_cards
from mekoy.compile import CompileReport, SearchSpace, compile_system
from mekoy.dataset import Split, TaskExample, coerce_label, split_examples
from mekoy.errors import CompileError, ModelUnreachableError
from mekoy.harness import Decode, extract
from mekoy.runtime import Completer, OllamaCompleter
from mekoy.tasks import Task, task_by_name, task_for_row
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
    """Per-app store, optional injected completer, and artifact destination."""

    store: Store
    completer: Completer | None
    artifacts: ArtifactStore = field(default_factory=store_from_env)


def get_ctx() -> AppContext:
    """Overridden per create_app() instance."""
    msg = "app context is not configured"
    raise CompileError(message=msg)


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


@_router.post("/v1/systems", tags=["systems"])
def create_system(
    body: CreateSystemRequest,
    ctx: Annotated[AppContext, Depends(get_ctx)],
) -> SystemCreated:
    """Store a job and labeled examples as a draft System."""
    task, examples = _task_examples(body.examples)
    record = ctx.store.create(task=body.task, examples=examples, task_name=task.name)
    return system_out(record)


def _task_examples(
    rows: tuple[ExampleRow, ...],
) -> tuple[Task, tuple[TaskExample, ...]]:
    """Detect the task from the row keys and validate every label against it.

    Rejecting an entire request because one row is shaped for another task would
    be worse than saying which row: the error names the row index.
    """
    first = rows[0].model_dump()
    task = task_for_row({k: v for k, v in first.items() if v is not None})
    out: list[TaskExample] = []
    for index, row in enumerate(rows):
        payload = {
            key: value
            for key, value in row.model_dump().items()
            if value is not None and key in {"outcome", "receipt", "label"}
        }
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


@_router.post("/v1/systems/{system_id}/evals", tags=["systems"])
def evaluate_system(
    system_id: str,
    body: EvalRequest,
    ctx: Annotated[AppContext, Depends(get_ctx)],
) -> EvalResponse:
    """Split examples and optionally unlock compile."""
    record = ctx.store.get_system(system_id)
    if body.approve:
        record = ctx.store.approve(system_id)
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
) -> RunResponse:
    """Start a compile and return at once with the run id.

    A compile is minutes of model calls. Running it inside the request meant every other
    caller queued behind it and the client's connection had to stay open for the whole
    thing, so closing a laptop tab lost the run and nothing else could be served. The
    work is handed to the background and the caller follows the run id.

    Everything that fails cheaply - the approval gate, the split, the task, the
    completer - is still resolved here, so a request that cannot compile says so now
    rather than leaving a run to die later.
    """
    run = ctx.store.create_run(system_id, quick=body.quick, model=body.model)
    record = ctx.store.get_system(system_id)
    split = split_examples(record.examples)
    task = task_by_name(record.task_name)
    space = (
        SearchSpace.single()
        if body.quick
        else SearchSpace.for_task(task, train_n=len(split.train))
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
        completer=completer,
        split=split,
        space=space,
        task=task,
    )
    return run_out(run)


def _run_compile(  # noqa: PLR0913 - the task carries what the request already resolved
    *,
    ctx: AppContext,
    run_id: str,
    completer: Completer,
    split: Split,
    space: SearchSpace,
    task: Task,
) -> None:
    """Run one compile and resolve the run whichever way it ends.

    Every exit has to resolve it, because a run left in `running` is worse than a failed
    one: the caller cannot tell a slow compile from a dead one, and nothing will ever
    move it. Nothing is raised past this point - the caller is gone by now, and a
    traceback would only reach the log while the run stayed open.
    """
    try:
        report = compile_system(completer, split, space, task=task)
    except CompileError as exc:
        _ = ctx.store.fail_run(run_id, exc.message)
        return
    except BaseException as exc:  # noqa: BLE001 - recorded, never re-raised
        _ = ctx.store.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        return
    _ = ctx.store.succeed_run(run_id, report)


@_router.get("/v1/runs/{run_id}", tags=["runs"])
def get_run(
    run_id: str,
    ctx: Annotated[AppContext, Depends(get_ctx)],
) -> RunResponse:
    """Poll a compile run."""
    return run_out(ctx.store.get_run(run_id))


@_router.post("/v1/systems/{system_id}/invoke", tags=["systems"])
def invoke_system(
    system_id: str,
    body: InvokeRequest,
    ctx: Annotated[AppContext, Depends(get_ctx)],
) -> InvokeOk | InvokeFail:
    """Extract one document with the compiled winner's harness."""
    record = ctx.store.require_compiled(system_id)
    split = split_examples(record.examples)
    shots = tuple(
        (row.text, row.outcome) for row in split.train[: record.winner.config.k_shot]
    )
    latest = ctx.store.latest_run(system_id)
    # Replay the model and the harness the compile measured, not defaults.
    model = body.model or (latest.model if latest is not None else "") or _DEFAULT_MODEL
    completer = _completer(ctx, model=model, base_url=body.base_url)
    task = task_by_name(record.task_name)
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
def list_systems(ctx: Annotated[AppContext, Depends(get_ctx)]) -> SystemListResponse:
    """Every System this control plane knows about.

    A catalog cannot be built on an API that cannot enumerate its own Systems, and a
    user cannot find what they built yesterday without it.
    """
    records = ctx.store.list_systems()
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
            for latest in (ctx.store.latest_run(record.id),)
        ),
        n=len(records),
    )


@_router.get("/v1/systems/{system_id}", tags=["systems"])
def get_system(
    system_id: str,
    ctx: Annotated[AppContext, Depends(get_ctx)],
) -> SystemDetail:
    """A System's phase and, once compiled, its latest run."""
    record = ctx.store.get_system(system_id)
    latest = ctx.store.latest_run(system_id)
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
) -> ReportResponse:
    """The compile card for the latest run."""
    _ = ctx.store.require_compiled(system_id)
    latest = ctx.store.latest_run(system_id)
    if latest is None or not latest.report:
        msg = f"system {system_id!r} has no compile report yet"
        raise NotFoundError(message=msg)
    return ReportResponse(system_id=system_id, run_id=latest.id, report=latest.report)


@_router.post("/v1/systems/{system_id}/deploy", tags=["systems"])
def deploy_system(
    system_id: str,
    body: DeployRequest,
    ctx: Annotated[AppContext, Depends(get_ctx)],
) -> DeployResponse:
    """Write a downloadable bundle, or refuse hosting.

    PLAN 23 declares hosted, self_host, and download. Hosting is not part of the
    local MVP, so those two modes say so rather than pretending. `download` writes
    spec.json, report.txt, README.md, and docker-compose.yml next to the run.
    """
    record = ctx.store.require_compiled(system_id)
    if body.mode != "download":
        return DeployResponse(
            mode=body.mode,
            detail="not hosted; use download or invoke",
        )
    latest = ctx.store.latest_run(system_id)
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
    out = write_bundle_to(
        ctx.artifacts,
        system_id,
        spec,
        latest.report,
        examples=record.examples,
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
) -> ComparisonOut:
    """Compare two compiled Systems on their measured test numbers."""
    left = _report_text(ctx, system_id)
    right = _report_text(ctx, other)
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


def _report_text(ctx: AppContext, system_id: str) -> str:
    _ = ctx.store.require_compiled(system_id)
    latest = ctx.store.latest_run(system_id)
    if latest is None or not latest.report:
        msg = f"system {system_id!r} has no compile report yet"
        raise NotFoundError(message=msg)
    return latest.report


@_router.post("/v1/chat/completions", tags=["invoke"])
def chat_completions(
    body: ChatCompletionRequest,
    ctx: Annotated[AppContext, Depends(get_ctx)],
) -> ChatCompletionResponse:
    """OpenAI-compatible invoke. `model` names the System, not a base model.

    PLAN 23 calls for invoke to be reachable this way so an existing OpenAI client
    can point at a System without new code. The last user message is the document.
    """
    record = ctx.store.require_compiled(body.model)
    document = next(
        (m.content for m in reversed(body.messages) if m.role == "user"), ""
    )
    split = split_examples(record.examples)
    shots = tuple(
        (row.text, row.outcome) for row in split.train[: record.winner.config.k_shot]
    )
    latest = ctx.store.latest_run(body.model)
    # `model` on this route names the System, so the base model comes from the run.
    base_model = (latest.model if latest is not None else "") or _DEFAULT_MODEL
    task = task_by_name(record.task_name)
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
    ctx = AppContext(store=Store(db_path=_default_db_path()), completer=completer)
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

    def _provide() -> AppContext:
        return ctx

    application.dependency_overrides[get_ctx] = _provide
    return application


app = create_app()
