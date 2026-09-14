"""OpenAPI request and response bodies."""

import os
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from mekoy.api.store import Phase, RunRecord, RunStatus, SystemRecord, phase_of
from mekoy.outcome import RestaurantOutcome

#: Model host defaults. In a container `127.0.0.1` is the container, so a
#: deployment has to say where its model lives; a caller may also override either
#: field per request.
_DEFAULT_URL = os.environ.get("MEKOY_MODEL_BASE_URL", "http://127.0.0.1:11434/v1")
_DEFAULT_MODEL = os.environ.get("MEKOY_MODEL_ID", "qwen2.5:7b")


class HealthResponse(BaseModel):
    """Liveness payload."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    status: Literal["ok"] = "ok"


class ExampleRow(BaseModel):
    """One labeled row. The label key names the task class.

    `outcome` is the restaurant label, `receipt` and `label` are the other two.
    Rows stay generic here because the control plane serves every task the
    compiler implements, not just the first one — a receipt-shaped row used to be
    rejected with a 422, which is what stopped the portal's compile flow.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    text: str = Field(min_length=1)
    outcome: object | None = None
    receipt: object | None = None
    label: object | None = None


class CreateSystemRequest(BaseModel):
    """Job plus labeled examples. Extra keys such as schema and slos are ignored."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    task: str = Field(min_length=1)
    #: Three, not two: the split needs one row for train, dev, and test. Allowing
    #: two here only moved the failure to the eval call.
    examples: tuple[ExampleRow, ...] = Field(min_length=3)


class EvalRequest(BaseModel):
    """Approve the stored examples as the eval snapshot."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    approve: bool = False


class CompileRequest(BaseModel):
    """Start a compile. quick is CLI --quick (one arm)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    quick: bool = True
    model: str = _DEFAULT_MODEL
    base_url: str = _DEFAULT_URL


class InvokeRequest(BaseModel):
    """One document to extract through the compiled winner.

    `model` is optional: leaving it unset replays the model the System was
    compiled against, which is what makes it the System that was measured.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    text: str = Field(min_length=1)
    model: str | None = None
    base_url: str = _DEFAULT_URL


class SystemCreated(BaseModel):
    """New System id and draft phase."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    id: str
    task: str
    phase: Phase
    n_examples: int


class EvalResponse(BaseModel):
    """Train/dev/test split and whether compile is unlocked."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    id: str
    approved: bool
    n_examples: int
    n_train: int
    n_dev: int
    n_test: int
    #: Compatibility alias for `n_test`. Kept so a caller written before the
    #: three-way split does not silently read `undefined`.
    n_holdout: int = 0
    phase: Phase


class WinnerOut(BaseModel):
    """Harness config that won dev search, with its measured cost and latency."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    config: str
    k_shot: int
    retries: int
    constrained: bool
    prompt: str
    quality: float
    schema_rate: float
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class RunResponse(BaseModel):
    """Compile run snapshot."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    id: str
    system_id: str
    status: RunStatus
    quick: bool
    report: str | None = None
    winner: WinnerOut | None = None
    error: str | None = None


class InvokeOk(BaseModel):
    """Verified extraction."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    ok: Literal[True] = True
    outcome: RestaurantOutcome


class InvokeFail(BaseModel):
    """Model output failed schema or numeric checks."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    ok: Literal[False] = False
    error: str


class SystemDetail(BaseModel):
    """A System and, once compiled, its latest run."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    id: str
    task: str
    #: Which task class the compiler will run: restaurant, receipt, banking77.
    task_name: str = "restaurant"
    phase: Phase
    n_examples: int
    run: RunResponse | None = None


class ReportResponse(BaseModel):
    """The compile card, as text."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    system_id: str
    run_id: str
    report: str


class DeployRequest(BaseModel):
    """Where the System should live. Hosting is not part of the local MVP."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    mode: Literal["hosted", "self_host", "download"] = "download"
    model: str = _DEFAULT_MODEL


class DeployResponse(BaseModel):
    """What deploy produced. A bundle path for `download`, a refusal otherwise."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    mode: str
    detail: str
    path: str | None = None


class ComparisonOut(BaseModel):
    """Two Systems on their measured test numbers."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    left: str
    right: str
    left_quality: float
    right_quality: float
    quality_winner: str
    faster: str
    n_test: int
    verdict: str


class ChatMessage(BaseModel):
    """One OpenAI-style message."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible invoke. `model` names the mekoy System, not a base model."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    model: str = Field(min_length=1)
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    base_url: str = _DEFAULT_URL


class ChatCompletionChoice(BaseModel):
    """One completion. `content` holds the extracted JSON."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop"] = "stop"


class ChatCompletionResponse(BaseModel):
    """Enough of the OpenAI shape for a standard client to parse."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    model: str
    choices: tuple[ChatCompletionChoice, ...]


def system_out(record: SystemRecord) -> SystemCreated:
    """Wire body for a created System."""
    return SystemCreated(
        id=record.id,
        task=record.task,
        phase=phase_of(record),
        n_examples=len(record.examples),
    )


def run_out(run: RunRecord) -> RunResponse:
    """Wire body for a compile run."""
    winner = None
    if run.winner is not None:
        winner = WinnerOut(
            config=run.winner.config.label,
            k_shot=run.winner.config.k_shot,
            retries=run.winner.config.retries,
            constrained=run.winner.config.constrained,
            prompt=run.winner.config.prompt,
            quality=run.winner.quality,
            schema_rate=run.winner.schema_rate,
            cost_usd=run.winner.cost_usd,
            latency_ms=run.winner.latency_ms,
        )
    return RunResponse(
        id=run.id,
        system_id=run.system_id,
        status=run.status,
        quick=run.quick,
        report=run.report,
        winner=winner,
        error=run.error,
    )
