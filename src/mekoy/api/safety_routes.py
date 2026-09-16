"""The safety surface: scan a compiled System, and read what it found.

This is the door a reviewer uses before a System becomes callable by strangers. It
reports; it does not act. Publication is refused elsewhere, in `mekoy.review`, because
the review is where every gate already lives and a second refusal path would eventually
disagree with the first.

One thing in here is structural rather than conventional: `SafetyOut` carries both
rates, and there is no request that asks for one of them. A caller cannot obtain a
harmful-completion rate without the false-positive-on-benign rate beside it, because the
report type they come from has no way to hold one number.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, ClassVar

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from mekoy.api.main import AppContext, get_ctx
from mekoy.api.store import CompiledSystem, SafetyCheck
from mekoy.dataset import split_examples
from mekoy.errors import CompileError
from mekoy.harness import Decode, extract
from mekoy.safety import (
    BENIGN_PROBES,
    NOT_APPLICABLE,
    RISKY_PROBES,
    Category,
    Completions,
    ModelVerifier,
    Probe,
    SafetyReport,
    Verifier,
    build_probes,
    plain_words,
    probe_set,
    scan,
    suggested_categories,
    write_probe_set,
)
from mekoy.tasks import task_by_name
from mekoy.verify import VerifyFail, VerifyOk

_router = APIRouter()

_DEFAULT_URL = "http://127.0.0.1:11434/v1"
#: The judge. A 7B open model on this machine: bigger than the compiled Systems are
#: likely to be, and local, because sending the user's documents to a closed provider
#: for a safety opinion would trade their privacy for our assurance.
_DEFAULT_JUDGE = "qwen2.5:14b"


class SafetyRequest(BaseModel):
    """How to probe. No field here selects a metric."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    #: The judge. Local by default and local by intent - see
    #: `safety.ModelVerifier.local`. Named as the concrete class rather than the
    #: `Verifier` protocol: the protocol declares `judge`, and the classmethod that
    #: builds a model-backed one lives on the implementation.
    judge_model: str = _DEFAULT_JUDGE
    judge_base_url: str = _DEFAULT_URL
    #: Override the model under test. Unset replays the one the System was compiled
    #: against, which is what makes the probe a measurement of this System.
    model: str | None = None
    base_url: str = _DEFAULT_URL
    #: Write the probe set next to the bundle so a reviewer can open the evidence.
    publish_probes: bool = False


class FindingOut(BaseModel):
    """One category's finding, with the user's own row quoted."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    category: str
    #: The labeled row the probe was shaped after - the user's own text, which is what
    #: makes the finding checkable instead of arguable.
    evidence_row: str
    probe: str
    response: str
    reason: str


class NotApplicableOut(BaseModel):
    """A category considered and excluded, with the reason it cannot apply here."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    category: str
    why_not: str


class CheckOut(BaseModel):
    """One thing the engine will measure, in plain words.

    `plain` is the sentence a user reads before they trust the number, and it is not
    optional: a rate whose check cannot be read is an assertion, which is what this
    surface exists to replace.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    category: str
    #: The category as a person would say it, for the label on the screen.
    label: str
    #: What the check measures, in one sentence.
    plain: str
    #: True when this job's own text suggested the category.
    suggested: bool = False
    #: True when the user added it; a built-in check has neither this nor `suggested`.
    added: bool = False
    #: The user's own text the probe is shaped after, when they supplied any.
    text: str = ""


class ChecksOut(BaseModel):
    """What the engine checks, what this job suggests, and what the user added.

    Three groups rather than one list with flags, because the three answer different
    questions: `engine` is the fixed set every System is measured against, `suggested`
    is a proposal derived from this job and these rows, and `added` is the user's own
    decision. Collapsing them would hide which checks the user actually chose.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    system_id: str
    #: Every category the engine measures, with the plain-words sentence for each.
    engine: tuple[CheckOut, ...]
    #: Categories this job's own words point at, ready to accept.
    suggested: tuple[CheckOut, ...]
    #: Categories the engine deliberately does not assess, with the reason.
    not_applicable: tuple[NotApplicableOut, ...]
    #: The checks the user has added, in the order they added them.
    added: tuple[CheckOut, ...]


class AddCheckRequest(BaseModel):
    """A check to add. `category` must be one the engine can measure."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    category: str
    #: The user's own text to shape the probe after. Empty means their rows shape it.
    text: str = ""


class SafetyOut(BaseModel):
    """A safety report. Both rates are required fields.

    Not "both are usually reported": they are non-defaulted, so a response missing
    either one cannot be constructed. That is the difference between a convention and a
    shape, and this module is the shape.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    system_id: str
    harmful_completed: int
    risky_probed: int
    harmful_completion_rate: float
    benign_refused: int
    benign_probed: int
    false_positive_rate: float
    flagged: bool
    findings: tuple[FindingOut, ...]
    unclassified: tuple[str, ...]
    unavailable: str
    #: The one-line summary, which by construction carries both rates.
    summary: str
    #: Categories deliberately not assessed at this layer, so a reviewer comparing this
    #: to a frontier-lab list sees the reasoning rather than a gap.
    not_applicable: tuple[NotApplicableOut, ...]
    probe_set_path: str = ""


def _out(report: SafetyReport, *, probe_path: Path | None = None) -> SafetyOut:
    return SafetyOut(
        system_id=report.system_id,
        harmful_completed=report.harmful_completed,
        risky_probed=report.risky_probed,
        harmful_completion_rate=report.harmful_completion_rate,
        benign_refused=report.benign_refused,
        benign_probed=report.benign_probed,
        false_positive_rate=report.false_positive_rate,
        flagged=report.flagged,
        findings=tuple(
            FindingOut(
                category=finding.category.label,
                evidence_row=finding.evidence_row,
                probe=finding.probe,
                response=finding.response,
                reason=finding.reason,
            )
            for finding in report.findings
        ),
        unclassified=report.unclassified,
        unavailable=report.unavailable,
        summary=report.summary(),
        not_applicable=tuple(
            NotApplicableOut(category=name, why_not=why) for name, why in NOT_APPLICABLE
        ),
        probe_set_path=str(probe_path) if probe_path is not None else "",
    )


def _probes_for(
    record: CompiledSystem, *, added: Sequence[SafetyCheck] = ()
) -> tuple[Probe, ...]:
    """The fixed probe set, the job-shaped probe, and one probe per added check.

    The fixed set measures the five categories a structured extractor can realise.
    The shaped probes add the job's own input form, because "given a request in the
    shape of your inputs, this System returned X" is a measurement and "this job could
    be misused" is a speculation.

    An added check is measured, not merely listed. A check a user adds and then does not
    find in the rates is decoration on a settings screen, and the whole reason they are
    allowed to choose is so the number that comes back covers what they asked to have
    covered.
    """
    split = split_examples(record.examples)
    samples = tuple(split.train[:1])
    shaped = tuple(
        probe
        for row in samples
        for probe in build_probes(_category_of(record), text=row.text)
    )
    chosen = tuple(
        probe
        for check in added
        for probe in build_probes(
            Category(check.category),
            text=check.text or (samples[0].text if samples else record.task),
        )
    )
    return (*RISKY_PROBES, *BENIGN_PROBES, *shaped, *chosen)


def _category_of(record: CompiledSystem) -> object:
    """Which category the shaped probe should claim.

    Fraud, unless the job's own text says otherwise: drafting outreach is the most
    likely accidental case, and it is where a legitimate job and a harmful one are
    written the same way.
    """
    text = f"{record.task} {record.task_name}".casefold()
    if "person" in text or "background" in text:
        return Category.PERSON_TARGETING
    return Category.FRAUD


def _invoke(
    ctx: AppContext,
    record: CompiledSystem,
    probe: Probe,
    *,
    model: str,
    base_url: str,
) -> str:
    """One probe through the compiled winner's harness. Empty text means nothing."""
    split = split_examples(record.examples)
    shots = tuple(
        (row.text, row.outcome) for row in split.train[: record.winner.config.k_shot]
    )
    task = task_by_name(record.task_name)
    completer = ctx.completer
    if completer is None:
        from mekoy.runtime import OllamaCompleter  # noqa: PLC0415 - test seam

        completer = OllamaCompleter(base_url=base_url, model=model)
    result = extract(
        completer,
        text=probe.text,
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
            return outcome.model_dump_json()
        case VerifyFail():
            # A refusal produced by the gate is a refusal, and an empty response is how
            # the scorer reads one. Returning the gate's reasons instead would be a
            # response the System never made.
            return ""
    return ""


@_router.post("/v1/systems/{system_id}/safety", tags=["safety"])
def scan_system(
    system_id: str,
    body: SafetyRequest,
    ctx: Annotated[AppContext, Depends(get_ctx)],
) -> SafetyOut:
    """Probe a compiled System and report both directions.

    Probes are shaped like the job's own inputs, and the judge is local. A judge that
    cannot answer produces `unclassified` for that probe, which is a reported gap rather
    than a pass.
    """
    record = ctx.store.require_compiled(system_id)
    probes = _probes_for(record, added=ctx.store.safety_checks(system_id))
    latest = ctx.store.latest_run(system_id)
    model = body.model or (latest.model if latest is not None else "") or "qwen2.5:7b"
    risky = tuple(probe for probe in probes if not probe.benign)
    benign = tuple(probe for probe in probes if probe.benign)
    try:
        completions = Completions(
            risky=tuple(
                _invoke(ctx, record, probe, model=model, base_url=body.base_url)
                for probe in risky
            ),
            benign=tuple(
                _invoke(ctx, record, probe, model=model, base_url=body.base_url)
                for probe in benign
            ),
        )
        verifier = (
            ModelVerifier.local(base_url=body.judge_base_url, model=body.judge_model)
            if ctx.completer is None
            else _injected(ctx)
        )
        report = scan(
            tuple(row.text for row in record.examples),
            completions=completions,
            verifier=verifier,
            system_id=system_id,
            probes=probes,
        )
    except Exception as exc:  # noqa: BLE001 - a failed probe is a reported gap
        # Fail visible: a scan that could not run reports that, and reports it through
        # the same shape as a scan that did, so a caller cannot mistake one for the
        # other by looking at the status code.
        report = SafetyReport(
            system_id=system_id,
            harmful_completed=0,
            risky_probed=len(risky),
            benign_refused=len(benign),
            benign_probed=len(benign),
            findings=(),
            unavailable=f"could not classify: {type(exc).__name__}: {exc}",
        )
    path = None
    if body.publish_probes:
        root = Path(ctx.artifacts.root()) / system_id
        path = write_probe_set(root / "probes.jsonl")
    return _out(report, probe_path=path)


@_router.get("/v1/systems/{system_id}/safety/checks", tags=["safety"])
def get_checks(
    system_id: str,
    ctx: Annotated[AppContext, Depends(get_ctx)],
) -> ChecksOut:
    """What the engine will check, what this job suggests, and what the user added.

    The point of the route is that a person can read what is measured *before* a number
    means anything to them. So `engine` is every category with its plain-words sentence,
    `suggested` is derived from their job line and their own labeled rows rather than
    from a stock list, and `added` is what they chose themselves.

    Nothing here invents a category: `engine` and `not_applicable` are the engine's two
    lists, and the split between them is the engine's decision, shown rather than
    restated.
    """
    record = ctx.store.require_compiled(system_id)
    return _checks_out(ctx, record)


@_router.post("/v1/systems/{system_id}/safety/checks", tags=["safety"])
def add_check(
    system_id: str,
    body: AddCheckRequest,
    ctx: Annotated[AppContext, Depends(get_ctx)],
) -> ChecksOut:
    """Add a check of the user's own, and return the list it now appears in.

    The category has to be one the engine can realise, so an unknown one is refused with
    the known list rather than stored and silently never measured. The user's own text,
    when they give any, is what the probe is shaped after - which is the difference
    between adding a check and picking one.
    """
    record = ctx.store.require_compiled(system_id)
    try:
        category = Category(body.category)
    except ValueError as exc:
        known = ", ".join(item.value for item in Category)
        msg = f"unknown category {body.category!r}; the engine measures: {known}"
        raise CompileError(message=msg) from exc
    _ = ctx.store.add_safety_check(system_id, category=category.value, text=body.text)
    return _checks_out(ctx, record)


def _checks_out(ctx: AppContext, record: CompiledSystem) -> ChecksOut:
    """Build the three groups from the engine's taxonomy and this System's own words."""
    # Every row is read for suggestions, not just the held-out ones: the suggestion is
    # about what the job is, and the user's rows are the job.
    all_rows = tuple(row.text for row in record.examples)
    proposed = set(suggested_categories(record.task, all_rows))
    # The engine already reads this job to decide the category its job-shaped probe
    # claims, so that reading is offered too. Without it, a job whose words match no
    # marker is offered nothing and "pick from suggestions" becomes "guess" - and a
    # stock list would be worse, because it would look derived when it is not.
    proposed.add(_category_of(record))
    added = ctx.store.safety_checks(record.id)
    return ChecksOut(
        system_id=record.id,
        engine=tuple(
            _check_out(category, suggested=category in proposed)
            for category in Category
        ),
        suggested=tuple(
            _check_out(category, suggested=True)
            for category in Category
            if category in proposed and _not_added(category, added)
        ),
        not_applicable=tuple(
            NotApplicableOut(category=name, why_not=why) for name, why in NOT_APPLICABLE
        ),
        added=tuple(
            _check_out(Category(check.category), added=True, text=check.text)
            for check in added
        ),
    )


def _check_out(
    category: Category, *, suggested: bool = False, added: bool = False, text: str = ""
) -> CheckOut:
    return CheckOut(
        category=category.value,
        label=category.label,
        plain=plain_words(category),
        suggested=suggested,
        added=added,
        text=text,
    )


def _not_added(category: Category, added: Sequence[SafetyCheck]) -> bool:
    """Whether the user has not already added this category.

    A category already on their list is not offered again: an accept button for a check
    that is already measured is a button that does nothing, and the screen would be
    claiming otherwise.
    """
    return all(check.category != category.value for check in added)


def _injected(ctx: AppContext) -> Verifier:
    """A verifier over an injected completer, for tests and in-process callers."""
    return ModelVerifier(ctx.completer)  # type: ignore[arg-type]


@_router.get("/v1/safety/probes", tags=["safety"])
def get_probe_set() -> dict[str, object]:
    """The probe set itself, so the claim is checkable rather than ours.

    The research is direct that publishing the probe set and the scorer is what makes
    our assessment different from our assertion, and it is what a reviewer who does not
    trust us can use.
    """
    lines = [
        line for line in probe_set().splitlines() if line and not line.startswith("#")
    ]
    return {
        "text": probe_set(),
        "n": len(lines),
        "categories": sorted({probe.category.value for probe in RISKY_PROBES}),
        "note": NOT_APPLICABLE,
        "judge_env": "MEKOY_JUDGE_MODEL",
    }


def default_judge_model() -> str:
    """The judge a deployment uses, overridable for hardware that cannot hold 14B."""
    return os.environ.get("MEKOY_JUDGE_MODEL", _DEFAULT_JUDGE)
