"""Catalog and orchestrator routes.

The control plane's third door: one URL can hit a private System,
a catalog System by id, or the orchestrator that routes a *job* to the best listed
System the caller allowed. This module is the second and third of those.

Publication is refused more often than it succeeds right now, and that is intended
rather than a bug to work around. A listing is ranked by a score, and nobody outside
this repository can recompute ours: the held-out corpora are not published and no
command re-scores a bundle. Ranking Systems on a number a caller cannot check would
assert evidence we do not have, so publishing returns a review whose blockers say
exactly that. See `mekoy.review`.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from mekoy.api.auth import Principal
from mekoy.api.main import AppContext, get_ctx, get_principal, owner_scope
from mekoy.catalog import ListingError, Visibility
from mekoy.errors import CompileError
from mekoy.orchestrator import Refusal, RouteDecision, Selectors, route
from mekoy.review import OptIn, SafetyDecision, review
from mekoy.spec import load_spec

_router = APIRouter()

#: A spec requires a model id and a compiled System does not always record one, because
#: the request may have used the server's default. Saying "unrecorded" is honest; an
#: empty string is invalid and makes the reviewer raise instead of refusing cleanly.
_UNKNOWN_MODEL = "unrecorded"


class PublishRequest(BaseModel):
    """Ask for one System to be listed.

    Both confirmations are separate fields on purpose: publication asks twice because
    the first answer is often reflexive and the second is where someone reads what they
    are agreeing to.

    The safety fields are the third confirmation, and they are separate for the same
    reason. `safety_scanned` says a scan was run; `safety_acknowledged` says the
    publisher was shown a specific finding and accepted it. One flag cannot tell those
    apart, and a publisher who ticked one box has read nothing.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    visibility: Visibility = Visibility.UNLISTED
    consent: bool = False
    acknowledged_data_becomes_public: bool = False
    #: Set only by a caller that has actually re-scored the bundle. Absent means the
    #: score is unverifiable, which refuses publication.
    bundle_verified: bool = False
    harness_tools: tuple[str, ...] = ()
    #: A scan was run and its findings are on file. Absent refuses publication, because
    #: an unscanned System is not a System nobody worried about - it is one nobody
    #: measured.
    safety_scanned: bool = False
    #: The publisher read the findings named in the refusal and accepted them anyway.
    safety_acknowledged: bool = False
    #: The categories the scan flagged, as the labels a scan report carries. Passed in
    #: rather than re-derived here so the refusal names what actually triggered it
    #: instead of a second guess at the same question.
    safety_flagged: tuple[str, ...] = ()


class ReviewOut(BaseModel):
    """What the reviewer concluded, and every reason it did not pass."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    published: bool
    decision: str
    explanation: str
    blockers: tuple[str, ...] = ()


class ListingOut(BaseModel):
    """One listed System, as a caller sees it."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    system_id: str
    task: str
    task_name: str
    visibility: str
    license: str
    data_source: str
    quality: float | None
    cost_per_doc: float
    latency_ms: float


class CatalogOut(BaseModel):
    """The public pool the orchestrator routes across."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    listings: tuple[ListingOut, ...]
    n: int
    #: Why the pool is empty, when it is. An empty catalog with no explanation reads
    #: as a broken endpoint rather than as an honest refusal.
    note: str = ""


class RouteRequest(BaseModel):
    """A job to route, and how the caller narrows the candidates."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    task_name: str
    system: str | None = None
    named_set: tuple[str, ...] = ()
    allowlist: tuple[str, ...] = ()
    denylist: tuple[str, ...] = ()


class RouteOut(BaseModel):
    """The System to call, or the reason there is none."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    routed: bool
    system_id: str | None = None
    refusal: str | None = None
    reason: str
    considered: tuple[str, ...] = ()


def _to_out(listing: object) -> ListingOut:
    return ListingOut(
        system_id=listing.system_id,  # type: ignore[attr-defined]
        task=listing.task,  # type: ignore[attr-defined]
        task_name=listing.task_name,  # type: ignore[attr-defined]
        visibility=str(listing.visibility),  # type: ignore[attr-defined]
        license=listing.license,  # type: ignore[attr-defined]
        data_source=listing.data_source,  # type: ignore[attr-defined]
        quality=listing.quality,  # type: ignore[attr-defined]
        cost_per_doc=listing.cost_per_doc,  # type: ignore[attr-defined]
        latency_ms=listing.latency_ms,  # type: ignore[attr-defined]
    )


@_router.get("/v1/catalog", tags=["catalog"])
def catalog(ctx: Annotated[AppContext, Depends(get_ctx)]) -> CatalogOut:
    """The public pool, best reviewed score first.

    Unscoped on purpose, and that is a design choice rather than a missing filter: a
    listing only exists once its author opted in twice and the review passed, and the
    point of publishing is that any caller may route to it. An unpublished System
    cannot appear here, because `Catalog.public()` reads only published listings.
    """
    listings = tuple(_to_out(entry) for entry in ctx.catalog.public())
    note = "" if listings else str(_why_empty(ctx))
    return CatalogOut(listings=listings, n=len(listings), note=note)


def _why_empty(ctx: AppContext) -> str:
    """Say why the pool is empty, since an empty list alone reads as a fault."""
    every = ctx.catalog.listings()
    if not every:
        return (
            "no System is published. Publication requires a score a third party can "
            "recompute, and no `verify-bundle` command or published corpus exists yet, "
            "so the review refuses every submission."
        )
    return (
        f"{len(every)} listing(s) exist but none is public with a verifiable score, "
        "so none may be routed to."
    )


@_router.post("/v1/systems/{system_id}/publish", tags=["catalog"])
def publish(
    system_id: str,
    body: PublishRequest,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> ReviewOut:
    """Submit a compiled System for listing. Refusals are the common case, by design.

    Owner-filtered, because this is the write path: without the filter a tenant could
    publish under another tenant's system id, which is worse than reading one - the
    listing is harder to walk back once somebody has routed to it.
    """
    owner = owner_scope(principal)
    record = ctx.store.require_compiled(system_id, owner=owner)
    spec_path = None
    try:
        bundle_dir = ctx.artifacts.root() / system_id
        spec_path = bundle_dir / "spec.json"
        spec = (
            load_spec(spec_path)
            if spec_path.is_file()
            else _spec_from_record(record, system_id, ctx)
        )
    except CompileError as exc:
        return ReviewOut(
            published=False,
            decision="refused",
            explanation=f"no spec to review for {system_id}: {exc.message}",
            blockers=(
                "the System has not written a bundle, so nothing can be reviewed",
            ),
        )

    checked = review(
        spec,
        opt_in=OptIn(
            consented=body.consent,
            acknowledged_data_becomes_public=body.acknowledged_data_becomes_public,
        ),
        bundle_dir=bundle_dir
        if spec_path is not None and spec_path.is_file()
        else None,
        score_is_recomputable=body.bundle_verified,
        harness_tools=body.harness_tools,
        safety=SafetyDecision(
            scanned=body.safety_scanned,
            acknowledged=body.safety_acknowledged,
            flagged=body.safety_flagged,
        ),
    )
    if not checked.listable:
        return ReviewOut(
            published=False,
            decision=str(checked.decision),
            explanation=checked.explain(),
            blockers=tuple(f"{f.check.value}: {f.detail}" for f in checked.blockers),
        )
    try:
        _ = ctx.catalog.publish(
            system_id,
            spec,
            visible=body.visibility,
            checked=checked,
            quality=record.winner.quality,
            task_name=record.task_name,
        )
    except ListingError as exc:
        return ReviewOut(
            published=False,
            decision="refused",
            explanation=exc.message,
            blockers=(exc.message,),
        )
    return ReviewOut(
        published=True,
        decision=str(checked.decision),
        explanation=checked.explain(),
    )


def _spec_from_record(record: object, system_id: str, ctx: AppContext) -> object:
    """Build a spec from a stored System when no bundle has been written yet.

    Imports the store's own machinery so the reviewer sees the same spec a bundle would
    carry, rather than a second, drifting definition of one.

    The model comes from the run that produced the winner, because a compiled System
    does not carry it directly and an empty model id is not a valid spec - which is how
    this was found: `spec_for` rejected the spec the first version built, and the route
    raised instead of refusing cleanly.
    """
    from mekoy.api.store import CompiledSystem  # noqa: PLC0415
    from mekoy.bundle import spec_for  # noqa: PLC0415
    from mekoy.compile import CompileReport  # noqa: PLC0415

    if not isinstance(record, CompiledSystem):
        msg = f"{system_id} is not compiled"
        raise CompileError(message=msg)
    latest = ctx.store.latest_run(system_id)
    model_id = (
        (latest.model if latest is not None else "")
        or record.winner.config.model
        or _UNKNOWN_MODEL
    )
    report = CompileReport(
        winner=record.winner,
        trials=(record.winner,),
        test=record.winner,
        stopped_early=False,
    )
    return spec_for(report, task=record.task, model_id=model_id)


@_router.post("/v1/route", tags=["catalog"])
def route_job(
    body: RouteRequest,
    ctx: Annotated[AppContext, Depends(get_ctx)],
) -> RouteOut:
    """Pick a listed System for this job, or refuse with the reason.

    Refusing is the point. A router that always returns something returns the wrong
    thing confidently, and a caller cannot tell "best available" from "nothing fits".

    Unscoped on purpose: the only Systems a router may see are *published* listings,
    and publication is a deliberate, double-confirmed, reviewed act whose whole point
    is that strangers may call the result. An unpublished System is not in
    `catalog.listings()` at all, so this route cannot read one - which is why it needs
    no owner filter, unlike the store-backed routes.
    """
    outcome = route(
        ctx.catalog,
        task_name=body.task_name,
        selectors=Selectors(
            system=body.system,
            named_set=body.named_set,
            allowlist=body.allowlist,
            denylist=body.denylist,
        ),
    )
    if outcome.decision is RouteDecision.ROUTED and outcome.listing is not None:
        return RouteOut(
            routed=True,
            system_id=outcome.listing.system_id,
            reason=outcome.explain(),
            considered=outcome.considered,
        )
    return RouteOut(
        routed=False,
        refusal=str(outcome.refusal or Refusal.CATALOG_EMPTY),
        reason=outcome.reason,
        considered=outcome.considered,
    )
