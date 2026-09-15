"""Routing a job to a listed System, or refusing to.

PLAN §24b draws a boundary this module has to respect:

    Do not train the orchestrator until there are enough listed Systems that random
    choice is worse than a router. Until then, `POST /v1/run` requires `system=`.

So this is not a learned router and does not pretend to be. It is the *policy* half:
given a job and a caller's allowlist/denylist/named set, which listed Systems are
eligible, which is best, and - the part that matters most - when the honest answer is
that none of them fits.

Refusing is a first-class outcome rather than an error path. PLAN clause 3: "If nothing
fits: refuse or offer compile, do not guess." A router that always returns something
returns the wrong thing confidently, and the caller cannot tell "best available" from
"nothing here was remotely relevant".

The blocker on honest fit is a per-task difficulty baseline, which does not exist.
Without one a score has no reference point: 0.94 tells you nothing about whether 0.94 is
good for *that* job. So eligibility is decided by what can be checked - does the listing
declare the task, did the caller allow it, is its score recomputable - and anything
outside that is refused with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from mekoy.catalog import Catalog, Listing, Visibility

__all__ = [
    "Refusal",
    "Route",
    "RouteDecision",
    "Selectors",
    "route",
]


class RouteDecision(StrEnum):
    """What routing concluded."""

    ROUTED = "routed"
    REFUSED = "refused"


class Refusal(StrEnum):
    """Why nothing was routed. Distinct values so the caller can act on it."""

    #: The caller asked for a System id that is not in the catalog.
    NO_SUCH_SYSTEM = "no_such_system"
    #: There are no listings at all, so there is nothing to route across.
    CATALOG_EMPTY = "catalog_empty"
    #: Listings exist but the caller's own filters excluded every one.
    EXCLUDED_BY_CALLER = "excluded_by_caller"
    #: Listings were eligible but none declares this task.
    NO_TASK_MATCH = "no_task_match"
    #: Eligible and on-task, but no score that anybody could recompute.
    SCORE_NOT_VERIFIABLE = "score_not_verifiable"


@dataclass(frozen=True, slots=True)
class Selectors:
    """How a caller narrows what may be picked.

    Mirrors PLAN §24b: pin one System, pin a set, allowlist/denylist by id, or send
    nothing and use the public catalog.
    """

    #: Pin one System by id. Routing is off; this id is used or refused.
    system: str | None = None
    #: A named combination the caller configured.
    named_set: tuple[str, ...] = ()
    #: If either is non-empty, only ids in it may be chosen.
    allowlist: tuple[str, ...] = ()
    denylist: tuple[str, ...] = ()

    def permits(self, listing: Listing) -> bool:
        """Whether this listing survives the caller's own filters."""
        if listing.system_id in self.denylist:
            return False
        allowed = set(self.allowlist) | set(self.named_set)
        if allowed:
            return listing.system_id in allowed
        return True


@dataclass(frozen=True, slots=True)
class Route:
    """What to call, or why not, with the candidates that were considered."""

    decision: RouteDecision
    listing: Listing | None = None
    refusal: Refusal | None = None
    reason: str = ""
    considered: tuple[str, ...] = field(default=())

    @property
    def routed(self) -> bool:
        """Whether a System was chosen."""
        return self.decision is RouteDecision.ROUTED

    def explain(self) -> str:
        """One line, naming the System or the reason there is none."""
        if self.routed and self.listing is not None:
            score = (
                f"{self.listing.quality:.3f}"
                if self.listing.quality is not None
                else "n/a"
            )
            return (
                f"routed to {self.listing.system_id} "
                f"({self.listing.task}, quality {score})"
            )
        why = self.refusal.value if self.refusal else "unknown"
        return f"refused ({why}): {self.reason}"


def _pick(candidates: tuple[Listing, ...]) -> Listing | None:
    """Best eligible candidate: reviewed score, then cheaper, then faster.

    Every term is a property of the listing, not of a timing, so the choice is the same
    on two machines.
    """
    if not candidates:
        return None

    def key(entry: Listing) -> tuple[float, float, float]:
        return (entry.quality or 0.0, -entry.cost_per_doc, -entry.latency_ms)

    return max(candidates, key=key)


def route(  # noqa: PLR0911 - each refusal is a distinct, named outcome
    catalog: Catalog,
    *,
    task_name: str,
    selectors: Selectors | None = None,
) -> Route:
    """Choose a listed System for this job, or refuse with the reason.

    A pinned `system` bypasses the public pool: the caller named it, so it is used if it
    exists, whatever its visibility. Everything else routes across public listings only,
    because PLAN is explicit that private Systems never enter the global pool.
    """
    chosen = selectors or Selectors()
    every = catalog.listings()
    if not every:
        return Route(
            decision=RouteDecision.REFUSED,
            refusal=Refusal.CATALOG_EMPTY,
            reason=(
                "no Systems are published, so there is nothing to route to. "
                "Compile one, then publish it."
            ),
        )

    if chosen.system is not None:
        match = next((e for e in every if e.system_id == chosen.system), None)
        if match is None:
            return Route(
                decision=RouteDecision.REFUSED,
                refusal=Refusal.NO_SUCH_SYSTEM,
                reason=f"{chosen.system} is not a published System",
            )
        return Route(
            decision=RouteDecision.ROUTED,
            listing=match,
            considered=(match.system_id,),
        )

    # Public listings the caller allows, and separately the ones that are public but
    # whose score nobody can recompute. They are counted apart on purpose: collapsing
    # them makes a missing score report as "your filters excluded everything", which
    # sends the caller looking for a filter that is not the problem.
    public = catalog.public()
    routable = [e for e in public if chosen.permits(e)]
    if not routable:
        unverifiable_public = [
            e for e in catalog.listings() if e.visibility is Visibility.PUBLIC
        ]
        if unverifiable_public and not public:
            return Route(
                decision=RouteDecision.REFUSED,
                refusal=Refusal.SCORE_NOT_VERIFIABLE,
                reason=(
                    "every public listing has a score nobody can recompute, and "
                    "ranking on one would assert evidence we do not have"
                ),
                considered=tuple(e.system_id for e in unverifiable_public),
            )
        return Route(
            decision=RouteDecision.REFUSED,
            refusal=Refusal.EXCLUDED_BY_CALLER,
            reason=(
                "no listing survives the caller's allowlist, denylist, or named set"
            ),
            considered=tuple(e.system_id for e in every),
        )

    on_task = [e for e in routable if e.task_name == task_name]
    if not on_task:
        declared = sorted({e.task_name for e in routable})
        return Route(
            decision=RouteDecision.REFUSED,
            refusal=Refusal.NO_TASK_MATCH,
            reason=(
                f"no published System handles {task_name!r}; the catalog declares "
                f"{declared}. Compile one for this job rather than routing to a "
                "different job and hoping."
            ),
            considered=tuple(e.system_id for e in routable),
        )

    scored = [e for e in on_task if e.quality is not None]
    if not scored:
        return Route(
            decision=RouteDecision.REFUSED,
            refusal=Refusal.SCORE_NOT_VERIFIABLE,
            reason=(
                "every candidate for this task has an unverifiable score, and ranking "
                "on a number nobody can recompute would be asserting evidence we do "
                "not have"
            ),
            considered=tuple(e.system_id for e in on_task),
        )

    best = _pick(tuple(scored))
    if best is None:  # pragma: no cover - `scored` is non-empty by construction
        return Route(
            decision=RouteDecision.REFUSED,
            refusal=Refusal.NO_TASK_MATCH,
            reason="no candidate",
        )
    return Route(
        decision=RouteDecision.ROUTED,
        listing=best,
        considered=tuple(e.system_id for e in scored),
    )
