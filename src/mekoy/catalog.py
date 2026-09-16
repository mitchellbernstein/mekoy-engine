"""The catalog: which Systems are listed, and what a caller may ask of them.

A listing is not a System. It is a System *plus* the things a stranger needs before
trusting it: what it is for, what its base model is licensed under, where its data came
from, and a score somebody other than its author can recompute.

A badge is ours, never self-reported. That is why the only
way into this catalog is through `review.listable` - there is no `add` that skips the
gate, so a listing cannot exist without having passed it.

Deliberately absent: prices, payouts, and a platform cut. Those are billing, they need
accounts to exist first, and nothing here pretends to handle money.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from mekoy.errors import CompileError
from mekoy.review import Review
from mekoy.spec import SystemSpec

__all__ = [
    "Catalog",
    "Listing",
    "ListingError",
    "Visibility",
]


class ListingError(CompileError):
    """A listing was asked for that the review refuses, or that does not exist."""


class Visibility(StrEnum):
    """How a published System may be reached."""

    #: Appears in the catalog and the orchestrator may route to it.
    PUBLIC = "public"
    #: Reachable by id only. Not a route candidate. The default.
    UNLISTED = "unlisted"
    #: Listed, but the orchestrator must never pick it - only an explicit id.
    DIRECT_ONLY = "direct_only"


@dataclass(frozen=True, slots=True)
class Listing:
    """One published System, with everything a caller is owed about it."""

    system_id: str
    task: str
    task_name: str
    visibility: Visibility
    #: What the base model may be used for. Carried because some weights cannot be sold
    #: as weights, and a caller cannot know what they may do with a result without it.
    license: str
    #: Where the labeled examples came from.
    data_source: str
    #: The reviewed score, or None when it cannot be recomputed. None is honest, and
    #: is why the review gates on recomputability rather than on a score existing.
    quality: float | None
    cost_per_doc: float
    latency_ms: float
    #: The review findings, kept so a caller can see what was checked rather than trust
    #: a badge.
    findings: tuple[str, ...] = field(default=())

    @property
    def routable(self) -> bool:
        """Whether the orchestrator may pick this without being named."""
        return self.visibility is Visibility.PUBLIC and self.quality is not None


@dataclass(slots=True)
class Catalog:
    """Published Systems, and the review that admits them.

    In memory, like the compile store, because a catalog with no entries is the honest
    state of Phase III and a database would imply a scale this does not have.
    """

    _by_id: dict[str, Listing] = field(default_factory=dict)

    def publish(  # noqa: PLR0913 - a publication names everything the caller is owed
        self,
        system_id: str,
        spec: SystemSpec,
        *,
        visible: Visibility,
        checked: Review,
        quality: float | None,
        task_name: str = "restaurant",
    ) -> Listing:
        """Add a listing, but only one the review admits.

        There is deliberately no flag to bypass this. A catalog that can carry an
        unreviewed System is a catalog that will, and a badge is ours to give.
        """
        if not checked.listable:
            msg = (
                f"refusing to publish {system_id}: {checked.explain()}. "
                "Resolve the blockers and publish again."
            )
            raise ListingError(message=msg)
        listing = Listing(
            system_id=system_id,
            task=spec.task,
            task_name=task_name,
            visibility=visible,
            license=spec.license,
            data_source=spec.data_source,
            quality=quality,
            cost_per_doc=spec.slos.cost_per_doc,
            latency_ms=spec.slos.latency_ms,
            findings=tuple(f"{f.check.value}: {f.detail}" for f in checked.findings),
        )
        self._by_id[system_id] = listing
        return listing

    def get(self, system_id: str) -> Listing:
        """One listing or raise."""
        listing = self._by_id.get(system_id)
        if listing is None:
            msg = f"no listing for {system_id}"
            raise ListingError(message=msg)
        return listing

    def listings(self) -> tuple[Listing, ...]:
        """Every listing, newest last."""
        return tuple(self._by_id.values())

    def public(self) -> tuple[Listing, ...]:
        """Listings the orchestrator may route across, best score first.

        Sorted by the reviewed score rather than a self-reported one. A listing whose
        score could not be recomputed is never routable, so it cannot appear here, which
        is the whole reason the review gates on recomputability.
        """
        routable = [entry for entry in self._by_id.values() if entry.routable]
        return tuple(sorted(routable, key=lambda e: e.quality or 0.0, reverse=True))

    def withdraw(self, system_id: str) -> None:
        """Remove a listing. Revoke has to be possible."""
        if system_id not in self._by_id:
            msg = f"no listing for {system_id}"
            raise ListingError(message=msg)
        del self._by_id[system_id]

    def relabel(self, system_id: str, visible: Visibility) -> Listing:
        """Change how a listing may be reached, without re-running review."""
        current = self.get(system_id)
        updated = replace(current, visibility=visible)
        self._by_id[system_id] = updated
        return updated
