"""The publication gate, the catalog, and the orchestrator's refusals.

These tests exist to hold one line: **nothing may rank on a number a third party cannot
recompute.** The interesting cases are therefore the refusals, not the happy path - a
catalog that admits an unverifiable System is the failure this whole area is built to
prevent, and it would fail silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mekoy.api.main import create_app, get_ctx
from mekoy.api.store import CompiledSystem
from mekoy.bundle import spec_for, write_bundle
from mekoy.catalog import Catalog, ListingError, Visibility
from mekoy.compile import CompileReport
from mekoy.orchestrator import Refusal, RouteDecision, Selectors, route
from mekoy.review import Check, Decision, OptIn, SafetyDecision, review
from mekoy.spec import OwnershipFlags, Slos, SystemSpec, load_spec


def _spec(
    *, license_: str = "apache-2.0", data: str = "customer-owned calls"
) -> SystemSpec:
    return SystemSpec(
        task="Read one restaurant-call transcript and return one record.",
        json_schema={"type": "object", "properties": {"booked": {"type": "boolean"}}},
        slos=Slos(quality=0.94, cost_per_doc=0.0, latency_ms=8000.0),
        model_id="qwen2.5:7b",
        k_shot=4,
        retries=0,
        ownership=OwnershipFlags(runtime_owned=True, downloadable=True),
        license=license_,
        data_source=data,
    )


def _opted_in() -> OptIn:
    return OptIn(consented=True, acknowledged_data_becomes_public=True)


#: A clean scan, so a test about licences or attestation is not also testing safety.
#: Safety has its own file; here it is the baseline the other checks sit on.
def _scanned() -> SafetyDecision:
    return SafetyDecision(scanned=True)


def _bundle(tmp_path: Path) -> Path:
    """A directory standing in for a written bundle, which is what gets re-scored."""
    made = tmp_path / "bundle"
    made.mkdir(exist_ok=True)
    _ = (made / "spec.json").write_text("{}", encoding="utf-8")
    return made


# --- the gate -------------------------------------------------------------------


def test_an_unverifiable_score_refuses_publication() -> None:
    """The rule the whole area exists for.

    Our published 0.940 cannot be recomputed by anyone outside this repository: the
    corpus is not shipped and no command re-scores a bundle. A catalog ranking Systems
    by such a number would assert evidence it does not have.
    """
    checked = review(
        _spec(), opt_in=_opted_in(), score_is_recomputable=False, safety=_scanned()
    )
    assert checked.decision is Decision.REFUSED
    assert not checked.listable

    blocker = next(f for f in checked.blockers if f.check is Check.VERIFIABLE)
    assert "cannot be recomputed" in blocker.detail
    assert "verify-bundle" in blocker.detail


def test_publication_needs_both_confirmations(tmp_path: Path) -> None:
    """Publication asks twice, and one answer must not stand in for the other."""
    for opt_in in (
        OptIn(consented=True),
        OptIn(acknowledged_data_becomes_public=True),
        OptIn(),
    ):
        checked = review(
            _spec(), opt_in=opt_in, score_is_recomputable=True, safety=_scanned()
        )
        assert checked.decision is Decision.REFUSED
        assert any(f.check is Check.OPT_IN for f in checked.blockers)


def test_a_missing_licence_is_refused(tmp_path: Path) -> None:
    """A listing without a licence cannot be routed to honestly."""
    checked = review(
        _spec(license_=""),
        opt_in=_opted_in(),
        score_is_recomputable=True,
        safety=_scanned(),
    )
    assert any(f.check is Check.LICENSE for f in checked.blockers)


def test_unattested_data_is_refused(tmp_path: Path) -> None:
    """No publication without proving the data was licensed."""
    checked = review(
        _spec(data=""),
        opt_in=_opted_in(),
        bundle_dir=_bundle(tmp_path),
        score_is_recomputable=True,
        safety=_scanned(),
    )
    assert any(f.check is Check.ATTESTATION for f in checked.blockers)


def test_harness_tools_are_reported_rather_than_allowed(tmp_path: Path) -> None:
    """The v1 harness has no tools, so anything present is unexpected."""
    clean = review(
        _spec(),
        opt_in=_opted_in(),
        bundle_dir=_bundle(tmp_path),
        score_is_recomputable=True,
        safety=_scanned(),
    )
    assert clean.listable

    tainted = review(
        _spec(),
        opt_in=_opted_in(),
        bundle_dir=_bundle(tmp_path),
        score_is_recomputable=True,
        harness_tools=("http_get",),
        safety=_scanned(),
    )
    assert not tainted.listable
    assert any(f.check is Check.HARNESS for f in tainted.blockers)


def test_every_blocker_is_named_in_the_explanation() -> None:
    """'Refused' without a reason is indistinguishable from a bug."""
    checked = review(
        _spec(license_="", data=""),
        opt_in=OptIn(),
        score_is_recomputable=False,
        safety=_scanned(),
    )
    said = checked.explain()
    assert said.startswith("refused:")
    for finding in checked.blockers:
        assert finding.check.value in said


def test_a_fully_checkable_system_is_listable(tmp_path: Path) -> None:
    """The gate must be able to pass, or it is a wall rather than a check."""
    checked = review(
        _spec(),
        opt_in=_opted_in(),
        bundle_dir=_bundle(tmp_path),
        score_is_recomputable=True,
        safety=_scanned(),
    )
    assert checked.listable
    assert checked.explain() == "listable: all review checks passed."


# --- the catalog ----------------------------------------------------------------


def test_the_catalog_refuses_an_unreviewed_system(tmp_path: Path) -> None:
    """There is no way in that skips the gate."""
    catalog = Catalog()
    checked = review(
        _spec(), opt_in=_opted_in(), score_is_recomputable=False, safety=_scanned()
    )
    with pytest.raises(ListingError, match="refusing to publish"):
        _ = catalog.publish(
            "sys_x", _spec(), visible=Visibility.PUBLIC, checked=checked, quality=0.94
        )
    assert catalog.listings() == (), "nothing was admitted"


def test_publishing_and_withdrawing(tmp_path: Path) -> None:
    """Revoke has to be possible."""
    catalog = Catalog()
    checked = review(
        _spec(),
        opt_in=_opted_in(),
        bundle_dir=_bundle(tmp_path),
        score_is_recomputable=True,
        safety=_scanned(),
    )
    listing = catalog.publish(
        "sys_a", _spec(), visible=Visibility.PUBLIC, checked=checked, quality=0.94
    )
    assert listing.license == "apache-2.0"
    assert listing.findings, "a caller can see what was checked, not just a badge"

    catalog.withdraw("sys_a")
    assert catalog.listings() == ()
    with pytest.raises(ListingError):
        catalog.withdraw("sys_a")


def test_a_listing_without_a_recomputable_score_is_not_routable(tmp_path: Path) -> None:
    """Unlisted and unverifiable both keep a System out of the public pool."""
    catalog = Catalog()
    checked = review(
        _spec(),
        opt_in=_opted_in(),
        bundle_dir=_bundle(tmp_path),
        score_is_recomputable=True,
        safety=_scanned(),
    )
    _ = catalog.publish(
        "sys_unlisted",
        _spec(),
        visible=Visibility.UNLISTED,
        checked=checked,
        quality=0.99,
    )
    _ = catalog.publish(
        "sys_direct",
        _spec(),
        visible=Visibility.DIRECT_ONLY,
        checked=checked,
        quality=0.99,
    )
    assert catalog.public() == (), "neither may be routed to by the global pool"

    # and a public listing whose score nobody can check is likewise excluded
    _ = catalog.publish(
        "sys_noscore", _spec(), visible=Visibility.PUBLIC, checked=checked, quality=None
    )
    assert catalog.public() == ()


# --- the orchestrator -----------------------------------------------------------


def _catalog_with_public(
    tmp_path: Path, *specs: tuple[str, str, float | None]
) -> Catalog:
    catalog = Catalog()
    checked = review(
        _spec(),
        opt_in=_opted_in(),
        bundle_dir=_bundle(tmp_path),
        score_is_recomputable=True,
        safety=_scanned(),
    )
    for system_id, task_name, quality in specs:
        _ = catalog.publish(
            system_id,
            _spec(),
            visible=Visibility.PUBLIC,
            checked=checked,
            quality=quality,
            task_name=task_name,
        )
    return catalog


def test_an_empty_catalog_refuses_and_says_so() -> None:
    """A router that always returns something returns the wrong thing confidently."""
    outcome = route(Catalog(), task_name="restaurant")
    assert outcome.decision is RouteDecision.REFUSED
    assert outcome.refusal is Refusal.CATALOG_EMPTY
    assert "nothing to route to" in outcome.reason


def test_it_refuses_rather_than_routing_to_a_different_job(tmp_path: Path) -> None:
    """The point of the refusal: a plausible-looking wrong answer is the bug."""
    catalog = _catalog_with_public(tmp_path, ("sys_receipts", "receipt", 0.91))
    outcome = route(catalog, task_name="restaurant")
    assert outcome.decision is RouteDecision.REFUSED
    assert outcome.refusal is Refusal.NO_TASK_MATCH
    assert "receipt" in outcome.reason, "it names what the catalog does handle"
    assert outcome.considered == ("sys_receipts",)


def test_a_pinned_system_is_used_when_it_exists(tmp_path: Path) -> None:
    catalog = _catalog_with_public(tmp_path, ("sys_a", "restaurant", 0.90))
    outcome = route(
        catalog, task_name="restaurant", selectors=Selectors(system="sys_a")
    )
    assert outcome.routed
    assert outcome.listing is not None
    assert outcome.listing.system_id == "sys_a"


def test_a_pinned_system_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    catalog = _catalog_with_public(tmp_path, ("sys_a", "restaurant", 0.90))
    outcome = route(
        catalog, task_name="restaurant", selectors=Selectors(system="sys_gone")
    )
    assert outcome.decision is RouteDecision.REFUSED
    assert outcome.refusal is Refusal.NO_SUCH_SYSTEM


def test_denylist_and_allowlist_are_respected(tmp_path: Path) -> None:
    catalog = _catalog_with_public(
        tmp_path, ("sys_a", "restaurant", 0.99), ("sys_b", "restaurant", 0.80)
    )
    denied = route(
        catalog, task_name="restaurant", selectors=Selectors(denylist=("sys_a",))
    )
    assert denied.routed
    assert denied.listing is not None
    assert denied.listing.system_id == "sys_b"

    allowed = route(
        catalog, task_name="restaurant", selectors=Selectors(allowlist=("sys_b",))
    )
    assert allowed.listing is not None
    assert allowed.listing.system_id == "sys_b"

    nothing = route(
        catalog,
        task_name="restaurant",
        selectors=Selectors(denylist=("sys_a", "sys_b")),
    )
    assert nothing.decision is RouteDecision.REFUSED
    assert nothing.refusal is Refusal.EXCLUDED_BY_CALLER


def test_a_named_set_is_an_allowlist(tmp_path: Path) -> None:
    catalog = _catalog_with_public(
        tmp_path, ("sys_a", "restaurant", 0.99), ("sys_b", "restaurant", 0.80)
    )
    outcome = route(
        catalog, task_name="restaurant", selectors=Selectors(named_set=("sys_b",))
    )
    assert outcome.listing is not None
    assert outcome.listing.system_id == "sys_b"


def test_it_refuses_when_no_score_is_verifiable(tmp_path: Path) -> None:
    """Ranking on an unverifiable number is asserting evidence we do not have.

    Built directly rather than through the helper, because the helper publishes only
    listings whose score is present - and this case is precisely the one where the score
    is missing while everything else checks out.
    """
    catalog = Catalog()
    checked = review(
        _spec(),
        opt_in=_opted_in(),
        bundle_dir=_bundle(tmp_path),
        score_is_recomputable=True,
        safety=_scanned(),
    )
    _ = catalog.publish(
        "sys_a",
        _spec(),
        visible=Visibility.PUBLIC,
        checked=checked,
        quality=None,
    )
    outcome = route(catalog, task_name="restaurant")
    assert outcome.decision is RouteDecision.REFUSED
    assert outcome.refusal is Refusal.SCORE_NOT_VERIFIABLE
    assert "recompute" in outcome.reason


def test_the_best_recomputable_score_wins(tmp_path: Path) -> None:
    catalog = _catalog_with_public(
        tmp_path,
        ("sys_low", "restaurant", 0.71),
        ("sys_high", "restaurant", 0.94),
        ("sys_mid", "restaurant", 0.88),
    )
    outcome = route(catalog, task_name="restaurant")
    assert outcome.routed
    assert outcome.listing is not None
    assert outcome.listing.system_id == "sys_high"
    assert set(outcome.considered) == {"sys_low", "sys_mid", "sys_high"}


def test_the_route_explain_names_the_system_or_the_reason(tmp_path: Path) -> None:
    catalog = _catalog_with_public(tmp_path, ("sys_a", "restaurant", 0.94))
    routed = route(catalog, task_name="restaurant")
    assert "sys_a" in routed.explain()
    assert "0.940" in routed.explain()

    refused = route(catalog, task_name="banking77")
    assert refused.explain().startswith("refused")
    assert "no_task_match" in refused.explain()


# --- the routes -----------------------------------------------------------------


def _client() -> TestClient:
    """A control plane with its own empty store and catalog."""
    return TestClient(create_app())


def test_the_catalog_route_explains_an_empty_pool() -> None:
    """An empty list with no reason reads as a broken endpoint, not a refusal."""
    body = _client().get("/v1/catalog").json()
    assert body["n"] == 0
    assert body["listings"] == []
    assert "verify-bundle" in body["note"]
    assert "refuses" in body["note"]


def test_routing_refuses_when_nothing_is_published() -> None:
    """The live refusal a caller would hit today."""
    body = _client().post("/v1/route", json={"task_name": "restaurant"}).json()
    assert body["routed"] is False
    assert body["refusal"] == "catalog_empty"
    assert "nothing to route to" in body["reason"]
    assert body["system_id"] is None


def test_publishing_without_a_recomputable_score_is_refused() -> None:
    """The gate a caller hits today: everything else right, and it still refuses."""
    client = _client()
    created = client.post(
        "/v1/systems", json={"task": "extract", "examples": _API_ROWS}
    ).json()
    sid = created["id"]
    _ = client.post(f"/v1/systems/{sid}/evals", json={"approve": True})
    # Publishing a System that was never compiled is refused earlier, with a different
    # reason, so the compile has to happen for this to be the unverifiable-score case.
    started = client.post(f"/v1/systems/{sid}/compile", json={"quick": True}).json()
    settled = client.get(f"/v1/runs/{started['id']}").json()
    assert settled["status"] in {"succeeded", "failed"}

    body = client.post(
        f"/v1/systems/{sid}/publish",
        json={
            "visibility": "public",
            "consent": True,
            "acknowledged_data_becomes_public": True,
            "bundle_verified": False,
        },
    ).json()

    assert body["published"] is False
    assert any("verifiable" in b for b in body["blockers"])
    assert "cannot be recomputed" in body["explanation"]
    assert client.get("/v1/catalog").json()["n"] == 0


_API_ROWS = [
    {
        "text": f"Uchi, table for {i + 2} Friday 7pm under Maya.",
        "outcome": {
            "restaurant": "Uchi",
            "intent": "reservation",
            "status": "confirmed",
            "party_size": i + 2,
            "when": "Friday 7pm",
            "under_name": "Maya",
            "evidence": "table for two",
            "booked": True,
        },
    }
    for i in range(3)
]


def test_the_gate_opens_when_the_prerequisites_are_met(tmp_path: Path) -> None:
    """A gate that cannot open is a wall, and a wall teaches nobody anything.

    Every other test here checks a refusal. This one checks that the refusals are
    requirements rather than a permanent no: with a licence, an attestation, both
    confirmations, and a verified bundle, the same System lists and becomes routable.
    """
    client = _client()
    app = client.app
    ctx = app.dependency_overrides[get_ctx]()

    rows = _API_ROWS
    sid = client.post(
        "/v1/systems", json={"task": "extraction", "examples": rows}
    ).json()["id"]
    _ = client.post(f"/v1/systems/{sid}/evals", json={"approve": True})
    started = client.post(f"/v1/systems/{sid}/compile", json={"quick": True}).json()
    for _ in range(60):
        if client.get(f"/v1/runs/{started['id']}").json()["status"] != "running":
            break

    record = ctx.store.require_compiled(sid)
    assert isinstance(record, CompiledSystem)
    report = CompileReport(
        winner=record.winner,
        trials=(record.winner,),
        test=record.winner,
        stopped_early=False,
    )
    spec = spec_for(report, task=record.task, model_id="qwen2.5:7b")
    bundle = tmp_path / "bundle"
    _ = write_bundle(bundle, spec, record.report, examples=record.examples)

    # The two fields the reviewer needs that a compile does not know.
    payload = json.loads((bundle / "spec.json").read_text())
    payload["license"] = "apache-2.0"
    payload["data_source"] = "customer-owned transcripts"
    _ = (bundle / "spec.json").write_text(json.dumps(payload), encoding="utf-8")

    # Review the bundle directly: the route reads from the artifact store, and this is
    # the same spec it would find there.
    checked = review(
        load_spec(bundle),
        opt_in=OptIn(consented=True, acknowledged_data_becomes_public=True),
        bundle_dir=bundle,
        score_is_recomputable=True,
        safety=_scanned(),
    )
    assert checked.listable, checked.explain()

    catalog = Catalog()
    listing = catalog.publish(
        "sys_ok",
        load_spec(bundle),
        visible=Visibility.PUBLIC,
        checked=checked,
        quality=0.94,
        task_name="restaurant",
    )
    assert listing.routable

    outcome = route(catalog, task_name="restaurant")
    assert outcome.routed
    assert outcome.listing is not None
    assert outcome.listing.system_id == "sys_ok"
