"""The business layer: identity, tenancy, quotas, metering, metrics.

The load-bearing tests are the pair at the bottom of the tenancy section: a second
principal gets a 404 on another tenant's System, **and** a keyless local control plane
still gets a 200 on a System nobody owns. Fixing the first by breaking the second is the
failure mode that would make this a worse product, so both are asserted together.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mekoy.api import main as api_main
from mekoy.api.auth import (
    API_KEY_ENV,
    API_KEYS_ENV,
    AuthSettings,
    Principal,
    settings_from_env,
)
from mekoy.api.limits import (
    CONCURRENT_ENV,
    DOCUMENTS_PER_DAY_ENV,
    Quota,
    QuotaRefusedError,
    UsageLog,
    quota_from_env,
)
from mekoy.api.main import create_app, get_ctx
from mekoy.api.observe import CompileMetering, Metrics
from mekoy.api.store import DraftSystem, NotFoundError, Store
from mekoy.dataset import ExampleRecord, TaskExample, load_examples

A_KEY = "key-tenant-a"
B_KEY = "key-tenant-b"
_KEYS = f"tenant-a:{A_KEY},tenant-b:{B_KEY}"

_ROWS = [
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


def _payload(task: str = "extract restaurant outcomes") -> dict[str, object]:
    return {"task": task, "examples": _ROWS}


def _auth(key: str) -> dict[str, str]:
    return {"authorization": f"Bearer {key}"}


# --- C3a: one principal resolution point -----------------------------------------


def test_local_mode_is_keyless_and_names_one_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No keys configured means the local principal, and the API stays open."""
    monkeypatch.delenv(API_KEYS_ENV, raising=False)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    settings = settings_from_env()
    assert settings.enabled is False
    assert settings.principal_for(None) is None
    with TestClient(create_app()) as client:
        assert client.post("/v1/systems", json=_payload()).status_code == 200


def test_hosted_keys_each_name_a_principal() -> None:
    """`id:key` pairs resolve to distinct principals, and a wrong key to none."""
    settings = AuthSettings(keys={A_KEY: "tenant-a", B_KEY: "tenant-b"})
    assert settings.principal_for(A_KEY) == Principal(
        id="tenant-a", key_name="tenant-a"
    )
    assert settings.principal_for(B_KEY) == Principal(
        id="tenant-b", key_name="tenant-b"
    )
    assert settings.principal_for("nope") is None
    assert settings.accepts(A_KEY) is True
    assert settings.accepts("nope") is False


def test_the_legacy_single_key_still_names_one_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`MEKOY_API_KEY` keeps working and is a single-tenant hosted mode."""
    monkeypatch.delenv(API_KEYS_ENV, raising=False)
    monkeypatch.setenv(API_KEY_ENV, "one-key")
    settings = settings_from_env()
    assert settings.enabled is True
    assert settings.principal_for("one-key") == Principal(
        id="default", key_name="default"
    )


def test_a_malformed_key_entry_does_not_take_the_plane_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo in one pair must not stop a running control plane from booting."""
    monkeypatch.setenv(API_KEYS_ENV, "broken,tenant-a:key-a,:empty,no-key:")
    settings = settings_from_env()
    assert settings.principal_for("key-a") == Principal(
        id="tenant-a", key_name="tenant-a"
    )
    # The malformed entries name nobody, so they cannot be used to get in.
    assert settings.principal_for("broken") is None
    assert settings.principal_for("empty") is None


def test_a_keyless_request_never_gets_a_caller_on_a_guarded_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hosted mode with no key is a 401, not a silently-local caller."""
    monkeypatch.setenv(API_KEYS_ENV, _KEYS)
    with TestClient(create_app()) as client:
        assert client.post("/v1/systems", json=_payload()).status_code == 401
        assert client.get("/health").status_code == 200


# --- C3b: the cross-tenant read, closed ------------------------------------------


@pytest.fixture
def hosted(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A hosted control plane with two principals, keys read from the environment."""
    monkeypatch.setenv(API_KEYS_ENV, _KEYS)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with TestClient(create_app()) as client:
        yield client


def test_a_second_tenant_cannot_read_the_firsts_system(hosted: TestClient) -> None:
    """The hole: GET /v1/systems/{id} had no ownership check at all."""
    created = hosted.post("/v1/systems", json=_payload(), headers=_auth(A_KEY)).json()
    sid = created["id"]

    as_owner = hosted.get(f"/v1/systems/{sid}", headers=_auth(A_KEY))
    assert as_owner.status_code == 200
    assert "outcomes" in as_owner.json()["task"]

    as_other = hosted.get(f"/v1/systems/{sid}", headers=_auth(B_KEY))
    # 404 rather than 403 on purpose: a 403 confirms the id exists, which is an
    # enumeration oracle.
    assert as_other.status_code == 404
    assert "outcomes" not in as_other.text


def test_a_second_tenant_cannot_list_the_firsts_system(hosted: TestClient) -> None:
    sid = hosted.post("/v1/systems", json=_payload(), headers=_auth(A_KEY)).json()["id"]
    listed = hosted.get("/v1/systems", headers=_auth(B_KEY)).json()
    assert listed == {"systems": [], "n": 0}
    assert sid not in json.dumps(listed)


def test_every_store_backed_route_is_owner_filtered(hosted: TestClient) -> None:
    """Each route that resolves a System has to answer with the same 404."""
    sid = hosted.post("/v1/systems", json=_payload(), headers=_auth(A_KEY)).json()["id"]
    paths = (
        f"/v1/systems/{sid}",
        f"/v1/systems/{sid}/report",
        f"/v1/systems/{sid}/compare?other={sid}",
    )
    for path in paths:
        assert hosted.get(path, headers=_auth(B_KEY)).status_code == 404, path


def test_a_second_tenant_cannot_write_through_the_firsts_system(
    hosted: TestClient,
) -> None:
    """The write path: publish. Worse than a read, because a listing is hard to undo."""
    sid = hosted.post("/v1/systems", json=_payload(), headers=_auth(A_KEY)).json()["id"]
    body = {"consent": True, "acknowledged_data_becomes_public": True}
    blocked = hosted.post(f"/v1/systems/{sid}/publish", json=body, headers=_auth(B_KEY))
    assert blocked.status_code == 404
    # The id is echoed because the caller sent it; what must not come back is tenant
    # A's task text or a review result computed from A's record.
    assert "outcomes" not in blocked.text
    assert "blockers" not in blocked.text


def test_a_tenant_cannot_open_a_run_on_an_unowned_system(hosted: TestClient) -> None:
    sid = hosted.post("/v1/systems", json=_payload(), headers=_auth(A_KEY)).json()["id"]
    compile_response = hosted.post(
        f"/v1/systems/{sid}/compile", json={"quick": True}, headers=_auth(B_KEY)
    )
    invoke_response = hosted.post(
        f"/v1/systems/{sid}/invoke", json={"text": "hi"}, headers=_auth(B_KEY)
    )
    assert compile_response.status_code == 404
    assert invoke_response.status_code == 404


def test_a_keyless_local_plane_still_reads_an_unowned_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trap: local mode has no principals, so nothing may break there.

    A System inserted straight into the store has no owner. In local mode it must stay
    reachable, because that is the mode a self-hoster runs and the mode an older
    database is in.
    """
    monkeypatch.delenv(API_KEYS_ENV, raising=False)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with TestClient(create_app()) as client:
        ctx = client.app.dependency_overrides[get_ctx]()
        unowned = ctx.store.create(
            task="extract restaurant outcomes",
            examples=(
                TaskExample(text="a", outcome={"restaurant": "x"}),
                TaskExample(text="b", outcome={"restaurant": "y"}),
            ),
        )
        assert unowned.principal_id == ""
        response = client.get(f"/v1/systems/{unowned.id}")
        assert response.status_code == 200
        assert "outcomes" in response.json()["task"]


def test_a_tenant_sees_its_own_system_and_only_its_own(hosted: TestClient) -> None:
    """Tenancy must not hide a caller's own rows; that would be the opposite bug."""
    a = hosted.post("/v1/systems", json=_payload("a job"), headers=_auth(A_KEY)).json()
    b = hosted.post("/v1/systems", json=_payload("b job"), headers=_auth(B_KEY)).json()
    listed = hosted.get("/v1/systems", headers=_auth(A_KEY)).json()
    assert [s["id"] for s in listed["systems"]] == [a["id"]]
    assert hosted.get(f"/v1/systems/{b['id']}", headers=_auth(A_KEY)).status_code == 404


# --- the store's own ownership filter --------------------------------------------


def _draft(store: Store, *, principal_id: str) -> DraftSystem:
    return store.create(
        task="t",
        examples=(TaskExample(text="a", outcome={}),),
        principal_id=principal_id,
    )


def test_a_named_owner_cannot_read_another_owners_row() -> None:
    store = Store()
    row = _draft(store, principal_id="tenant-a")
    assert store.get_system(row.id, owner="tenant-a").id == row.id
    with pytest.raises(NotFoundError):
        _ = store.get_system(row.id, owner="tenant-b")


def test_the_unfiltered_scope_sees_everything() -> None:
    """`owner=None` is local mode, and local mode hides nothing."""
    store = Store()
    row = _draft(store, principal_id="tenant-a")
    assert store.get_system(row.id, owner=None).id == row.id
    assert len(store.list_systems(owner=None)) == 1


def test_ownership_survives_approval(tmp_path: Path) -> None:
    """A phase change that dropped the owner would make a compiled row unreachable."""
    store = Store(db_path=tmp_path / "s.db")
    row = _draft(store, principal_id="tenant-a")
    approved = store.approve(row.id, owner="tenant-a")
    assert approved.principal_id == "tenant-a"
    with pytest.raises(NotFoundError):
        _ = store.approve(row.id, owner="tenant-b")


def test_an_older_database_gains_the_column(tmp_path: Path) -> None:
    """A file written before ownership existed must migrate, not crash."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        "CREATE TABLE systems ("
        "id TEXT PRIMARY KEY, task TEXT NOT NULL, task_name TEXT NOT NULL, "
        "phase TEXT NOT NULL, examples TEXT NOT NULL, winner TEXT, report TEXT);"
        "CREATE TABLE runs ("
        "id TEXT PRIMARY KEY, system_id TEXT NOT NULL, status TEXT NOT NULL, "
        "quick INTEGER NOT NULL, model TEXT NOT NULL, report TEXT, winner TEXT, "
        "error TEXT, seq INTEGER NOT NULL);"
    )
    _ = conn.execute(
        "INSERT INTO systems VALUES ('sys_old', 'old job', 'restaurant', 'draft', "
        "'[]', NULL, NULL)"
    )
    conn.commit()
    conn.close()

    store = Store(db_path=path)
    restored = store.get_system("sys_old", owner=None)
    assert isinstance(restored, DraftSystem)
    assert restored.principal_id == ""


# --- C3c: quotas ------------------------------------------------------------------


def test_quota_defaults_are_unlimited(monkeypatch: pytest.MonkeyPatch) -> None:
    """A laptop has no tenants to share, so the default must not refuse anything."""
    monkeypatch.delenv(CONCURRENT_ENV, raising=False)
    monkeypatch.delenv(DOCUMENTS_PER_DAY_ENV, raising=False)
    quota = quota_from_env()
    assert quota.concurrent_compiles == 0
    assert quota.documents_per_day == 0
    UsageLog().check("whoever", quota, asked=10_000)


def test_a_bad_quota_value_reads_as_unlimited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONCURRENT_ENV, "two")
    monkeypatch.setenv(DOCUMENTS_PER_DAY_ENV, "-5")
    quota = quota_from_env()
    assert quota.concurrent_compiles == 0
    assert quota.documents_per_day == 0


def test_the_document_limit_refuses_and_names_the_limit_and_value() -> None:
    """A denial with no reason is indistinguishable from a bug."""
    log = UsageLog()
    _ = log.add("tenant-a", 3)
    with pytest.raises(QuotaRefusedError) as caught:
        log.check("tenant-a", Quota(documents_per_day=4), asked=2)
    message = str(caught.value)
    assert DOCUMENTS_PER_DAY_ENV in message
    assert "3 of 4" in message
    assert "needs 2 more" in message


def test_the_usage_log_counts_per_principal_and_per_day() -> None:
    log = UsageLog()
    assert log.add("tenant-a", 2) == 2
    assert log.add("tenant-a", 1) == 3
    assert log.used_today("tenant-b") == 0
    assert log.used_today("tenant-a", day="1999-01-01") == 0


def test_the_concurrent_compile_limit_refuses_with_the_environment_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The refusal names the limit and its value, so it cannot read as a bug."""
    monkeypatch.setenv(API_KEYS_ENV, _KEYS)
    monkeypatch.setenv(CONCURRENT_ENV, "1")
    monkeypatch.setenv("MEKOY_DB", str(tmp_path / "conc.db"))
    with TestClient(create_app()) as client:
        created = client.post(
            "/v1/systems", json=_payload(), headers=_auth(A_KEY)
        ).json()
        sid = created["id"]
        _ = client.post(
            f"/v1/systems/{sid}/evals", json={"approve": True}, headers=_auth(A_KEY)
        )
        with HeldCompile():
            first = client.post(
                f"/v1/systems/{sid}/compile", json={"quick": True}, headers=_auth(A_KEY)
            )
            assert first.status_code == 200
            second = client.post(
                f"/v1/systems/{sid}/compile", json={"quick": True}, headers=_auth(A_KEY)
            )
            assert second.status_code == 429
            assert CONCURRENT_ENV in second.text
            assert "1" in second.text


def test_one_tenants_compile_does_not_count_against_another(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The limit is per principal, not global."""
    monkeypatch.setenv(API_KEYS_ENV, _KEYS)
    monkeypatch.setenv(CONCURRENT_ENV, "1")
    monkeypatch.setenv("MEKOY_DB", str(tmp_path / "per.db"))
    with TestClient(create_app()) as client, HeldCompile():
        sid = client.post("/v1/systems", json=_payload(), headers=_auth(A_KEY)).json()[
            "id"
        ]
        _ = client.post(
            f"/v1/systems/{sid}/evals", json={"approve": True}, headers=_auth(A_KEY)
        )
        assert (
            client.post(
                f"/v1/systems/{sid}/compile",
                json={"quick": True},
                headers=_auth(A_KEY),
            ).status_code
            == 200
        )
        other = client.post(
            "/v1/systems", json=_payload(), headers=_auth(B_KEY)
        ).json()["id"]
        _ = client.post(
            f"/v1/systems/{other}/evals", json={"approve": True}, headers=_auth(B_KEY)
        )
        assert (
            client.post(
                f"/v1/systems/{other}/compile",
                json={"quick": True},
                headers=_auth(B_KEY),
            ).status_code
            == 200
        )


class HeldCompile:
    """Replace the background compile with a no-op so a run stays in flight.

    A context manager rather than a fixture so the swap is undone even when the body
    raises, and so it is obvious at each call site which requests see a stuck run.
    """

    def __enter__(self) -> None:
        self._original = api_main._run_compile
        api_main._run_compile = _do_nothing  # type: ignore[assignment]

    def __exit__(self, *exc: object) -> None:
        api_main._run_compile = self._original  # type: ignore[assignment]


def _do_nothing(**_kwargs: object) -> None:
    return None


# --- C3d: metering and metrics ----------------------------------------------------


def _metering(
    *, run_id: str = "run_1", principal_id: str = "tenant-a"
) -> CompileMetering:
    return CompileMetering(
        run_id=run_id,
        principal_id=principal_id,
        system_id="sys_1",
        seconds=12.5,
        model_calls=40,
        documents=54,
        model="qwen2.5:7b",
    )


def test_a_metering_row_records_what_a_bill_needs(tmp_path: Path) -> None:
    metrics = Metrics(db_path=tmp_path / "m.db")
    row = metrics.record_compile(_metering())
    assert row.finished_at
    snapshot = metrics.snapshot(principal_id="tenant-a")
    assert snapshot.compiles_metered == 1
    assert snapshot.documents_scored == 54


def test_metering_survives_a_new_process(tmp_path: Path) -> None:
    path = tmp_path / "meter.db"
    _ = Metrics(db_path=path).record_compile(_metering())
    assert Metrics(db_path=path).snapshot().compiles_metered == 1


def test_a_failed_compile_is_visible_in_the_metrics() -> None:
    """A failed compile is otherwise a run row nothing queries and nothing alerts on."""
    metrics = Metrics()
    _ = metrics.record_failure(
        run_id="run_bad",
        principal_id="tenant-a",
        system_id="sys_1",
        error="ModelUnreachableError: connection refused",
    )
    snapshot = metrics.snapshot(principal_id="tenant-a")
    assert snapshot.counters["compile_failed"] == 1
    assert snapshot.failures[0].run_id == "run_bad"
    assert "connection refused" in snapshot.failures[0].error


def test_the_metrics_endpoint_reports_a_failure_and_is_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host reaches this over HTTP; another tenant does not see the error text."""
    monkeypatch.setenv(API_KEYS_ENV, _KEYS)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with TestClient(create_app()) as client:
        ctx = client.app.dependency_overrides[get_ctx]()
        _ = ctx.metrics.record_failure(
            run_id="run_bad",
            principal_id="tenant-a",
            system_id="sys_1",
            error="exploded",
        )
        mine = client.get("/v1/metrics", headers=_auth(A_KEY)).json()
        assert mine["counters"]["compile_failed"] == 1
        assert mine["failures"][0]["run_id"] == "run_bad"
        theirs = client.get("/v1/metrics", headers=_auth(B_KEY)).json()
        assert theirs["counters"]["compile_failed"] == 0
        assert theirs["failures"] == []


def test_the_metrics_endpoint_is_open_and_global_in_local_mode() -> None:
    """A self-hoster is the only user and wants every number."""
    with TestClient(create_app()) as client:
        body = client.get("/v1/metrics")
        assert body.status_code == 200
        assert body.json()["scope"] == "local"


def test_the_metrics_endpoint_needs_a_key_in_hosted_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_KEYS_ENV, _KEYS)
    with TestClient(create_app()) as client:
        assert client.get("/v1/metrics").status_code == 401


class _GoldEcho:
    """Answers each fixture row with its own label, so a quick compile succeeds."""

    local: bool = True

    def __init__(self, rows: tuple[ExampleRecord, ...]) -> None:
        self._by_text = {row.text: row.outcome.model_dump_json() for row in rows}

    def complete(self, *, system: str, user: str, **_rest: object) -> str:
        del system
        tail = user.rsplit("Text:\n", 1)[-1].rsplit("\nJSON:", 1)[0]
        return self._by_text.get(tail, "{}")


def test_a_completed_compile_writes_metering_and_a_success_counter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real path: a compile that finishes leaves a billable row behind."""
    monkeypatch.setenv(API_KEYS_ENV, _KEYS)
    monkeypatch.setenv("MEKOY_DB", str(tmp_path / "compile.db"))
    rows = load_examples(Path("examples/bucko-restaurant/examples.jsonl"))
    application = create_app(completer=_GoldEcho(rows))
    with TestClient(application) as client:
        payload = {
            "task": "extract restaurant outcomes",
            "examples": [row.model_dump(mode="json") for row in rows],
        }
        created = client.post("/v1/systems", json=payload, headers=_auth(A_KEY)).json()
        sid = created["id"]
        _ = client.post(
            f"/v1/systems/{sid}/evals", json={"approve": True}, headers=_auth(A_KEY)
        )
        started = client.post(
            f"/v1/systems/{sid}/compile", json={"quick": True}, headers=_auth(A_KEY)
        ).json()
        run = {"status": "running"}
        for _ in range(120):
            run = client.get(f"/v1/runs/{started['id']}", headers=_auth(A_KEY)).json()
            if run["status"] != "running":
                break
        assert run["status"] == "succeeded", run

        ctx = application.dependency_overrides[get_ctx]()
        snapshot = ctx.metrics.snapshot(principal_id="tenant-a")
        assert snapshot.compiles_metered == 1
        assert snapshot.counters["compile_succeeded"] == 1
        assert snapshot.documents_scored >= 1


def test_the_document_quota_refuses_the_invoke_that_would_cross_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End to end: a compiled System, then the quota refuses the next document."""
    monkeypatch.setenv(API_KEYS_ENV, _KEYS)
    monkeypatch.setenv(DOCUMENTS_PER_DAY_ENV, "1")
    monkeypatch.setenv("MEKOY_DB", str(tmp_path / "quota.db"))
    rows = load_examples(Path("examples/bucko-restaurant/examples.jsonl"))
    application = create_app(completer=_GoldEcho(rows))
    with TestClient(application) as client:
        payload = {
            "task": "extract restaurant outcomes",
            "examples": [row.model_dump(mode="json") for row in rows],
        }
        sid = client.post("/v1/systems", json=payload, headers=_auth(A_KEY)).json()[
            "id"
        ]
        _ = client.post(
            f"/v1/systems/{sid}/evals", json={"approve": True}, headers=_auth(A_KEY)
        )
        started = client.post(
            f"/v1/systems/{sid}/compile", json={"quick": True}, headers=_auth(A_KEY)
        ).json()
        for _ in range(120):
            run = client.get(f"/v1/runs/{started['id']}", headers=_auth(A_KEY)).json()
            if run["status"] != "running":
                break
        assert run["status"] == "succeeded", run

        first = client.post(
            f"/v1/systems/{sid}/invoke",
            json={"text": rows[0].text},
            headers=_auth(A_KEY),
        )
        assert first.status_code == 200
        second = client.post(
            f"/v1/systems/{sid}/invoke",
            json={"text": rows[0].text},
            headers=_auth(A_KEY),
        )
        assert second.status_code == 429
        assert DOCUMENTS_PER_DAY_ENV in second.text
        assert "1" in second.text
        # The other tenant is unaffected by tenant A's quota.
        third = client.post(
            f"/v1/systems/{sid}/invoke",
            json={"text": rows[0].text},
            headers=_auth(B_KEY),
        )
        assert third.status_code == 404
