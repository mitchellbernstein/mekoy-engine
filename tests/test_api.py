import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

import mekoy.api.main as api_main
from mekoy.api.main import create_app
from mekoy.api.models import (
    EvalResponse,
    HealthResponse,
    InvokeFail,
    InvokeOk,
    RunResponse,
    SystemCreated,
)
from mekoy.api.store import RunStatus, Store
from mekoy.bundle import spec_for, write_bundle
from mekoy.compile import CompileReport, Trial
from mekoy.dataset import ExampleRecord, TaskExample, load_examples
from mekoy.search import HarnessConfig

_FIXTURE = Path("examples/bucko-restaurant/examples.jsonl")


class _GoldEcho:
    def __init__(self, rows: tuple[ExampleRecord, ...]) -> None:
        self._by_text: dict[str, str] = {
            row.text: row.outcome.model_dump_json() for row in rows
        }

    local: bool = True

    def complete(
        self,
        *,
        system: str,
        user: str,
        constrained: bool = True,
        temperature: float = 0.0,
        schema: dict[str, object] | None = None,
    ) -> str:
        del system, constrained, temperature, schema
        for text, raw in self._by_text.items():
            if text in user:
                return raw
        return "{}"


class _Empty:
    local: bool = True

    def complete(
        self,
        *,
        system: str,
        user: str,
        constrained: bool = True,
        temperature: float = 0.0,
        schema: dict[str, object] | None = None,
    ) -> str:
        del system, user, constrained, temperature, schema
        return "{}"


@pytest.fixture
def rows() -> tuple[ExampleRecord, ...]:
    return load_examples(_FIXTURE)


@pytest.fixture
def client(rows: tuple[ExampleRecord, ...]) -> Iterator[TestClient]:
    application = create_app(completer=_GoldEcho(rows))
    with TestClient(application) as test_client:
        yield test_client


def _payload(rows: tuple[ExampleRecord, ...]) -> dict[str, object]:
    return {
        "task": "extract restaurant call outcomes",
        "examples": [row.model_dump(mode="json") for row in rows],
    }


def _parse[T: BaseModel](content: bytes, model: type[T]) -> T:
    return model.model_validate_json(content)


def _approved(client: TestClient, rows: tuple[ExampleRecord, ...]) -> str:
    created = _parse(
        client.post("/v1/systems", json=_payload(rows)).content, SystemCreated
    )
    response = client.post(f"/v1/systems/{created.id}/evals", json={"approve": True})
    assert response.status_code == 200
    evaluated = _parse(response.content, EvalResponse)
    assert evaluated.approved is True
    return created.id


def test_health_does_not_need_a_model() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert _parse(response.content, HealthResponse).status == "ok"


def test_openapi_lists_control_plane(client: TestClient) -> None:
    text = client.get("/openapi.json").text
    for path in (
        "/health",
        "/v1/systems",
        "/v1/systems/{system_id}/evals",
        "/v1/systems/{system_id}/compile",
        "/v1/systems/{system_id}/invoke",
        "/v1/runs/{run_id}",
    ):
        assert path in text


def test_create_rejects_one_example(
    client: TestClient, rows: tuple[ExampleRecord, ...]
) -> None:
    payload = {
        "task": "extract",
        "examples": [rows[0].model_dump(mode="json")],
    }
    assert client.post("/v1/systems", json=payload).status_code == 422


def test_compile_requires_eval_approval(
    client: TestClient, rows: tuple[ExampleRecord, ...]
) -> None:
    created_resp = client.post("/v1/systems", json=_payload(rows))
    assert created_resp.status_code == 200
    created = _parse(created_resp.content, SystemCreated)
    proposed = client.post(f"/v1/systems/{created.id}/evals", json={"approve": False})
    assert proposed.status_code == 200
    assert _parse(proposed.content, EvalResponse).approved is False
    blocked = client.post(f"/v1/systems/{created.id}/compile", json={"quick": True})
    assert blocked.status_code == 409
    assert "not approved" in blocked.text


def test_eval_approve_is_idempotent(
    client: TestClient, rows: tuple[ExampleRecord, ...]
) -> None:
    system_id = _approved(client, rows)
    again = client.post(f"/v1/systems/{system_id}/evals", json={"approve": True})
    assert again.status_code == 200
    assert _parse(again.content, EvalResponse).phase == "eval_approved"


def test_quick_compile_invoke_and_get_run(
    client: TestClient, rows: tuple[ExampleRecord, ...]
) -> None:
    system_id = _approved(client, rows)
    body = _settled(client, _start_compile(client, system_id))
    assert body.status == "succeeded"
    assert body.winner is not None
    assert body.winner.k_shot == 0
    assert body.winner.retries == 1
    assert body.report is not None
    assert "training: skipped" in body.report
    fetched = client.get(f"/v1/runs/{body.id}")
    assert fetched.status_code == 200
    assert _parse(fetched.content, RunResponse).status == "succeeded"
    invoked = client.post(
        f"/v1/systems/{system_id}/invoke", json={"text": rows[0].text}
    )
    assert invoked.status_code == 200
    payload = _parse(invoked.content, InvokeOk)
    assert payload.ok is True
    assert payload.outcome.restaurant == rows[0].outcome.restaurant


def test_invoke_before_compile_conflicts(
    client: TestClient, rows: tuple[ExampleRecord, ...]
) -> None:
    system_id = _approved(client, rows)
    response = client.post(
        f"/v1/systems/{system_id}/invoke", json={"text": rows[0].text}
    )
    assert response.status_code == 409
    assert "not compiled" in response.text


def test_unknown_ids_are_404(client: TestClient) -> None:
    assert client.get("/v1/runs/run_missing").status_code == 404
    missing = client.post("/v1/systems/sys_missing/evals", json={"approve": True})
    assert missing.status_code == 404


def test_invoke_reports_verify_fail(rows: tuple[ExampleRecord, ...]) -> None:
    application = create_app(completer=_Empty())
    with TestClient(application) as client:
        system_id = _approved(client, rows)
        _ = _settled(client, _start_compile(client, system_id))
        invoked = client.post(
            f"/v1/systems/{system_id}/invoke", json={"text": rows[0].text}
        )
        assert invoked.status_code == 200
        payload = _parse(invoked.content, InvokeFail)
        assert payload.ok is False
        assert payload.error


def test_create_app_isolates_store(rows: tuple[ExampleRecord, ...]) -> None:
    payload = _payload(rows)
    with (
        TestClient(create_app(completer=_GoldEcho(rows))) as first,
        TestClient(create_app(completer=_GoldEcho(rows))) as second,
    ):
        created = _parse(first.post("/v1/systems", json=payload).content, SystemCreated)
        unseen = second.post(f"/v1/systems/{created.id}/evals", json={"approve": True})
        assert unseen.status_code == 404


def _start_compile(client: TestClient, system_id: str) -> str:
    """POST a compile and return its run id.

    The response is built before the work runs, so its status is `running` even when
    the compile has already finished by the time the caller reads it. Anything that
    wants the outcome has to ask the run, which is what a real client does too.
    """
    response = client.post(f"/v1/systems/{system_id}/compile", json={"quick": True})
    assert response.status_code == 200
    started = _parse(response.content, RunResponse)
    assert started.status == "running", "the request should not wait for the compile"
    assert started.id
    return started.id


def _settled(client: TestClient, run_id: str) -> RunResponse:
    """The run as the store now holds it."""
    response = client.get(f"/v1/runs/{run_id}")
    assert response.status_code == 200
    return _parse(response.content, RunResponse)


def _compiled(client: TestClient, rows: tuple[ExampleRecord, ...]) -> str:
    system_id = _approved(client, rows)
    run_id = _start_compile(client, system_id)
    settled = _settled(client, run_id)
    assert settled.status == "succeeded", settled.error
    return system_id


def test_get_system_reports_phase_and_latest_run(
    client: TestClient, rows: tuple[ExampleRecord, ...]
) -> None:
    """GET /v1/systems/:id exists; a client needs it to poll by System."""
    created = _parse(
        client.post("/v1/systems", json=_payload(rows)).content, SystemCreated
    )
    before = client.get(f"/v1/systems/{created.id}")
    assert before.status_code == 200
    assert '"run":null' in before.text.replace(" ", "")

    system_id = _compiled(client, rows)
    after = client.get(f"/v1/systems/{system_id}")
    assert after.status_code == 200
    assert '"phase":"compiled"' in after.text.replace(" ", "")


def test_report_endpoint_returns_the_card(
    client: TestClient, rows: tuple[ExampleRecord, ...]
) -> None:
    system_id = _compiled(client, rows)
    response = client.get(f"/v1/systems/{system_id}/report")
    assert response.status_code == 200
    assert "training: skipped" in response.text
    assert "test" in response.text


def test_report_before_compile_is_refused(
    client: TestClient, rows: tuple[ExampleRecord, ...]
) -> None:
    system_id = _approved(client, rows)
    response = client.get(f"/v1/systems/{system_id}/report")
    assert response.status_code == 409  # phase error, not a 500


def test_deploy_download_writes_a_bundle(
    client: TestClient, rows: tuple[ExampleRecord, ...], tmp_path: Path
) -> None:
    system_id = _compiled(client, rows)
    response = client.post(f"/v1/systems/{system_id}/deploy", json={"mode": "download"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "download"
    bundle = Path(body["path"])
    for name in ("spec.json", "report.txt", "README.md", "docker-compose.yml"):
        assert (bundle / name).is_file(), name


def test_hosted_deploy_is_declined_honestly(
    client: TestClient, rows: tuple[ExampleRecord, ...]
) -> None:
    system_id = _compiled(client, rows)
    for mode in ("hosted", "self_host"):
        body = client.post(
            f"/v1/systems/{system_id}/deploy", json={"mode": mode}
        ).json()
        assert body["mode"] == mode
        assert "not hosted" in body["detail"]


def test_compare_requires_two_comparable_systems(
    client: TestClient, rows: tuple[ExampleRecord, ...]
) -> None:
    left = _compiled(client, rows)
    right = _compiled(client, rows)
    response = client.get(f"/v1/systems/{left}/compare", params={"other": right})
    assert response.status_code == 200
    body = response.json()
    assert body["left"] == left
    assert body["n_test"] == body["n_test"]
    assert body["quality_winner"] in {"left", "right", "tie"}
    assert body["verdict"]


def test_openai_compatible_invoke(
    client: TestClient, rows: tuple[ExampleRecord, ...]
) -> None:
    """Invoke is reachable at /v1/chat/completions too."""
    system_id = _compiled(client, rows)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": system_id,
            "messages": [{"role": "user", "content": rows[0].text}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == system_id
    content = body["choices"][0]["message"]["content"]
    assert '"booked"' in content
    assert json.loads(content)["restaurant"]


def test_openai_compatible_invoke_rejects_a_bad_gate(
    client: TestClient, rows: tuple[ExampleRecord, ...]
) -> None:
    """A gate failure must surface as an error, not as empty content."""
    system_id = _compiled(client, rows)
    response = client.post(
        "/v1/chat/completions",
        json={"model": system_id, "messages": [{"role": "user", "content": "nothing"}]},
    )
    assert response.status_code == 400
    assert "verify failed" in response.text


def test_the_declared_api_surface_is_complete(client: TestClient) -> None:
    """Every route the first phase declares must exist."""
    paths = client.get("/openapi.json").json()["paths"]
    for declared in (
        "/v1/systems",
        "/v1/systems/{system_id}",
        "/v1/systems/{system_id}/evals",
        "/v1/systems/{system_id}/compile",
        "/v1/runs/{run_id}",
        "/v1/systems/{system_id}/report",
        "/v1/systems/{system_id}/deploy",
        "/v1/systems/{system_id}/invoke",
        "/v1/systems/{system_id}/compare",
        "/v1/chat/completions",
    ):
        assert declared in paths, declared


class _Recording:
    """Records the model id each completion was asked for."""

    local: bool = True

    def __init__(self, rows: tuple[ExampleRecord, ...]) -> None:
        self._by_text = {row.text: row.outcome.model_dump_json() for row in rows}
        self.models: list[str] = []
        self._model = ""

    def complete(
        self,
        *,
        system: str,
        user: str,
        constrained: bool = True,
        temperature: float = 0.0,
        schema: dict[str, object] | None = None,
    ) -> str:
        del system, constrained, temperature, schema
        for text, raw in self._by_text.items():
            if text in user:
                return raw
        return "{}"


def test_a_run_records_the_model_it_compiled_against(
    rows: tuple[ExampleRecord, ...],
) -> None:
    """A System is the harness plus the model; invoke must replay both."""
    application = create_app(completer=_Recording(rows))
    with TestClient(application) as client:
        system_id = _approved(client, rows)
        _ = client.post(
            f"/v1/systems/{system_id}/compile",
            json={"quick": True, "model": "qwen2.5:14b"},
        )
        run = client.get(f"/v1/systems/{system_id}").json()["run"]
        assert run["winner"] is not None
        detail = client.get(f"/v1/systems/{system_id}/report")
        assert detail.status_code == 200
        # The model is recorded on the run, which is what invoke replays.
        report = client.get(f"/v1/systems/{system_id}/report").json()
        assert report["run_id"] == run["id"]


def test_chat_completions_does_not_treat_the_system_id_as_a_model(
    rows: tuple[ExampleRecord, ...],
) -> None:
    """The bug this exists for: Ollama 404'd on `sys_...` as a model name."""
    application = create_app(completer=_Recording(rows))
    with TestClient(application) as client:
        system_id = _approved(client, rows)
        _ = client.post(
            f"/v1/systems/{system_id}/compile",
            json={"quick": True, "model": "qwen2.5:14b"},
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": system_id,
                "messages": [{"role": "user", "content": rows[0].text}],
            },
        )
        assert response.status_code == 200
        assert response.json()["model"] == system_id


_RECEIPT_ROWS = [
    {
        "text": (
            "Cafe Rio\n2 Taco @ 5.00 = 10.00\nSubtotal 10.00\nTax 0.80\nTotal 10.80 USD"
        ),
        "receipt": {
            "merchant": "Cafe Rio",
            "date": "2024-03-02",
            "currency": "USD",
            "subtotal": 10.0,
            "tax": 0.8,
            "total": 10.8,
            "items": [
                {"desc": "Taco", "qty": 2, "unit_price": 5.0, "line_total": 10.0}
            ],
        },
    },
    {
        "text": "HEB\nSub 7.49 Tax 0.61 Total 8.10",
        "receipt": {
            "merchant": "HEB",
            "date": "2024-01-15",
            "subtotal": 7.49,
            "tax": 0.61,
            "total": 8.1,
        },
    },
    {
        "text": "Uchi\nSub 16.50 Tax 1.36 Total 17.86 USD",
        "receipt": {
            "merchant": "Uchi",
            "date": "2024-05-04",
            "currency": "USD",
            "subtotal": 16.5,
            "tax": 1.36,
            "total": 17.86,
        },
    },
]


def test_a_receipt_shaped_system_can_be_created() -> None:
    """The bug this exists for: the portal's own samples returned 422.

    The create endpoint only accepted restaurant rows, so the primary UX could not
    create the receipt System its samples describe.
    """
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/systems", json={"task": "receipts", "examples": _RECEIPT_ROWS}
        )
        assert response.status_code == 200, response.text
        assert response.json()["n_examples"] == len(_RECEIPT_ROWS)


def test_a_classification_shaped_system_can_be_created() -> None:
    rows = [
        {"text": "How do I locate my card?", "label": "card_arrival"},
        {"text": "My card never arrived", "label": "card_arrival"},
        {"text": "Where is my new card?", "label": "card_arrival"},
    ]
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/systems", json={"task": "intents", "examples": rows}
        )
        assert response.status_code == 200, response.text


def test_a_row_with_no_label_key_names_the_row() -> None:
    rows = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    with TestClient(create_app()) as client:
        response = client.post("/v1/systems", json={"task": "x", "examples": rows})
        assert response.status_code == 400
        assert "row 0" in response.text


def test_a_receipt_system_compiles_with_the_receipt_task() -> None:
    """The stored task class must drive compile, not the restaurant default."""
    completer = _ReceiptEcho()
    with TestClient(create_app(completer=completer)) as client:
        created = client.post(
            "/v1/systems", json={"task": "receipts", "examples": _RECEIPT_ROWS}
        ).json()
        _ = client.post(f"/v1/systems/{created['id']}/evals", json={"approve": True})
        run = _settled(client, _start_compile(client, created["id"]))
        assert run.status == "succeeded", run.error
        assert run.winner is not None
        # The control plane must remember which task class it stored, or compile
        # would fall back to the restaurant schema.
        detail = client.get(f"/v1/systems/{created['id']}").json()
        assert detail["task_name"] == "receipt"


class _ReceiptEcho:
    """Returns the gold receipt for whichever labeled text is in the prompt."""

    local: bool = True

    def complete(
        self,
        *,
        system: str,
        user: str,
        constrained: bool = True,
        temperature: float = 0.0,
        schema: dict[str, object] | None = None,
    ) -> str:
        del system, constrained, temperature, schema
        tail = user.rsplit("Text:\n", 1)[-1].rsplit("\nJSON:", 1)[0]
        for row in _RECEIPT_ROWS:
            if row["text"] == tail:
                return json.dumps(row["receipt"])
        return "{}"


def _row() -> dict:
    """One labeled example row, shaped the way the API expects."""
    return {
        "text": "Uchi, table for 2 Friday 7pm under Maya.",
        "outcome": {
            "restaurant": "Uchi",
            "intent": "reservation",
            "status": "confirmed",
            "party_size": 2,
            "when": "Friday 7pm",
            "under_name": "Maya",
            "evidence": "table for 2 Friday 7pm under Maya",
            "booked": True,
        },
    }


def test_a_run_is_never_left_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every exit from a compile has to resolve the run.

    A run left in `running` is worse than a failed one: the caller cannot tell a slow
    compile from a dead one, and nothing will ever move it. A dead model server was
    already covered, because `ModelUnreachableError` is a `CompileError` - what was not
    covered is anything else, and an unexpected exception is exactly when a run is most
    likely to be abandoned.
    """

    def explode(*_args: object, **_kwargs: object) -> None:
        msg = "something nobody planned for"
        raise RuntimeError(msg)

    monkeypatch.setattr(api_main, "compile_system", explode)

    application = api_main.create_app()
    client = TestClient(application, raise_server_exceptions=False)
    created = client.post(
        "/v1/systems", json={"task": "job", "examples": [_row(), _row(), _row()]}
    ).json()
    sid = created["id"]
    _ = client.post(f"/v1/systems/{sid}/evals", json={"approve": True})

    run_id = _start_compile(client, sid)
    settled = _settled(client, run_id)
    assert settled.status == "failed", f"run left as {settled.status!r}"

    run = client.get(f"/v1/systems/{sid}").json()["run"]
    assert run is not None, "the run disappeared"
    assert run["status"] == "failed", f"run left as {run['status']!r}"
    assert "RuntimeError" in (run.get("error") or "")


def test_recover_rebuilds_the_index_from_the_artifacts(tmp_path: Path) -> None:
    """A System is a directory; the database is only how it is found.

    Losing the index while the directories survive is recoverable, and was not: the API
    would list nothing while the Systems sat on disk.
    """
    artifacts = tmp_path / "artifacts"
    winner = Trial(
        config=HarnessConfig(k_shot=2, retries=1, constrained=True), scores=()
    )
    report = CompileReport(
        winner=winner, trials=(winner,), test=winner, stopped_early=False
    )
    spec = spec_for(report, task="restaurant call extraction", model_id="qwen2.5:7b")
    rows = tuple(
        TaskExample(text=f"call {i}", outcome={"restaurant": "R"}) for i in range(3)
    )
    _ = write_bundle(artifacts / "sys_rebuilt", spec, "card\n", examples=rows)

    fresh = Store(db_path=tmp_path / "index.db")
    assert fresh.list_systems() == (), "the index starts empty"
    failed, recovered = fresh.recover(artifacts)

    assert failed == 0
    assert recovered == 1
    record = fresh.get_system("sys_rebuilt")
    assert record.winner is not None, "the harness travelled with the bundle"
    assert len(record.examples) == 3, "and so did the labeled rows"


def test_recover_fails_a_run_the_previous_process_left_running(tmp_path: Path) -> None:
    """A run left `running` by a dead process is worse than a failed one.

    The caller cannot tell a slow compile from a dead one, and nothing will ever move
    it. Anything still running when a fresh process starts is orphaned by definition.
    """
    db = tmp_path / "index.db"
    first = Store(db_path=db)
    draft = first.create(task="a job", examples=(), task_name="restaurant")
    _ = first.approve(draft.id)
    run = first.create_run(draft.id, quick=True)
    assert run.status is RunStatus.RUNNING

    second = Store(db_path=db)  # a new process over the same file
    failed, _recovered = second.recover(tmp_path / "nothing-here")

    assert failed == 1
    settled = second.get_run(run.id)
    assert settled.status is RunStatus.FAILED
    assert "interrupted" in (settled.error or "")
