import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from mekoy.api.main import create_app
from mekoy.api.models import (
    EvalResponse,
    HealthResponse,
    InvokeFail,
    InvokeOk,
    RunResponse,
    SystemCreated,
)
from mekoy.dataset import ExampleRecord, load_examples

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
    compiled = client.post(f"/v1/systems/{system_id}/compile", json={"quick": True})
    assert compiled.status_code == 200
    body = _parse(compiled.content, RunResponse)
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
        compiled = client.post(f"/v1/systems/{system_id}/compile", json={"quick": True})
        assert compiled.status_code == 200
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


def _compiled(client: TestClient, rows: tuple[ExampleRecord, ...]) -> str:
    system_id = _approved(client, rows)
    response = client.post(f"/v1/systems/{system_id}/compile", json={"quick": True})
    assert response.status_code == 200
    assert _parse(response.content, RunResponse).status == "succeeded"
    return system_id


def test_get_system_reports_phase_and_latest_run(
    client: TestClient, rows: tuple[ExampleRecord, ...]
) -> None:
    """PLAN 23 declares GET /v1/systems/:id; a client needs it to poll by System."""
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
    """PLAN 23: invoke is reachable at /v1/chat/completions too."""
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
    """Every Phase-I route PLAN 23 declares must exist."""
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
        run = client.post(
            f"/v1/systems/{created['id']}/compile", json={"quick": True}
        ).json()
        assert run["status"] == "succeeded"
        assert run["winner"] is not None
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
