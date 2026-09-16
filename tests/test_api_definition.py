"""A job the caller defined, compiled through the control plane.

The guided flow asks a user what should come out and what a dangerous answer is. Before
this seam those answers had nowhere to go: `POST /v1/systems` took a job sentence and
rows shaped like one of three shipped classes, and every other key was dropped by
`extra="ignore"`. A card that collects an answer nobody uses is a card that lies, so the
tests here are what proves the answers arrive somewhere they are used.

Three claims, one per test:

- the fields a caller declares are the schema their rows are validated against, so a row
  shaped for their own job is not read as a restaurant call;
- a dangerous answer the caller names is the reason the gate rejects, so "the weights do
  not add up" is a real rejection rather than a sentence in a UI;
- a rebuilt System (through SQLite, which is what a restart does) still runs the
  caller's own gate rather than falling back to the shipped class.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from mekoy.api.main import create_app
from mekoy.api.models import RunResponse
from mekoy.api.store import DraftSystem, Store, task_for
from mekoy.dataset import TaskExample
from mekoy.jobs import JobDefinition
from mekoy.tasks import Task, task_from_definition

#: A job the engine has never seen: a delivery manifest whose weights must add up.
_MANIFEST: dict[str, object] = {
    "name": "manifest",
    "task": "Read a shipping manifest and report the carrier and the weights.",
    "fields": [
        {"name": "carrier", "type": "string"},
        {"name": "weight_a", "type": "number"},
        {"name": "weight_b", "type": "number"},
        {"name": "total_weight", "type": "number"},
    ],
    "checks": [
        {
            "kind": "required",
            "fields": ["carrier", "weight_a", "weight_b", "total_weight"],
            "message": "the manifest left a field out",
        },
        {
            "kind": "arithmetic",
            "fields": ["weight_a", "weight_b", "total_weight"],
            "parts": ["weight_a", "weight_b"],
            "total": "total_weight",
            "message": "the weights do not add up",
        },
    ],
}


def _rows() -> list[dict[str, object]]:
    """Four labeled manifests that satisfy the checks above."""
    return [
        {
            "text": f"{carrier} manifest: {a}kg + {b}kg",
            "outcome": {
                "carrier": carrier,
                "weight_a": a,
                "weight_b": b,
                "total_weight": a + b,
            },
        }
        for carrier, a, b in (
            ("DHL", 12.0, 5.5),
            ("FedEx", 3.0, 4.0),
            ("UPS", 20.0, 1.5),
            ("USPS", 8.0, 2.0),
        )
    ]


class _GoldEcho:
    """Returns the gold label for whichever document is in the prompt."""

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
        for row in _rows():
            if str(row["text"]) in user:
                return json.dumps(row["outcome"])
        return "{}"


class _Off:
    """Always answers with weights that do not add up, so a check has work to do."""

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
        return json.dumps(
            {
                "carrier": "NOT-A-CARRIER",
                "weight_a": 1.0,
                "weight_b": 1.0,
                "total_weight": 5.0,
            }
        )


def _client(completer: object | None = None) -> TestClient:
    return TestClient(create_app(completer=completer) if completer else create_app())


def _create(client: TestClient) -> str:
    response = client.post(
        "/v1/systems",
        json={"task": "manifests", "examples": _rows(), "definition": _MANIFEST},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _parse_run(client: TestClient, run_id: str) -> RunResponse:
    response = client.get(f"/v1/runs/{run_id}")
    assert response.status_code == 200
    return RunResponse.model_validate_json(response.content)


def test_a_caller_defined_field_set_is_the_schema_rows_are_checked_against() -> None:
    """The fields the user proposed are the schema, not a shipped class.

    The same rows without a definition are a 400: `carrier` is not a restaurant label.
    With one they are the caller's own job, which is what makes "what should come out"
    an answer the compiler uses.
    """
    with _client() as client:
        rejected = client.post(
            "/v1/systems", json={"task": "manifests", "examples": _rows()}
        )
        assert rejected.status_code == 400
        assert "not a valid restaurant label" in rejected.text

        assert _create(client).startswith("sys_")


def test_a_required_check_accepts_an_honest_no() -> None:
    """A boolean answer of `false` is complete, not missing.

    The guided flow lets a user require a claim field such as `booked`, and the first
    cut of `_missing` treated a falsy value as absent — so a required check over a
    boolean rejected every honest "no" and accepted only an answer claiming the thing
    happened. That inverts what a safety check is for, which is why this is pinned.
    """
    definition = {
        "name": "reservation",
        "task": "Report whether a booking happened.",
        "fields": [
            {"name": "restaurant", "type": "string"},
            {"name": "booked", "type": "boolean"},
        ],
        "checks": [
            {
                "kind": "required",
                "fields": ["restaurant", "booked"],
                "message": "missing",
            }
        ],
    }
    task = task_from_definition(JobDefinition.model_validate(definition))
    honest = task.model.model_validate({"restaurant": "Uchi", "booked": False})
    assert task.gate(honest) == ()
    absent = task.model.model_validate({"restaurant": "Uchi", "booked": None})
    assert task.gate(absent) != ()


def test_the_declared_checks_are_the_gate_a_compile_runs() -> None:
    """A dangerous answer the caller named is the reason the gate rejects.

    `_Off` answers with weights that do not add up. That is the arithmetic check the
    caller declared, so the run reports the rejection rather than scoring a wrong answer
    as fine.
    """
    with _client(_Off()) as client:
        system_id = _create(client)
        _ = client.post(f"/v1/systems/{system_id}/evals", json={"approve": True})
        started = client.post(f"/v1/systems/{system_id}/compile", json={"quick": True})
        assert started.status_code == 200
        run = _parse_run(client, str(started.json()["id"]))
        assert run.status == "succeeded", run.error
        assert run.report is not None
        # The reason string the gate produced, naming the parts and the total the caller
        # wrote. This is the check firing, not a message the report invented.
        assert "weight_a+weight_b=2.00 != total_weight=5.00" in run.report


def test_a_defined_job_survives_a_restart_and_keeps_its_own_gate(
    tmp_path: Path,
) -> None:
    """Recovery must rebuild the caller's task, not the shipped default.

    SQLite is what a restart reads. If the definition were not stored, a reloaded System
    would be scored by the restaurant gate, which is a different measurement than the
    one that was run, reported under the same System id.
    """
    db = tmp_path / "store.sqlite"
    store = Store(db_path=db)
    task = task_from_definition(JobDefinition.model_validate(_MANIFEST))
    created = store.create(
        task="manifests",
        examples=tuple(_example(task, row) for row in _rows()),
        task_name=task.name,
        definition=_MANIFEST,
    )
    assert isinstance(created, DraftSystem)

    reloaded = Store(db_path=db)
    other = task_for(reloaded.get_system(created.id))
    assert other.name == "manifest"
    bad = {"carrier": "DHL", "weight_a": 1.0, "weight_b": 1.0, "total_weight": 5.0}
    reasons = other.gate(other.model.model_validate(bad))
    assert reasons, "a manifest that does not add up must be rejected"


def _example(task: Task, row: dict[str, object]) -> TaskExample:
    """One stored example, labeled with the caller's own model."""
    return TaskExample(
        text=str(row["text"]), outcome=task.model.model_validate(row["outcome"])
    )


def test_a_bar_the_caller_set_changes_when_the_search_stops() -> None:
    """An SLO the caller names reaches the budget rather than being dropped.

    `_GoldEcho` is perfect, so a bar of zero is met by the first candidate and the
    search stops instead of pricing the rest. Without the wiring the request still
    succeeds, which is exactly the silent drop this test exists to catch.
    """
    with TestClient(create_app(completer=_GoldEcho())) as client:
        system_id = _create(client)
        _ = client.post(f"/v1/systems/{system_id}/evals", json={"approve": True})
        response = client.post(
            f"/v1/systems/{system_id}/compile",
            json={
                "quick": True,
                "slos": {
                    "quality": 0.0,
                    "cost_per_doc": 1.0,
                    "latency_ms": 100000.0,
                },
            },
        )
        assert response.status_code == 200, response.text
        run = _parse_run(client, str(response.json()["id"]))
    assert run.status == "succeeded", run.error
    assert run.report is not None
    assert "stopped_on_slo=True" in run.report
