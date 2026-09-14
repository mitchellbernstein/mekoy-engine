"""API-key auth and the artifact store. PLAN 41.17, local form."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mekoy.api.auth import API_KEY_ENV, AuthSettings, settings_from_env
from mekoy.api.main import create_app
from mekoy.artifacts import (
    ARTIFACTS_DIR_ENV,
    LocalArtifacts,
    store_from_env,
    write_bundle_to,
)
from mekoy.bundle import spec_for
from mekoy.compile import Budget, CompileReport, SearchSpace, compile_system
from mekoy.dataset import ExampleRecord, load_examples, split_examples
from mekoy.errors import CompileError
from mekoy.search import HarnessConfig, Trial

_KEY = "test-key-123"


@pytest.fixture
def keyed_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv(API_KEY_ENV, _KEY)
    return TestClient(create_app())


def test_auth_is_off_without_a_key() -> None:
    """A laptop must keep working; open is the documented default."""
    assert AuthSettings(api_key="").enabled is False
    with TestClient(create_app()) as client:
        assert client.post("/v1/systems", json={}).status_code != 401


def test_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    assert settings_from_env().enabled is False
    monkeypatch.setenv(API_KEY_ENV, _KEY)
    assert settings_from_env().api_key == _KEY


def test_key_comparison_is_exact() -> None:
    settings = AuthSettings(api_key=_KEY)
    assert settings.accepts(_KEY) is True
    assert settings.accepts(_KEY[:-1]) is False
    assert settings.accepts("") is False
    assert settings.accepts(None) is False


def test_v1_needs_the_bearer_key(keyed_client: TestClient) -> None:
    anonymous = keyed_client.post("/v1/systems", json={})
    assert anonymous.status_code == 401
    assert "API key" in anonymous.text

    wrong = keyed_client.post(
        "/v1/systems", json={}, headers={"authorization": "Bearer nope"}
    )
    assert wrong.status_code == 401

    # With the right key the request gets past auth and fails validation instead.
    right = keyed_client.post(
        "/v1/systems",
        json={},
        headers={"authorization": f"Bearer {_KEY}"},
    )
    assert right.status_code != 401


def test_x_api_key_works_too(keyed_client: TestClient) -> None:
    """curl and browser fetch are friendlier with a plain header."""
    response = keyed_client.post("/v1/systems", json={}, headers={"x-api-key": _KEY})
    assert response.status_code != 401


def test_health_stays_open(keyed_client: TestClient) -> None:
    """A liveness probe that needs a credential cannot be probed."""
    assert keyed_client.get("/health").status_code == 200


def test_artifacts_default_to_a_local_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ARTIFACTS_DIR_ENV, raising=False)
    store = store_from_env()
    assert isinstance(store, LocalArtifacts)
    assert store.locator("sys_1").endswith("sys_1")


def test_the_artifact_root_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ARTIFACTS_DIR_ENV, str(tmp_path / "bundles"))
    store = store_from_env()
    assert Path(store.locator("sys_1")).parent == tmp_path / "bundles"


def test_a_bundle_is_written_through_the_store(tmp_path: Path) -> None:
    class _Echo:
        local: bool = True

        def __init__(self, rows: tuple[ExampleRecord, ...]) -> None:
            self._by_text = {r.text: r.outcome.model_dump_json() for r in rows}

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
            return self._by_text.get(tail, "{}")

    rows = load_examples(Path("examples/bucko-restaurant/examples.jsonl"))
    report = compile_system(
        _Echo(rows),
        split_examples(rows),
        SearchSpace.single(),
        Budget(trials=1),
    )
    store = LocalArtifacts(base=tmp_path / "store")
    spec = spec_for(report, task="restaurant", model_id="qwen2.5:7b")
    out = write_bundle_to(store, "sys_1", spec, "card")
    assert (out / "spec.json").is_file()
    assert out.parent == store.root()


def test_a_bundle_path_outside_the_root_is_refused(tmp_path: Path) -> None:
    """A System id is caller-supplied, so the path it resolves to is checked."""

    class _Escaping(LocalArtifacts):
        def locator(self, system_id: str) -> str:
            return str(tmp_path / "outside" / system_id)

    trial = Trial(config=HarnessConfig(), scores=())
    report = CompileReport(
        winner=trial, trials=(trial,), test=trial, stopped_early=False
    )
    spec = spec_for(report, task="t", model_id="m")
    with pytest.raises(CompileError, match="outside"):
        write_bundle_to(_Escaping(base=tmp_path / "store"), "sys_1", spec, "card")


def test_the_api_image_installs_a_websocket_library() -> None:
    """Bare uvicorn answers an upgrade with 'No supported WebSocket library
    detected', so a streaming client sees a broken endpoint rather than a 404."""
    root = Path(__file__).resolve().parents[1]
    image = (root / "Dockerfile.api").read_text()
    assert "uvicorn[standard]" in image
