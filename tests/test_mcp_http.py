from fastapi.testclient import TestClient

from mekoy.api.main import create_app


def test_http_mcp_initialize_and_list_tools() -> None:
    client = TestClient(create_app())
    init = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    )
    assert init.status_code == 200
    name = init.json()["result"]["serverInfo"]["name"]
    assert name == "mekoy"
    listed = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    names = {tool["name"] for tool in listed.json()["result"]["tools"]}
    assert "compile_system" in names
    assert "train_model" not in names
