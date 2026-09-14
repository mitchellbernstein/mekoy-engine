# How to add Mekoy (every host)

See `connectors/INSTALL.md` for Claude.ai, ChatGPT, Codex, Cursor, Grok, and local stdio.

# Local stdio (Claude Code / Cursor on your GPU)

Claude Code and Cursor call Mekoy over stdio JSON-RPC. The server uses the stdlib. It does not add a Python package.

## Claude Code

```bash
claude mcp add mekoy -- uv run python -m mekoy.mcp_server
```

## Cursor

Put this in `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "mekoy": {
      "command": "uv",
      "args": ["run", "python", "-m", "mekoy.mcp_server"]
    }
  }
}
```

## Tools

`inspect_task`, `propose_eval`, `compile_system`, `get_compile_status`, `get_report`, `deploy_system`, `invoke_system`, `list_systems`.

To unlock compile, call `propose_eval` with `approve` set to `true`. `compile_system` refuses without that yes. The stamp is `.eval-approved` next to the examples file, same as `mekoy eval --approve`.

If `deploy_system` is called with `hosted` or `self_host`, it returns `not hosted; use download/invoke`. Use `download` or `invoke_system` on this machine.

The connector does not expose `train_model`.
