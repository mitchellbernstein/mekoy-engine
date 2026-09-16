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

`inspect_task`, `ask_questions`, `record_answers`, `propose_eval`, `compile_system`, `get_compile_status`, `get_report`, `compare_models`, `deploy_system`, `invoke_system`, `list_systems`.

`ask_questions` asks and changes nothing. It returns the six questions a compile needs,
marks which the job and examples already answered, and states the honest cost of running
a comparison. Put them to the user, then call `record_answers`.

`record_answers` refuses without `dangerous` and `good_enough`. One becomes the hard
gate; the other is the search's stop condition. A compile on a default the user never
chose is worse than a refusal. The answers persist beside the examples in `.intake.json`,
so a connector process that dies between turns does not lose them, and `run_comparison`
carries the user's preference for running the other models first.

To unlock compile, call `propose_eval` with `approve` set to `true`. `compile_system` refuses without that yes. The stamp is `.eval-approved` next to the examples file, same as `mekoy eval --approve`.

`compare_models` scores this System and one other model on the same held-out rows with the
harness held fixed, so the score is comparable rather than asserted. It needs a compiled
System: call `compile_system` first.

If `deploy_system` is called with `hosted` or `self_host`, it returns `not hosted; use download/invoke`. Use `download` or `invoke_system` on this machine.

The connector does not expose `train_model`.
