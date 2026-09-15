# Connect Mekoy to your assistant

The same tools, two transports. Pick the one that matches your setup.

| Your setup | Transport | Works with |
|---|---|---|
| Local hardware (Ollama, Apple Silicon, a GPU) | stdio MCP | Claude Code, Cursor, Codex CLI, Grok Build |
| No local model hardware | Streamable HTTP MCP | Claude.ai connector, ChatGPT plugin, Cursor plugin by URL |

**There is no hosted endpoint to point at.** Mekoy runs on your machine, so the server
is yours to start. If your assistant needs a URL rather than a command, start the local
API and expose it yourself:

```bash
uv run uvicorn mekoy.api.main:app --host 127.0.0.1 --port 8787
ngrok http 8787
# then use https://<your-id>.ngrok.app/mcp
```

## Start here: stdio, no URL needed

```bash
uv run --directory /path/to/mekoy-engine python -m mekoy.mcp_server
```

### Claude Code

```bash
claude mcp add --scope user mekoy -- uv run --directory /path/to/mekoy-engine python -m mekoy.mcp_server
```

Type `/mcp` (not `/mcp mekoy`). Enable Mekoy. Start a new session.

### Cursor

Settings → MCP → add a command server, or put this in `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "mekoy": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mekoy-engine", "python", "-m", "mekoy.mcp_server"]
    }
  }
}
```

### Codex CLI

Use the packaged plugin in `plugins/mekoy/`. `/plugins` after it is in a marketplace.

## If your assistant needs a URL

Start the local API and tunnel it as shown above, then point the connector at
`https://<your-id>.ngrok.app/mcp`.

### Claude.ai

Customize → Connectors → Add custom connector → Web → your tunnel URL. There is no
sign-in for the local server.

### ChatGPT

1. Enable developer mode.
2. Settings → Plugins → create app, or add a custom MCP.
3. Use your tunnel URL.

### Grok

```bash
grok mcp add --transport http mekoy https://<your-id>.ngrok.app/mcp
```

Grok web and iOS reject localhost, so a tunnel is required there rather than optional.

## What the assistant gets

Eight MCP tools. The assistant drives them; it supplies its own model. Mekoy does not
need an API key from you, because the assistant you are already talking to is the model.

The eval gate applies over MCP exactly as it does in the CLI: a compile is refused until
the examples are approved.
