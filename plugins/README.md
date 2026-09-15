# Mekoy Codex / ChatGPT plugin

Layout follows [Package your plugin](https://developers.openai.com/plugins/build/plugins):

```
plugins/mekoy/
  .codex-plugin/plugin.json   required manifest
  .mcp.json                   Streamable HTTP MCP
  skills/compile/SKILL.md
```

Required entry is `.codex-plugin/plugin.json`, not a root `plugin.json`.

## Local, just you

Docs: personal marketplace at `~/.agents/plugins/marketplace.json` plus plugin copy under `~/.codex/plugins/`. CLI: `codex plugin marketplace add <dir>`.

Already wired on this machine. Restart the ChatGPT desktop app, open **Plugins**, pick the **Mitchell local** marketplace, install **Mekoy**.

Do not use Plugin Creator to generate a new plugin. This folder is the plugin.

## Later: public directory

1. Stand up a public HTTPS MCP endpoint, since the local server is stdio-only.
2. Open the [plugin submission portal](https://chatgpt.com/plugins), Create plugin, remote MCP submission.
3. Publishing to your ChatGPT workspace (Personal → ⋯ → Publish) is **not** the universal Codex/ChatGPT directory.
