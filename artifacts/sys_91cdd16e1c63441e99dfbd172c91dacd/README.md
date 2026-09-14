# System

extract restaurant call outcomes

## Run it

1. Start the model server:

```
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:7b
```

2. Install the harness (the compiled System is the harness, not the
   weights):

```
uv tool install mekoy   # or: pip install -e <repo>
```

3. Put a document in a file and extract:

```
mekoy invoke document.txt --model qwen2.5:7b --k-shot 0 --retries 1
```

To build the all-in-one image instead of using compose, use the
repository Dockerfile, which serves Ollama and the CLI together.

## What is in here

- `spec.json` — schema, SLOs, ownership flags, and the harness knobs the
  compile selected.
- `report.txt` — the compile card: what was tried and what it scored.
- `docker-compose.yml` — the model server.
