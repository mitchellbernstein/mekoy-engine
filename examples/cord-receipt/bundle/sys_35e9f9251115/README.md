# System

receipt extract-and-verify

## Invoke

Install Ollama.
Pull `qwen2.5:7b`.

Put the document text in a file, then run:

```
mekoy invoke receipt.txt --model qwen2.5:7b --k-shot 0 --retries 1
```

`spec.json` holds the schema, SLOs, and ownership flags.
`report.txt` is the compile card.

## Optional compose

```yaml
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
```

Pull `qwen2.5:7b` after the container starts, then invoke as above.
