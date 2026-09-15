"""Check the local setup and say what is slowing it down, and what it cannot afford.

The engine's measured weakness is wall-clock time, not accuracy: a compile that scores
0.940 at $0/doc still takes eight seconds a document, and most of that is waiting on a
model server that answers one request at a time.

Two separate things decide how much of that waiting can be removed, and neither is
visible from inside the engine:

1. **How many requests the server will serve at once.** Ollama passes one slot by
   default, so requests queue no matter how many the engine sends.
2. **How much memory those slots cost.** A KV cache grows linearly with the context
   reserved for it, so raising the request count multiplies memory. On a machine with
   a few gigabytes spare, the honest answer to "run four at once" is no.

So the check computes both before recommending anything. The arithmetic is the formula
from the model's own config — `layers x KV heads x head dimension x 2 for K and V x
bytes per value` — which makes the recommendation a calculation rather than an opinion.

There is a third thing, and `--measure` exists because of it: **a wider pool does not
always buy speed.** On this machine, four slots moved a batch from 16.2s to 15.4s while
per-document latency rose from 2.0s to 7.5s. That is time-slicing, not parallelism, and
it is what memory-bandwidth-bound hardware does — the published 2.8x for this setting
came from hardware with bandwidth to spare and did not reproduce here. Both outcomes are
real, nothing in the server's configuration distinguishes them, and only a timing run
settles which one a given machine is. So `probe` measures instead of quoting.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from mekoy.search import _concurrency  # the setting and its reader live together

#: Bytes per cached value. Local models are served in float16, so two bytes.
_BYTES_PER_VALUE = 2

#: The widest pool worth pricing. Past this, memory is the whole story.
_MAX_SENSIBLE_WIDTH = 8

#: Below this ratio, concurrency bought nothing worth its memory. Set above 1.0 because
#: an identical second run varies by a few percent on a shared machine.
_SPEEDUP_WORTH_KEEPING = 1.15

#: Context the engine actually occupies. A brief plus a few examples plus one document
#: is a few thousand tokens, and the KV cache is priced per reserved token, so reserving
#: the model's full context multiplies memory for room we never use.
_OUR_CONTEXT = 8192


@dataclass(frozen=True)
class Finding:
    """One thing the check looked at, and what it means for speed."""

    name: str
    detail: str
    advice: str | None = None

    @property
    def is_problem(self) -> bool:
        """True when this is costing the user time right now."""
        return self.advice is not None


@dataclass(frozen=True)
class ModelShape:
    """Just enough of a model's config to price its KV cache."""

    name: str
    layers: int
    kv_heads: int
    head_dim: int
    context: int

    @property
    def kv_bytes_per_token(self) -> int:
        """KV memory for one token: layers x KV heads x head dim x 2 x bytes per value.

        Every factor earns its place. Each layer keeps its own K and V, the cache grows
        linearly with tokens, and nothing is shared between requests unless the server
        implements prefix sharing. For the 7B this machine runs that is
        `28 x 4 x 128 x 2 x 2 = 56 KB per token`, so one 32K context costs 1.9 GB before
        a second request exists.
        """
        return self.layers * self.kv_heads * self.head_dim * 2 * _BYTES_PER_VALUE

    def kv_bytes(self, *, tokens: int, width: int) -> int:
        """KV memory for `width` requests each holding up to `tokens` cached tokens.

        The server multiplies the context by the slot count: a server told to hold
        8192 tokens per request and to serve four at once reserves 32768 tokens of KV
        cache, not 8192. Pricing one slot and multiplying by hand gets the same answer;
        assuming the context is shared between slots does not.
        """
        return self.kv_bytes_per_token * tokens * width


def _base_url() -> str:
    return os.environ.get("MEKOY_MODEL_BASE_URL", "http://127.0.0.1:11434/v1")


def _curl(path: str) -> object | None:
    """GET or POST the server and parse JSON, or None when the server cannot be read."""
    host = _base_url().rstrip("/")
    root = host.removesuffix("/v1")
    args = ["curl", "-sS", "-m", "20", f"{root}{path}"]
    if path.endswith("/show"):
        args += ["-d", json.dumps({"model": _configured_model()})]
    try:
        response = subprocess.run(  # noqa: S603 - fixed argv, no shell
            args, capture_output=True, text=True, timeout=30, check=False
        )
        return json.loads(response.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def _configured_model() -> str:
    return os.environ.get("MEKOY_MODEL", "qwen2.5:7b")


def server_slots() -> int | None:
    """How many requests the local model server serves at once, if we can tell.

    llama.cpp is launched with `-np N` and the server passes its own default through, so
    the running process's command line is the honest answer. Reading it is local and
    read-only, and returning None when the server is remote keeps this a diagnostic
    rather than a dependency.
    """
    try:
        listing = subprocess.run(
            ["pgrep", "-fl", "llama-server"],  # noqa: S607 - pgrep is on PATH
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"-np (\d+)", listing.stdout)
    return int(match.group(1)) if match else None


def model_shape(model: str | None = None) -> ModelShape | None:
    """Read a model's real shape from the server, or None when it cannot be read."""
    info = _curl("/api/show")
    if not isinstance(info, dict):
        return None
    return _shape_from_info(model or _configured_model(), info.get("model_info", {}))


def _shape_from_info(model: str, info: object) -> ModelShape | None:
    """Pull layers, KV heads, head dimension and context out of a server's metadata."""
    if not isinstance(info, dict):
        return None

    def pick(*names: str) -> int | None:
        for key, value in info.items():
            if any(key.endswith(name) for name in names) and isinstance(value, int):
                return value
        return None

    layers = pick("block_count")
    kv_heads = pick("head_count_kv")
    heads = pick("head_count")
    embedding = pick("embedding_length")
    if not layers or not kv_heads or not heads or not embedding:
        return None
    return ModelShape(
        name=model,
        layers=layers,
        kv_heads=kv_heads,
        head_dim=embedding // heads,
        context=pick("context_length") or _OUR_CONTEXT,
    )


def _memory() -> tuple[int, int]:
    """Total and realistically available bytes.

    Free pages alone understate what a new process can get, because macOS counts idle
    file cache as inactive; inactive and speculative pages are reclaimable, so they are
    included and active and wired pages are not.
    """

    def sysctl(name: str) -> int:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["sysctl", "-n", name],  # noqa: S607 - sysctl is on PATH
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            return int(out.stdout.strip() or 0)
        except ValueError:
            return 0

    total = sysctl("hw.memsize")
    vm = subprocess.run(
        ["vm_stat"],  # noqa: S607 - vm_stat is on PATH
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    page = 16384
    usable = 0
    for line in vm.splitlines():
        head = line.strip().split(":")[0]
        if head in {"Pages free", "Pages inactive", "Pages speculative"}:
            digits = "".join(c for c in line.split(":")[1] if c.isdigit())
            usable += (int(digits) if digits else 0) * page
    return total, usable


def _loaded_models() -> set[str]:
    """Names the server currently holds in memory."""
    listing = _curl("/api/ps")
    if not isinstance(listing, dict):
        return set()
    names = set()
    for entry in listing.get("models", []):
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            names.add(entry["name"])
    return names


def _model_bytes(model: str) -> int:
    """The model's real size on disk, which is what it costs in memory once loaded.

    An unreadable size stays zero rather than becoming a guess, because a made-up number
    here would silently decide how many requests the machine is allowed to run at once.
    """
    listed = _curl("/api/tags")
    if not isinstance(listed, dict):
        return 0
    for entry in listed.get("models", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("name") == model or entry.get("model") == model:
            size = entry.get("size")
            if isinstance(size, int):
                return size
    return 0


def plan(model: str | None = None, *, width: int | None = None) -> list[Finding]:
    """What each width would cost in memory, and whether the machine can afford it."""
    name = model or _configured_model()
    shape = model_shape(name)
    if shape is None:
        return [
            Finding(
                name="memory plan",
                detail=(
                    f"could not read the shape of {name}, so the memory cost is unknown"
                ),
            )
        ]

    total, usable = _memory()
    weights = _model_bytes(name)
    chosen = _concurrency() if width is None else width

    findings = [
        Finding(
            name="model",
            detail=f"{shape.name}: {shape.layers} layers, {shape.kv_heads} KV heads, "
            f"head dim {shape.head_dim}, serves up to {shape.context} context",
        ),
        Finding(
            name="KV cache",
            detail=f"{shape.kv_bytes_per_token / 1024:.0f} KB per token "
            f"({shape.layers} layers x {shape.kv_heads} KV heads x "
            f"{shape.head_dim} head dim x 2 for K and V x {_BYTES_PER_VALUE} bytes)",
        ),
        Finding(
            name="memory",
            detail=f"{usable / 1e9:.1f} GB usable of {total / 1e9:.1f} GB total, "
            f"model weights {weights / 1e9:.1f} GB",
        ),
    ]

    # If the model is already resident, its weights are part of the memory that is
    # already gone, so `usable` excludes them and adding them again would double-count.
    # Only a model that is not loaded costs its full size on top of what is free.
    resident = name in _loaded_models()
    findings.append(
        Finding(
            name="model is loaded",
            detail="yes, so its weights are already counted in the memory above"
            if resident
            else "no, so starting it costs its full size on top of the memory above",
        )
    )

    affordable = 0
    for candidate in range(1, _MAX_SENSIBLE_WIDTH + 1):
        kv = shape.kv_bytes(tokens=_OUR_CONTEXT, width=candidate)
        fits = kv <= usable if resident else weights + kv <= usable
        if fits:
            affordable = candidate
        if candidate == chosen:
            detail = (
                f"width {candidate} at {_OUR_CONTEXT} context per request needs "
                f"{kv / 1e9:.2f} GB of KV cache"
                + (
                    f", on top of the {weights / 1e9:.1f} GB the loaded model already "
                    "holds"
                    if resident
                    else f" plus {weights / 1e9:.1f} GB of weights, "
                    f"{(weights + kv) / 1e9:.2f} GB total"
                )
            )
            findings.append(
                Finding(
                    name="chosen width",
                    detail=detail + (" (fits)" if fits else " (does not fit)"),
                    advice=None
                    if fits
                    else "the engine asks for more than this machine can hold; "
                    "lower MEKOY_CONCURRENCY or the server's context length",
                )
            )

    if affordable >= 1:
        kv = shape.kv_bytes(tokens=_OUR_CONTEXT, width=affordable)
        findings.append(
            Finding(
                name="largest width that fits",
                detail=f"{affordable} at {_OUR_CONTEXT} context "
                f"({kv / 1e9:.2f} GB of KV cache)",
                advice=None
                if affordable >= chosen
                else f"the engine asks for {chosen}; {affordable} is what fits here",
            )
        )
    return findings


def check() -> list[Finding]:
    """Every finding, in the order a user should read them."""
    width = _concurrency()
    slots = server_slots()
    env_parallel = os.environ.get("OLLAMA_NUM_PARALLEL", "").strip()

    findings = [
        Finding(
            name="rows scored at once",
            detail=f"{width} (MEKOY_CONCURRENCY)",
            advice=None if width > 1 else "width 1 scores one document at a time",
        ),
        Finding(
            name="server serves at once",
            detail="unknown (server is remote, or not llama.cpp)"
            if slots is None
            else str(slots),
            advice=None
            if slots is None or slots >= width
            else (
                f"the server serves {slots} request(s) but the engine sends {width}; "
                "extra requests queue instead of running together"
            ),
        ),
    ]
    findings.extend(plan(width=width))

    if slots == 1:
        findings.append(
            Finding(
                name="the fix",
                detail=(
                    "set OLLAMA_NUM_PARALLEL to match MEKOY_CONCURRENCY, and set the "
                    "context to what the engine uses rather than the model's maximum, "
                    "then restart the model server"
                ),
                advice=(
                    "a local 7B measured 2.8x throughput at width 4, with per-request "
                    "latency rising from 24.4s to 25.7s — but that measurement assumed "
                    "memory this machine does not have at the model's full context, so "
                    "size the width from the memory plan above"
                ),
            )
        )
    if env_parallel and slots == 1:
        findings.append(
            Finding(
                name="note",
                detail=f"OLLAMA_NUM_PARALLEL is set to {env_parallel} in this shell",
                advice=(
                    "the Ollama desktop app starts its own server and ignores this; "
                    "the setting only applies to a server you start yourself"
                ),
            )
        )
    return findings


def render(findings: list[Finding]) -> str:
    """A short report, problems first."""
    problems = [f for f in findings if f.is_problem]
    lines: list[str] = []
    if not problems:
        lines.append("nothing is slowing this setup down.")
    else:
        lines.append(f"{len(problems)} thing(s) to look at:\n")
        for finding in problems:
            lines.append(f"  {finding.name}: {finding.detail}")
            lines.append(f"    -> {finding.advice}\n")
    lines.append("measured, not guessed:")
    lines.extend(f"  {finding.name} = {finding.detail}" for finding in findings)
    return "\n".join(lines)


@dataclass(frozen=True)
class Probe:
    """What concurrency actually bought here, measured rather than assumed."""

    probes: int
    serial_seconds: float
    parallel_seconds: float
    width: int

    @property
    def speedup(self) -> float:
        """Serial time over parallel time. 1.0 means concurrency bought nothing."""
        return (
            self.serial_seconds / self.parallel_seconds
            if self.parallel_seconds
            else 0.0
        )

    def explain(self) -> str:
        """One line stating the measurement and what it implies for the width."""
        if self.speedup < _SPEEDUP_WORTH_KEEPING:
            return (
                f"{self.probes} calls: {self.serial_seconds:.1f}s serial vs "
                f"{self.parallel_seconds:.1f}s at width {self.width} — "
                f"{self.speedup:.2f}x, so concurrency buys nothing here and the extra "
                "memory is wasted. This is what bandwidth-bound hardware looks like: "
                "several streams share one pipe, so each request takes proportionally "
                "longer while the total stays the same."
            )
        return (
            f"{self.probes} calls: {self.serial_seconds:.1f}s serial vs "
            f"{self.parallel_seconds:.1f}s at width {self.width} — "
            f"{self.speedup:.2f}x, so width {self.width} is worth keeping."
        )


def probe(*, width: int = 4, calls: int = 8, model: str | None = None) -> Probe | None:
    """Time the same calls serially and concurrently, and report the real speedup.

    A number lifted from someone else's hardware is a guess about yours. The published
    figure for this setting is 2.8x and the measurement on this machine was 1.06x, so
    the tool measures instead of quoting: same model, same calls, two orderings.
    """
    name = model or _configured_model()
    host = _base_url().rstrip("/")

    def one() -> None:
        args = [
            f"{host}/chat/completions",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(
                {
                    "model": name,
                    "messages": [{"role": "user", "content": "Say ok."}],
                    "max_tokens": 4,
                }
            ),
        ]
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(  # noqa: S603 - fixed argv, no shell
                ["curl", "-sS", "-m", "120", "-o", "/dev/null", *args],  # noqa: S607
                capture_output=True,
                timeout=180,
                check=False,
            )

    one()  # warm the model, so the first ordering is not charged the load

    start = time.perf_counter()
    for _ in range(calls):
        one()
    serial = time.perf_counter() - start

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=width) as pool:
        list(pool.map(lambda _x: one(), range(calls)))
    parallel = time.perf_counter() - start

    if serial <= 0 or parallel <= 0:
        return None
    return Probe(
        probes=calls, serial_seconds=serial, parallel_seconds=parallel, width=width
    )
