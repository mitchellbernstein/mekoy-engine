"""Check the local setup and say what is slowing it down.

The engine's measured weakness is wall-clock time, not accuracy: a compile that scores
0.940 at $0/doc still takes eight seconds a document, and most of that is waiting on a
model server that answers one request at a time.

That waiting is invisible from inside the engine. The server is up, the answers are
right, and the only symptom is that everything takes longer than it should. So the check
lives here: it looks at the running server, reads how many requests it will serve at
once, and says so. A user who cannot see the bottleneck cannot fix it.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

from mekoy.search import (
    _concurrency,
)


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


def server_slots() -> int | None:
    """How many requests the local model server will serve at once, if we can tell.

    llama.cpp is launched with `-np N` and Ollama passes its own default through, so the
    running process's command line is the honest answer. Reading it is local-only and
    read-only, and returning None when the server is remote or unreadable keeps this a
    diagnostic rather than a dependency.
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

    if slots == 1:
        findings.append(
            Finding(
                name="the fix",
                detail=(
                    "set OLLAMA_NUM_PARALLEL to match MEKOY_CONCURRENCY, then "
                    "restart the model server"
                ),
                advice=(
                    "a local 7B measured 2.8x throughput at width 4, with per-request "
                    "latency rising from 24.4s to 25.7s. On a 400-call compile that is "
                    "roughly 222 minutes down to 57"
                ),
            )
        )
    if env_parallel and slots == 1:
        findings.append(
            Finding(
                name="note",
                detail=f"OLLAMA_NUM_PARALLEL is set to {env_parallel} in this shell",
                advice=(
                    "the Ollama desktop app starts its own server and ignores "
                    "this; the setting only applies to a server you start yourself"
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
