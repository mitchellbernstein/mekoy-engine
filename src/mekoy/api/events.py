"""A run's progress, as events that outlive the process that made them.

The portal's live pane used to fake granular progress, because the API told it nothing
between "started" and "finished". A compile is minutes of work and the interesting part
is in the middle: which arm is being measured, what it scored, which candidates were
dropped. Those are events, and a client that attaches halfway through should still see
the ones it missed.

The log is a table in the same SQLite file the Systems already live in, so history is
replayed from disk rather than kept in memory. Nothing here imports the control plane:
the store hands over a connection and this module writes to it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

#: Created beside the Systems table. Kept here rather than in the store's schema so the
#: event log owns its own layout and cannot drift from what this module reads.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_events (
    run_id TEXT NOT NULL,
    seq    INTEGER NOT NULL,
    kind   TEXT NOT NULL,
    data   TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""


class EventKind(StrEnum):
    """Every step worth telling a waiting client about."""

    RUN_STARTED = "run_started"
    ARM_STARTED = "arm_started"
    ARM_SCORED = "arm_scored"
    RUNG_PRUNED = "rung_pruned"
    SEARCH_SETTLED = "search_settled"
    TEST_SCORED = "test_scored"
    RUN_FINISHED = "run_finished"


#: Events after which the run is over, so a stream can close instead of waiting.
TERMINAL: frozenset[EventKind] = frozenset({EventKind.RUN_FINISHED})


@dataclass(frozen=True, slots=True)
class RunEvent:
    """One thing that happened during a compile."""

    kind: EventKind
    data: dict[str, object] = field(default_factory=dict)
    #: Set by the log on append. A client reconnecting sends the last one it saw.
    seq: int = 0

    def to_sse(self) -> str:
        """This event as one server-sent-events frame.

        The id is the sequence number, which is what makes `Last-Event-ID` work: a
        client that drops reconnects and the server resumes after the last event it
        received, with no bookkeeping on either side.
        """
        payload = json.dumps({"kind": str(self.kind), **self.data})
        return f"id: {self.seq}\nevent: {self.kind}\ndata: {payload}\n\n"


class EventLog:
    """Append-only events for one run, in SQLite.

    In memory when `db_path` is None, which is what tests and a one-shot CLI run want.
    """

    def __init__(self, *, db_path: Path | None = None) -> None:
        """Open the log, creating the table if this is a fresh file."""
        self._rows: dict[str, list[RunEvent]] = {}
        self._db: sqlite3.Connection | None = None
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(db_path), check_same_thread=False)
            self._db.executescript(_SCHEMA)

    def append(self, run_id: str, event: RunEvent) -> RunEvent:
        """Record one event and return it with its sequence number."""
        if self._db is None:
            rows = self._rows.setdefault(run_id, [])
            stored = RunEvent(kind=event.kind, data=event.data, seq=len(rows) + 1)
            rows.append(stored)
            return stored
        with self._db:
            cursor = self._db.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM run_events WHERE run_id = ?",
                (run_id,),
            )
            seq = int(next(iter(cursor))[0])
            self._db.execute(
                "INSERT INTO run_events (run_id, seq, kind, data) VALUES (?, ?, ?, ?)",
                (run_id, seq, str(event.kind), json.dumps(event.data)),
            )
        return RunEvent(kind=event.kind, data=event.data, seq=seq)

    def since(self, run_id: str, *, after: int = 0) -> tuple[RunEvent, ...]:
        """Every event after `after`, oldest first. `after=0` replays the whole run."""
        if self._db is None:
            rows = self._rows.get(run_id, [])
            return tuple(e for e in rows if e.seq > after)
        cursor = self._db.execute(
            "SELECT seq, kind, data FROM run_events WHERE run_id = ? AND seq > ? "
            "ORDER BY seq",
            (run_id, after),
        )
        return tuple(
            RunEvent(kind=EventKind(kind), data=json.loads(data), seq=seq)
            for seq, kind, data in cursor
        )

    def finished(self, run_id: str) -> bool:
        """Whether this run has already reached a terminal event."""
        return any(e.kind in TERMINAL for e in self.since(run_id))
