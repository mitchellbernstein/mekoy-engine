"""Metering and operational metrics, as tables in the file everything else uses.

Two different jobs, deliberately in one module because they are written from the same
place and read from the same place:

**Metering** is a row per completed compile. Compilation is what a price attaches to,
and a price needs a unit: compile seconds, model calls, documents scored, and which
model. Writing
it at completion rather than reconstructing it later means billing is a query, not a
re-instrumentation of the whole compile path.

**Operational metrics** answer "is it healthy" for a host. The specific failure that
motivates this: a compile that dies leaves a run marked failed, and a failed run is
invisible unless something counts it and a host can reach the count. So failures are
counted and listed, and `GET /v1/metrics` returns them.

No metrics dependency. A self-hoster runs one process; adding Prometheus means adding a
scrape target, a port, and a second thing to keep alive. A table plus an endpoint is the
same information with none of that. A hosted deployment can export from the table.

The table is opened the way `EventLog` opens its own — a connection to the same SQLite
file — so there is still exactly one file to back up.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "CompileMetering",
    "Failure",
    "Metrics",
    "MetricsSnapshot",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS metering (
    run_id       TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL,
    system_id    TEXT NOT NULL,
    seconds      REAL NOT NULL,
    model_calls  INTEGER NOT NULL,
    documents    INTEGER NOT NULL,
    model        TEXT NOT NULL,
    finished_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS counters (
    name         TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    value        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (name, principal_id)
);
CREATE TABLE IF NOT EXISTS failures (
    run_id       TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL,
    system_id    TEXT NOT NULL,
    error        TEXT NOT NULL,
    at           TEXT NOT NULL
);
"""

#: How many recent failures an operator sees. Enough to spot a pattern, capped so the
#: endpoint stays a page rather than a log dump.
_FAILURES_SHOWN = 20


def _now() -> str:
    """A UTC timestamp, so two hosts writing the same file agree on order."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class CompileMetering:
    """One completed compile, as a billable unit."""

    run_id: str
    principal_id: str
    system_id: str
    seconds: float
    model_calls: int
    documents: int
    model: str
    finished_at: str = ""


@dataclass(frozen=True, slots=True)
class Failure:
    """One failed compile, kept so a host can see it happened."""

    run_id: str
    principal_id: str
    system_id: str
    error: str
    at: str


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """What a host sees: counters, recent failures, and metering totals."""

    counters: dict[str, int]
    failures: tuple[Failure, ...]
    compiles_metered: int
    documents_scored: int


class Metrics:
    """Metering rows, counters, and failures for one control plane.

    In memory when `db_path` is None, which is what tests and a one-shot run want.
    """

    def __init__(self, *, db_path: Path | None = None) -> None:
        """Open the tables, creating them if this is a fresh file."""
        self._metering: dict[str, CompileMetering] = {}
        self._counters: dict[tuple[str, str], int] = {}
        self._failures: list[Failure] = []
        self._db: sqlite3.Connection | None = None
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(db_path), check_same_thread=False)
            self._db.executescript(_SCHEMA)

    # --- writes -----------------------------------------------------------

    def record_compile(self, entry: CompileMetering) -> CompileMetering:
        """Write the metering row for a completed compile.

        Called on success, which is what makes the row billable: a compile that failed
        produced no System, and charging for it is the kind of thing a customer
        notices. Failures go to `record_failure` instead, which is a metric rather
        than an invoice.
        """
        stored = CompileMetering(
            run_id=entry.run_id,
            principal_id=entry.principal_id,
            system_id=entry.system_id,
            seconds=entry.seconds,
            model_calls=entry.model_calls,
            documents=entry.documents,
            model=entry.model,
            finished_at=entry.finished_at or _now(),
        )
        if self._db is None:
            self._metering[stored.run_id] = stored
        else:
            with self._db:
                self._db.execute(
                    "INSERT OR REPLACE INTO metering "
                    "(run_id, principal_id, system_id, seconds, model_calls, "
                    "documents, model, finished_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        stored.run_id,
                        stored.principal_id,
                        stored.system_id,
                        stored.seconds,
                        stored.model_calls,
                        stored.documents,
                        stored.model,
                        stored.finished_at,
                    ),
                )
        return stored

    def record_failure(
        self, *, run_id: str, principal_id: str, system_id: str, error: str
    ) -> Failure:
        """Record a failed compile, and count it.

        This is the row a host looks for. Without it a failed compile exists only as a
        run whose status is `failed`, which is a row nobody queries and nothing alerts
        on.
        """
        failure = Failure(
            run_id=run_id,
            principal_id=principal_id,
            system_id=system_id,
            error=error,
            at=_now(),
        )
        if self._db is None:
            self._failures = [f for f in self._failures if f.run_id != run_id] + [
                failure
            ]
        else:
            with self._db:
                self._db.execute(
                    "INSERT OR REPLACE INTO failures "
                    "(run_id, principal_id, system_id, error, at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        failure.run_id,
                        failure.principal_id,
                        failure.system_id,
                        failure.error,
                        failure.at,
                    ),
                )
        _ = self.bump("compile_failed", principal_id)
        return failure

    def bump(self, name: str, principal_id: str, *, by: int = 1) -> int:
        """Increment a counter and return its new value."""
        if self._db is None:
            key = (name, principal_id)
            self._counters[key] = self._counters.get(key, 0) + by
            return self._counters[key]
        with self._db:
            self._db.execute(
                "INSERT INTO counters (name, principal_id, value) VALUES (?, ?, ?) "
                "ON CONFLICT (name, principal_id) DO UPDATE SET value = value + ?",
                (name, principal_id, by, by),
            )
        return self.counter(name, principal_id)

    # --- reads ------------------------------------------------------------

    def counter(self, name: str, principal_id: str) -> int:
        """One counter's value for a principal."""
        if self._db is None:
            return self._counters.get((name, principal_id), 0)
        cursor = self._db.execute(
            "SELECT value FROM counters WHERE name = ? AND principal_id = ?",
            (name, principal_id),
        )
        row = next(iter(cursor), None)
        return int(row[0]) if row is not None else 0

    def snapshot(self, *, principal_id: str | None = None) -> MetricsSnapshot:
        """Counters, recent failures, and metering totals for a host.

        `principal_id=None` is the whole control plane, which is the local,
        single-tenant case: the host is the only user and wants every number.
        """
        names = ("compile_succeeded", "compile_failed")
        counters = {
            name: self._sum_counter(name, principal_id=principal_id) for name in names
        }
        failures = self._recent_failures(principal_id)
        metered, documents = self._totals(principal_id)
        counters["compiles_metered"] = metered
        counters["documents_scored"] = documents
        return MetricsSnapshot(
            counters=counters,
            failures=failures,
            compiles_metered=metered,
            documents_scored=documents,
        )

    def _sum_counter(self, name: str, *, principal_id: str | None) -> int:
        if self._db is None:
            return sum(
                value
                for (counter, owner), value in self._counters.items()
                if counter == name and (principal_id is None or owner == principal_id)
            )
        if principal_id is None:
            cursor = self._db.execute(
                "SELECT COALESCE(SUM(value), 0) FROM counters WHERE name = ?", (name,)
            )
        else:
            cursor = self._db.execute(
                "SELECT COALESCE(SUM(value), 0) FROM counters "
                "WHERE name = ? AND principal_id = ?",
                (name, principal_id),
            )
        return int(next(iter(cursor))[0])

    def _recent_failures(self, principal_id: str | None) -> tuple[Failure, ...]:
        """The newest failures in scope, newest first.

        Two statements rather than one built by string concatenation, so the scope is
        visible in the SQL a reader sees rather than assembled at runtime.
        """
        if self._db is None:
            rows = [
                f
                for f in self._failures
                if principal_id is None or f.principal_id == principal_id
            ]
            return tuple(rows[-_FAILURES_SHOWN:])
        columns = "SELECT run_id, principal_id, system_id, error, at FROM failures"
        tail = f" ORDER BY at DESC LIMIT {_FAILURES_SHOWN}"
        if principal_id is None:
            cursor = self._db.execute(columns + tail)
        else:
            cursor = self._db.execute(
                columns + " WHERE principal_id = ?" + tail, (principal_id,)
            )
        return tuple(Failure(*row) for row in cursor)

    def _totals(self, principal_id: str | None) -> tuple[int, int]:
        """(compiles metered, documents scored) in scope."""
        if self._db is None:
            rows = [
                m
                for m in self._metering.values()
                if principal_id is None or m.principal_id == principal_id
            ]
            return (len(rows), sum(m.documents for m in rows))
        head = "SELECT COUNT(*), COALESCE(SUM(documents), 0) FROM metering"
        if principal_id is None:
            cursor = self._db.execute(head)
        else:
            cursor = self._db.execute(head + " WHERE principal_id = ?", (principal_id,))
        count, documents = next(iter(cursor))
        return (int(count), int(documents))
