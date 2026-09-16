"""Per-principal quotas, and a refusal that says which limit and what the value is.

Two limits matter and both are built here: concurrent compiles, and documents per day.
Both are enforced at the only place they can be — before the work starts — and both
refuse
by naming the limit and the number, because a denial with no reason is
indistinguishable from a bug. A caller who cannot tell why they were refused will file
it as a defect and route around the limit, which is worse than the limit.

The daily counter is a table in the same SQLite file everything else uses, opened the
way `EventLog` opens it: the control plane stays one process and one file, which is
what a self-hoster has. No Redis, no broker, no quota service.

Unlimited by default. A laptop has no tenants to share, and a default limit that
cannot be turned off would be a bug rather than a policy. `0` means unlimited
everywhere here, so "set it and forget it" reads the same as "never configured".
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mekoy.errors import CompileError

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "CONCURRENT_ENV",
    "DOCUMENTS_PER_DAY_ENV",
    "Quota",
    "QuotaRefusedError",
    "UsageLog",
    "quota_from_env",
]

CONCURRENT_ENV = "MEKOY_MAX_CONCURRENT_COMPILES"
DOCUMENTS_PER_DAY_ENV = "MEKOY_MAX_DOCUMENTS_PER_DAY"

#: The daily counter. One row per principal per UTC day, so the window is the
#: calendar day a caller would look at rather than a rolling one they cannot see.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_usage (
    principal_id TEXT NOT NULL,
    day          TEXT NOT NULL,
    documents    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (principal_id, day)
);
"""


class QuotaRefusedError(CompileError):
    """A limit was hit. The message names the limit and the value."""


@dataclass(frozen=True, slots=True)
class Quota:
    """How much one principal may have in flight and per day. `0` is unlimited."""

    concurrent_compiles: int = 0
    documents_per_day: int = 0

    def explain_concurrent(self, limit: int) -> str:
        """The refusal for too many compiles, naming the limit and the value."""
        return (
            f"concurrent compile limit reached: {limit} compile(s) already running "
            f"for this principal (limit {CONCURRENT_ENV}={limit}). "
            "Wait for one to finish, or raise the limit."
        )

    def explain_documents(self, limit: int, used: int, asked: int) -> str:
        """The refusal for too many documents, naming the limit and the value."""
        return (
            f"document quota exceeded: {used} of {limit} document(s) already used "
            f"today, this request needs {asked} more "
            f"(limit {DOCUMENTS_PER_DAY_ENV}={limit}). "
            "Try again tomorrow, or raise the limit."
        )


def quota_from_env() -> Quota:
    """Read both limits from the environment. A bad value reads as unlimited."""
    return Quota(
        concurrent_compiles=_int_env(CONCURRENT_ENV),
        documents_per_day=_int_env(DOCUMENTS_PER_DAY_ENV),
    )


def _int_env(name: str) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _today() -> str:
    """The UTC day key. UTC rather than local so two hosts agree on the window."""
    return datetime.now(UTC).strftime("%Y-%m-%d")


class UsageLog:
    """Documents consumed per principal per day.

    In memory when `db_path` is None, which is what tests and a one-shot run want.
    """

    def __init__(self, *, db_path: Path | None = None) -> None:
        """Open the log, creating its table if this is a fresh file."""
        self._rows: dict[tuple[str, str], int] = {}
        self._db: sqlite3.Connection | None = None
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(db_path), check_same_thread=False)
            self._db.executescript(_SCHEMA)

    def used_today(self, principal_id: str, *, day: str | None = None) -> int:
        """Documents this principal has consumed today."""
        key = (principal_id, day or _today())
        if self._db is None:
            return self._rows.get(key, 0)
        cursor = self._db.execute(
            "SELECT documents FROM daily_usage WHERE principal_id = ? AND day = ?",
            key,
        )
        row = next(iter(cursor), None)
        return int(row[0]) if row is not None else 0

    def add(self, principal_id: str, count: int, *, day: str | None = None) -> int:
        """Charge `count` documents and return the new total for the day."""
        key = (principal_id, day or _today())
        if self._db is None:
            total = self._rows.get(key, 0) + count
            self._rows[key] = total
            return total
        with self._db:
            self._db.execute(
                "INSERT INTO daily_usage (principal_id, day, documents) "
                "VALUES (?, ?, ?) ON CONFLICT (principal_id, day) "
                "DO UPDATE SET documents = documents + ?",
                (key[0], key[1], count, count),
            )
        return self.used_today(principal_id, day=key[1])

    def check(
        self, principal_id: str, quota: Quota, *, asked: int, day: str | None = None
    ) -> None:
        """Refuse when charging `asked` documents would cross the daily limit.

        Checked before the work, so a caller is told no rather than told yes and
        charged a failure. Unlimited (`0`) short-circuits, so the common case costs
        nothing.
        """
        limit = quota.documents_per_day
        if limit <= 0:
            return
        used = self.used_today(principal_id, day=day)
        if used + asked > limit:
            raise QuotaRefusedError(message=quota.explain_documents(limit, used, asked))
