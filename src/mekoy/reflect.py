"""Our own reflective optimizer: read the failures, write a better instruction.

This replaces the borrowed optimizer. The search over k-shot, decode, model, and
schema was always ours; the piece we rented was *reflection* — the step that reads
why a candidate failed and proposes a fix. That is where the advantage lives.

The idea is one sentence: **a score tells you that a candidate failed; the trace
tells you why.**

Four patterns here were learned by reading other people's implementations, and each
replaced something that was wrong or missing in the first version:

- **Acceptance is a minibatch improvement, not Pareto dominance.** GEPA's default is
  `sum(after) > sum(before)` over the sampled batch. The first version of this file
  required a candidate to be at least as good on *every* validation row, which for
  19 rows and 7 fields is 133 comparisons — almost nothing in the space can satisfy
  that, so children were never accepted and the run was inert. Dominance is the
  wrong test for admission; it is the right test for *keeping*.
- **Parent selection reads a per-example Pareto front** with a seeded RNG, rather
  than cycling round-robin. Two candidates good at different rows are both worth
  reflecting on, and randomness is what stops the search settling.
- **Minibatches are epoch-shuffled**, so every row is seen before any row is seen
  twice. Sampling with replacement leaves rows unexamined and over-fits the ones
  that keep coming up.
- **Stopping is a rule, not a loop counter** — budget, no-improvement, or target.
  A run that stops when it stops improving spends its budget where it still can.

Optional **islands** follow OpenEvolve: several sub-populations that occasionally
exchange members. That is the standard defence against search collapse, where one
behaviour crowds out every alternative.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from mekoy.errors import CompileError
from mekoy.harness import Decode, extract
from mekoy.search import HarnessConfig, Trial, _system_for, evaluate
from mekoy.tasks import Task
from mekoy.verify import VerifyFail, VerifyOk

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from mekoy.dataset import ExampleRecord
    from mekoy.runtime import Completer

__all__ = [
    "DEFAULT_METRIC_CALLS",
    "Candidate",
    "ReflectionResult",
    "diagnose_prompt",
    "pareto",
    "parse_reply",
    "reflect",
    "sample_batch",
    "select_parent",
    "should_stop",
]

#: Metric calls a reflection run may spend. One call is one document scored.
DEFAULT_METRIC_CALLS = 60
#: Documents per batch. Small on purpose: the point is a cheap signal about *which*
#: rows disagree, not a precise average.
DEFAULT_BATCH = 8
#: How many failures to show the reflection model at once.
_MAX_EXAMPLES = 4
#: A reply shorter than this is not an instruction, it is a non-answer.
_MIN_INSTRUCTION_CHARS = 40
#: Instructions longer than this are almost always the model rambling.
_MAX_INSTRUCTION_CHARS = 6000
#: Stop after this many accepted rounds with no gain (GEPA's NoImprovementStopper).
DEFAULT_PATIENCE = 4
#: A batch the model already aces teaches nothing. Re-sample this many times
#: before concluding the instruction has no diagnosable failures left. Counting an
#: uninformative batch as "no improvement" stops a run that has simply not been
#: shown anything hard yet.
DEFAULT_UNINFORMATIVE = 3
#: Sub-populations. One island is plain evolution; several resist collapse.
DEFAULT_ISLANDS = 2
#: Rounds between migration when islands is greater than one.
DEFAULT_MIGRATION = 3
#: An island needs at least this many members before it can send a champion.
_ISLAND_MIN = 2


@dataclass(frozen=True, slots=True)
class Candidate:
    """One instruction, what it scored, and what it was told along the way."""

    instructions: str
    #: Score per validation row, in order. The Pareto front reads these.
    per_example: tuple[float, ...]
    parent: int | None = None
    #: Lessons inherited from ancestors, carried into the next rewrite prompt.
    lessons: tuple[str, ...] = ()
    label: str = "seed"
    island: int = 0

    @property
    def score(self) -> float:
        """Mean score. Used for reporting and stopping, not for admission."""
        if not self.per_example:
            return 0.0
        return sum(self.per_example) / len(self.per_example)

    @property
    def total(self) -> float:
        """Sum of per-row scores, matching the batch admission test."""
        return sum(self.per_example)

    def beats(self, other: Candidate) -> bool:
        """Pareto dominance: no worse everywhere, better somewhere.

        Used to keep a candidate, not to admit one. Admission uses the batch total.
        """
        no_worse = all(
            a >= b for a, b in zip(self.per_example, other.per_example, strict=False)
        )
        strictly_better = any(
            a > b for a, b in zip(self.per_example, other.per_example, strict=False)
        )
        return no_worse and strictly_better


@dataclass(frozen=True, slots=True)
class ReflectionResult:
    """What reflection produced, and everything it considered."""

    best: Candidate
    pool: tuple[Candidate, ...]
    metric_calls: int
    baseline: float
    rounds: int = 0
    stopped_by: str = "budget"

    @property
    def improved(self) -> bool:
        """Whether the winner is better than where the run started."""
        return self.best.score > self.baseline

    @property
    def lessons(self) -> tuple[str, ...]:
        """Everything the winner inherited from its ancestors."""
        return self.best.lessons

    def summary(self) -> str:
        """One line for a compile card."""
        verdict = "improved" if self.improved else "no improvement"
        return (
            f"reflection: {verdict} {self.baseline:.3f} -> {self.best.score:.3f} "
            f"({len(self.pool)} candidates, {self.rounds} rounds, "
            f"{self.metric_calls} metric calls, stopped by {self.stopped_by})"
        )


def pareto(pool: Sequence[Candidate]) -> tuple[Candidate, ...]:
    """Candidates that no other candidate beats outright."""
    return tuple(
        candidate
        for candidate in pool
        if not any(other.beats(candidate) for other in pool if other is not candidate)
    )


def per_example_front(
    pool: Sequence[Candidate],
) -> dict[int, tuple[int, ...]]:
    """Map each row index to the candidates that are best on it.

    GEPA calls this the Pareto front mapping. It is what lets selection pick a
    candidate that is *best at something* rather than merely high on average.
    """
    front: dict[int, list[int]] = {}
    for index, candidate in enumerate(pool):
        for row, value in enumerate(candidate.per_example):
            current = front.setdefault(row, [])
            if not current:
                current.append(index)
                continue
            best = max(pool[i].per_example[row] for i in current)
            if value > best:
                front[row] = [index]
            elif value == best:
                current.append(index)
    return {row: tuple(sorted(set(who))) for row, who in front.items()}


def select_parent(
    pool: Sequence[Candidate], *, rng: random.Random, epsilon: float = 0.25
) -> Candidate:
    """Pick who to reflect on: usually a front member, sometimes the incumbent.

    Taken from GEPA's epsilon-greedy selector. Pure front selection explores
    forever; pure best-only selection collapses onto one behaviour. The mixture is
    the point.
    """
    if rng.random() < epsilon:
        return max(pool, key=lambda c: c.score)
    front = pareto(pool)
    mapping = per_example_front(front)
    if not mapping:
        return rng.choice(list(front))
    row = rng.choice(sorted(mapping))
    return front[rng.choice(mapping[row])]


def sample_batch(
    rows: Sequence[ExampleRecord], *, epoch: int, size: int, rng: random.Random
) -> tuple[ExampleRecord, ...]:
    """Epoch-shuffled sample: every row is drawn before any row repeats.

    GEPA's EpochShuffledBatchSampler, in miniature. Sampling with replacement
    leaves rows unexamined and over-weights whichever ones keep coming up.
    """
    del epoch  # the advancing rng state already varies the order
    order = list(rows)
    rng.shuffle(order)
    return tuple(order[: min(size, len(order))])


def should_stop(  # noqa: PLR0913 - it reports on several stop conditions
    *,
    used: int,
    metric_calls: int,
    rounds_without_gain: int,
    patience: int,
    best_score: float,
    target: float | None,
) -> str | None:
    """Why the run should end, or None to continue.

    GEPA composes stop conditions this way (budget, no-improvement, threshold).
    A single loop counter spends budget after the search has stopped paying.
    """
    if target is not None and best_score >= target:
        return "target"
    if rounds_without_gain >= patience:
        return "no improvement"
    if used >= metric_calls:
        return "budget"
    return None


def diagnose_prompt(
    candidate: Candidate,
    failures: Sequence[tuple[str, object, str]],
) -> str:
    """Ask for a diagnosis first, then a rewritten instruction.

    Diagnosis before rewriting is deliberate: a model asked only for a new
    instruction rewrites prose, and a model asked what went wrong makes a claim we
    can read, keep, or discard.
    """
    blocks: list[str] = []
    for text, gold, produced in failures[:_MAX_EXAMPLES]:
        blocks.append(
            "CASE\n"
            f"document: {text[:600]}\n"
            f"correct answer: {_as_text(gold)}\n"
            f"current output: {produced[:400]}"
        )
    lessons = "\n".join(f"- {lesson}" for lesson in candidate.lessons) or "- none yet"
    return (
        f"{candidate.instructions}\n\n"
        "---\n"
        "The instruction above produced wrong answers on these cases:\n\n"
        f"{chr(10).join(blocks)}\n\n"
        f"Lessons already learned (do not lose these):\n{lessons}\n\n"
        "Reply in exactly this shape and nothing else:\n"
        "DIAGNOSIS: one sentence naming the single mistake these cases share.\n"
        "INSTRUCTION: the complete revised instruction.\n"
    )


def _as_text(value: object) -> str:
    dump = getattr(value, "model_dump_json", None)
    return dump() if callable(dump) else str(value)


def parse_reply(reply: str, *, fallback: str) -> tuple[str, str | None]:
    """Split a reply into (instruction, diagnosis).

    A model that answers with prose instead of the requested shape keeps its
    instruction and loses only the diagnosis; refusing the whole reply would throw
    away usable work.
    """
    diagnosis: str | None = None
    instruction = ""
    for line in reply.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("DIAGNOSIS:"):
            diagnosis = stripped.partition(":")[2].strip() or None
        elif stripped.upper().startswith("INSTRUCTION:"):
            instruction = stripped.partition(":")[2].strip()
    if not instruction:
        candidate = reply.strip()
        instruction = (
            candidate
            if _MIN_INSTRUCTION_CHARS < len(candidate) <= _MAX_INSTRUCTION_CHARS
            else ""
        )
    return (instruction or fallback), diagnosis


def _with_instruction(task: Task, instructions: str, label: str) -> Task:
    """A copy of the task whose prompt is this instruction."""
    variants = dict(task.prompt_variants)
    variants[label] = instructions
    return replace(task, prompt_variants=variants, prompt=instructions)


def _failures(
    task: Task,
    candidate: Candidate,
    rows: Sequence[ExampleRecord],
    trials: Sequence[Trial],
    completer: Completer,
) -> list[tuple[str, object, str]]:
    """Collect the rows a candidate got wrong, with what it actually said.

    The output text matters as much as the score. A quoted wrong field is a
    diagnosis the model can act on; a bare zero is not.
    """
    out: list[tuple[str, object, str]] = []
    for row, trial in zip(rows, trials, strict=False):
        if trial.quality >= 1.0:
            continue
        out.append(
            (row.text, row.outcome, _one(task, candidate.instructions, row, completer))
        )
    return out


def _one(
    task: Task,
    instructions: str,
    row: ExampleRecord,
    completer: Completer,
) -> str:
    """Run one document and return the raw output text, or the gate's reasons."""
    scoped = _with_instruction(task, instructions, "reflected")
    result = extract(
        completer,
        text=row.text,
        shots=(),
        retries=1,
        decode=Decode(system=instructions, constrained=True),
        task=scoped,
    )
    if isinstance(result, VerifyFail):
        return "rejected: " + "; ".join(result.reasons)
    return cast("VerifyOk", result).outcome.model_dump_json()


def reflect(  # noqa: PLR0913, PLR0915, C901 - the loop is the algorithm
    task: Task,
    *,
    completer: Completer,
    train: tuple[ExampleRecord, ...],
    val: tuple[ExampleRecord, ...],
    config: HarnessConfig | None = None,
    metric_calls: int = DEFAULT_METRIC_CALLS,
    batch: int = DEFAULT_BATCH,
    patience: int = DEFAULT_PATIENCE,
    uninformative_limit: int = DEFAULT_UNINFORMATIVE,
    islands: int = DEFAULT_ISLANDS,
    target: float | None = None,
    seed: int = 0,
) -> ReflectionResult:
    """Evolve the task's instruction against the task's own gate.

    `val` is the only data that decides anything; `train` is where failure cases
    come from. The budget counts documents scored, so a run is bounded work rather
    than an open-ended loop.
    """
    if not val:
        msg = "reflection needs a non-empty validation slice"
        raise CompileError(message=msg)
    spend = config or HarnessConfig(k_shot=0, retries=1, constrained=True)
    # Seeded on purpose: a reflection run must be reproducible.
    rng = random.Random(seed)  # noqa: S311 - reproducibility, not cryptography
    used = 0

    def score(instructions: str, rows: tuple[ExampleRecord, ...]) -> Candidate:
        nonlocal used
        scoped = _with_instruction(task, instructions, "reflected")
        trial = evaluate(
            completer,
            rows,
            replace(spend, prompt="reflected"),
            shots=(),
            task=scoped,
        )
        used += len(rows)
        return Candidate(
            instructions=instructions,
            per_example=tuple(s.quality for s in trial.scores),
        )

    seed_candidate = score(_system_for(spend, task), val)
    # One seed per island: diversity starts at the front, not halfway through.
    pool: list[Candidate] = [
        replace(seed_candidate, island=i, label="seed" if i == 0 else f"seed{i}")
        for i in range(max(1, islands))
    ]
    baseline = seed_candidate.score
    best_score = max(c.score for c in pool)
    no_gain = 0
    uninformative = 0
    epoch = 0
    stopped = "budget"

    while True:
        reason = should_stop(
            used=used,
            metric_calls=metric_calls,
            rounds_without_gain=no_gain,
            patience=patience,
            best_score=best_score,
            target=target,
        )
        if reason is not None:
            stopped = reason
            break

        epoch += 1
        island = epoch % max(1, islands)
        # Islands evolve separately and only occasionally exchange members, which
        # is what keeps several behaviours alive (OpenEvolve's pattern).
        residents = [c for c in pool if c.island == island] or pool
        parent = select_parent(residents, rng=rng)

        rows = sample_batch(train, epoch=epoch, size=batch, rng=rng)
        scoped = _with_instruction(task, parent.instructions, "reflected")
        trial = evaluate(
            completer,
            rows,
            replace(spend, prompt="reflected"),
            shots=(),
            task=scoped,
        )
        used += len(rows)
        before = sum(s.quality for s in trial.scores)
        failures = _failures(
            task,
            replace(parent, per_example=tuple(s.quality for s in trial.scores)),
            rows,
            trial.scores,
            completer,
        )
        if not failures:
            # This batch was too easy to say anything about. That is not evidence
            # the instruction cannot be improved, so it does not consume patience.
            uninformative += 1
            if uninformative >= uninformative_limit:
                stopped = "no failures found"
                break
            continue
        uninformative = 0

        reply = completer.complete(
            system=task.prompt,
            user=diagnose_prompt(parent, failures),
            constrained=False,
            temperature=0.7,
        )
        instructions, diagnosis = parse_reply(reply, fallback=parent.instructions)
        if instructions == parent.instructions:
            no_gain += 1
            continue

        child = replace(
            score(instructions, val),
            parent=pool.index(parent),
            lessons=parent.lessons + ((diagnosis,) if diagnosis else ()),
            label=f"r{epoch}",
            island=island,
        )
        # Admission: the batch must improve, the same test GEPA uses. A row-level
        # dominance requirement here would reject nearly everything.
        gained = _batch_gain(task, child, rows, completer, spend)
        if gained > before:
            pool.append(child)
            if child.score > best_score:
                best_score = child.score
                no_gain = 0
            else:
                no_gain += 1
        else:
            no_gain += 1

        if islands > 1 and epoch % DEFAULT_MIGRATION == 0:
            pool = _migrate(pool, islands=islands)

    front = pareto(pool)
    best = max(front, key=lambda c: c.score)
    return ReflectionResult(
        best=best,
        pool=tuple(pool),
        metric_calls=used,
        baseline=baseline,
        rounds=epoch,
        stopped_by=stopped,
    )


def _batch_gain(
    task: Task,
    child: Candidate,
    rows: tuple[ExampleRecord, ...],
    completer: Completer,
    spend: HarnessConfig,
) -> float:
    """Score the child on the same rows the parent was just scored on.

    The comparison has to be like-for-like, so this re-runs the batch rather than
    reading the child's validation score. `val` is deliberately not a parameter:
    admission must not consult the data the winner is judged on.
    """
    scoped = _with_instruction(task, child.instructions, "reflected")
    trial = evaluate(
        completer, rows, replace(spend, prompt="reflected"), shots=(), task=scoped
    )
    return sum(s.quality for s in trial.scores)


def _migrate(pool: list[Candidate], *, islands: int) -> list[Candidate]:
    """Move the best member of each island into the next one.

    Without this, sub-populations drift apart and a good idea found in one never
    reaches the others (OpenEvolve's migration step).
    """
    champions = [
        (island, max([c for c in pool if c.island == island], key=lambda c: c.score))
        for island in range(islands)
        if len([c for c in pool if c.island == island]) >= _ISLAND_MIN
    ]
    return [
        *pool,
        *(
            replace(
                champion,
                island=(island + 1) % islands,
                label=f"{champion.label}->i{(island + 1) % islands}",
            )
            for island, champion in champions
        ),
    ]
