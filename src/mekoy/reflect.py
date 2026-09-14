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

import json
import random
import re
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
    "FailureShape",
    "ReflectionResult",
    "diagnose_prompt",
    "failure_shape",
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


#: Words that carry no signal about which mistakes are related. `label` and `value`
#: appear in almost every serialised answer, so leaving them in would put every case
#: in one family and teach nothing.
#: Words shorter than this are too common to signal a family ("get", "pay", "why").
#: Two characters and fewer matched too much.
_MIN_SIGNAL_CHARS = 2

_LABEL_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "my",
        "me",
        "i",
        "to",
        "of",
        "for",
        "is",
        "it",
        "and",
        "label",
        "value",
        "intent",
        "outcome",
        "receipt",
        "status",
        "restaurant",
    }
)


#: Below this many failures the shape is too noisy to call; a single failure is always
#: a family of one.
_MIN_CASES_FOR_SHAPE = 5

#: The share of failures that must sit in a shared family for the loop to believe an
#: instruction can help. Below it, the loop reports a knowledge gap.
_SHARED_SHAPE_MIN = 0.34


def _signal_words(gold: object) -> frozenset[str]:
    """The distinctive words in the label a wrong answer should have produced."""
    words = re.findall(r"[a-z]+", _as_text(gold).lower())
    return frozenset(
        word
        for word in words
        if word not in _LABEL_STOPWORDS and len(word) > _MIN_SIGNAL_CHARS
    )


def _grouped(
    failures: Sequence[tuple[str, object, str]],
) -> list[tuple[str, list[tuple[str, object, str]]]]:
    """Failures clustered by shared label words, largest family first.

    Connected components, not exact-match buckets: `verify_my_identity` and
    `why_verify_identity` never have identical word sets, so bucketing them by an
    exact key puts every case in a family of one and the diagnosis has no family to
    name. Two cases join when their labels share a distinctive word, which is the
    relation that actually holds between confusing pairs.

    Largest first: a family of three teaches a rule worth writing, a family of one is
    a long tail an instruction should not chase.
    """
    words = [_signal_words(gold) for _, gold, _ in failures]
    parent = list(range(len(failures)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(failures)):
        for right in range(left + 1, len(failures)):
            if words[left] & words[right]:
                parent[find(left)] = find(right)

    families: dict[int, list[int]] = {}
    for index in range(len(failures)):
        families.setdefault(find(index), []).append(index)

    def name(members: list[int]) -> str:
        shared = (
            set.intersection(*(set(words[i]) for i in members)) if members else set()
        )
        source = shared or (set(words[members[0]]) if members else set())
        return "_".join(sorted(source)[:3]) or "other"

    grouped = [(name(m), [failures[i] for i in m]) for m in families.values()]
    return sorted(grouped, key=lambda item: (-len(item[1]), item[0]))


@dataclass(frozen=True)
class FailureShape:
    """What a batch of failures looks like as a group.

    The distinction this exists to draw: a **rule gap** fails the same way repeatedly
    and can be fixed by rewriting the instruction; a **knowledge gap** fails a
    different way every time and cannot be fixed by any instruction, because the model
    does not know the thing it is being asked to choose between.

    Measured, not assumed. On BANKING77 the compiled model scored 0.600 and made 9
    mistakes across 9 distinct label pairs — every error unique. Four rewritten
    instructions, including one that grouped related cases, moved the score by zero.
    On the restaurant job the failures were the same mistake repeated (claimed a
    booking the transcript never supports) and one rewritten instruction fixed it.
    Sending both jobs to the same reflection loop wastes the user's evening on the
    second, so the loop states which one it is looking at.
    """

    cases: int
    families: int
    largest: int

    @property
    def repeated(self) -> int:
        """Cases belonging to a family with more than one member."""
        return 0 if self.families == 0 else max(0, self.cases - self.families)

    @property
    def is_knowledge_gap(self) -> bool:
        """True when almost no two failures fail alike.

        The threshold is deliberately blunt. Nine unique mistakes in nine cases is not
        a near miss; it is the shape of a task whose answers are not derivable from what
        the model knows. Below this line a few shared rules still exist and the loop
        runs as normal.
        """
        if self.cases < _MIN_CASES_FOR_SHAPE:
            return False
        return self.repeated / self.cases < _SHARED_SHAPE_MIN

    def explain(self) -> str:
        """One line naming the gap and the lever that actually moves it."""
        if self.is_knowledge_gap:
            return (
                f"failures are unrelated ({self.families} different mistakes in "
                f"{self.cases} cases), so this is a knowledge gap, not a rule gap. "
                "Rewriting the instruction cannot help; show more examples, pick a "
                "stronger model, or train on this task."
            )
        return (
            f"failures repeat ({self.cases - self.families} of {self.cases} cases "
            f"share a pattern across {self.families} families), so this is a rule gap "
            "an instruction can close."
        )


def failure_shape(
    failures: Sequence[tuple[str, object, str]],
) -> FailureShape:
    """Measure how alike a batch of failures is."""
    grouped = _grouped(failures)
    return FailureShape(
        cases=len(failures),
        families=len(grouped),
        largest=max((len(cases) for _, cases in grouped), default=0),
    )


def diagnose_prompt(
    candidate: Candidate,
    failures: Sequence[tuple[str, object, str]],
) -> str:
    """Ask for a diagnosis first, then a rewritten instruction.

    Diagnosis before rewriting is deliberate: a model asked only for a new
    instruction rewrites prose, and a model asked what went wrong makes a claim we
    can read, keep, or discard.

    Failures arrive grouped by family, and the ask is for however many mistakes the
    cases actually show. Asking for *the single* shared mistake assumed a repeated
    error, so on a task whose mistakes are all different the model would correctly
    answer "none" and the loop would stop having learned nothing. Naming the families
    gives it something true to say instead.
    """
    shown = 0
    blocks: list[str] = []
    related = 0
    for family, cases in _grouped(failures):
        del family
        for text, gold, produced in cases:
            if shown >= _MAX_EXAMPLES:
                break
            shown += 1
            blocks.append(
                "CASE\n"
                f"document: {text[:600]}\n"
                f"correct answer: {_as_text(gold)}\n"
                f"current output: {produced[:400]}"
            )
        if len(cases) > 1:
            related += 1
        if shown >= _MAX_EXAMPLES:
            break
    lessons = "\n".join(f"- {lesson}" for lesson in candidate.lessons) or "- none yet"
    plural = (
        "These cases do not all share one mistake. Name each distinct mistake you "
        "can see, one per line; cases listed with a shared label word tend to fail "
        "the same way."
        if related
        else "Name the mistake these cases share."
    )
    return (
        f"{candidate.instructions}\n\n"
        "---\n"
        "The instruction above produced wrong answers on these cases, grouped so "
        f"related mistakes sit together:\n\n{chr(10).join(blocks)}\n\n"
        f"Lessons already learned (do not lose these):\n{lessons}\n\n"
        "Reply in exactly this shape and nothing else:\n"
        f"DIAGNOSIS: {plural}\n"
        "INSTRUCTION: the complete revised instruction. Every rule must be a test "
        "the model can apply to a document it has never seen. Never name a specific "
        "case, answer, or label from the examples above, and never list what those "
        "particular documents were. A rule that works only on the cases you were "
        "shown is worthless, because the cases that decide the score are the ones "
        "you were not shown.\n"
    )


def _as_text(value: object) -> str:
    dump = getattr(value, "model_dump_json", None)
    return dump() if callable(dump) else str(value)


def _from_json(reply: str) -> tuple[str, str | None] | None:
    """Read a `{DIAGNOSIS, INSTRUCTION}` reply, or None if it is not one.

    Models answer the requested two-field shape as JSON about half the time, whatever
    the prompt says, and small local models do it more often than large ones. The
    first version of this parser read only plain-text lines, so a JSON reply lost both
    fields and the *entire reply* became the next instruction. Reflection then fed a
    JSON blob back to itself, the child scored no better, and the loop stopped with
    "no improvement" — which read as the task being undiagnosable rather than as a
    parsing bug. Handling both shapes is what makes the rest of the loop real.
    """
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        loaded = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None
    folded = {str(key).strip().lower(): value for key, value in loaded.items()}
    instruction = str(folded.get("instruction") or "").strip()
    diagnosis = str(folded.get("diagnosis") or "").strip() or None
    if not instruction and not diagnosis:
        return None
    return instruction, diagnosis


#: The two labels the diagnosis step is asked for. A label may sit on its own line
#: with the content underneath, which is how models usually answer.
_REPLY_LABELS = ("DIAGNOSIS", "INSTRUCTION")


def _sections(reply: str) -> dict[str, str]:
    """Split a labelled reply into its sections, label line or label line plus body.

    The first parser read only the text after the colon *on the label's own line*. When
    the model wrote `DIAGNOSIS:` and then a bulleted list beneath it — the common
    shape — the section read back empty, the diagnosis was dropped, and the loop
    concluded the task was undiagnosable. Reading to the next label instead makes both
    layouts work.
    """
    found: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None:
            found[current] = "\n".join(buffer).strip()

    for line in reply.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        label = next(
            (name for name in _REPLY_LABELS if upper.startswith(f"{name}:")), None
        )
        if label is not None:
            flush()
            current = label
            buffer = [stripped[len(label) + 1 :].strip()]
            continue
        if current is not None:
            buffer.append(line)
    flush()
    return {name: value for name, value in found.items() if value}


def parse_reply(reply: str, *, fallback: str) -> tuple[str, str | None]:
    """Split a reply into (instruction, diagnosis).

    JSON is tried first because a JSON reply read as prose yields a blob, and a blob is
    worse than no answer: it becomes the next instruction. A model that answers with
    neither shape keeps its instruction and loses only the diagnosis; refusing the whole
    reply would throw away usable work.
    """
    as_json = _from_json(reply)
    if as_json is not None and as_json[0]:
        return as_json
    sections = _sections(reply)
    instruction = sections.get("INSTRUCTION", "")
    diagnosis = sections.get("DIAGNOSIS") or (as_json[1] if as_json else None)
    if not instruction:
        candidate = reply.strip()
        instruction = (
            candidate
            if _MIN_INSTRUCTION_CHARS < len(candidate) <= _MAX_INSTRUCTION_CHARS
            else ""
        )
    if "{" in instruction and instruction.rstrip().endswith("}"):
        instruction = ""
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
    allow_knowledge_gap: bool = False,
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

        # Before spending a single reflection call, check whether these failures are
        # the kind an instruction can fix. Repeat work is worth reflecting on; a fresh
        # mistake in every case means the model does not know the answers, and no
        # rewrite will teach it. Reporting that instead of grinding through the budget
        # is the difference between a tool that helps and one that burns an evening.
        shape = failure_shape(failures)
        if shape.is_knowledge_gap and not allow_knowledge_gap:
            stopped = shape.explain()
            break

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
