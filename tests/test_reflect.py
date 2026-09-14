"""Our own reflection optimiser. The loop is offline; the model is faked."""

import inspect
from pathlib import Path

import pytest

from mekoy.dataset import ExampleRecord, load_examples, split_examples
from mekoy.errors import CompileError
from mekoy.outcome import RestaurantOutcome
from mekoy.reflect import (
    Candidate,
    _grouped,
    diagnose_prompt,
    failure_shape,
    pareto,
    parse_reply,
    reflect,
)
from mekoy.tasks import RESTAURANT

_FIXTURE = Path("examples/bucko-restaurant/generated.jsonl")
_GOOD = RestaurantOutcome(
    restaurant="Uchi",
    intent="availability",
    status="confirmed",
    party_size=2,
    when="Friday",
    under_name=None,
    evidence="A table for two is open Friday.",
    booked=False,
)


def _row(index: int) -> ExampleRecord:
    return ExampleRecord(text=f"call {index}", outcome=_GOOD)


class _Reflector:
    """Scores one phrase in the instruction, and rewrites when asked."""

    local: bool = True

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.systems: list[str] = []
        self.writes = 0

    def complete(
        self,
        *,
        system: str,
        user: str,
        constrained: bool = True,
        temperature: float = 0.0,
        schema: dict[str, object] | None = None,
    ) -> str:
        del constrained, schema, temperature
        self.systems.append(system)
        if user.strip().startswith(system.strip()[:40]):
            # This is the extraction call, not the reflection call.
            marker = "GOOD" in system
            return (
                _GOOD.model_dump_json().replace('"confirmed"', '"confirmed"')
                if marker
                else "{}"
            )
        self.writes += 1
        return self._replies.pop(0) if self._replies else "ok"


def test_pareto_keeps_complementary_candidates() -> None:
    """Averaging would drop one of these; the front keeps both."""
    a = Candidate(instructions="A", per_example=(1.0, 0.5))
    b = Candidate(instructions="B", per_example=(0.5, 1.0))
    dominated = Candidate(instructions="C", per_example=(0.4, 0.4))
    assert {c.instructions for c in pareto([a, b, dominated])} == {"A", "B"}


def test_dominance_is_not_a_mean_comparison() -> None:
    a = Candidate(instructions="A", per_example=(1.0, 0.0))
    b = Candidate(instructions="B", per_example=(0.5, 0.5))
    assert a.score == b.score
    assert not a.beats(b)
    assert not b.beats(a)


def test_parse_reply_reads_the_requested_shape() -> None:
    text = (
        "DIAGNOSIS: confuses a time with a headcount.\n"
        "INSTRUCTION: Rewrite it and be careful."
    )
    instruction, diagnosis = parse_reply(text, fallback="KEEP")
    assert instruction == "Rewrite it and be careful."
    assert diagnosis == "confuses a time with a headcount."


def test_parse_reply_reads_a_label_with_its_body_on_the_following_lines() -> None:
    """The shape that silently discarded every rewrite.

    Models usually put `DIAGNOSIS:` on its own line and the content underneath. Reading
    only what follows the colon on the label's own line returned an empty string, so
    both fields were dropped and the loop concluded the task could not be diagnosed.
    """
    text = (
        "DIAGNOSIS: \n"
        "- Confuses a transfer charge with a top-up limit.\n"
        "- Reads a completed top-up as still pending.\n"
        "\n"
        "INSTRUCTION:\n"
        "- A transfer charge is not a top-up limit.\n"
        "- A completed top-up is never pending.\n"
    )
    instruction, diagnosis = parse_reply(text, fallback="KEEP")
    assert "A transfer charge is not a top-up limit." in instruction
    assert "A completed top-up is never pending." in instruction
    assert diagnosis is not None
    assert "Confuses a transfer charge" in diagnosis
    assert instruction != "KEEP"


def test_parse_reply_reads_a_json_answer() -> None:
    """Small local models answer the two-field shape as JSON about half the time.

    Read as prose, a JSON reply made the *entire blob* the next instruction, so the
    child scored no better and the loop reported "no improvement" — a parsing failure
    dressed up as a property of the task.
    """
    text = (
        '{"DIAGNOSIS": "Names the wrong field.", '
        '"INSTRUCTION": "Always read the time before the headcount."}'
    )
    instruction, diagnosis = parse_reply(text, fallback="KEEP")
    assert instruction == "Always read the time before the headcount."
    assert diagnosis == "Names the wrong field."


def test_a_json_blob_never_becomes_the_next_instruction() -> None:
    """A failed parse must keep the parent, not adopt the raw reply."""
    blob = '{"unrelated": "' + ("x" * 200) + '"}'
    instruction, _ = parse_reply(blob, fallback="KEEP")
    assert instruction == "KEEP"


def test_failure_shape_separates_a_rule_gap_from_a_knowledge_gap() -> None:
    """Scattered mistakes cannot be instructed away; repeated ones can.

    Nine unique mistakes in nine cases is what BANKING77 actually produced, and four
    rewritten instructions moved it by zero. Saying so beats spending the budget.
    """
    scattered = [
        ("t", f'{{"label":"{name}"}}', "wrong")
        for name in ("a_b_c", "d_e_f", "g_h_i", "j_k_l", "m_n_o", "p_q_r")
    ]
    assert failure_shape(scattered).is_knowledge_gap
    assert "knowledge gap" in failure_shape(scattered).explain()

    repeated = [("t", '{"label":"booked_true"}', "wrong") for _ in range(6)]
    assert not failure_shape(repeated).is_knowledge_gap
    assert "rule gap" in failure_shape(repeated).explain()


def test_failure_shape_is_not_called_on_too_few_cases() -> None:
    """One unique failure is not evidence of anything."""
    assert not failure_shape([("t", '{"label":"a_b"}', "wrong")]).is_knowledge_gap


def test_related_labels_land_in_the_same_family() -> None:
    """`verify_my_identity` and `why_verify_identity` are the same kind of mistake."""
    failures = [
        ("t", '{"label":"why_verify_identity"}', "wrong"),
        ("t", '{"label":"verify_my_identity"}', "wrong"),
        ("t", '{"label":"order_physical_card"}', "wrong"),
        ("t", '{"label":"get_physical_card"}', "wrong"),
    ]
    grouped = _grouped(failures)
    assert len(grouped) == 2
    assert all(len(cases) == 2 for _, cases in grouped)


def test_parse_reply_keeps_usable_work_from_a_prose_answer() -> None:
    prose = (
        "Here is a better instruction that tells the model to always check the "
        "time field before it decides anything at all about the party size."
    )
    instruction, diagnosis = parse_reply(prose, fallback="KEEP")
    assert instruction.startswith("Here is a better instruction")
    assert diagnosis is None


def test_parse_reply_falls_back_on_a_non_answer() -> None:
    assert parse_reply("ok", fallback="KEEP") == ("KEEP", None)


def test_the_diagnosis_prompt_shows_failures_and_asks_for_a_reason() -> None:
    candidate = Candidate(
        instructions="Extract the call.",
        per_example=(0.5,),
        lessons=("times are not sizes",),
    )
    prompt = diagnose_prompt(candidate, [("a call", _GOOD, "{}")])
    assert "wrong answers" in prompt
    assert "DIAGNOSIS:" in prompt
    assert "INSTRUCTION:" in prompt
    assert "times are not sizes" in prompt, "ancestor lessons must survive"


def test_lessons_accumulate_across_generations() -> None:
    candidate = Candidate(
        instructions="x",
        per_example=(0.5,),
        lessons=("first lesson", "second lesson"),
    )
    prompt = diagnose_prompt(candidate, [("a call", _GOOD, "{}")])
    assert "first lesson" in prompt
    assert "second lesson" in prompt


def test_reflection_needs_validation_data() -> None:
    with pytest.raises(CompileError, match="validation slice"):
        reflect(RESTAURANT, completer=_Reflector([]), train=(_row(1),), val=())


def test_the_budget_is_a_hard_cap_on_documents_scored() -> None:
    """A reflection run is bounded work, not an open-ended loop."""
    rows = load_examples(_FIXTURE)
    split = split_examples(rows)
    result = reflect(
        RESTAURANT,
        completer=_Reflector([]),
        train=split.train,
        val=split.dev,
        metric_calls=1,
        seed=0,
    )
    assert result.metric_calls <= 1 + len(split.dev)
    assert result.pool


def test_the_run_is_reproducible() -> None:
    rows = load_examples(_FIXTURE)
    split = split_examples(rows)
    first = reflect(
        RESTAURANT,
        completer=_Reflector(
            ["DIAGNOSIS: d.\nINSTRUCTION: A completely new and different instruction."]
        ),
        train=split.train,
        val=split.dev,
        metric_calls=20,
        seed=7,
    )
    second = reflect(
        RESTAURANT,
        completer=_Reflector(
            ["DIAGNOSIS: d.\nINSTRUCTION: A completely new and different instruction."]
        ),
        train=split.train,
        val=split.dev,
        metric_calls=20,
        seed=7,
    )
    assert first.best.instructions == second.best.instructions
    assert first.metric_calls == second.metric_calls


def test_the_summary_line_reports_both_ends() -> None:
    rows = load_examples(_FIXTURE)
    split = split_examples(rows)
    result = reflect(
        RESTAURANT,
        completer=_Reflector([]),
        train=split.train,
        val=split.dev,
        metric_calls=1,
    )
    text = result.summary()
    assert "reflection:" in text
    assert "metric calls" in text


def test_failures_are_drawn_from_train_not_from_validation() -> None:
    """Reflecting on val failures would fit the instruction to its own test."""
    rows = load_examples(_FIXTURE)
    split = split_examples(rows)
    completer = _Reflector(
        ["DIAGNOSIS: d.\nINSTRUCTION: Another instruction long enough to count."]
    )
    _ = reflect(
        RESTAURANT,
        completer=completer,
        train=split.train,
        val=split.dev,
        metric_calls=40,
        seed=3,
    )
    seen = "\n".join(completer.systems)
    val_texts = [r.text for r in split.dev]
    assert not any(text in seen for text in val_texts), (
        "a validation row reached the prompt"
    )


def test_no_dependency_on_an_external_optimiser() -> None:
    """The whole point: reflection is ours."""
    source = Path(inspect.getfile(reflect)).read_text()
    imports = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    joined = "\n".join(imports)
    assert "dspy" not in joined
    assert "gepa" not in joined
