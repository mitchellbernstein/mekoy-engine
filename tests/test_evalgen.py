from pathlib import Path

from mekoy.dataset import audit_coverage, load_examples, split_examples
from mekoy.evalgen import audit, build_scenarios, write_examples
from mekoy.verify import VerifyOk, parse_and_verify


def test_generated_labels_satisfy_the_gate() -> None:
    rows = build_scenarios()
    assert len(rows) >= 90
    assert audit(rows) == ()


def test_generated_labels_are_all_verifiable() -> None:
    for row in build_scenarios():
        assert isinstance(parse_and_verify(row.outcome().model_dump_json()), VerifyOk)


def test_generation_is_deterministic() -> None:
    first = [row.text for row in build_scenarios()]
    second = [row.text for row in build_scenarios()]
    assert first == second


def test_generated_corpus_has_a_balanced_test_split(tmp_path: Path) -> None:
    path = write_examples(tmp_path / "generated.jsonl", build_scenarios())
    split = split_examples(load_examples(path))
    assert len(split.test) >= 10
    assert audit_coverage(split) == ()


def test_audit_catches_a_booked_violation() -> None:
    """The audit is only useful if it actually fails on bad labels."""
    good = build_scenarios()[0]
    bad = type(good)(
        text=good.text,
        restaurant=good.restaurant,
        intent="availability",
        status="confirmed",
        party_size=2,
        when=None,
        under_name=None,
        booked=True,
        evidence="x",
    )
    assert audit((bad,)) != ()


def test_every_restaurant_appears_in_its_transcript() -> None:
    assert audit(build_scenarios()) == ()
