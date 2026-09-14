from mekoy.harness import Decode, extract
from mekoy.outcome import RestaurantOutcome
from mekoy.verify import VerifyFail, VerifyOk


class _Scripted:
    def __init__(self, replies: list[str]) -> None:
        self._replies: list[str] = replies
        self.seen_constrained: list[bool] = []

    local: bool = True

    def complete(
        self,
        *,
        system: str,
        user: str,
        constrained: bool = True,
        temperature: float = 0.0,
        schema: dict[str, object] | None = None,
    ) -> str:
        del system, user
        self.seen_constrained.append(constrained)
        return self._replies.pop(0)


def _ok_json() -> str:
    return RestaurantOutcome(
        restaurant="North Loop Bistro",
        intent="reservation",
        status="confirmed",
        party_size=4,
        when="Friday 7pm",
        under_name="Mitchell",
        evidence="The host confirmed a reservation for four.",
        booked=True,
    ).model_dump_json()


def _bad_json() -> str:
    return (
        '{"restaurant":"X","intent":"availability","status":"confirmed",'
        '"party_size":2,"when":null,"under_name":null,'
        '"evidence":"open","booked":true}'
    )


def test_extract_returns_ok_on_valid_json() -> None:
    result = extract(_Scripted([_ok_json()]), text="booked four", retries=0)
    assert isinstance(result, VerifyOk)
    assert result.outcome.restaurant == "North Loop Bistro"


def test_extract_retries_after_verify_fail() -> None:
    result = extract(
        _Scripted([_bad_json(), _ok_json()]), text="booked four", retries=1
    )
    assert isinstance(result, VerifyOk)


def test_extract_stays_failed_without_retries() -> None:
    result = extract(_Scripted([_bad_json()]), text="booked four", retries=0)
    assert isinstance(result, VerifyFail)


def test_decode_defaults_to_constrained() -> None:
    completer = _Scripted([_ok_json()])
    _ = extract(completer, text="booked four", retries=0)
    assert completer.seen_constrained == [True]


def test_unconstrained_decode_is_passed_through() -> None:
    completer = _Scripted([_ok_json()])
    _ = extract(
        completer,
        text="booked four",
        retries=0,
        decode=Decode(system="x", constrained=False),
    )
    assert completer.seen_constrained == [False]
