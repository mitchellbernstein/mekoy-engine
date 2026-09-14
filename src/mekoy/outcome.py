"""Bucko restaurant-call result. Frozen fields Luna currently writes."""

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

Intent = Literal["availability", "reservation"]
Status = Literal["confirmed", "unknown", "unavailable"]


class RestaurantOutcome(BaseModel):
    """What staff actually confirmed. Not a booking agent."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    restaurant: str
    intent: Intent
    status: Status
    party_size: int | None = Field(default=None, ge=1)
    when: str | None = None
    under_name: str | None = None
    evidence: str
    booked: bool
