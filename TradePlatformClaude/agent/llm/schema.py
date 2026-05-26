from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Action = Literal["OPEN_LONG", "OPEN_SHORT", "ADD", "REDUCE", "CLOSE", "HOLD"]


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    conviction: float = Field(ge=0.0, le=1.0)
    size_pct: float = Field(ge=0.0, le=1.0)
    leverage: int = Field(ge=1)
    stop_loss_pct: float = Field(ge=0.0)
    take_profit_pct: float | None = None
    reasoning: str


def hold(reason: str) -> Decision:
    return Decision(
        action="HOLD",
        conviction=0.0,
        size_pct=0.0,
        leverage=1,
        stop_loss_pct=0.0,
        take_profit_pct=None,
        reasoning=reason,
    )
