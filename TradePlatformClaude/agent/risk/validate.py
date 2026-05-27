from dataclasses import dataclass

from agent.llm.schema import Decision, hold
from agent.risk.limits import RiskLimits
from agent.state.store import Position

_OPENING = ("OPEN_LONG", "OPEN_SHORT", "ADD")


@dataclass
class AccountState:
    equity: float
    day_start_equity: float
    position: Position


def validate(
    decision: Decision, account: AccountState, limits: RiskLimits
) -> tuple[Decision, list[str]]:
    adjustments: list[str] = []
    opening = decision.action in _OPENING

    loss = 0.0
    if account.day_start_equity > 0:
        loss = (account.day_start_equity - account.equity) / account.day_start_equity

    if loss >= limits.max_daily_loss_pct and opening:
        return (
            hold(f"kill switch: daily loss {loss:.2%}"),
            ["kill switch blocked open"],
        )

    d = decision

    if d.leverage > limits.max_leverage:
        adjustments.append(f"leverage {d.leverage} -> {limits.max_leverage}")
        d = d.model_copy(update={"leverage": limits.max_leverage})

    if opening and limits.require_stop_loss and d.stop_loss_pct <= 0.0:
        return hold("open rejected: missing stop loss"), ["missing stop loss"]

    if (
        limits.single_position
        and d.action in ("OPEN_LONG", "OPEN_SHORT")
        and account.position.side != "FLAT"
    ):
        return (
            hold("open rejected: single_position, position already open"),
            ["single_position blocked open"],
        )

    return d, adjustments
