from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    max_leverage: int
    max_position_pct: float       # max margin per position as fraction of equity
    require_stop_loss: bool
    max_daily_loss_pct: float     # kill-switch threshold, e.g. 0.10 = 10%
    single_position: bool
