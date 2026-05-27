from agent.llm.schema import Decision
from agent.risk.limits import RiskLimits
from agent.risk.validate import validate, AccountState
from agent.state.store import Position, FLAT

LIMITS = RiskLimits(
    max_leverage=5, max_position_pct=0.5, require_stop_loss=True,
    max_daily_loss_pct=0.10, single_position=True,
)


def _decision(**kw):
    base = dict(
        action="OPEN_LONG", conviction=0.8, size_pct=0.5, leverage=3,
        stop_loss_pct=0.02, take_profit_pct=None, reasoning="x",
    )
    base.update(kw)
    return Decision.model_validate(base)


def _account(equity=10000.0, day_start=10000.0, position=FLAT):
    return AccountState(equity=equity, day_start_equity=day_start, position=position)


def test_leverage_clamped_to_max():
    d, adj = validate(_decision(leverage=20), _account(), LIMITS)
    assert d.leverage == 5
    assert any("leverage" in a for a in adj)


def test_leverage_within_cap_unchanged():
    d, adj = validate(_decision(leverage=4), _account(), LIMITS)
    assert d.leverage == 4
    assert adj == []


def test_open_without_stop_loss_downgraded_to_hold():
    d, adj = validate(_decision(stop_loss_pct=0.0), _account(), LIMITS)
    assert d.action == "HOLD"


def test_kill_switch_blocks_open_after_daily_loss():
    acct = _account(equity=8900.0, day_start=10000.0)
    d, adj = validate(_decision(), acct, LIMITS)
    assert d.action == "HOLD"
    assert "kill switch" in d.reasoning


def test_kill_switch_allows_close_after_daily_loss():
    acct = _account(equity=8900.0, day_start=10000.0, position=Position(
        side="LONG", qty=1.0, entry=100.0, leverage=3, liq_price=70.0))
    d, adj = validate(_decision(action="CLOSE"), acct, LIMITS)
    assert d.action == "CLOSE"


def test_single_position_blocks_second_open():
    acct = _account(position=Position(
        side="LONG", qty=1.0, entry=100.0, leverage=3, liq_price=70.0))
    d, adj = validate(_decision(action="OPEN_SHORT"), acct, LIMITS)
    assert d.action == "HOLD"
