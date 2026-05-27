from agent.execution.base import Fill, ExecutionResult
from agent.execution.paper import PaperExecutor
from agent.llm.schema import Decision
from agent.state.store import Position, FLAT


def _executor(cash=10000.0, position=FLAT, realized=0.0):
    return PaperExecutor(
        cash=cash, position=position, realized_pnl=realized,
        max_position_pct=0.5, slippage_bps=0.0, taker_fee_bps=0.0,
    )


def _decision(**kw):
    base = dict(
        action="HOLD", conviction=0.5, size_pct=0.5, leverage=2,
        stop_loss_pct=0.02, take_profit_pct=None, reasoning="x",
    )
    base.update(kw)
    return Decision.model_validate(base)


def test_hold_on_flat_account_keeps_position_flat():
    result = _executor().apply(_decision(action="HOLD"), mark_price=100.0)
    assert isinstance(result, ExecutionResult)
    assert result.position == FLAT
    assert result.equity == 10000.0


def test_open_long_creates_position_and_charges_no_fee_at_zero_bps():
    result = _executor(cash=10000.0).apply(
        _decision(action="OPEN_LONG", size_pct=1.0, leverage=2), mark_price=100.0)
    pos = result.position
    # margin = 1.0 * 0.5 * 10000 = 5000; notional = 10000; qty = 100
    assert pos.side == "LONG"
    assert pos.qty == 100.0
    assert pos.entry == 100.0
    assert pos.liq_price == 50.0      # 100 * (1 - 1/2)


def test_close_long_realizes_profit():
    pos = Position(side="LONG", qty=100.0, entry=100.0, leverage=2, liq_price=50.0)
    result = _executor(cash=5000.0, position=pos).apply(
        _decision(action="CLOSE"), mark_price=110.0)
    # pnl = 100 * (110 - 100) = 1000
    assert result.position == FLAT
    assert result.cash == 6000.0
    assert result.realized_pnl == 1000.0


def test_long_force_closed_on_liquidation():
    pos = Position(side="LONG", qty=100.0, entry=100.0, leverage=2, liq_price=50.0)
    result = _executor(cash=5000.0, position=pos).apply(
        _decision(action="HOLD"), mark_price=49.0)
    assert result.position == FLAT
    assert any("LIQUIDATED" in n for n in result.notes)


def test_open_ignored_when_position_already_open():
    pos = Position(side="LONG", qty=10.0, entry=100.0, leverage=2, liq_price=50.0)
    result = _executor(position=pos).apply(
        _decision(action="OPEN_SHORT"), mark_price=100.0)
    assert result.position == pos
    assert any("already open" in n for n in result.notes)


def test_accrue_funding_long_pays_when_rate_positive():
    pos = Position(side="LONG", qty=10.0, entry=100.0, leverage=2, liq_price=50.0)
    ex = _executor(cash=5000.0, position=pos)
    cost = ex.accrue_funding(funding_rate=0.001, mark_price=100.0)
    # notional = 10 * 100 = 1000; payment = 1.0; long pays
    assert cost == 1.0
    assert ex.cash == 4999.0


def test_accrue_funding_is_noop_when_flat():
    ex = _executor()
    assert ex.accrue_funding(0.001, 100.0) == 0.0
    assert ex.cash == 10000.0
