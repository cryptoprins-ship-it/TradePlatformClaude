from datetime import datetime, timezone

from agent.config import Settings
from agent.risk.limits import RiskLimits
from agent.cycle import run_cycle
from agent.data.market import MarketData, FuturesMetrics
from agent.data.news import NewsFeed
from agent.state.store import Store


def _settings(tmp_path):
    return Settings(
        symbol="BTC/USDT:USDT", timeframe="1h", candle_limit=60,
        slippage_bps=0.0, taker_fee_bps=0.0, starting_equity=10000.0,
        anthropic_model="claude-opus-4-7", anthropic_api_key="k",
        db_path=str(tmp_path / "c.db"), news_rss_url="x",
        risk=RiskLimits(max_leverage=5, max_position_pct=0.5,
                        require_stop_loss=True, max_daily_loss_pct=0.10,
                        single_position=True),
    )


class FakeMarket:
    def __init__(self, fail=False):
        self.fail = fail

    def fetch(self):
        if self.fail:
            raise RuntimeError("exchange down")
        candles = [[i, 100.0, 101.0, 99.0, 100.0 + i * 0.1, 10.0] for i in range(60)]
        return MarketData(
            candles=candles, mark_price=106.0,
            metrics=FuturesMetrics(funding_rate=0.0, open_interest=0.0,
                                   long_short_ratio=0.0),
        )


class FakeDecider:
    def __init__(self, decision, raw):
        self._decision = decision
        self._raw = raw

    def decide(self, user_prompt):
        return self._decision, self._raw


def test_cycle_persists_decision_and_equity(tmp_path):
    from agent.llm.schema import Decision
    settings = _settings(tmp_path)
    store = Store(settings.db_path)
    decision = Decision.model_validate({
        "action": "OPEN_LONG", "conviction": 0.8, "size_pct": 0.5,
        "leverage": 3, "stop_loss_pct": 0.02, "take_profit_pct": None,
        "reasoning": "uptrend",
    })
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)

    run_cycle(settings, FakeMarket(), NewsFeed(None),
              FakeDecider(decision, '{"raw":1}'), store, now=now)

    assert store.load_position().side == "LONG"
    assert store.last_cash(0.0) > 0
    store.close()


def test_cycle_skips_on_data_fetch_failure(tmp_path):
    from agent.llm.schema import hold
    settings = _settings(tmp_path)
    store = Store(settings.db_path)
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)

    run_cycle(settings, FakeMarket(fail=True), NewsFeed(None),
              FakeDecider(hold("x"), "{}"), store, now=now)

    # nothing persisted, position still flat
    assert store.load_position().side == "FLAT"
    assert store.last_cash(-1.0) == -1.0
    store.close()
