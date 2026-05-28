from datetime import datetime, timezone

from agent.config import Settings
from agent.risk.limits import RiskLimits
from agent.cycle import run_cycle
from agent.data.market import MarketFeed
from agent.data.news import NewsFeed
from agent.llm.decide import Decider
from agent.state.store import Store


class FakeExchange:
    def __init__(self, mark):
        self.mark = mark

    def fetch_ohlcv(self, symbol, timeframe, limit):
        return [[i, 100.0, 101.0, 99.0, 100.0 + i * 0.05, 10.0] for i in range(limit)]

    def fetch_ticker(self, symbol):
        return {"last": self.mark}

    def fetch_funding_rate(self, symbol):
        return {"fundingRate": 0.0001}

    def fetch_open_interest(self, symbol):
        return {"openInterestAmount": 5000.0}


class _Content:
    def __init__(self, text):
        self.text = text


class _Response:
    def __init__(self, text):
        self.content = [_Content(text)]


class FakeAnthropic:
    def __init__(self, text):
        self._text = text

    class _Messages:
        def __init__(self, text):
            self._text = text

        def create(self, **kwargs):
            return _Response(self._text)

    @property
    def messages(self):
        return FakeAnthropic._Messages(self._text)


def _settings(tmp_path):
    return Settings(
        symbol="BTC/USDT:USDT", timeframe="1h", candle_limit=60,
        slippage_bps=2.0, taker_fee_bps=5.0, starting_equity=10000.0,
        anthropic_model="claude-opus-4-7", anthropic_api_key="k",
        db_path=str(tmp_path / "e2e.db"), news_rss_url="x",
        risk=RiskLimits(max_leverage=5, max_position_pct=0.5,
                        require_stop_loss=True, max_daily_loss_pct=0.10,
                        single_position=True),
    )


def test_full_cycle_open_then_resume_and_close(tmp_path):
    settings = _settings(tmp_path)
    store = Store(settings.db_path)

    open_json = (
        '{"action":"OPEN_LONG","conviction":0.8,"size_pct":0.5,"leverage":3,'
        '"stop_loss_pct":0.02,"take_profit_pct":0.06,"reasoning":"uptrend"}'
    )
    close_json = (
        '{"action":"CLOSE","conviction":0.6,"size_pct":1.0,"leverage":1,'
        '"stop_loss_pct":0.0,"take_profit_pct":null,"reasoning":"take profit"}'
    )

    # Cycle 1: open a long.
    market1 = MarketFeed(settings.symbol, "1h", 60, exchange=FakeExchange(mark=100.0))
    decider1 = Decider("k", settings.anthropic_model, client=FakeAnthropic(open_json))
    run_cycle(settings, market1, NewsFeed(None), decider1, store,
              now=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))

    assert store.load_position().side == "LONG"

    # Cycle 2: a fresh decider/market, agent resumes from SQLite and closes.
    market2 = MarketFeed(settings.symbol, "1h", 60, exchange=FakeExchange(mark=110.0))
    decider2 = Decider("k", settings.anthropic_model, client=FakeAnthropic(close_json))
    run_cycle(settings, market2, NewsFeed(None), decider2, store,
              now=datetime(2026, 5, 20, 13, 0, tzinfo=timezone.utc))

    assert store.load_position().side == "FLAT"
    # closed in profit -> equity above the starting balance
    assert store.last_cash(0.0) > 10000.0

    decisions = store.conn.execute("SELECT action FROM decisions ORDER BY ts").fetchall()
    assert [r["action"] for r in decisions] == ["OPEN_LONG", "CLOSE"]
    store.close()
