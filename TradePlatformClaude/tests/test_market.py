from agent.data.market import MarketFeed, MarketData, FuturesMetrics


class FakeExchange:
    def fetch_ohlcv(self, symbol, timeframe, limit):
        return [[i, 100.0, 101.0, 99.0, 100.5, 10.0] for i in range(limit)]

    def fetch_ticker(self, symbol):
        return {"last": 100.5}

    def fetch_funding_rate(self, symbol):
        return {"fundingRate": 0.0001}

    def fetch_open_interest(self, symbol):
        return {"openInterestAmount": 1234.0}


def test_fetch_returns_market_data():
    feed = MarketFeed("BTC/USDT:USDT", "1h", 30, exchange=FakeExchange())
    data = feed.fetch()
    assert isinstance(data, MarketData)
    assert len(data.candles) == 30
    assert data.mark_price == 100.5
    assert data.metrics.funding_rate == 0.0001
    assert data.metrics.open_interest == 1234.0


def test_fetch_tolerates_missing_futures_metrics():
    class Partial(FakeExchange):
        def fetch_funding_rate(self, symbol):
            raise RuntimeError("not supported")

        def fetch_open_interest(self, symbol):
            raise RuntimeError("not supported")

    feed = MarketFeed("BTC/USDT:USDT", "1h", 10, exchange=Partial())
    data = feed.fetch()
    assert data.metrics.funding_rate == 0.0
    assert data.metrics.open_interest == 0.0
