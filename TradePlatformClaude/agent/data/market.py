from dataclasses import dataclass
from typing import Any

import ccxt


@dataclass
class FuturesMetrics:
    funding_rate: float
    open_interest: float
    long_short_ratio: float


@dataclass
class MarketData:
    candles: list[list[float]]
    mark_price: float
    metrics: FuturesMetrics


class MarketFeed:
    def __init__(
        self, symbol: str, timeframe: str, candle_limit: int,
        exchange: Any = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.candle_limit = candle_limit
        self.exchange = exchange or ccxt.mexc({"options": {"defaultType": "swap"}})

    def fetch(self) -> MarketData:
        candles = self.exchange.fetch_ohlcv(
            self.symbol, timeframe=self.timeframe, limit=self.candle_limit
        )
        ticker = self.exchange.fetch_ticker(self.symbol)
        mark_price = float(ticker["last"])

        funding = 0.0
        try:
            funding = float(self.exchange.fetch_funding_rate(self.symbol)
                            .get("fundingRate") or 0.0)
        except Exception:
            pass

        open_interest = 0.0
        try:
            open_interest = float(self.exchange.fetch_open_interest(self.symbol)
                                  .get("openInterestAmount") or 0.0)
        except Exception:
            pass

        metrics = FuturesMetrics(
            funding_rate=funding,
            open_interest=open_interest,
            long_short_ratio=0.0,   # no unified ccxt method for MEXC
        )
        return MarketData(candles=candles, mark_price=mark_price, metrics=metrics)
