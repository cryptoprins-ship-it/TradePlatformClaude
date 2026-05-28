from agent.data.indicators import compute, Indicators


def _candles(closes):
    # build OHLCV rows; high/low padded around close
    return [[i, c, c + 1, c - 1, c, 100.0] for i, c in enumerate(closes)]


def test_compute_returns_indicators():
    closes = [float(x) for x in range(100, 160)]   # steady uptrend
    ind = compute(_candles(closes))
    assert isinstance(ind, Indicators)
    # uptrend -> fast SMA above slow SMA
    assert ind.sma_fast > ind.sma_slow
    # steady gains -> RSI high
    assert ind.rsi > 70


def test_rsi_all_losses_is_low():
    closes = [float(x) for x in range(160, 100, -1)]   # steady downtrend
    ind = compute(_candles(closes))
    assert ind.rsi < 30


def test_atr_is_positive():
    closes = [100.0 + (i % 5) for i in range(60)]
    ind = compute(_candles(closes))
    assert ind.atr > 0
