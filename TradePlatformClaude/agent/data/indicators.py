from dataclasses import dataclass


@dataclass
class Indicators:
    rsi: float
    sma_fast: float
    sma_slow: float
    macd: float
    macd_signal: float
    atr: float


def _sma(values: list[float], period: int) -> float:
    window = values[-period:] if len(values) >= period else values
    return sum(window) / len(window)


def _ema_series(values: list[float], period: int) -> list[float]:
    k = 2 / (period + 1)
    ema = [values[0]]
    for v in values[1:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> float:
    trs = []
    for i in range(1, len(closes)):
        trs.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    if not trs:
        return 0.0
    window = trs[-period:] if len(trs) >= period else trs
    return sum(window) / len(window)


def compute(candles: list[list[float]]) -> Indicators:
    closes = [c[4] for c in candles]
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    ema_fast = _ema_series(closes, 12)
    ema_slow = _ema_series(closes, 26)
    macd_vals = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal = _ema_series(macd_vals, 9)
    return Indicators(
        rsi=_rsi(closes),
        sma_fast=_sma(closes, 20),
        sma_slow=_sma(closes, 50),
        macd=macd_vals[-1],
        macd_signal=signal[-1],
        atr=_atr(highs, lows, closes),
    )
