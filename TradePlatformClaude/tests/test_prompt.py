from agent.llm.prompt import SYSTEM, build_user_prompt
from agent.data.indicators import Indicators
from agent.data.market import FuturesMetrics
from agent.data.news import NewsItem
from agent.risk.validate import AccountState
from agent.state.store import FLAT


def test_system_prompt_states_news_is_secondary_and_untrusted():
    assert "SECONDARY" in SYSTEM
    assert "untrusted" in SYSTEM.lower()


def test_user_prompt_contains_all_four_blocks():
    candles = [[i, 100.0, 101.0, 99.0, 100.5, 10.0] for i in range(20)]
    ind = Indicators(rsi=55.0, sma_fast=100.0, sma_slow=99.0,
                     macd=0.5, macd_signal=0.3, atr=2.0)
    metrics = FuturesMetrics(funding_rate=0.0001, open_interest=1000.0,
                             long_short_ratio=0.0)
    news = [NewsItem(headline="BTC ETF inflows rise", summary="...")]
    account = AccountState(equity=10000.0, day_start_equity=10000.0, position=FLAT)

    prompt = build_user_prompt("BTC/USDT:USDT", candles, ind, metrics, news, account)

    assert "PRIMARY SIGNAL" in prompt
    assert "SECONDARY" in prompt
    assert "ACCOUNT STATE" in prompt
    assert "TASK" in prompt
    assert "BTC ETF inflows rise" in prompt


def test_user_prompt_shows_placeholder_when_no_news():
    candles = [[i, 100.0, 101.0, 99.0, 100.5, 10.0] for i in range(20)]
    ind = Indicators(rsi=55.0, sma_fast=100.0, sma_slow=99.0,
                     macd=0.5, macd_signal=0.3, atr=2.0)
    metrics = FuturesMetrics(funding_rate=0.0, open_interest=0.0,
                             long_short_ratio=0.0)
    account = AccountState(equity=10000.0, day_start_equity=10000.0, position=FLAT)

    prompt = build_user_prompt("BTC/USDT:USDT", candles, ind, metrics, [], account)
    assert "(none)" in prompt
