import time

from agent.config import load_settings
from agent.cycle import run_cycle
from agent.data.market import MarketFeed
from agent.data.news import NewsFeed, rss_source
from agent.llm.decide import Decider
from agent.state.store import Store

_TIMEFRAME_SECONDS = {"1h": 3600, "4h": 14400}


def seconds_until_next_close(timeframe: str, now: float) -> float:
    period = _TIMEFRAME_SECONDS[timeframe]
    remainder = now % period
    return period if remainder == 0 else period - remainder


def main() -> None:
    settings = load_settings()
    market = MarketFeed(settings.symbol, settings.timeframe, settings.candle_limit)
    news = NewsFeed(rss_source(settings.news_rss_url))
    decider = Decider(settings.anthropic_api_key, settings.anthropic_model)
    store = Store(settings.db_path)
    print(f"agent started: {settings.symbol} {settings.timeframe}")
    try:
        while True:
            wait = seconds_until_next_close(settings.timeframe, time.time()) + 5
            print(f"sleeping {wait:.0f}s until next candle close")
            time.sleep(wait)
            try:
                run_cycle(settings, market, news, decider, store)
            except Exception as exc:
                print(f"cycle crashed (contained): {exc}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
