from datetime import datetime, timezone

from agent.config import Settings
from agent.data.indicators import compute
from agent.data.market import MarketFeed
from agent.data.news import NewsFeed
from agent.execution.paper import PaperExecutor
from agent.llm.decide import Decider
from agent.llm.prompt import build_user_prompt
from agent.risk.validate import AccountState, validate
from agent.state.store import Store


def run_cycle(
    settings: Settings,
    market: MarketFeed,
    news: NewsFeed,
    decider: Decider,
    store: Store,
    now: datetime | None = None,
) -> None:
    now = now or datetime.now(timezone.utc)

    # 1. Market data
    try:
        data = market.fetch()
    except Exception as exc:
        print(f"[{now.isoformat()}] data fetch failed: {exc}; skipping cycle")
        return

    indicators = compute(data.candles)

    try:
        news_items = news.fetch()
    except Exception as exc:
        news_items = []
        print(f"[{now.isoformat()}] news fetch failed: {exc}; continuing")

    # 2. Account state, reconstructed from persisted state
    position = store.load_position()
    cash = store.last_cash(settings.starting_equity)
    realized = store.last_realized(0.0)
    executor = PaperExecutor(
        cash=cash, position=position, realized_pnl=realized,
        max_position_pct=settings.risk.max_position_pct,
        slippage_bps=settings.slippage_bps, taker_fee_bps=settings.taker_fee_bps,
    )
    # Accrue funding once per cycle while a position is open (spec §9).
    executor.accrue_funding(data.metrics.funding_rate, data.mark_price)
    account = AccountState(
        equity=executor.equity(data.mark_price),
        day_start_equity=store.day_start_equity(now, settings.starting_equity),
        position=position,
    )

    # 3. Prompt
    user_prompt = build_user_prompt(
        settings.symbol, data.candles, indicators, data.metrics,
        news_items, account,
    )

    # 4. LLM decision
    decision, raw = decider.decide(user_prompt)

    # 5. Risk validation
    decision, adjustments = validate(decision, account, settings.risk)

    # 6. Execute
    result = executor.apply(decision, data.mark_price)

    # 7. Persist
    store.record_decision(now, raw, decision)
    for fill in result.fills:
        store.record_trade(now, fill.side, fill.qty, fill.price, fill.fee)
    store.save_position(now, result.position)
    store.record_equity(
        now, result.equity, result.cash,
        result.realized_pnl, result.unrealized_pnl,
    )

    print(
        f"[{now.isoformat()}] action={decision.action} "
        f"adjustments={adjustments} notes={result.notes} "
        f"equity={result.equity:.2f}"
    )
