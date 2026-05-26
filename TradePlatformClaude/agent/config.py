import os
from dataclasses import dataclass

import yaml

from agent.risk.limits import RiskLimits


@dataclass(frozen=True)
class Settings:
    symbol: str
    timeframe: str
    candle_limit: int
    slippage_bps: float
    taker_fee_bps: float
    starting_equity: float
    anthropic_model: str
    anthropic_api_key: str
    db_path: str
    news_rss_url: str
    risk: RiskLimits


def load_settings(config_path: str = "config.yaml") -> Settings:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    r = raw["risk"]
    risk = RiskLimits(
        max_leverage=int(r["max_leverage"]),
        max_position_pct=float(r["max_position_pct"]),
        require_stop_loss=bool(r["require_stop_loss"]),
        max_daily_loss_pct=float(r["max_daily_loss_pct"]),
        single_position=bool(r["single_position"]),
    )
    return Settings(
        symbol=str(raw["symbol"]),
        timeframe=str(raw["timeframe"]),
        candle_limit=int(raw["candle_limit"]),
        slippage_bps=float(raw["slippage_bps"]),
        taker_fee_bps=float(raw["taker_fee_bps"]),
        starting_equity=float(raw["starting_equity"]),
        anthropic_model=str(raw["anthropic_model"]),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        db_path=str(raw["db_path"]),
        news_rss_url=str(raw["news_rss_url"]),
        risk=risk,
    )
