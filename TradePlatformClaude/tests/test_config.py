import textwrap

from agent.config import load_settings


def test_load_settings_reads_yaml_and_env(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent("""
        symbol: "BTC/USDT:USDT"
        timeframe: "4h"
        candle_limit: 50
        slippage_bps: 1.0
        taker_fee_bps: 4.0
        starting_equity: 5000.0
        anthropic_model: "claude-opus-4-7"
        db_path: "x.db"
        news_rss_url: "https://example.com/rss"
        risk:
          max_leverage: 3
          max_position_pct: 0.4
          require_stop_loss: true
          max_daily_loss_pct: 0.08
          single_position: true
    """))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    s = load_settings(str(cfg))

    assert s.symbol == "BTC/USDT:USDT"
    assert s.timeframe == "4h"
    assert s.starting_equity == 5000.0
    assert s.anthropic_api_key == "test-key"
    assert s.risk.max_leverage == 3
    assert s.risk.require_stop_loss is True
