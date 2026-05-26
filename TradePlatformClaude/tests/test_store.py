from datetime import datetime, timezone

from agent.llm.schema import hold
from agent.state.store import Store, Position, FLAT


def _store(tmp_path):
    return Store(str(tmp_path / "t.db"))


def test_load_position_defaults_to_flat(tmp_path):
    s = _store(tmp_path)
    assert s.load_position() == FLAT
    s.close()


def test_save_and_load_position_roundtrip(tmp_path):
    s = _store(tmp_path)
    ts = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    pos = Position(side="LONG", qty=1.5, entry=60000.0, leverage=3, liq_price=40000.0)
    s.save_position(ts, pos)
    assert s.load_position() == pos
    s.close()


def test_record_decision_and_equity(tmp_path):
    s = _store(tmp_path)
    ts = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    s.record_decision(ts, '{"raw":1}', hold("x"))
    s.record_equity(ts, equity=10050.0, cash=10050.0, realized=50.0, unrealized=0.0)
    assert s.last_cash(0.0) == 10050.0
    assert s.last_realized(0.0) == 50.0
    s.close()


def test_day_start_equity_returns_first_row_of_utc_day(tmp_path):
    s = _store(tmp_path)
    morning = datetime(2026, 5, 20, 1, 0, tzinfo=timezone.utc)
    noon = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    s.record_equity(morning, equity=10000.0, cash=10000.0, realized=0.0, unrealized=0.0)
    s.record_equity(noon, equity=9000.0, cash=9000.0, realized=-1000.0, unrealized=0.0)
    assert s.day_start_equity(noon, fallback=1.0) == 10000.0
    s.close()


def test_day_start_equity_fallback_when_no_rows(tmp_path):
    s = _store(tmp_path)
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    assert s.day_start_equity(now, fallback=777.0) == 777.0
    s.close()
