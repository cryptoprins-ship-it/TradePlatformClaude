import pytest

from agent.main import seconds_until_next_close


def test_seconds_until_next_close_1h():
    # 100 seconds past the hour -> 3500 left
    assert seconds_until_next_close("1h", now=3600 + 100) == 3500


def test_seconds_until_next_close_4h():
    assert seconds_until_next_close("4h", now=14400 + 400) == 14000


def test_seconds_until_next_close_exact_boundary():
    assert seconds_until_next_close("1h", now=7200) == 3600


def test_unknown_timeframe_raises():
    with pytest.raises(KeyError):
        seconds_until_next_close("13m", now=0)
