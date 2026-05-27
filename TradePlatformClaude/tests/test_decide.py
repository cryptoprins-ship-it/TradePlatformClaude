from agent.llm.decide import Decider


class _Content:
    def __init__(self, text):
        self.text = text


class _Response:
    def __init__(self, text):
        self.content = [_Content(text)]


class FakeClient:
    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = 0

    class _Messages:
        def __init__(self, outer):
            self._outer = outer

        def create(self, **kwargs):
            self._outer.calls += 1
            return _Response(self._outer._texts.pop(0))

    @property
    def messages(self):
        return FakeClient._Messages(self)


_GOOD = (
    '{"action":"OPEN_LONG","conviction":0.7,"size_pct":0.4,"leverage":3,'
    '"stop_loss_pct":0.02,"take_profit_pct":null,"reasoning":"uptrend"}'
)


def test_decide_parses_valid_json():
    decider = Decider("k", "claude-opus-4-7", client=FakeClient([_GOOD]))
    decision, raw = decider.decide("prompt")
    assert decision.action == "OPEN_LONG"
    assert decision.leverage == 3


def test_decide_extracts_json_embedded_in_prose():
    decider = Decider("k", "claude-opus-4-7",
                      client=FakeClient([f"Here is my call: {_GOOD} done."]))
    decision, _ = decider.decide("prompt")
    assert decision.action == "OPEN_LONG"


def test_decide_retries_once_then_holds_on_garbage():
    decider = Decider("k", "claude-opus-4-7",
                      client=FakeClient(["not json", "still not json"]))
    decision, _ = decider.decide("prompt")
    assert decision.action == "HOLD"
    assert decider.client.calls == 2


def test_decide_recovers_on_second_attempt():
    decider = Decider("k", "claude-opus-4-7",
                      client=FakeClient(["garbage", _GOOD]))
    decision, _ = decider.decide("prompt")
    assert decision.action == "OPEN_LONG"
