from agent.execution.base import ExecutionResult, Fill
from agent.llm.schema import Decision
from agent.state.store import FLAT, Position


def liq_price(side: str, entry: float, leverage: int) -> float:
    # Simplified approximation: ignores maintenance margin and fees.
    if side == "LONG":
        return entry * (1 - 1 / leverage)
    return entry * (1 + 1 / leverage)


def unrealized(position: Position, mark: float) -> float:
    if position.side == "LONG":
        return position.qty * (mark - position.entry)
    if position.side == "SHORT":
        return position.qty * (position.entry - mark)
    return 0.0


class PaperExecutor:
    def __init__(
        self, cash: float, position: Position, realized_pnl: float,
        max_position_pct: float, slippage_bps: float, taker_fee_bps: float,
    ) -> None:
        self.cash = cash
        self.position = position
        self.realized_pnl = realized_pnl
        self.max_position_pct = max_position_pct
        self.slippage_bps = slippage_bps
        self.taker_fee_bps = taker_fee_bps

    def equity(self, mark_price: float) -> float:
        return self.cash + unrealized(self.position, mark_price)

    def accrue_funding(self, funding_rate: float, mark_price: float) -> float:
        # Simplified Phase-1 model: apply funding once per cycle while a
        # position is open. funding_rate > 0 => longs pay shorts.
        if self.position.side == "FLAT" or funding_rate == 0.0:
            return 0.0
        notional = self.position.qty * mark_price
        payment = notional * funding_rate
        cost = payment if self.position.side == "LONG" else -payment
        self.cash -= cost
        self.realized_pnl -= cost
        return cost

    def _fill_price(self, side: str, mark: float) -> float:
        slip = self.slippage_bps / 10_000
        return mark * (1 + slip) if side == "BUY" else mark * (1 - slip)

    def _fee(self, qty: float, price: float) -> float:
        return abs(qty) * price * self.taker_fee_bps / 10_000

    def apply(self, decision: Decision, mark_price: float) -> ExecutionResult:
        notes: list[str] = []
        fills: list[Fill] = []

        # 1. Liquidation check before acting on the decision.
        if self.position.side != "FLAT":
            crossed = (
                (self.position.side == "LONG" and mark_price <= self.position.liq_price)
                or
                (self.position.side == "SHORT" and mark_price >= self.position.liq_price)
            )
            if crossed:
                fills.append(self._close(self.position.liq_price))
                notes.append(f"LIQUIDATED at {self.position.liq_price:.2f}")
                return self._result(mark_price, fills, notes)

        action = decision.action
        if action == "HOLD":
            notes.append("hold")
        elif action in ("OPEN_LONG", "OPEN_SHORT"):
            if self.position.side != "FLAT":
                notes.append("open ignored: position already open")
            else:
                fills.append(self._open(action, decision, mark_price))
        elif action == "ADD":
            if self.position.side == "FLAT":
                notes.append("add ignored: no position")
            else:
                fills.append(self._add(decision, mark_price))
        elif action == "REDUCE":
            if self.position.side == "FLAT":
                notes.append("reduce ignored: no position")
            else:
                fills.append(self._reduce(decision, mark_price))
        elif action == "CLOSE":
            if self.position.side == "FLAT":
                notes.append("close ignored: no position")
            else:
                exit_side = "SELL" if self.position.side == "LONG" else "BUY"
                fills.append(self._close(self._fill_price(exit_side, mark_price)))

        return self._result(mark_price, fills, notes)

    def _open(self, action: str, d: Decision, mark: float) -> Fill:
        side = "LONG" if action == "OPEN_LONG" else "SHORT"
        buy_sell = "BUY" if side == "LONG" else "SELL"
        price = self._fill_price(buy_sell, mark)
        margin = d.size_pct * self.max_position_pct * self.cash
        qty = (margin * d.leverage) / price
        fee = self._fee(qty, price)
        self.cash -= fee
        self.realized_pnl -= fee
        self.position = Position(
            side=side, qty=qty, entry=price, leverage=d.leverage,
            liq_price=liq_price(side, price, d.leverage),
        )
        return Fill(side=buy_sell, qty=qty, price=price, fee=fee)

    def _add(self, d: Decision, mark: float) -> Fill:
        pos = self.position
        buy_sell = "BUY" if pos.side == "LONG" else "SELL"
        price = self._fill_price(buy_sell, mark)
        margin = d.size_pct * self.max_position_pct * self.cash
        add_qty = (margin * pos.leverage) / price
        fee = self._fee(add_qty, price)
        self.cash -= fee
        self.realized_pnl -= fee
        new_qty = pos.qty + add_qty
        new_entry = (pos.entry * pos.qty + price * add_qty) / new_qty
        self.position = Position(
            side=pos.side, qty=new_qty, entry=new_entry, leverage=pos.leverage,
            liq_price=liq_price(pos.side, new_entry, pos.leverage),
        )
        return Fill(side=buy_sell, qty=add_qty, price=price, fee=fee)

    def _reduce(self, d: Decision, mark: float) -> Fill:
        pos = self.position
        buy_sell = "SELL" if pos.side == "LONG" else "BUY"
        price = self._fill_price(buy_sell, mark)
        qty = pos.qty * d.size_pct
        if pos.side == "LONG":
            pnl = qty * (price - pos.entry)
        else:
            pnl = qty * (pos.entry - price)
        fee = self._fee(qty, price)
        self.cash += pnl - fee
        self.realized_pnl += pnl - fee
        remaining = pos.qty - qty
        if remaining <= 0:
            self.position = FLAT
        else:
            self.position = Position(
                side=pos.side, qty=remaining, entry=pos.entry,
                leverage=pos.leverage, liq_price=pos.liq_price,
            )
        return Fill(side=buy_sell, qty=qty, price=price, fee=fee)

    def _close(self, price: float) -> Fill:
        pos = self.position
        buy_sell = "SELL" if pos.side == "LONG" else "BUY"
        pnl = unrealized(pos, price)
        fee = self._fee(pos.qty, price)
        self.cash += pnl - fee
        self.realized_pnl += pnl - fee
        qty = pos.qty
        self.position = FLAT
        return Fill(side=buy_sell, qty=qty, price=price, fee=fee)

    def _result(
        self, mark: float, fills: list[Fill], notes: list[str]
    ) -> ExecutionResult:
        unreal = unrealized(self.position, mark)
        return ExecutionResult(
            fills=fills, position=self.position, cash=self.cash,
            equity=self.cash + unreal, unrealized_pnl=unreal,
            realized_pnl=self.realized_pnl, notes=notes,
        )
