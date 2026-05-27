from dataclasses import dataclass
from typing import Protocol

from agent.llm.schema import Decision
from agent.state.store import Position


@dataclass
class Fill:
    side: str       # "BUY" | "SELL"
    qty: float
    price: float
    fee: float


@dataclass
class ExecutionResult:
    fills: list[Fill]
    position: Position
    cash: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    notes: list[str]


class Executor(Protocol):
    def apply(self, decision: Decision, mark_price: float) -> ExecutionResult: ...
