import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from agent.llm.schema import Decision


@dataclass
class Position:
    side: str           # "LONG" | "SHORT" | "FLAT"
    qty: float
    entry: float
    leverage: int
    liq_price: float


FLAT = Position(side="FLAT", qty=0.0, entry=0.0, leverage=1, liq_price=0.0)


class Store:
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                ts TEXT, raw_json TEXT, action TEXT, reasoning TEXT
            );
            CREATE TABLE IF NOT EXISTS trades (
                ts TEXT, side TEXT, qty REAL, price REAL, fee REAL
            );
            CREATE TABLE IF NOT EXISTS positions (
                ts TEXT, side TEXT, qty REAL, entry REAL,
                leverage INTEGER, liq_price REAL
            );
            CREATE TABLE IF NOT EXISTS equity (
                ts TEXT, equity REAL, cash REAL, realized REAL, unrealized REAL
            );
            """
        )
        self.conn.commit()

    def record_decision(self, ts: datetime, raw_json: str, decision: Decision) -> None:
        self.conn.execute(
            "INSERT INTO decisions (ts, raw_json, action, reasoning) VALUES (?,?,?,?)",
            (ts.isoformat(), raw_json, decision.action, decision.reasoning),
        )
        self.conn.commit()

    def record_trade(
        self, ts: datetime, side: str, qty: float, price: float, fee: float
    ) -> None:
        self.conn.execute(
            "INSERT INTO trades (ts, side, qty, price, fee) VALUES (?,?,?,?,?)",
            (ts.isoformat(), side, qty, price, fee),
        )
        self.conn.commit()

    def save_position(self, ts: datetime, pos: Position) -> None:
        self.conn.execute(
            "INSERT INTO positions (ts, side, qty, entry, leverage, liq_price) "
            "VALUES (?,?,?,?,?,?)",
            (ts.isoformat(), pos.side, pos.qty, pos.entry, pos.leverage, pos.liq_price),
        )
        self.conn.commit()

    def load_position(self) -> Position:
        row = self.conn.execute(
            "SELECT side, qty, entry, leverage, liq_price "
            "FROM positions ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return FLAT
        return Position(
            side=row["side"], qty=row["qty"], entry=row["entry"],
            leverage=row["leverage"], liq_price=row["liq_price"],
        )

    def record_equity(
        self, ts: datetime, equity: float, cash: float,
        realized: float, unrealized: float,
    ) -> None:
        self.conn.execute(
            "INSERT INTO equity (ts, equity, cash, realized, unrealized) "
            "VALUES (?,?,?,?,?)",
            (ts.isoformat(), equity, cash, realized, unrealized),
        )
        self.conn.commit()

    def last_cash(self, fallback: float) -> float:
        row = self.conn.execute(
            "SELECT cash FROM equity ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        return row["cash"] if row else fallback

    def last_realized(self, fallback: float) -> float:
        row = self.conn.execute(
            "SELECT realized FROM equity ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        return row["realized"] if row else fallback

    def day_start_equity(self, now: datetime, fallback: float) -> float:
        day = now.astimezone(timezone.utc).date().isoformat()
        row = self.conn.execute(
            "SELECT equity FROM equity WHERE ts >= ? ORDER BY ts ASC LIMIT 1",
            (day,),
        ).fetchone()
        return row["equity"] if row else fallback

    def close(self) -> None:
        self.conn.close()
