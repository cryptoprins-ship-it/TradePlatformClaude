# Markov Regime Detection — Design Spec (Addendum)

**Date:** 2026-05-22
**Project:** TradePlatformClaude
**Scope:** Addendum to the trading agent engine — adds an HMM market-regime signal.
**Status:** Approved design, ready for implementation planning.
**Parent spec:** `docs/superpowers/specs/2026-05-20-trading-agent-design.md`

---

## 1. Summary

The agent gains a **Markov regime-switching signal**. Each cycle a Gaussian
Hidden Markov Model is fit on recent price history and classifies the market
into one of three directional regimes — **BEAR**, **RANGE**, or **BULL** — with
per-regime probabilities. The estimate is added to the prompt as part of the
PRIMARY SIGNAL. The LLM weighs it; it has **no deterministic effect** on the
risk layer or execution.

This is an addendum: it extends the parent spec without changing any of its
approved decisions. The regime signal is purely additive context.

## 2. Goal & success criteria

The regime feature is done when:

- A 3-state Gaussian HMM is fit each cycle on a configurable long candle
  window and produces a labelled regime plus probabilities.
- The estimate is reproducible — a fixed `random_state` makes `classify()`
  deterministic for a given input.
- The estimate reaches the LLM inside the PRIMARY SIGNAL block.
- Any HMM failure degrades to `UNAVAILABLE` and never aborts a cycle.
- CI stays green: install, typecheck, lint, build, test.

Profitability impact is **not** a success criterion — same as the parent spec,
the Phase-1 bar is "runs unattended, produces sane decisions".

## 3. Key decisions

| Decision           | Choice                                                       |
|--------------------|--------------------------------------------------------------|
| Regime role        | Prompt context only — no risk-layer or execution effect      |
| Model              | Gaussian HMM via `hmmlearn` (`GaussianHMM`)                  |
| Regimes            | 3-state directional: BEAR / RANGE / BULL                     |
| Labelling          | Deterministic post-fit — sort states by fitted mean return   |
| Fit strategy       | Refit every cycle on a dedicated long window; stateless      |
| Failure mode       | Degrade to `UNAVAILABLE`; cycle continues                    |

**Why prompt-context-only.** The parent spec keeps the risk layer as
deterministic, pure functions (parent §8). A regime estimate from an HMM is
noisy; using it as a deterministic gate or cap modulator would whipsaw and
would make the risk layer stateful and harder to test. The existing
`max_daily_loss_pct` kill switch already provides the deterministic crisis
brake. The LLM is the right place to weigh a noisy probabilistic signal.

## 4. Architecture

The regime signal slots into the existing pipeline between the data layer and
the prompt builder. No new pipeline stage — it is a derived market signal,
computed alongside the technical indicators.

```
candle close
   │
   ▼
[Data layer] ── OHLCV (long window) + indicators + futures metrics + news
   │                         │
   │                         ▼
   │                  [Regime] ── HMM fit -> RegimeEstimate
   │                         │
   ▼                         ▼
[Account state]     [Prompt builder] ── regime joins the PRIMARY SIGNAL block
   │                         │
   └─────────────────────────┘
                  ▼
            [LLM] -> [Risk layer] -> [Paper executor] -> [State store]
```

The regime estimate is consumed only by the prompt builder. The risk layer,
executor, and account state are untouched by this addendum.

## 5. New module — `agent/data/regime.py`

The regime module lives in `data/` because it is a derived market signal, the
same category as `indicators.py`. Public surface:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeEstimate:
    label: str                       # "BEAR" | "RANGE" | "BULL" | "UNAVAILABLE"
    probabilities: dict[str, float]  # last-bar posterior per label
    confidence: float                # = max(probabilities); 0.0 when unavailable
    state_volatility: float          # per-bar return stdev of the assigned state
    available: bool


def classify(
    candles: list[list[float]],
    *,
    random_state: int = 42,
    min_returns: int = 120,
) -> RegimeEstimate:
    ...
```

`classify()` is a pure function. `candles` uses the project-wide ccxt OHLCV
layout `[timestamp_ms, open, high, low, close, volume]`, consistent with
`indicators.py`.

### 5.1 Algorithm

1. Extract closes; compute log-returns `r_t = ln(close_t / close_{t-1})`.
2. **Guard:** if fewer than `min_returns` returns, or any return is non-finite
   (a non-positive close), return `UNAVAILABLE` without attempting a fit.
3. Fit `GaussianHMM(n_components=3, covariance_type="diag",
   random_state=random_state)` on the returns reshaped to `(-1, 1)`.
4. Read the three fitted state means (`model.means_`). Sort the state indices
   by mean ascending and build the index→label map: lowest mean → `BEAR`,
   middle → `RANGE`, highest → `BULL`.
5. `model.predict_proba(returns)` → take the **last bar's** row. Remap its
   three entries through the label map into the `probabilities` dict.
6. `label` = argmax of `probabilities`; `confidence` = that max value.
7. `state_volatility` = √(variance of the assigned state), from
   `model.covars_`.
8. Return `RegimeEstimate(..., available=True)`.

Steps 3–7 run inside a `try/except`: **any** exception returns the
`UNAVAILABLE` estimate. A fixed `random_state` makes the fit reproducible, so
`classify()` is deterministic for a given input — required for unit testing.

`UNAVAILABLE` estimate: `label="UNAVAILABLE"`, `probabilities={}`,
`confidence=0.0`, `state_volatility=0.0`, `available=False`.

### 5.2 Why `n_states` is not configurable

The directional labelling in step 4 (BEAR / RANGE / BULL) hardcodes three
states. `n_states` is therefore structural, not a tunable — it is not exposed
in config.

## 6. Configuration

`config.yaml` gains a `regime` block:

```yaml
regime:
  enabled: true
  fit_candle_limit: 720    # candles fetched for the HMM fit (~30 days at 1h)
  random_state: 42
  min_returns: 120         # below this many returns -> UNAVAILABLE
```

`agent/config.py` gains a `RegimeConfig` frozen dataclass, nested in
`Settings` as `Settings.regime`:

```python
@dataclass(frozen=True)
class RegimeConfig:
    enabled: bool
    fit_candle_limit: int
    random_state: int
    min_returns: int
```

When `enabled` is `false`, `classify()` is never called and the prompt shows
the regime as `UNAVAILABLE`.

## 7. Data layer integration — `agent/data/market.py`

The HMM needs far more history than the parent spec's `candle_limit` (100).
`MarketFeed` is given the regime fit window and fetches once:

- `MarketFeed.__init__` gains a `regime_candle_limit: int` parameter.
- `fetch()` calls `fetch_ohlcv` with
  `limit = max(candle_limit, regime_candle_limit)`.
- `MarketData` gains a `regime_candles: list[list[float]]` field holding the
  full long window.
- `MarketData.candles` stays the **last `candle_limit`** rows — existing
  indicator and prompt consumers are unchanged.

A single `fetch_ohlcv` call serves both needs. When `regime.enabled` is
`false`, `regime_candle_limit` is passed as `0` and the feed fetches only
`candle_limit` rows; `regime_candles` is then the same trimmed slice and is
simply not used.

## 8. Prompt integration — `agent/llm/prompt.py`

The regime renders **inside the PRIMARY SIGNAL block** — it is trusted,
price-derived data, unlike the untrusted NEWS block.

`build_user_prompt()` gains a `regime: RegimeEstimate` parameter. Rendering:

```
Regime (HMM 3-state): BULL  conf=0.78  [BEAR 0.05 / RANGE 0.17 / BULL 0.78]  state_vol=0.0121
```

When `available` is `false`:

```
Regime: UNAVAILABLE
```

The `SYSTEM` constant gains one rule line: the regime is an HMM market-state
estimate derived from price history, it is part of the PRIMARY SIGNAL, it
contextualizes trend strength, and it is an estimate rather than a guarantee.
Because `SYSTEM` stays static across cycles it remains prompt-cacheable
(parent §7.2).

## 9. Cycle integration — `agent/cycle.py`

In `run_cycle()`, after the indicators are computed:

```python
regime = (
    classify(
        market.regime_candles,
        random_state=settings.regime.random_state,
        min_returns=settings.regime.min_returns,
    )
    if settings.regime.enabled
    else RegimeEstimate("UNAVAILABLE", {}, 0.0, 0.0, available=False)
)
```

`classify()` contains its own failure handling, so the cycle needs no extra
`try/except` around it. The estimate is passed straight to
`build_user_prompt()`.

## 10. State store — `agent/state/store.py`

The `decisions` table gains a `regime TEXT` column so the future TS dashboard
can show which regime drove each trade. `record_decision()` gains a
`regime_label: str` parameter and writes it. This is one cheap column with
clear analytical value; no other table changes.

## 11. Dependency

`hmmlearn>=0.3` is added to `pyproject.toml` `[project].dependencies`. It pulls
`numpy`, `scipy`, and `scikit-learn` — a heavier CI install, which is
acceptable. `hmmlearn` ships no type stubs; the existing
`[tool.mypy] ignore_missing_imports = true` already covers it.

## 12. Error handling

Every regime failure mode is contained, consistent with parent spec §13:

| Failure                                  | Behaviour                          |
|------------------------------------------|------------------------------------|
| Fewer than `min_returns` returns         | `UNAVAILABLE`, no fit attempted    |
| Non-finite return (non-positive close)   | `UNAVAILABLE`, no fit attempted    |
| HMM fit / `predict_proba` raises         | `UNAVAILABLE`                      |
| Degenerate input (zero-variance series)  | `UNAVAILABLE`                      |
| Long OHLCV fetch fails                   | Handled by existing cycle retry/skip (parent §13) |
| `regime.enabled` is `false`              | `classify()` skipped; prompt shows `UNAVAILABLE` |

A regime failure **never** aborts a cycle. The LLM simply decides on price,
indicators, and futures metrics as it would without the regime signal.

## 13. Security

The regime signal is derived entirely from trusted exchange price data — there
is no new untrusted input surface. It renders inside the PRIMARY SIGNAL block,
not the untrusted NEWS block. No new secrets, no new network endpoints beyond
the existing OHLCV fetch.

## 14. Testing

New `tests/test_regime.py`:

- **Synthetic regime series** — concatenate a strong uptrend, a flat segment,
  and a downtrend; assert `classify()` returns a `RegimeEstimate`, the
  probabilities sum to ≈ 1.0, and `confidence` equals the max probability.
- **Label ordering** — a series with three clearly distinct mean-return
  segments yields `BEAR` mean < `RANGE` mean < `BULL` mean.
- **Too few candles** — fewer than `min_returns` → `UNAVAILABLE`,
  `available is False`.
- **Constant-price series** — zero-variance input does not crash; returns
  `UNAVAILABLE`.
- **Determinism** — the same input classified twice yields an identical
  `RegimeEstimate` (fixed `random_state`).

Updated tests:

- `tests/test_config.py` — `load_settings()` parses the `regime` block into
  `Settings.regime`.
- `tests/test_market.py` — `MarketData.regime_candles` is populated; the long
  window is fetched once.
- `tests/test_prompt.py` — the regime line renders in the PRIMARY SIGNAL
  block; an unavailable estimate renders as `Regime: UNAVAILABLE`.

No live HMM training in CI beyond these deterministic fixture-based tests; each
runs in well under a second.

## 15. Plan integration

The parent 17-task implementation plan
(`docs/superpowers/plans/2026-05-20-trading-agent.md`) is not yet implemented.
The regime feature touches several of its tasks:

| Parent task            | Regime change                                        |
|------------------------|------------------------------------------------------|
| Task 3 — config        | `RegimeConfig`, `regime` block in `config.yaml`      |
| Task 5 — state store   | `regime` column; `record_decision()` parameter       |
| Task 7 — market feed   | `regime_candle_limit`, `regime_candles` field        |
| Task 11 — prompt builder | regime parameter + rendering; `SYSTEM` rule line   |
| Cycle task             | `classify()` call wired into `run_cycle()`           |
| **New task**           | `agent/data/regime.py` + `tests/test_regime.py`      |

The implementation plan produced from this addendum is a focused regime plan
whose steps slot into the parent tasks above, plus the one new module task.

## 16. Out of scope

- Regime-driven risk-layer modulation or hard gating — deliberately excluded
  (§3); may be revisited after paper-trading results.
- Model persistence / refit-every-N-cycles — the stateless refit-per-cycle
  approach was chosen for simplicity.
- Volatility-axis or 2-state regimes — the directional 3-state model was
  chosen; the LLM already sees ATR for volatility context.
