# Agent Task Claims — TradePlatformClaude

This file is the coordination channel between parallel Claude Code sessions working on the trading agent plan (`docs/superpowers/plans/2026-05-20-trading-agent.md`).

**Rule:** Before starting a task, read this file. Append a claim line. Commit the claim. Then do the work. Do not start work on a task another agent has claimed and not yet released.

## Current claims

| Task | Status | Agent | Session | Claimed | Released |
|------|--------|-------|---------|---------|----------|
| 1 | done | Opus 4.7 (this session) | A | 2026-05-22 | 2026-05-22 (commit `61327d6`) |
| 2 | done | Opus 4.7 (this session) | A | 2026-05-22 | 2026-05-22 (commit `2cf8763`) |
| 3 | done | Opus 4.7 (this session) | A | 2026-05-26 | 2026-05-26 (commit `e5931c9`) |
| 4 | done | Opus 4.7 (other session) | B | — | 2026-05-? (commit `b0588c4`) |
| 5 | done | Opus 4.7 (other session) | B | — | 2026-05-? (commit `01e8c5b`) |
| 6 | done | Opus 4.7 (other session) | B | — | 2026-05-? (commit `a81effc`) |
| 7 | free | — | — | — | — |
| 8 | claimed | Opus 4.7 (this session) | A | 2026-05-27 | — |
| 9 | claimed | Opus 4.7 (this session) | A | 2026-05-27 | — |
| 10 | free | — | — | — | — |
| 11 | free | — | — | — | — |
| 12 | free | — | — | — | — |
| 13 | free | — | — | — | — |
| 14 | free | — | — | — | — |
| 15 | free | — | — | — | — |

## Session A intent

Session A takes **Tasks 8 (News feed + sanitizer) and 9 (Risk validation) only**. Session B is free to take any task marked `free`.

Dependencies session A cares about:
- Task 8 has no dependencies on other plan tasks. OK to start.
- Task 9 depends on done Tasks 3 (RiskLimits), 4 (Decision schema), and 5 (Position dataclass). OK to start.

Session B: please claim any of `7, 10-15` you intend to work on by editing this file and committing it before starting.

## How to release a claim

After committing a task, update the row: `claimed` → `done`, fill in the release date and commit SHA, and commit this file update alongside or right after the task commit.

## Conflict protocol

If two sessions both have a row marked `claimed` for the same task at the same time, the session with the **earlier `Claimed` timestamp** keeps the task; the later session deletes its row.
