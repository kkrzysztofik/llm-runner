# llama_manager Simplification — Overview

**Goal**: Reduce `llama_manager` from ~84 files / ~16,135 LOC to ~12,000 LOC.

**Current state**: 28 top-level modules + 7 subpackages, 62-field `Config` god dataclass,
982-line `ServerManager` god class, 3 redaction implementations, 4 port validators,
5 overlapping error types, 216 root `__init__.py` re-exports (~140 dead).

## Revision Notes

- Original Task 1 (circular import `errors.py` ↔ `server.py`) was a **false positive** —
  exploration confirmed the chain is acyclic: `enums.py` → `errors.py` → `server.py`.

## Phase Summary

| Phase | Tasks | Est. Hours | Dependencies |
|-------|-------|------------|--------------|
| 1 — Quick wins | Tasks 1, 2, 4 | ~5h | None |
| 2 — API hygiene | Tasks 3, 5, 12 | ~4h | None |
| 3 — Error + Config | Tasks 6, 7 | ~7h | None |
| 4 — Structural | Tasks 8, 9, 10 | ~17h | Tasks 1, 5, 6, 7 |
| 5 — Cleanup | Tasks 11, 13 | ~6h | None |
| **Total** | **13 tasks** | **~39h** | |

## File Index

- `phase-1-quick-wins.md` — Redaction, port validation, metadata/__init__
- `phase-2-api-hygiene.md` — Slot ID normalization, private exports, filename sanitization
- `phase-3-error-config.md` — Error type consolidation, spec decode extraction
- `phase-4-structural.md` — Config split, ServerManager split, root __init__ pruning
- `phase-5-cleanup.md` — level_zero split, profile store merge

## Task Quick Reference

| # | Task | Priority | Effort | Files |
|---|------|----------|--------|-------|
| 1 | Consolidate redaction | Critical | ~2h | 4 |
| 2 | Consolidate port validation | High | ~2h | 4 |
| 3 | Slot ID normalization | Medium | ~1h | 2 |
| 4 | Move metadata/__init__ logic | Low | ~30min | 1 |
| 5 | Remove private exports | Medium | ~2h | 3+tests |
| 6 | Consolidate error types | High | ~4h | ~10 |
| 7 | Extract spec decode | High | ~3h | ~6 |
| 8 | Split Config dataclass | High | ~6h | ~15 |
| 9 | Split ServerManager | Critical | ~8h | ~8 |
| 10 | Prune root __init__ | High | ~3h | 1+tests |
| 11 | Split level_zero.py | Medium | ~4h | 1→4 |
| 12 | Filename sanitization | Low | ~1h | 2 |
| 13 | Merge profile stores | Low | ~2h | 3 |
