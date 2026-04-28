# Research: Superseded TUI Key Handling Design

**Date**: 2026-04-23  
**Feature**: M4 — Operational Hardening and Smoke Verification  
**Status**: Superseded by the Textual TUI migration

---

## Current Decision

The TUI now uses Textual for terminal input, rendering, resize events, and key bindings.
Hardware and VRAM acknowledgement prompts are routed through Textual key handlers into the
existing controller state machine (`handle_keypress` and `_process_keypresses`). The controller
keeps prompt logic testable without owning raw terminal mode.

## Consequences

- Textual owns cross-platform keyboard handling and terminal restoration.
- There is no manual stdin polling thread in the active TUI implementation.
- There is no raw-terminal-mode helper in the active TUI implementation.
- Tests should simulate prompt choices via controller key dispatch or Textual `run_test()`.
- Blocking subprocess and log I/O must stay off the Textual app thread.
- UI updates should flow through Textual widgets, app messages, or controller snapshots.

## Retained Requirements

- `y` continues risky or nonstandard hardware launches.
- `n`, `q`, Escape, or Ctrl+C aborts safely.
- Enter without a valid choice keeps the prompt active.
- Timeout defaults to abort.
- Cleanup still terminates child processes on TUI exit.
