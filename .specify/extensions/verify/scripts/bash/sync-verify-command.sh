#!/usr/bin/env bash
set -euo pipefail

CANONICAL_FILE=".specify/extensions/verify/commands/verify.md"
GENERATED_FILE=".opencode/command/speckit.verify.run.md"
GENERATED_NOTE="<!-- GENERATED FILE: Do not edit directly. Source: .specify/extensions/verify/commands/verify.md -->"

if [[ ! -f "$CANONICAL_FILE" ]]; then
  printf 'error: canonical file not found: %s\n' "$CANONICAL_FILE" >&2
  exit 1
fi

tmp_file="$(mktemp)"
cleanup() {
  rm -f "$tmp_file"
}
trap cleanup EXIT

python3 - "$CANONICAL_FILE" "$tmp_file" "$GENERATED_NOTE" <<'PY'
from pathlib import Path
import sys

canonical_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
generated_note = sys.argv[3]

canonical_text = canonical_path.read_text(encoding="utf-8")

frontmatter_end = canonical_text.find("\n---\n", 4)
if frontmatter_end == -1:
    raise SystemExit("error: canonical file is missing frontmatter terminator")

insert_pos = frontmatter_end + len("\n---\n")
generated_text = (
    canonical_text[:insert_pos]
    + "\n"
    + generated_note
    + "\n"
    + canonical_text[insert_pos:]
)

output_path.write_text(generated_text, encoding="utf-8")
PY

if [[ "${1:-}" == "--check" ]]; then
  if [[ ! -f "$GENERATED_FILE" ]]; then
    printf 'error: generated file missing: %s\n' "$GENERATED_FILE" >&2
    exit 1
  fi

  if ! cmp -s "$tmp_file" "$GENERATED_FILE"; then
    printf 'error: verify command docs are out of sync\n' >&2
    printf 'run: .specify/extensions/verify/scripts/bash/sync-verify-command.sh\n' >&2
    exit 1
  fi

  printf 'verify command docs are in sync\n'
  exit 0
fi

cp "$tmp_file" "$GENERATED_FILE"
printf 'synced %s -> %s\n' "$CANONICAL_FILE" "$GENERATED_FILE"
