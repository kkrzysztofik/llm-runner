#!/usr/bin/env bash
set -euo pipefail

# Benchmarks Qwen 3.6 MTP vs non-MTP using llama-server + /completion.
# Uses MTP settings from llama.cpp PR #22673:
#   --spec-type mtp --spec-draft-n-max 3
#
# Outputs:
# - raw_runs.csv      : per-run prompt/decode tok/s
# - delta_summary.csv : direct delta % (MTP vs non-MTP)

LLAMA_SERVER_BIN="/home/kmk/src/llama.cpp/build_cuda/bin/llama-server"
OUT_DIR="${1:-/tmp/llama-bench-mtp-vs-nonmtp}"
mkdir -p "$OUT_DIR"

RAW_CSV="$OUT_DIR/raw_runs.csv"
SUMMARY_CSV="$OUT_DIR/delta_summary.csv"

HOST="127.0.0.1"
BASE_PORT="18080"
REPETITIONS="${REPETITIONS:-3}"
WARMUP="${WARMUP:-1}"
CTX_SIZE="${CTX_SIZE:-10000}"
PROMPT_REPEAT="${PROMPT_REPEAT:-60}"
RUN_PID=""

cleanup_pid() {
  if [[ -n "${RUN_PID:-}" ]]; then
    kill "$RUN_PID" >/dev/null 2>&1 || true
    wait "$RUN_PID" 2>/dev/null || true
    RUN_PID=""
  fi
}

trap cleanup_pid EXIT

PROMPT_FILE="${OUT_DIR}/prompt.txt"
python3 - "$PROMPT_REPEAT" <<'PY' > "$PROMPT_FILE"
import sys

repeat = int(sys.argv[1])

text = """You are a meticulous coding assistant.
Review the following pseudocode and explain complexity, edge cases, and possible bugs.

def transform(data):
    out = []
    for i in range(len(data)):
        if i % 3 == 0:
            out.append(data[i] * 2)
        elif i % 5 == 0:
            out.append(data[i] - 1)
        else:
            out.append(data[i])
    return out

Now propose tests for correctness, performance, and adversarial inputs.
"""

# Repeat to produce a stable prefill chunk
print((text + "\n") * repeat)
PY

NON_MTP_27B="/home/kmk/models/unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf"
MTP_27B="/home/kmk/models/Qwen3.6-27B-MTP-unsloth-pattern-imatrix.gguf"

NON_MTP_35B="/home/kmk/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
MTP_35B="/home/kmk/models/Qwen3.6-35BA3B-MTP-unsloth-pattern-imatrix.gguf"

for f in "$NON_MTP_27B" "$MTP_27B" "$NON_MTP_35B" "$MTP_35B"; do
  if [[ ! -f "$f" ]]; then
    echo "error: model not found: $f" >&2
    exit 1
  fi
done

if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
  echo "error: llama-server not found or not executable: $LLAMA_SERVER_BIN" >&2
  exit 1
fi

echo "size,kind,run,prompt_tok_s,decode_tok_s,spec_type,spec_draft_n_max" > "$RAW_CSV"

start_server() {
  local model="$1"
  local port="$2"
  local spec_type="$3"
  local draft_max="$4"

  local cmd=(
    "$LLAMA_SERVER_BIN"
    --model "$model"
    --host "$HOST"
    --port "$port"
    --ctx-size "$CTX_SIZE"
    --n-predict 256
    --flash-attn on
    --cache-type-k q8_0
    --cache-type-v q8_0
    --batch-size 2048
    --ubatch-size 1024
    --threads 12
    --poll 0
    --n-gpu-layers 99
    --split-mode layer
    --parallel 1
    --mmap
    --no-webui
    --no-cache-prompt
    --chat-template-kwargs '{"preserve_thinking":true}'
  )

  if [[ "$spec_type" != "none" ]]; then
    cmd+=(--spec-type "$spec_type")
  fi
  if [[ "$draft_max" != "0" ]]; then
    cmd+=(--spec-draft-n-max "$draft_max")
  fi

  "${cmd[@]}" > "$OUT_DIR/server_${port}.log" 2>&1 &
  echo $!
}

wait_server_ready() {
  local port="$1"
  local attempts=90
  local i
  for ((i=0; i<attempts; i++)); do
    if curl -sS "http://${HOST}:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "error: server did not become ready on port $port" >&2
  return 1
}

run_one_request() {
  local port="$1"
  local out_json="$2"

  python3 - "$PROMPT_FILE" > "$OUT_DIR/request_${port}.json" <<'PY'
import json
import sys

prompt = open(sys.argv[1], encoding="utf-8").read()
payload = {
    "prompt": prompt,
    "n_predict": 256,
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "min_p": 0.0,
    "cache_prompt": False,
    "ignore_eos": True,
}
print(json.dumps(payload))
PY

  local attempts=120
  local i
  for ((i=0; i<attempts; i++)); do
    if ! curl -sS "http://${HOST}:${port}/completion" \
      -H 'Content-Type: application/json' \
      --data-binary "@$OUT_DIR/request_${port}.json" > "$out_json"; then
      sleep 1
      continue
    fi

    # During load/warmup server can return 503 "Loading model".
    # Retry until we get a response with valid timings.
    if python3 - "$out_json" <<'PY'
import json
import sys

obj = json.load(open(sys.argv[1], encoding="utf-8"))
timings = obj.get("timings", {})
tokens_predicted = int(obj.get("tokens_predicted") or 0)
predicted_per_second = timings.get("predicted_per_second")
ok = (
    isinstance(timings, dict)
    and "prompt_per_second" in timings
    and "predicted_per_second" in timings
    and predicted_per_second is not None
    and float(predicted_per_second) < 999999.0
    and tokens_predicted > 1
)
raise SystemExit(0 if ok else 1)
PY
    then
      return 0
    fi

    sleep 1
  done

  echo "error: completion did not return timings after retries: $out_json" >&2
  return 1
}

record_run() {
  local size="$1"
  local kind="$2"
  local run_idx="$3"
  local spec_type="$4"
  local draft_max="$5"
  local json_file="$6"

  python3 - "$RAW_CSV" "$size" "$kind" "$run_idx" "$spec_type" "$draft_max" "$json_file" <<'PY'
import csv
import json
import sys

raw_csv, size, kind, run_idx, spec_type, draft_max, json_file = sys.argv[1:]
data = json.load(open(json_file, encoding="utf-8"))

timings = data.get("timings", {})
pp = timings.get("prompt_per_second")
tg = timings.get("predicted_per_second")

if pp is None or tg is None:
    raise SystemExit(f"missing timings in response: {json_file}")

with open(raw_csv, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        size,
        kind,
        run_idx,
        f"{float(pp):.6f}",
        f"{float(tg):.6f}",
        spec_type,
        draft_max,
    ])
PY
}

run_case() {
  local size="$1"
  local kind="$2"
  local model="$3"
  local spec_type="$4"
  local draft_max="$5"
  local port="$6"

  local pid
  pid="$(start_server "$model" "$port" "$spec_type" "$draft_max")"
  RUN_PID="$pid"

  wait_server_ready "$port"

  local i out
  for ((i=1; i<=WARMUP; i++)); do
    out="$OUT_DIR/${size}_${kind}_warmup_${i}.json"
    run_one_request "$port" "$out"
  done

  for ((i=1; i<=REPETITIONS; i++)); do
    out="$OUT_DIR/${size}_${kind}_run_${i}.json"
    run_one_request "$port" "$out"
    record_run "$size" "$kind" "$i" "$spec_type" "$draft_max" "$out"
  done

  cleanup_pid
}

echo "Running benchmarks (server mode, MTP-aware)..."

# PR #22673: best steady-state tg reported around spec_draft_n_max=3
run_case "27b" "nonmtp" "$NON_MTP_27B" "none" "0" "$BASE_PORT"
run_case "27b" "mtp"    "$MTP_27B"     "mtp"  "3" "$((BASE_PORT+1))"

run_case "35b" "nonmtp" "$NON_MTP_35B" "none" "0" "$((BASE_PORT+2))"
run_case "35b" "mtp"    "$MTP_35B"     "mtp"  "3" "$((BASE_PORT+3))"

python3 - "$RAW_CSV" "$SUMMARY_CSV" <<'PY'
import csv
import statistics
import sys

raw_csv, summary_csv = sys.argv[1:]

groups = {}
with open(raw_csv, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        key = (row["size"], row["kind"])
        groups.setdefault(key, {"pp": [], "tg": []})
        groups[key]["pp"].append(float(row["prompt_tok_s"]))
        groups[key]["tg"].append(float(row["decode_tok_s"]))

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "size",
        "nonmtp_prompt_tok_s",
        "mtp_prompt_tok_s",
        "prompt_delta_pct_mtp_vs_nonmtp",
        "nonmtp_decode_tok_s",
        "mtp_decode_tok_s",
        "decode_delta_pct_mtp_vs_nonmtp",
    ])

    for size in sorted({k[0] for k in groups.keys()}):
        non = groups[(size, "nonmtp")]
        mtp = groups[(size, "mtp")]

        non_pp = statistics.mean(non["pp"])
        mtp_pp = statistics.mean(mtp["pp"])
        non_tg = statistics.mean(non["tg"])
        mtp_tg = statistics.mean(mtp["tg"])

        pp_delta = ((mtp_pp - non_pp) / non_pp) * 100.0
        tg_delta = ((mtp_tg - non_tg) / non_tg) * 100.0

        w.writerow([
            size,
            f"{non_pp:.4f}",
            f"{mtp_pp:.4f}",
            f"{pp_delta:+.2f}",
            f"{non_tg:.4f}",
            f"{mtp_tg:.4f}",
            f"{tg_delta:+.2f}",
        ])

print("\nDelta summary (MTP vs non-MTP):")
with open(summary_csv, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        print(
            f"- {row['size']}: "
            f"prompt {row['nonmtp_prompt_tok_s']} -> {row['mtp_prompt_tok_s']} tok/s "
            f"({row['prompt_delta_pct_mtp_vs_nonmtp']}%), "
            f"decode {row['nonmtp_decode_tok_s']} -> {row['mtp_decode_tok_s']} tok/s "
            f"({row['decode_delta_pct_mtp_vs_nonmtp']}%)"
        )
PY

echo
echo "Done."
echo "Raw runs:      $RAW_CSV"
echo "Delta summary: $SUMMARY_CSV"
echo "Logs:          $OUT_DIR/server_*.log"
