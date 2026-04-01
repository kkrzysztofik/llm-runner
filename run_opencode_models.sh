#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CONFIGURATION
# ============================================================

# Paths
LLAMA_CPP_ROOT="/home/kmk/src/llama.cpp"
LLAMA_SERVER_BIN_INTEL="$LLAMA_CPP_ROOT/build/bin/llama-server"
LLAMA_SERVER_BIN_NVIDIA="$LLAMA_CPP_ROOT/build_cuda/bin/llama-server"
MODEL_SUMMARY_BALANCED="/home/kmk/models/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-IQ4_XS.gguf"
MODEL_SUMMARY_FAST="/home/kmk/models/unsloth/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf"
MODEL_OMNICODER9B="/home/kmk/models/Tesslate/OmniCoder-9B-GGUF/omnicoder-9b-q6_k.gguf"
MODEL_QWEN35="/home/kmk/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf"
MODEL_QWEN35_BOTH="/home/kmk/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf"

# Network
HOST="127.0.0.1"
SUMMARY_BALANCED_PORT="8080"
SUMMARY_FAST_PORT="8082"
QWEN35_PORT="8081"
OMNICODER9B_PORT="8083"

# Model-specific defaults
SUMMARY_BALANCED_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'

# Server defaults
DEFAULT_N_GPU_LAYERS=99                  # Max GPU layers for fastest inference
DEFAULT_CTX_SIZE_SUMMARY=16144           # 16k with headroom for summary assistants
DEFAULT_CTX_SIZE_QWEN35=262144           # Match NVIDIA qwen35 single-run config
DEFAULT_CTX_SIZE_OMNICODER9B=100000      # Stable high-context coding setup
DEFAULT_CTX_SIZE_BOTH_SUMMARY=16144      # Keep summarizer at 16k in dual-run mode
DEFAULT_CTX_SIZE_BOTH_QWEN35=262144      # Match NVIDIA qwen35 dual-run config
DEFAULT_N_GPU_LAYERS_QWEN35=all
DEFAULT_N_GPU_LAYERS_QWEN35_BOTH=all
DEFAULT_UBATCH_SIZE_SUMMARY_BALANCED=1024
DEFAULT_UBATCH_SIZE_SUMMARY_FAST=512
DEFAULT_UBATCH_SIZE_QWEN35=1024
DEFAULT_UBATCH_SIZE_OMNICODER9B=1024
DEFAULT_UBATCH_SIZE_QWEN35_BOTH=1024
DEFAULT_UBATCH_SIZE_OMNICODER9B_BOTH=1024
DEFAULT_THREADS_SUMMARY_BALANCED=8
DEFAULT_THREADS_SUMMARY_FAST=8
DEFAULT_THREADS_QWEN35=12
DEFAULT_THREADS_OMNICODER9B=8
DEFAULT_THREADS_QWEN35_BOTH=12
DEFAULT_THREADS_OMNICODER9B_BOTH=8
DEFAULT_CACHE_TYPE_SUMMARY_K=q8_0
DEFAULT_CACHE_TYPE_SUMMARY_V=q8_0
DEFAULT_CACHE_TYPE_QWEN35_K=q8_0
DEFAULT_CACHE_TYPE_QWEN35_V=q8_0
DEFAULT_CACHE_TYPE_QWEN35_BOTH_K=q8_0
DEFAULT_CACHE_TYPE_QWEN35_BOTH_V=q8_0
DEFAULT_CACHE_TYPE_OMNICODER9B_K=q8_0
DEFAULT_CACHE_TYPE_OMNICODER9B_V=q8_0
DEFAULT_CACHE_TYPE_OMNICODER9B_BOTH_K=q8_0
DEFAULT_CACHE_TYPE_OMNICODER9B_BOTH_V=q8_0

# ============================================================
# GLOBAL STATE
# ============================================================

COLOR_ENABLED=0
PIDS=()
SHUTTING_DOWN=0

# Cache color codes for efficiency
declare -A COLOR_CACHE=()

# ============================================================
# COLOR UTILITIES
# ============================================================

init_colors() {
  if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    COLOR_ENABLED=1
  fi
}

server_color_code() {
  local server_name="$1"
  
  # Return cached value if available
  if [[ -n "${COLOR_CACHE[$server_name]:-}" ]]; then
    printf '%s' "${COLOR_CACHE[$server_name]}"
    return
  fi
  
  local code
  case "$server_name" in
    summary-balanced) code='1;34' ;;
    summary-fast)     code='1;33' ;;
    omnicoder-9b)     code='1;35' ;;
    qwen35-coding)  code='1;32' ;;
    *)              code='1;36' ;;
  esac
  
  COLOR_CACHE["$server_name"]="$code"
  printf '%s' "$code"
}

# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

validate_port() {
  local port="$1"
  local name="${2:-port}"
  
  if ! [[ "$port" =~ ^[0-9]+$ ]]; then
    echo "error: $name must be a number, got: $port" >&2
    return 1
  fi
  
  if (( port < 1 || port > 65535 )); then
    echo "error: $name must be between 1 and 65535, got: $port" >&2
    return 1
  fi
  
  return 0
}

validate_threads() {
  local threads="$1"
  local name="${2:-threads}"
  
  if ! [[ "$threads" =~ ^[0-9]+$ ]]; then
    echo "error: $name must be a positive integer, got: $threads" >&2
    return 1
  fi
  
  if (( threads < 1 )); then
    echo "error: $name must be greater than 0, got: $threads" >&2
    return 1
  fi
  
  return 0
}

require_model() {
  local model_path="$1"
  if [[ ! -f "$model_path" ]]; then
    echo "error: model not found: $model_path" >&2
    return 1
  fi
  return 0
}

require_executable() {
  local bin_path="$1"
  local name="${2:-binary}"
  if [[ ! -x "$bin_path" ]]; then
    echo "error: $name not found or not executable: $bin_path" >&2
    return 1
  fi
  return 0
}

validate_ports() {
  local port1="$1"
  local port2="$2"
  local name1="${3:-port1}"
  local name2="${4:-port2}"
  
  if [[ "$port1" == "$port2" ]]; then
    echo "error: $name1 and $name2 must be different, got: $port1" >&2
    return 1
  fi
  
  return 0
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

build_server_cmd() {
  local -n cmd_ref="$1"
  local model="$2"
  local alias_name="$3"
  local device="$4"
  local port="$5"
  local ctx_size="$6"
  local ubatch_size="$7"
  local threads="$8"
  local tensor_split="${9:-}"
  local reasoning_mode="${10-auto}"
  local reasoning_format="${11-none}"
  local chat_template_kwargs="${12:-}"
  local reasoning_budget="${13:-}"
  local use_jinja="${14:-false}"
  local cache_type_k="${15:-q8_0}"
  local cache_type_v="${16:-q8_0}"
  local n_gpu_layers="${17:-$DEFAULT_N_GPU_LAYERS}"
  local server_bin="${18:-$LLAMA_SERVER_BIN_INTEL}"
  
  cmd_ref=(
    "$server_bin"
    --model "$model"
    --alias "$alias_name"
  )

  [[ -n "$device" ]] && cmd_ref+=(--device "$device")

  cmd_ref+=(
    --n-gpu-layers "$n_gpu_layers"
    --split-mode layer
  )

  [[ -n "$reasoning_mode" ]] && cmd_ref+=(--reasoning "$reasoning_mode")
  [[ -n "$reasoning_format" ]] && cmd_ref+=(--reasoning-format "$reasoning_format")
  
  [[ -n "$tensor_split" ]] && cmd_ref+=(--tensor-split "$tensor_split")
  [[ -n "$chat_template_kwargs" ]] && cmd_ref+=(--chat-template-kwargs "$chat_template_kwargs")
  [[ -n "$reasoning_budget" ]] && cmd_ref+=(--reasoning-budget "$reasoning_budget")
  [[ "$use_jinja" == "true" ]] && cmd_ref+=(--jinja)
  
  cmd_ref+=(
    --ctx-size "$ctx_size"
    --flash-attn on
    --cache-type-k "$cache_type_k"
    --cache-type-v "$cache_type_v"
    --batch-size 2048
    --ubatch-size "$ubatch_size"
    --threads "$threads"
    --poll 50
    --mmap
    --host "$HOST"
    --port "$port"
    --no-webui
  )
}

prefix_output() {
  local server_name="$1"
  local line
  local timestamp
  local color_code
  
  color_code="$(server_color_code "$server_name")"
  
  while IFS= read -r line || [[ -n "$line" ]]; do
    timestamp="$(date +"%H:%M:%S")"
    if ((COLOR_ENABLED)) && [[ -n "$color_code" ]]; then
      printf '\033[%sm[%s][%s]\033[0m %s\n' "$color_code" "$timestamp" "$server_name" "$line"
    else
      printf '[%s][%s] %s\n' "$timestamp" "$server_name" "$line"
    fi
  done
}

start_server_background() {
  local server_name="$1"
  local -n cmd_ref="$2"
  
  "${cmd_ref[@]}" \
    > >(prefix_output "$server_name") \
    2> >(prefix_output "$server_name" >&2) &
  
  PIDS+=("$!")
}

exec_server() {
  local server_name="$1"
  local -n cmd_ref="$2"
  
  "${cmd_ref[@]}" \
    > >(prefix_output "$server_name") \
    2> >(prefix_output "$server_name" >&2)
}

log_error() {
  local message="$1"
  echo "error: $message" >&2
}

log_warning() {
  local message="$1"
  echo "warning: $message" >&2
}

# ============================================================
# SIGNAL HANDLERS
# ============================================================

cleanup_servers() {
  ((SHUTTING_DOWN)) && return
  SHUTTING_DOWN=1
  
  local running_pids=()
  local pid
  
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      running_pids+=("$pid")
    fi
  done
  
  if [[ ${#running_pids[@]} -eq 0 ]]; then
    return
  fi
  
  log_warning "Sending TERM to ${#running_pids[@]} server(s)..."
  kill -TERM "${running_pids[@]}" 2>/dev/null || log_warning "Failed to send TERM"
  sleep 1
  
  local stubborn_pids=()
  for pid in "${running_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      stubborn_pids+=("$pid")
    fi
  done
  
  if [[ ${#stubborn_pids[@]} -gt 0 ]]; then
    log_warning "Killing ${#stubborn_pids[@]} stubborn server(s)..."
    kill -KILL "${stubborn_pids[@]}" 2>/dev/null || log_warning "Failed to send KILL"
  fi
  
  wait "${running_pids[@]}" 2>/dev/null || log_warning "Wait failed"
}

on_interrupt() {
  cleanup_servers
  exit 130
}

on_terminate() {
  cleanup_servers
  exit 143
}

# ============================================================
# SERVER STARTUP FUNCTIONS
# ============================================================

start_summary_balanced() {
  local port="$1"
  local cmd=()

  require_model "$MODEL_SUMMARY_BALANCED"
  build_server_cmd cmd "$MODEL_SUMMARY_BALANCED" "summary-balanced" "SYCL0" "$port" \
    "$DEFAULT_CTX_SIZE_SUMMARY" "$DEFAULT_UBATCH_SIZE_SUMMARY_BALANCED" "$DEFAULT_THREADS_SUMMARY_BALANCED" \
    "" off deepseek "$SUMMARY_BALANCED_CHAT_TEMPLATE_KWARGS" "" true "$DEFAULT_CACHE_TYPE_SUMMARY_K" "$DEFAULT_CACHE_TYPE_SUMMARY_V"

  exec_server "summary-balanced" cmd
}

start_summary_fast() {
  local port="$1"
  local cmd=()

  require_model "$MODEL_SUMMARY_FAST"
  build_server_cmd cmd "$MODEL_SUMMARY_FAST" "summary-fast" "SYCL0" "$port" \
    "$DEFAULT_CTX_SIZE_SUMMARY" "$DEFAULT_UBATCH_SIZE_SUMMARY_FAST" "$DEFAULT_THREADS_SUMMARY_FAST" \
    "" auto none "" "" false "$DEFAULT_CACHE_TYPE_SUMMARY_K" "$DEFAULT_CACHE_TYPE_SUMMARY_V"

  exec_server "summary-fast" cmd
}

start_omnicoder9b() {
  local port="$1"
  local cmd=()

  require_model "$MODEL_OMNICODER9B"
  build_server_cmd cmd "$MODEL_OMNICODER9B" "omnicoder-9b" "SYCL0" "$port" \
    "$DEFAULT_CTX_SIZE_OMNICODER9B" "$DEFAULT_UBATCH_SIZE_OMNICODER9B" "$DEFAULT_THREADS_OMNICODER9B" \
    "" auto none "" "" false "$DEFAULT_CACHE_TYPE_OMNICODER9B_K" "$DEFAULT_CACHE_TYPE_OMNICODER9B_V"

  exec_server "omnicoder-9b" cmd
}

start_qwen35() {
  local port="$1"
  local cmd=()
  
  require_model "$MODEL_QWEN35"
  require_executable "$LLAMA_SERVER_BIN_NVIDIA" "NVIDIA llama-server"
  build_server_cmd cmd "$MODEL_QWEN35" "qwen35-coding" "" "$port" \
    "$DEFAULT_CTX_SIZE_QWEN35" "$DEFAULT_UBATCH_SIZE_QWEN35" "$DEFAULT_THREADS_QWEN35" \
    "" "" "" "" "" false \
    "$DEFAULT_CACHE_TYPE_QWEN35_K" "$DEFAULT_CACHE_TYPE_QWEN35_V" "$DEFAULT_N_GPU_LAYERS_QWEN35" "$LLAMA_SERVER_BIN_NVIDIA"
  
  exec_server "qwen35-coding" cmd
}

start_both_qwen35() {
  local summary_balanced_port="$1"
  local qwen35_port="$2"
  local summary_balanced_cmd=()
  local qwen35_cmd=()

  require_model "$MODEL_SUMMARY_BALANCED"
  require_model "$MODEL_QWEN35_BOTH"
  require_executable "$LLAMA_SERVER_BIN_NVIDIA" "NVIDIA llama-server"

  build_server_cmd summary_balanced_cmd "$MODEL_SUMMARY_BALANCED" "summary-balanced" "SYCL0" "$summary_balanced_port" \
    "$DEFAULT_CTX_SIZE_BOTH_SUMMARY" "$DEFAULT_UBATCH_SIZE_SUMMARY_BALANCED" "$DEFAULT_THREADS_SUMMARY_BALANCED" \
    "" off deepseek "$SUMMARY_BALANCED_CHAT_TEMPLATE_KWARGS" "" true "$DEFAULT_CACHE_TYPE_SUMMARY_K" "$DEFAULT_CACHE_TYPE_SUMMARY_V"
  
  build_server_cmd qwen35_cmd "$MODEL_QWEN35_BOTH" "qwen35-coding" "" "$qwen35_port" \
    "$DEFAULT_CTX_SIZE_BOTH_QWEN35" "$DEFAULT_UBATCH_SIZE_QWEN35_BOTH" "$DEFAULT_THREADS_QWEN35_BOTH" \
    "" "" "" "" "" false \
    "$DEFAULT_CACHE_TYPE_QWEN35_BOTH_K" "$DEFAULT_CACHE_TYPE_QWEN35_BOTH_V" "$DEFAULT_N_GPU_LAYERS_QWEN35_BOTH" "$LLAMA_SERVER_BIN_NVIDIA"
  
  # Setup signal handlers BEFORE launching servers
  trap on_interrupt INT
  trap on_terminate TERM
  trap cleanup_servers EXIT
  
  # Launch servers in background
  start_server_background "summary-balanced" summary_balanced_cmd
  start_server_background "qwen35-coding" qwen35_cmd

  echo "summary-balanced: http://$HOST:$summary_balanced_port/v1"
  echo "qwen35-coding: http://$HOST:$qwen35_port/v1"
  
  wait -n "${PIDS[@]}"
}

start_both() {
  local summary_balanced_port="$1"
  local omnicoder9b_port="$2"
  local summary_balanced_cmd=()
  local omnicoder9b_cmd=()

  require_model "$MODEL_SUMMARY_BALANCED"
  require_model "$MODEL_OMNICODER9B"

  build_server_cmd summary_balanced_cmd "$MODEL_SUMMARY_BALANCED" "summary-balanced" "SYCL0" "$summary_balanced_port" \
    "$DEFAULT_CTX_SIZE_SUMMARY" "$DEFAULT_UBATCH_SIZE_SUMMARY_BALANCED" "$DEFAULT_THREADS_SUMMARY_BALANCED" \
    "" off deepseek "$SUMMARY_BALANCED_CHAT_TEMPLATE_KWARGS" "" true "$DEFAULT_CACHE_TYPE_SUMMARY_K" "$DEFAULT_CACHE_TYPE_SUMMARY_V"

  build_server_cmd omnicoder9b_cmd "$MODEL_OMNICODER9B" "omnicoder-9b" "SYCL1" "$omnicoder9b_port" \
    "$DEFAULT_CTX_SIZE_OMNICODER9B" "$DEFAULT_UBATCH_SIZE_OMNICODER9B_BOTH" "$DEFAULT_THREADS_OMNICODER9B_BOTH" \
    "" auto none "" "" false "$DEFAULT_CACHE_TYPE_OMNICODER9B_BOTH_K" "$DEFAULT_CACHE_TYPE_OMNICODER9B_BOTH_V"

  trap on_interrupt INT
  trap on_terminate TERM
  trap cleanup_servers EXIT

  start_server_background "summary-balanced" summary_balanced_cmd
  start_server_background "omnicoder-9b" omnicoder9b_cmd

  echo "summary-balanced: http://$HOST:$summary_balanced_port/v1"
  echo "omnicoder-9b: http://$HOST:$omnicoder9b_port/v1"

  wait -n "${PIDS[@]}"
}

# ============================================================
# HELP
# ============================================================

usage() {
  cat <<'EOF'
Usage:
  run_opencode_models.sh summary-balanced [port]
  run_opencode_models.sh summary-fast [port]
  run_opencode_models.sh qwen35 [port]
  run_opencode_models.sh omnicoder9b [port]
  run_opencode_models.sh both [summary_balanced_port omnicoder9b_port]
  run_opencode_models.sh both-qwen35 [summary_balanced_port qwen35_port]
  run_opencode_models.sh dry-run summary-balanced|summary-fast|qwen35|omnicoder9b|both|both-qwen35 [ports...]

Examples:
  run_opencode_models.sh summary-balanced
  run_opencode_models.sh summary-fast 8082
  run_opencode_models.sh qwen35 8080
  run_opencode_models.sh omnicoder9b 8083
  run_opencode_models.sh both 8080 8083
  run_opencode_models.sh both-qwen35 8080 8081
  run_opencode_models.sh dry-run both
EOF
}

# ============================================================
# MAIN
# ============================================================

check_prereqs() {
  require_executable "$LLAMA_SERVER_BIN_INTEL" "Intel llama-server" || exit 1
}

dry_run() {
  local mode="$1"
  local primary_port="${2:-}"
  local secondary_port="${3:-}"
  local summary_balanced_port="${primary_port:-$SUMMARY_BALANCED_PORT}"
  local summary_fast_port="${primary_port:-$SUMMARY_FAST_PORT}"
  local qwen35_port_single="${primary_port:-$QWEN35_PORT}"
  local qwen35_port_both="${secondary_port:-$QWEN35_PORT}"
  local omnicoder9b_port="${primary_port:-$OMNICODER9B_PORT}"

  echo "=== DRY RUN MODE ==="
  echo "Mode: $mode"
  echo "llama-server (Intel): $LLAMA_SERVER_BIN_INTEL"
  echo "llama-server (NVIDIA): $LLAMA_SERVER_BIN_NVIDIA"
  echo "summary-balanced model: $MODEL_SUMMARY_BALANCED"
  echo "summary-fast model: $MODEL_SUMMARY_FAST"
  echo "omnicoder-9b model: $MODEL_OMNICODER9B"
  echo "qwen35 model: $MODEL_QWEN35"
  echo "qwen35 both model: $MODEL_QWEN35_BOTH"
  echo ""

  case "$mode" in
    summary-balanced|llama32)
      echo "summary-balanced:"
      echo "  Port: $summary_balanced_port"
      echo "  Device: SYCL0"
      echo "  Context: $DEFAULT_CTX_SIZE_SUMMARY"
      echo "  Threads: $DEFAULT_THREADS_SUMMARY_BALANCED"
      echo "  UBatch: $DEFAULT_UBATCH_SIZE_SUMMARY_BALANCED"
      echo "  Reasoning: off"
      echo "  Reasoning Format: deepseek"
      echo "  Jinja: true"
      echo "  Chat Template Kwargs: $SUMMARY_BALANCED_CHAT_TEMPLATE_KWARGS"
      build_server_cmd tmp_cmd "$MODEL_SUMMARY_BALANCED" "summary-balanced" "SYCL0" "$summary_balanced_port" \
        "$DEFAULT_CTX_SIZE_SUMMARY" "$DEFAULT_UBATCH_SIZE_SUMMARY_BALANCED" "$DEFAULT_THREADS_SUMMARY_BALANCED" \
        "" off deepseek "$SUMMARY_BALANCED_CHAT_TEMPLATE_KWARGS" "" true
      echo "  Command: ${tmp_cmd[*]}"
      unset tmp_cmd
      ;;
    summary-fast)
      echo "summary-fast:"
      echo "  Port: $summary_fast_port"
      echo "  Device: SYCL0"
      echo "  Context: $DEFAULT_CTX_SIZE_SUMMARY"
      echo "  Threads: $DEFAULT_THREADS_SUMMARY_FAST"
      echo "  UBatch: $DEFAULT_UBATCH_SIZE_SUMMARY_FAST"
      build_server_cmd tmp_cmd "$MODEL_SUMMARY_FAST" "summary-fast" "SYCL0" "$summary_fast_port" \
        "$DEFAULT_CTX_SIZE_SUMMARY" "$DEFAULT_UBATCH_SIZE_SUMMARY_FAST" "$DEFAULT_THREADS_SUMMARY_FAST"
      echo "  Command: ${tmp_cmd[*]}"
      unset tmp_cmd
      ;;
    qwen35)
      echo "qwen35:"
      echo "  Port: $qwen35_port_single"
      echo "  Device: NVIDIA (CUDA)"
      echo "  Context: $DEFAULT_CTX_SIZE_QWEN35"
      echo "  Threads: $DEFAULT_THREADS_QWEN35"
      echo "  UBatch: $DEFAULT_UBATCH_SIZE_QWEN35"
      echo "  KV cache: $DEFAULT_CACHE_TYPE_QWEN35_K/$DEFAULT_CACHE_TYPE_QWEN35_V"
      echo "  n-gpu-layers: $DEFAULT_N_GPU_LAYERS_QWEN35"
      build_server_cmd tmp_cmd "$MODEL_QWEN35" "qwen35-coding" "" "$qwen35_port_single" \
        "$DEFAULT_CTX_SIZE_QWEN35" "$DEFAULT_UBATCH_SIZE_QWEN35" "$DEFAULT_THREADS_QWEN35" \
        "" "" "" "" "" false \
        "$DEFAULT_CACHE_TYPE_QWEN35_K" "$DEFAULT_CACHE_TYPE_QWEN35_V" "$DEFAULT_N_GPU_LAYERS_QWEN35" "$LLAMA_SERVER_BIN_NVIDIA"
      echo "  Command: ${tmp_cmd[*]}"
      unset tmp_cmd
      ;;
    omnicoder9b)
      echo "omnicoder-9b:"
      echo "  Port: $omnicoder9b_port"
      echo "  Device: SYCL0"
      echo "  Context: $DEFAULT_CTX_SIZE_OMNICODER9B"
      echo "  Threads: $DEFAULT_THREADS_OMNICODER9B"
      echo "  UBatch: $DEFAULT_UBATCH_SIZE_OMNICODER9B"
      build_server_cmd tmp_cmd "$MODEL_OMNICODER9B" "omnicoder-9b" "SYCL0" "$omnicoder9b_port" \
        "$DEFAULT_CTX_SIZE_OMNICODER9B" "$DEFAULT_UBATCH_SIZE_OMNICODER9B" "$DEFAULT_THREADS_OMNICODER9B"
      echo "  Command: ${tmp_cmd[*]}"
      unset tmp_cmd
      ;;
    both)
      echo "summary-balanced:"
      echo "  Port: $summary_balanced_port"
      echo "  Device: SYCL0"
      echo "  Context: $DEFAULT_CTX_SIZE_SUMMARY"
      echo "  Threads: $DEFAULT_THREADS_SUMMARY_BALANCED"
      echo "  UBatch: $DEFAULT_UBATCH_SIZE_SUMMARY_BALANCED"
      echo "  KV cache: $DEFAULT_CACHE_TYPE_SUMMARY_K/$DEFAULT_CACHE_TYPE_SUMMARY_V"
      echo "  Reasoning: off"
      echo "  Reasoning Format: deepseek"
      echo "  Jinja: true"
      echo "  Chat Template Kwargs: $SUMMARY_BALANCED_CHAT_TEMPLATE_KWARGS"
      build_server_cmd tmp_cmd "$MODEL_SUMMARY_BALANCED" "summary-balanced" "SYCL0" "$summary_balanced_port" \
        "$DEFAULT_CTX_SIZE_SUMMARY" "$DEFAULT_UBATCH_SIZE_SUMMARY_BALANCED" "$DEFAULT_THREADS_SUMMARY_BALANCED" \
        "" off deepseek "$SUMMARY_BALANCED_CHAT_TEMPLATE_KWARGS" "" true "$DEFAULT_CACHE_TYPE_SUMMARY_K" "$DEFAULT_CACHE_TYPE_SUMMARY_V"
      echo "  Command: ${tmp_cmd[*]}"
      unset tmp_cmd
      echo ""
      echo "omnicoder-9b:"
      echo "  Port: $qwen35_port_both"
      echo "  Device: SYCL1"
      echo "  Context: $DEFAULT_CTX_SIZE_OMNICODER9B"
      echo "  Threads: $DEFAULT_THREADS_OMNICODER9B_BOTH"
      echo "  UBatch: $DEFAULT_UBATCH_SIZE_OMNICODER9B_BOTH"
      echo "  KV cache: $DEFAULT_CACHE_TYPE_OMNICODER9B_BOTH_K/$DEFAULT_CACHE_TYPE_OMNICODER9B_BOTH_V"
      build_server_cmd tmp_cmd "$MODEL_OMNICODER9B" "omnicoder-9b" "SYCL1" "$qwen35_port_both" \
        "$DEFAULT_CTX_SIZE_OMNICODER9B" "$DEFAULT_UBATCH_SIZE_OMNICODER9B_BOTH" "$DEFAULT_THREADS_OMNICODER9B_BOTH" \
        "" auto none "" "" false "$DEFAULT_CACHE_TYPE_OMNICODER9B_BOTH_K" "$DEFAULT_CACHE_TYPE_OMNICODER9B_BOTH_V"
      echo "  Command: ${tmp_cmd[*]}"
      unset tmp_cmd
      ;;
    both-qwen35)
      echo "summary-balanced:"
      echo "  Port: $summary_balanced_port"
      echo "  Device: SYCL0"
      echo "  Context: $DEFAULT_CTX_SIZE_BOTH_SUMMARY"
      echo "  Threads: $DEFAULT_THREADS_SUMMARY_BALANCED"
      echo "  UBatch: $DEFAULT_UBATCH_SIZE_SUMMARY_BALANCED"
      echo "  KV cache: $DEFAULT_CACHE_TYPE_SUMMARY_K/$DEFAULT_CACHE_TYPE_SUMMARY_V"
      echo "  Reasoning: off"
      echo "  Reasoning Format: deepseek"
      echo "  Jinja: true"
      echo "  Chat Template Kwargs: $SUMMARY_BALANCED_CHAT_TEMPLATE_KWARGS"
      build_server_cmd tmp_cmd "$MODEL_SUMMARY_BALANCED" "summary-balanced" "SYCL0" "$summary_balanced_port" \
        "$DEFAULT_CTX_SIZE_BOTH_SUMMARY" "$DEFAULT_UBATCH_SIZE_SUMMARY_BALANCED" "$DEFAULT_THREADS_SUMMARY_BALANCED" \
        "" off deepseek "$SUMMARY_BALANCED_CHAT_TEMPLATE_KWARGS" "" true "$DEFAULT_CACHE_TYPE_SUMMARY_K" "$DEFAULT_CACHE_TYPE_SUMMARY_V"
      echo "  Command: ${tmp_cmd[*]}"
      unset tmp_cmd
      echo ""
      echo "qwen35:"
      echo "  Port: $qwen35_port_both"
      echo "  Device: NVIDIA (CUDA)"
      echo "  Context: $DEFAULT_CTX_SIZE_BOTH_QWEN35"
      echo "  Threads: $DEFAULT_THREADS_QWEN35_BOTH"
      echo "  UBatch: $DEFAULT_UBATCH_SIZE_QWEN35_BOTH"
      echo "  KV cache: $DEFAULT_CACHE_TYPE_QWEN35_BOTH_K/$DEFAULT_CACHE_TYPE_QWEN35_BOTH_V"
      echo "  n-gpu-layers: $DEFAULT_N_GPU_LAYERS_QWEN35_BOTH"
      build_server_cmd tmp_cmd "$MODEL_QWEN35_BOTH" "qwen35-coding" "" "$qwen35_port_both" \
        "$DEFAULT_CTX_SIZE_BOTH_QWEN35" "$DEFAULT_UBATCH_SIZE_QWEN35_BOTH" "$DEFAULT_THREADS_QWEN35_BOTH" \
        "" "" "" "" "" false \
        "$DEFAULT_CACHE_TYPE_QWEN35_BOTH_K" "$DEFAULT_CACHE_TYPE_QWEN35_BOTH_V" "$DEFAULT_N_GPU_LAYERS_QWEN35_BOTH" "$LLAMA_SERVER_BIN_NVIDIA"
      echo "  Command: ${tmp_cmd[*]}"
      unset tmp_cmd
      ;;
  esac
}

# Initialize global state
init_colors
check_prereqs
export ZES_ENABLE_SYSMAN=1

# Parse and validate arguments
mode="${1:-}"
if [[ "$mode" == "dry-run" ]]; then
  mode="${2:-}"
  if [[ -z "$mode" ]]; then
    echo "error: dry-run requires a mode argument (summary-balanced|summary-fast|qwen35|omnicoder9b|both|both-qwen35)" >&2
    usage
    exit 1
  fi
  dry_run "$mode" "${3:-}" "${4:-}"
  exit 0
fi
case "$mode" in
  summary-balanced|llama32)
    port="${2:-$SUMMARY_BALANCED_PORT}"
    if ! validate_port "$port" "summary-balanced port"; then
      usage
      exit 1
    fi
    echo "Starting summary-balanced at http://$HOST:$port/v1"
    start_summary_balanced "$port"
    ;;
  summary-fast)
    port="${2:-$SUMMARY_FAST_PORT}"
    if ! validate_port "$port" "summary-fast port"; then
      usage
      exit 1
    fi
    echo "Starting summary-fast at http://$HOST:$port/v1"
    start_summary_fast "$port"
    ;;
  qwen35)
    port="${2:-$QWEN35_PORT}"
    if ! validate_port "$port" "qwen35 port"; then
      usage
      exit 1
    fi
    echo "Starting qwen35-coding at http://$HOST:$port/v1 (NVIDIA CUDA)"
    start_qwen35 "$port"
    ;;
  omnicoder9b)
    port="${2:-$OMNICODER9B_PORT}"
    if ! validate_port "$port" "omnicoder9b port"; then
      usage
      exit 1
    fi
    echo "Starting omnicoder-9b at http://$HOST:$port/v1"
    start_omnicoder9b "$port"
    ;;
  both)
    port32="${2:-$SUMMARY_BALANCED_PORT}"
    port35="${3:-$OMNICODER9B_PORT}"

    if ! validate_port "$port32" "summary-balanced port"; then
      usage
      exit 1
    fi
    if ! validate_port "$port35" "omnicoder9b port"; then
      usage
      exit 1
    fi
    if ! validate_ports "$port32" "$port35" "summary-balanced port" "omnicoder9b port"; then
      usage
      exit 1
    fi

    start_both "$port32" "$port35"
    ;;
  both-qwen35)
    port32="${2:-$SUMMARY_BALANCED_PORT}"
    port35="${3:-$QWEN35_PORT}"

    if ! validate_port "$port32" "summary-balanced port"; then
      usage
      exit 1
    fi
    if ! validate_port "$port35" "qwen35 port"; then
      usage
      exit 1
    fi
    if ! validate_ports "$port32" "$port35" "summary-balanced port" "qwen35 port"; then
      usage
      exit 1
    fi

    start_both_qwen35 "$port32" "$port35"
    ;;
  *)
    usage
    exit 1
    ;;
esac
