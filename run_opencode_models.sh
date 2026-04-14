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
MODEL_SUMMARY_FAST="/home/kmk/models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf"
MODEL_QWEN35="/home/kmk/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf"
MODEL_QWEN35_BOTH="/home/kmk/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf"
MODEL_GEMMA4_E4B="/home/kmk/models/unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-UD-Q6_K_XL.gguf"
MODEL_GEMMA4_E4B_MMPROJ="/home/kmk/models/unsloth/gemma-4-E4B-it-GGUF/mmproj-BF16.gguf"
MODEL_GEMMA4_27B="/home/kmk/models/unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
MODEL_GEMMA4_27B_BOTH="$MODEL_GEMMA4_27B"
# Gemma 4 31B IT — IQ4_XS; max stable ctx ~82176 on RTX 3090 (f16 KV, flash-attn, single slot)
MODEL_GEMMA4_31B="/home/kmk/models/unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-IQ4_XS.gguf"

# Network
HOST="127.0.0.1"
SUMMARY_BALANCED_PORT="8080"
SUMMARY_FAST_PORT="8082"
QWEN35_PORT="8081"
GEMMA4_E4B_PORT="$SUMMARY_BALANCED_PORT"
GEMMA4_27B_PORT="$QWEN35_PORT"
GEMMA4_31B_PORT="8085"

# Model-specific defaults
SUMMARY_BALANCED_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'
GEMMA4_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'

# Server defaults
DEFAULT_N_GPU_LAYERS=99                  # Max GPU layers for fastest inference
DEFAULT_CTX_SIZE_SUMMARY=262144           # 256k with headroom for summary assistants
DEFAULT_CTX_SIZE_QWEN35=262144           # Match NVIDIA qwen35 single-run config
DEFAULT_CTX_SIZE_GEMMA4_E4B=131072       # Full 128k context for Gemma 4 E4B
# 27B Q4_K_XL: validated llama-completion warmup + gen at 262144 on RTX 3090 (f16 KV, -np 1)
DEFAULT_CTX_SIZE_GEMMA4_27B=262144
# 31B IQ4_XS: max stable ctx with default batch/ubatch (see probe notes in script header paths)
DEFAULT_CTX_SIZE_GEMMA4_31B=82176
DEFAULT_CTX_SIZE_BOTH_SUMMARY=262144      # Keep summarizer at 256k in dual-run mode
DEFAULT_CTX_SIZE_BOTH_QWEN35=262144      # Match NVIDIA qwen35 dual-run config
DEFAULT_CTX_SIZE_BOTH_GEMMA4_27B=262144   # Same as single-GPU 27B when paired with E4B on Intel
DEFAULT_N_GPU_LAYERS_QWEN35=all
DEFAULT_N_GPU_LAYERS_QWEN35_BOTH=all
DEFAULT_N_GPU_LAYERS_GEMMA4_27B=99
DEFAULT_N_GPU_LAYERS_GEMMA4_27B_BOTH=99
DEFAULT_N_GPU_LAYERS_GEMMA4_31B=99
DEFAULT_UBATCH_SIZE_SUMMARY_BALANCED=1024
DEFAULT_UBATCH_SIZE_SUMMARY_FAST=512
DEFAULT_UBATCH_SIZE_QWEN35=1024
DEFAULT_UBATCH_SIZE_GEMMA4_E4B=512
DEFAULT_UBATCH_SIZE_GEMMA4_27B=1024
DEFAULT_UBATCH_SIZE_GEMMA4_31B=512
DEFAULT_UBATCH_SIZE_QWEN35_BOTH=1024
DEFAULT_UBATCH_SIZE_GEMMA4_27B_BOTH=1024
DEFAULT_THREADS_SUMMARY_BALANCED=8
DEFAULT_THREADS_SUMMARY_FAST=8
DEFAULT_THREADS_QWEN35=12
DEFAULT_THREADS_GEMMA4_E4B=12
DEFAULT_THREADS_BATCH_GEMMA4_E4B=24
DEFAULT_THREADS_GEMMA4_27B=24
DEFAULT_THREADS_GEMMA4_31B=24
DEFAULT_THREADS_QWEN35_BOTH=12
DEFAULT_THREADS_GEMMA4_27B_BOTH=8
DEFAULT_POLL_MS_GEMMA4_E4B=0
DEFAULT_POLL_MS_QWEN35=0
DEFAULT_PARALLEL_GEMMA4_E4B=-1
# Single slot minimizes VRAM for parallel KV / server batching (matches llama-completion -np 1 probes)
DEFAULT_PARALLEL_GEMMA4_27B=1
DEFAULT_PARALLEL_GEMMA4_31B=1
# Intel SYCL summary: q8_0 KV cache for 256k context (memory savings outweigh slight overhead vs f16)
DEFAULT_CACHE_TYPE_SUMMARY_K=q8_0
DEFAULT_CACHE_TYPE_SUMMARY_V=q8_0
DEFAULT_CACHE_TYPE_QWEN35_K=q8_0
DEFAULT_CACHE_TYPE_QWEN35_V=q8_0
DEFAULT_CACHE_TYPE_QWEN35_BOTH_K=q8_0
DEFAULT_CACHE_TYPE_QWEN35_BOTH_V=q8_0
DEFAULT_CACHE_TYPE_GEMMA4_E4B_K=f16
DEFAULT_CACHE_TYPE_GEMMA4_E4B_V=f16
# f16 KV validated for max-context probes on RTX 3090
DEFAULT_CACHE_TYPE_GEMMA4_27B_K=f16
DEFAULT_CACHE_TYPE_GEMMA4_27B_V=f16
DEFAULT_CACHE_TYPE_GEMMA4_27B_BOTH_K=f16
DEFAULT_CACHE_TYPE_GEMMA4_27B_BOTH_V=f16
DEFAULT_CACHE_TYPE_GEMMA4_31B_K=f16
DEFAULT_CACHE_TYPE_GEMMA4_31B_V=f16
DEFAULT_N_PREDICT=32768
DEFAULT_SPEC_TYPE_GEMMA4_E4B=ngram-mod
DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_E4B=24
DEFAULT_DRAFT_MIN_GEMMA4_E4B=48
DEFAULT_DRAFT_MAX_GEMMA4_E4B=64
DEFAULT_SPEC_TYPE_GEMMA4_27B=ngram-mod
DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_27B=24
DEFAULT_DRAFT_MIN_GEMMA4_27B=48
DEFAULT_DRAFT_MAX_GEMMA4_27B=64
DEFAULT_SPEC_TYPE_GEMMA4_31B=ngram-mod
DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_31B=24
DEFAULT_DRAFT_MIN_GEMMA4_31B=48
DEFAULT_DRAFT_MAX_GEMMA4_31B=64

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
    qwen35-coding)  code='1;32' ;;
    gemma4-e4b)     code='1;36' ;;
    gemma4-27b-coding) code='1;92' ;;
    gemma4-31b-coding) code='1;95' ;;
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

require_optional_model() {
  local model_path="$1"
  local model_name="$2"
  if [[ ! -f "$model_path" ]]; then
    log_warning "$model_name not found: $model_path"
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
  local mmproj_path="${19:-}"
  local poll_ms="${20:-50}"
  
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
  [[ -n "$mmproj_path" ]] && cmd_ref+=(--mmproj "$mmproj_path")
  [[ "$use_jinja" == "true" ]] && cmd_ref+=(--jinja)
  
  cmd_ref+=(
    --ctx-size "$ctx_size"
    --n-predict "$DEFAULT_N_PREDICT"
    --flash-attn on
    --cache-type-k "$cache_type_k"
    --cache-type-v "$cache_type_v"
    --batch-size 2048
    --ubatch-size "$ubatch_size"
    --threads "$threads"
    --poll "$poll_ms"
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
    "" off "" "" "" false "$DEFAULT_CACHE_TYPE_SUMMARY_K" "$DEFAULT_CACHE_TYPE_SUMMARY_V"
  cmd+=(--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0.0)
  cmd+=(--presence-penalty 0.0 --repeat-penalty 1.0)

  exec_server "summary-balanced" cmd
}

start_summary_fast() {
  local port="$1"
  local cmd=()

  require_model "$MODEL_SUMMARY_FAST"
  build_server_cmd cmd "$MODEL_SUMMARY_FAST" "summary-fast" "SYCL0" "$port" \
    "$DEFAULT_CTX_SIZE_SUMMARY" "$DEFAULT_UBATCH_SIZE_SUMMARY_FAST" "$DEFAULT_THREADS_SUMMARY_FAST" \
    "" auto none "" "" false "$DEFAULT_CACHE_TYPE_SUMMARY_K" "$DEFAULT_CACHE_TYPE_SUMMARY_V"
  cmd+=(--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0.0)
  cmd+=(--presence-penalty 0.0 --repeat-penalty 1.0)

  exec_server "summary-fast" cmd
}

start_gemma4_e4b() {
  local port="$1"
  local cmd=()

  require_model "$MODEL_GEMMA4_E4B"

  build_server_cmd cmd "$MODEL_GEMMA4_E4B" "gemma4-e4b" "SYCL0" "$port" \
    "$DEFAULT_CTX_SIZE_GEMMA4_E4B" "$DEFAULT_UBATCH_SIZE_GEMMA4_E4B" "$DEFAULT_THREADS_GEMMA4_E4B" \
    "" on deepseek "$GEMMA4_CHAT_TEMPLATE_KWARGS" "" true "$DEFAULT_CACHE_TYPE_GEMMA4_E4B_K" "$DEFAULT_CACHE_TYPE_GEMMA4_E4B_V" \
    "$DEFAULT_N_GPU_LAYERS" "$LLAMA_SERVER_BIN_INTEL" "" "$DEFAULT_POLL_MS_GEMMA4_E4B"
  cmd+=(--no-mmproj)
  cmd+=(--threads-batch "$DEFAULT_THREADS_BATCH_GEMMA4_E4B")
  cmd+=(--parallel "$DEFAULT_PARALLEL_GEMMA4_E4B")
  cmd+=(
    --spec-type "$DEFAULT_SPEC_TYPE_GEMMA4_E4B"
    --spec-ngram-size-n "$DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_E4B"
    --draft-min "$DEFAULT_DRAFT_MIN_GEMMA4_E4B"
    --draft-max "$DEFAULT_DRAFT_MAX_GEMMA4_E4B"
  )
  cmd+=(--fit off)

  exec_server "gemma4-e4b" cmd
}

start_qwen35() {
  local port="$1"
  local cmd=()
  
  require_model "$MODEL_QWEN35"
  require_executable "$LLAMA_SERVER_BIN_NVIDIA" "NVIDIA llama-server"
  build_server_cmd cmd "$MODEL_QWEN35" "qwen35-coding" "" "$port" \
    "$DEFAULT_CTX_SIZE_QWEN35" "$DEFAULT_UBATCH_SIZE_QWEN35" "$DEFAULT_THREADS_QWEN35" \
    "" on deepseek '{"enable_thinking":true}' "" "false" \
    "$DEFAULT_CACHE_TYPE_QWEN35_K" "$DEFAULT_CACHE_TYPE_QWEN35_V" "$DEFAULT_N_GPU_LAYERS_QWEN35" "$LLAMA_SERVER_BIN_NVIDIA" "" "$DEFAULT_POLL_MS_QWEN35"
  cmd+=(--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0.0)
  cmd+=(--presence-penalty 0.0 --repeat-penalty 1.0)
  
  exec_server "qwen35-coding" cmd
}

start_gemma4_27b() {
  local port="$1"
  local cmd=()

  require_model "$MODEL_GEMMA4_27B"
  require_executable "$LLAMA_SERVER_BIN_NVIDIA" "NVIDIA llama-server"
  build_server_cmd cmd "$MODEL_GEMMA4_27B" "gemma4-27b-coding" "" "$port" \
    "$DEFAULT_CTX_SIZE_GEMMA4_27B" "$DEFAULT_UBATCH_SIZE_GEMMA4_27B" "$DEFAULT_THREADS_GEMMA4_27B" \
    "" "" "" "$GEMMA4_CHAT_TEMPLATE_KWARGS" "" false \
    "$DEFAULT_CACHE_TYPE_GEMMA4_27B_K" "$DEFAULT_CACHE_TYPE_GEMMA4_27B_V" "$DEFAULT_N_GPU_LAYERS_GEMMA4_27B" "$LLAMA_SERVER_BIN_NVIDIA" "" 0
  cmd+=(--fit off)
  cmd+=(--parallel "$DEFAULT_PARALLEL_GEMMA4_27B")
  cmd+=(
    --spec-type "$DEFAULT_SPEC_TYPE_GEMMA4_27B"
    --spec-ngram-size-n "$DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_27B"
    --draft-min "$DEFAULT_DRAFT_MIN_GEMMA4_27B"
    --draft-max "$DEFAULT_DRAFT_MAX_GEMMA4_27B"
  )

  exec_server "gemma4-27b-coding" cmd
}

start_gemma4_31b() {
  local port="$1"
  local cmd=()

  require_model "$MODEL_GEMMA4_31B"
  require_executable "$LLAMA_SERVER_BIN_NVIDIA" "NVIDIA llama-server"
  build_server_cmd cmd "$MODEL_GEMMA4_31B" "gemma4-31b-coding" "" "$port" \
    "$DEFAULT_CTX_SIZE_GEMMA4_31B" "$DEFAULT_UBATCH_SIZE_GEMMA4_31B" "$DEFAULT_THREADS_GEMMA4_31B" \
    "" "" "" "$GEMMA4_CHAT_TEMPLATE_KWARGS" "" false \
    "$DEFAULT_CACHE_TYPE_GEMMA4_31B_K" "$DEFAULT_CACHE_TYPE_GEMMA4_31B_V" "$DEFAULT_N_GPU_LAYERS_GEMMA4_31B" "$LLAMA_SERVER_BIN_NVIDIA" "" 0
  cmd+=(--fit off)
  cmd+=(--parallel "$DEFAULT_PARALLEL_GEMMA4_31B")
  cmd+=(
    --spec-type "$DEFAULT_SPEC_TYPE_GEMMA4_31B"
    --spec-ngram-size-n "$DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_31B"
    --draft-min "$DEFAULT_DRAFT_MIN_GEMMA4_31B"
    --draft-max "$DEFAULT_DRAFT_MAX_GEMMA4_31B"
  )

  exec_server "gemma4-31b-coding" cmd
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
    "" off "" "" "" false "$DEFAULT_CACHE_TYPE_SUMMARY_K" "$DEFAULT_CACHE_TYPE_SUMMARY_V"
  summary_balanced_cmd+=(--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0.0)
  summary_balanced_cmd+=(--presence-penalty 0.0 --repeat-penalty 1.0)
  
  build_server_cmd qwen35_cmd "$MODEL_QWEN35_BOTH" "qwen35-coding" "" "$qwen35_port" \
    "$DEFAULT_CTX_SIZE_BOTH_QWEN35" "$DEFAULT_UBATCH_SIZE_QWEN35_BOTH" "$DEFAULT_THREADS_QWEN35_BOTH" \
    "" on deepseek '{"enable_thinking":true}' "" "false" \
    "$DEFAULT_CACHE_TYPE_QWEN35_BOTH_K" "$DEFAULT_CACHE_TYPE_QWEN35_BOTH_V" "$DEFAULT_N_GPU_LAYERS_QWEN35_BOTH" "$LLAMA_SERVER_BIN_NVIDIA" "" "$DEFAULT_POLL_MS_QWEN35"
  qwen35_cmd+=(--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0.0)
  qwen35_cmd+=(--presence-penalty 0.0 --repeat-penalty 1.0)
  
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

start_both_gemma4_27b() {
  local gemma4_e4b_port="$1"
  local gemma4_27b_port="$2"
  local gemma4_e4b_cmd=()
  local gemma4_27b_cmd=()

  require_model "$MODEL_GEMMA4_E4B"
  require_model "$MODEL_GEMMA4_27B_BOTH"
  require_executable "$LLAMA_SERVER_BIN_NVIDIA" "NVIDIA llama-server"

  build_server_cmd gemma4_e4b_cmd "$MODEL_GEMMA4_E4B" "gemma4-e4b" "SYCL0" "$gemma4_e4b_port" \
    "$DEFAULT_CTX_SIZE_GEMMA4_E4B" "$DEFAULT_UBATCH_SIZE_GEMMA4_E4B" "$DEFAULT_THREADS_GEMMA4_E4B" \
    "" on deepseek "$GEMMA4_CHAT_TEMPLATE_KWARGS" "" true "$DEFAULT_CACHE_TYPE_GEMMA4_E4B_K" "$DEFAULT_CACHE_TYPE_GEMMA4_E4B_V" \
    "$DEFAULT_N_GPU_LAYERS" "$LLAMA_SERVER_BIN_INTEL" "" "$DEFAULT_POLL_MS_GEMMA4_E4B"
  gemma4_e4b_cmd+=(--no-mmproj)
  gemma4_e4b_cmd+=(--threads-batch "$DEFAULT_THREADS_BATCH_GEMMA4_E4B")
  gemma4_e4b_cmd+=(--parallel "$DEFAULT_PARALLEL_GEMMA4_E4B")
  gemma4_e4b_cmd+=(
    --spec-type "$DEFAULT_SPEC_TYPE_GEMMA4_E4B"
    --spec-ngram-size-n "$DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_E4B"
    --draft-min "$DEFAULT_DRAFT_MIN_GEMMA4_E4B"
    --draft-max "$DEFAULT_DRAFT_MAX_GEMMA4_E4B"
  )
  gemma4_e4b_cmd+=(--fit off)

  build_server_cmd gemma4_27b_cmd "$MODEL_GEMMA4_27B_BOTH" "gemma4-27b-coding" "" "$gemma4_27b_port" \
    "$DEFAULT_CTX_SIZE_BOTH_GEMMA4_27B" "$DEFAULT_UBATCH_SIZE_GEMMA4_27B_BOTH" "$DEFAULT_THREADS_GEMMA4_27B_BOTH" \
    "" "" "" "$GEMMA4_CHAT_TEMPLATE_KWARGS" "" false \
    "$DEFAULT_CACHE_TYPE_GEMMA4_27B_BOTH_K" "$DEFAULT_CACHE_TYPE_GEMMA4_27B_BOTH_V" "$DEFAULT_N_GPU_LAYERS_GEMMA4_27B_BOTH" "$LLAMA_SERVER_BIN_NVIDIA" "" 0
  gemma4_27b_cmd+=(--fit off)
  gemma4_27b_cmd+=(--parallel "$DEFAULT_PARALLEL_GEMMA4_27B")
  gemma4_27b_cmd+=(
    --spec-type "$DEFAULT_SPEC_TYPE_GEMMA4_27B"
    --spec-ngram-size-n "$DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_27B"
    --draft-min "$DEFAULT_DRAFT_MIN_GEMMA4_27B"
    --draft-max "$DEFAULT_DRAFT_MAX_GEMMA4_27B"
  )

  trap on_interrupt INT
  trap on_terminate TERM
  trap cleanup_servers EXIT

  start_server_background "gemma4-e4b" gemma4_e4b_cmd
  start_server_background "gemma4-27b-coding" gemma4_27b_cmd

  echo "gemma4-e4b: http://$HOST:$gemma4_e4b_port/v1"
  echo "gemma4-27b-coding: http://$HOST:$gemma4_27b_port/v1"

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
  run_opencode_models.sh gemma4-e4b [port]
  run_opencode_models.sh gemma4-27b [port]
  run_opencode_models.sh gemma4-31b [port]
  run_opencode_models.sh qwen35 [port]
  run_opencode_models.sh both-qwen35 [summary_balanced_port qwen35_port]
  run_opencode_models.sh both-gemma4-27b [gemma4_e4b_port gemma4_27b_port]
  run_opencode_models.sh dry-run summary-balanced|summary-fast|gemma4-e4b|gemma4-27b|gemma4-31b|qwen35|both-qwen35|both-gemma4-27b [ports...]

Examples:
  run_opencode_models.sh summary-balanced
  run_opencode_models.sh summary-fast 8082
  run_opencode_models.sh gemma4-e4b 8080
  run_opencode_models.sh gemma4-27b 8084
  run_opencode_models.sh gemma4-31b 8085
  run_opencode_models.sh qwen35 8080
  run_opencode_models.sh both-qwen35 8080 8081
  run_opencode_models.sh both-gemma4-27b 8080 8084
  run_opencode_models.sh dry-run both-qwen35
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
  local gemma4_27b_port_single="${primary_port:-$GEMMA4_27B_PORT}"
  local gemma4_27b_port_both="${secondary_port:-$GEMMA4_27B_PORT}"
  local gemma4_31b_port_single="${primary_port:-$GEMMA4_31B_PORT}"
  local gemma4_e4b_port="${primary_port:-$GEMMA4_E4B_PORT}"

  echo "=== DRY RUN MODE ==="
  echo "Mode: $mode"
  echo "llama-server (Intel): $LLAMA_SERVER_BIN_INTEL"
  echo "llama-server (NVIDIA): $LLAMA_SERVER_BIN_NVIDIA"
  echo "summary-balanced model: $MODEL_SUMMARY_BALANCED"
  echo "summary-fast model: $MODEL_SUMMARY_FAST"
  echo "gemma4-e4b model: $MODEL_GEMMA4_E4B"
  echo "gemma4-e4b mmproj: $MODEL_GEMMA4_E4B_MMPROJ"
  echo "gemma4-27b model: $MODEL_GEMMA4_27B"
  echo "gemma4-31b model: $MODEL_GEMMA4_31B"
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
      echo "  Reasoning Format: (disabled)"
      echo "  Jinja: false"
      echo "  Chat Template Kwargs: (none)"
      echo "  Sampling: temperature=0.6 top-p=0.95 top-k=20 min-p=0.0"
      echo "  Penalties: presence=0.0 repeat=1.0"
      build_server_cmd tmp_cmd "$MODEL_SUMMARY_BALANCED" "summary-balanced" "SYCL0" "$summary_balanced_port" \
        "$DEFAULT_CTX_SIZE_SUMMARY" "$DEFAULT_UBATCH_SIZE_SUMMARY_BALANCED" "$DEFAULT_THREADS_SUMMARY_BALANCED" \
        "" off "" "" "" false
      tmp_cmd+=(--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0.0)
      tmp_cmd+=(--presence-penalty 0.0 --repeat-penalty 1.0)
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
      echo "  Reasoning: auto"
      echo "  Reasoning Format: none"
      echo "  Sampling: temperature=0.6 top-p=0.95 top-k=20 min-p=0.0"
      echo "  Penalties: presence=0.0 repeat=1.0"
      build_server_cmd tmp_cmd "$MODEL_SUMMARY_FAST" "summary-fast" "SYCL0" "$summary_fast_port" \
        "$DEFAULT_CTX_SIZE_SUMMARY" "$DEFAULT_UBATCH_SIZE_SUMMARY_FAST" "$DEFAULT_THREADS_SUMMARY_FAST" \
        "" auto none "" "" false "$DEFAULT_CACHE_TYPE_SUMMARY_K" "$DEFAULT_CACHE_TYPE_SUMMARY_V"
      tmp_cmd+=(--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0.0)
      tmp_cmd+=(--presence-penalty 0.0 --repeat-penalty 1.0)
      echo "  Command: ${tmp_cmd[*]}"
      unset tmp_cmd
      ;;
    gemma4-e4b)
      echo "gemma4-e4b:"
      echo "  Port: $gemma4_e4b_port"
      echo "  Device: SYCL0"
      echo "  Context: $DEFAULT_CTX_SIZE_GEMMA4_E4B"
      echo "  Threads: $DEFAULT_THREADS_GEMMA4_E4B"
      echo "  Threads Batch: $DEFAULT_THREADS_BATCH_GEMMA4_E4B"
      echo "  Parallel Slots: auto (-1)"
      echo "  UBatch: $DEFAULT_UBATCH_SIZE_GEMMA4_E4B"
      echo "  Poll: $DEFAULT_POLL_MS_GEMMA4_E4B"
      echo "  Split Mode: layer"
      echo "  KV cache: $DEFAULT_CACHE_TYPE_GEMMA4_E4B_K/$DEFAULT_CACHE_TYPE_GEMMA4_E4B_V"
      echo "  Multimodal: disabled (--no-mmproj)"
      echo "  Speculative: $DEFAULT_SPEC_TYPE_GEMMA4_E4B (n=$DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_E4B, draft=$DEFAULT_DRAFT_MIN_GEMMA4_E4B..$DEFAULT_DRAFT_MAX_GEMMA4_E4B)"
      echo "  Reasoning: on"
      echo "  Reasoning Format: deepseek"
      echo "  Thinking: enabled"
      echo "  Chat Template Kwargs: $GEMMA4_CHAT_TEMPLATE_KWARGS"
      build_server_cmd tmp_cmd "$MODEL_GEMMA4_E4B" "gemma4-e4b" "SYCL0" "$gemma4_e4b_port" \
        "$DEFAULT_CTX_SIZE_GEMMA4_E4B" "$DEFAULT_UBATCH_SIZE_GEMMA4_E4B" "$DEFAULT_THREADS_GEMMA4_E4B" \
        "" on deepseek "$GEMMA4_CHAT_TEMPLATE_KWARGS" "" true "$DEFAULT_CACHE_TYPE_GEMMA4_E4B_K" "$DEFAULT_CACHE_TYPE_GEMMA4_E4B_V" \
        "$DEFAULT_N_GPU_LAYERS" "$LLAMA_SERVER_BIN_INTEL" "" "$DEFAULT_POLL_MS_GEMMA4_E4B"
      tmp_cmd+=(--no-mmproj)
      tmp_cmd+=(--threads-batch "$DEFAULT_THREADS_BATCH_GEMMA4_E4B")
      tmp_cmd+=(--parallel "$DEFAULT_PARALLEL_GEMMA4_E4B")
      tmp_cmd+=(
        --spec-type "$DEFAULT_SPEC_TYPE_GEMMA4_E4B"
        --spec-ngram-size-n "$DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_E4B"
        --draft-min "$DEFAULT_DRAFT_MIN_GEMMA4_E4B"
        --draft-max "$DEFAULT_DRAFT_MAX_GEMMA4_E4B"
      )
      tmp_cmd+=(--fit off)
      echo "  Command: ${tmp_cmd[*]}"
      unset tmp_cmd
      ;;
    gemma4-27b)
      echo "gemma4-27b-coding:"
      echo "  Port: $gemma4_27b_port_single"
      echo "  Device: NVIDIA (CUDA)"
      echo "  Context: $DEFAULT_CTX_SIZE_GEMMA4_27B"
      echo "  Threads: $DEFAULT_THREADS_GEMMA4_27B"
      echo "  UBatch: $DEFAULT_UBATCH_SIZE_GEMMA4_27B"
      echo "  Parallel slots: $DEFAULT_PARALLEL_GEMMA4_27B"
      echo "  KV cache: $DEFAULT_CACHE_TYPE_GEMMA4_27B_K/$DEFAULT_CACHE_TYPE_GEMMA4_27B_V"
      echo "  n-gpu-layers: $DEFAULT_N_GPU_LAYERS_GEMMA4_27B"
      echo "  Poll: 0"
      echo "  Speculative: $DEFAULT_SPEC_TYPE_GEMMA4_27B (n=$DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_27B, draft=$DEFAULT_DRAFT_MIN_GEMMA4_27B..$DEFAULT_DRAFT_MAX_GEMMA4_27B)"
      echo "  Thinking: enabled"
      echo "  Chat Template Kwargs: $GEMMA4_CHAT_TEMPLATE_KWARGS"
      build_server_cmd tmp_cmd "$MODEL_GEMMA4_27B" "gemma4-27b-coding" "" "$gemma4_27b_port_single" \
        "$DEFAULT_CTX_SIZE_GEMMA4_27B" "$DEFAULT_UBATCH_SIZE_GEMMA4_27B" "$DEFAULT_THREADS_GEMMA4_27B" \
        "" "" "" "$GEMMA4_CHAT_TEMPLATE_KWARGS" "" false \
        "$DEFAULT_CACHE_TYPE_GEMMA4_27B_K" "$DEFAULT_CACHE_TYPE_GEMMA4_27B_V" "$DEFAULT_N_GPU_LAYERS_GEMMA4_27B" "$LLAMA_SERVER_BIN_NVIDIA" "" 0
      tmp_cmd+=(--fit off)
      tmp_cmd+=(--parallel "$DEFAULT_PARALLEL_GEMMA4_27B")
      tmp_cmd+=(
        --spec-type "$DEFAULT_SPEC_TYPE_GEMMA4_27B"
        --spec-ngram-size-n "$DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_27B"
        --draft-min "$DEFAULT_DRAFT_MIN_GEMMA4_27B"
        --draft-max "$DEFAULT_DRAFT_MAX_GEMMA4_27B"
      )
      echo "  Command: ${tmp_cmd[*]}"
      unset tmp_cmd
      ;;
    gemma4-31b)
      echo "gemma4-31b-coding:"
      echo "  Port: $gemma4_31b_port_single"
      echo "  Device: NVIDIA (CUDA)"
      echo "  Context: $DEFAULT_CTX_SIZE_GEMMA4_31B"
      echo "  Threads: $DEFAULT_THREADS_GEMMA4_31B"
      echo "  UBatch: $DEFAULT_UBATCH_SIZE_GEMMA4_31B"
      echo "  Parallel slots: $DEFAULT_PARALLEL_GEMMA4_31B"
      echo "  KV cache: $DEFAULT_CACHE_TYPE_GEMMA4_31B_K/$DEFAULT_CACHE_TYPE_GEMMA4_31B_V"
      echo "  n-gpu-layers: $DEFAULT_N_GPU_LAYERS_GEMMA4_31B"
      echo "  Poll: 0"
      echo "  Speculative: $DEFAULT_SPEC_TYPE_GEMMA4_31B (n=$DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_31B, draft=$DEFAULT_DRAFT_MIN_GEMMA4_31B..$DEFAULT_DRAFT_MAX_GEMMA4_31B)"
      echo "  Thinking: enabled"
      echo "  Chat Template Kwargs: $GEMMA4_CHAT_TEMPLATE_KWARGS"
      build_server_cmd tmp_cmd "$MODEL_GEMMA4_31B" "gemma4-31b-coding" "" "$gemma4_31b_port_single" \
        "$DEFAULT_CTX_SIZE_GEMMA4_31B" "$DEFAULT_UBATCH_SIZE_GEMMA4_31B" "$DEFAULT_THREADS_GEMMA4_31B" \
        "" "" "" "$GEMMA4_CHAT_TEMPLATE_KWARGS" "" false \
        "$DEFAULT_CACHE_TYPE_GEMMA4_31B_K" "$DEFAULT_CACHE_TYPE_GEMMA4_31B_V" "$DEFAULT_N_GPU_LAYERS_GEMMA4_31B" "$LLAMA_SERVER_BIN_NVIDIA" "" 0
      tmp_cmd+=(--fit off)
      tmp_cmd+=(--parallel "$DEFAULT_PARALLEL_GEMMA4_31B")
      tmp_cmd+=(
        --spec-type "$DEFAULT_SPEC_TYPE_GEMMA4_31B"
        --spec-ngram-size-n "$DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_31B"
        --draft-min "$DEFAULT_DRAFT_MIN_GEMMA4_31B"
        --draft-max "$DEFAULT_DRAFT_MAX_GEMMA4_31B"
      )
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
      echo "  Reasoning: on"
      echo "  Reasoning Format: deepseek"
      echo "  Chat Template Kwargs: {\"enable_thinking\":true}"
      echo "  Poll: $DEFAULT_POLL_MS_QWEN35"
      build_server_cmd tmp_cmd "$MODEL_QWEN35" "qwen35-coding" "" "$qwen35_port_single" \
        "$DEFAULT_CTX_SIZE_QWEN35" "$DEFAULT_UBATCH_SIZE_QWEN35" "$DEFAULT_THREADS_QWEN35" \
        "" on deepseek '{"enable_thinking":true}' "" "false" \
        "$DEFAULT_CACHE_TYPE_QWEN35_K" "$DEFAULT_CACHE_TYPE_QWEN35_V" "$DEFAULT_N_GPU_LAYERS_QWEN35" "$LLAMA_SERVER_BIN_NVIDIA" "" "$DEFAULT_POLL_MS_QWEN35"
      tmp_cmd+=(--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0.0)
      tmp_cmd+=(--presence-penalty 0.0 --repeat-penalty 1.0)
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
      echo "  Reasoning Format: (disabled)"
      echo "  Jinja: false"
      echo "  Chat Template Kwargs: (none)"
      echo "  Sampling: temperature=0.6 top-p=0.95 top-k=20 min-p=0.0"
      echo "  Penalties: presence=0.0 repeat=1.0"
      build_server_cmd tmp_cmd "$MODEL_SUMMARY_BALANCED" "summary-balanced" "SYCL0" "$summary_balanced_port" \
        "$DEFAULT_CTX_SIZE_BOTH_SUMMARY" "$DEFAULT_UBATCH_SIZE_SUMMARY_BALANCED" "$DEFAULT_THREADS_SUMMARY_BALANCED" \
        "" off "" "" "" false "$DEFAULT_CACHE_TYPE_SUMMARY_K" "$DEFAULT_CACHE_TYPE_SUMMARY_V"
      tmp_cmd+=(--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0.0)
      tmp_cmd+=(--presence-penalty 0.0 --repeat-penalty 1.0)
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
      echo "  Reasoning: on"
      echo "  Reasoning Format: deepseek"
      echo "  Chat Template Kwargs: {\"enable_thinking\":true}"
      echo "  Poll: $DEFAULT_POLL_MS_QWEN35"
      build_server_cmd tmp_cmd "$MODEL_QWEN35_BOTH" "qwen35-coding" "" "$qwen35_port_both" \
        "$DEFAULT_CTX_SIZE_BOTH_QWEN35" "$DEFAULT_UBATCH_SIZE_QWEN35_BOTH" "$DEFAULT_THREADS_QWEN35_BOTH" \
        "" on deepseek '{"enable_thinking":true}' "" "false" \
        "$DEFAULT_CACHE_TYPE_QWEN35_BOTH_K" "$DEFAULT_CACHE_TYPE_QWEN35_BOTH_V" "$DEFAULT_N_GPU_LAYERS_QWEN35_BOTH" "$LLAMA_SERVER_BIN_NVIDIA" "" "$DEFAULT_POLL_MS_QWEN35"
      tmp_cmd+=(--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0.0)
      tmp_cmd+=(--presence-penalty 0.0 --repeat-penalty 1.0)
      echo "  Command: ${tmp_cmd[*]}"
      unset tmp_cmd
      ;;
    both-gemma4-27b)
      echo "gemma4-e4b:"
      echo "  Port: $summary_balanced_port"
      echo "  Device: SYCL0"
      echo "  Context: $DEFAULT_CTX_SIZE_GEMMA4_E4B"
      echo "  Threads: $DEFAULT_THREADS_GEMMA4_E4B"
      echo "  Threads Batch: $DEFAULT_THREADS_BATCH_GEMMA4_E4B"
      echo "  Parallel Slots: auto (-1)"
      echo "  UBatch: $DEFAULT_UBATCH_SIZE_GEMMA4_E4B"
      echo "  Poll: $DEFAULT_POLL_MS_GEMMA4_E4B"
      echo "  Split Mode: layer"
      echo "  KV cache: $DEFAULT_CACHE_TYPE_GEMMA4_E4B_K/$DEFAULT_CACHE_TYPE_GEMMA4_E4B_V"
      echo "  Multimodal: disabled (--no-mmproj)"
      echo "  Speculative: $DEFAULT_SPEC_TYPE_GEMMA4_E4B (n=$DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_E4B, draft=$DEFAULT_DRAFT_MIN_GEMMA4_E4B..$DEFAULT_DRAFT_MAX_GEMMA4_E4B)"
      echo "  Reasoning: on"
      echo "  Reasoning Format: deepseek"
      echo "  Thinking: enabled"
      echo "  Chat Template Kwargs: $GEMMA4_CHAT_TEMPLATE_KWARGS"
      build_server_cmd tmp_cmd "$MODEL_GEMMA4_E4B" "gemma4-e4b" "SYCL0" "$summary_balanced_port" \
        "$DEFAULT_CTX_SIZE_GEMMA4_E4B" "$DEFAULT_UBATCH_SIZE_GEMMA4_E4B" "$DEFAULT_THREADS_GEMMA4_E4B" \
        "" on deepseek "$GEMMA4_CHAT_TEMPLATE_KWARGS" "" true "$DEFAULT_CACHE_TYPE_GEMMA4_E4B_K" "$DEFAULT_CACHE_TYPE_GEMMA4_E4B_V" \
        "$DEFAULT_N_GPU_LAYERS" "$LLAMA_SERVER_BIN_INTEL" "" "$DEFAULT_POLL_MS_GEMMA4_E4B"
      tmp_cmd+=(--no-mmproj)
      tmp_cmd+=(--threads-batch "$DEFAULT_THREADS_BATCH_GEMMA4_E4B")
      tmp_cmd+=(--parallel "$DEFAULT_PARALLEL_GEMMA4_E4B")
      tmp_cmd+=(
        --spec-type "$DEFAULT_SPEC_TYPE_GEMMA4_E4B"
        --spec-ngram-size-n "$DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_E4B"
        --draft-min "$DEFAULT_DRAFT_MIN_GEMMA4_E4B"
        --draft-max "$DEFAULT_DRAFT_MAX_GEMMA4_E4B"
      )
      tmp_cmd+=(--fit off)
      echo "  Command: ${tmp_cmd[*]}"
      unset tmp_cmd
      echo ""
      echo "gemma4-27b-coding:"
      echo "  Port: $gemma4_27b_port_both"
      echo "  Device: NVIDIA (CUDA)"
      echo "  Context: $DEFAULT_CTX_SIZE_BOTH_GEMMA4_27B"
      echo "  Threads: $DEFAULT_THREADS_GEMMA4_27B_BOTH"
      echo "  UBatch: $DEFAULT_UBATCH_SIZE_GEMMA4_27B_BOTH"
      echo "  Parallel slots: $DEFAULT_PARALLEL_GEMMA4_27B"
      echo "  KV cache: $DEFAULT_CACHE_TYPE_GEMMA4_27B_BOTH_K/$DEFAULT_CACHE_TYPE_GEMMA4_27B_BOTH_V"
      echo "  n-gpu-layers: $DEFAULT_N_GPU_LAYERS_GEMMA4_27B_BOTH"
      echo "  Poll: 0"
      echo "  Speculative: $DEFAULT_SPEC_TYPE_GEMMA4_27B (n=$DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_27B, draft=$DEFAULT_DRAFT_MIN_GEMMA4_27B..$DEFAULT_DRAFT_MAX_GEMMA4_27B)"
      echo "  Thinking: enabled"
      echo "  Chat Template Kwargs: $GEMMA4_CHAT_TEMPLATE_KWARGS"
      build_server_cmd tmp_cmd "$MODEL_GEMMA4_27B_BOTH" "gemma4-27b-coding" "" "$gemma4_27b_port_both" \
        "$DEFAULT_CTX_SIZE_BOTH_GEMMA4_27B" "$DEFAULT_UBATCH_SIZE_GEMMA4_27B_BOTH" "$DEFAULT_THREADS_GEMMA4_27B_BOTH" \
        "" "" "" "$GEMMA4_CHAT_TEMPLATE_KWARGS" "" false \
        "$DEFAULT_CACHE_TYPE_GEMMA4_27B_BOTH_K" "$DEFAULT_CACHE_TYPE_GEMMA4_27B_BOTH_V" "$DEFAULT_N_GPU_LAYERS_GEMMA4_27B_BOTH" "$LLAMA_SERVER_BIN_NVIDIA" "" 0
      tmp_cmd+=(--fit off)
      tmp_cmd+=(--parallel "$DEFAULT_PARALLEL_GEMMA4_27B")
      tmp_cmd+=(
        --spec-type "$DEFAULT_SPEC_TYPE_GEMMA4_27B"
        --spec-ngram-size-n "$DEFAULT_SPEC_NGRAM_SIZE_N_GEMMA4_27B"
        --draft-min "$DEFAULT_DRAFT_MIN_GEMMA4_27B"
        --draft-max "$DEFAULT_DRAFT_MAX_GEMMA4_27B"
      )
      echo "  Command: ${tmp_cmd[*]}"
      unset tmp_cmd
      ;;
  esac
}

# Initialize global state
init_colors
export ZES_ENABLE_SYSMAN=1

# Parse and validate arguments
mode="${1:-}"
if [[ "$mode" == "dry-run" ]]; then
  mode="${2:-}"
  if [[ -z "$mode" ]]; then
    echo "error: dry-run requires a mode argument (summary-balanced|summary-fast|gemma4-e4b|gemma4-27b|gemma4-31b|qwen35|both-qwen35|both-gemma4-27b)" >&2
    usage
    exit 1
  fi
  dry_run "$mode" "${3:-}" "${4:-}"
  exit 0
fi

# Only check prerequisites for actual server starts, not dry-run
check_prereqs

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
  gemma4-e4b)
    port="${2:-$GEMMA4_E4B_PORT}"
    if ! validate_port "$port" "gemma4-e4b port"; then
      usage
      exit 1
    fi
    echo "Starting gemma4-e4b at http://$HOST:$port/v1"
    start_gemma4_e4b "$port"
    ;;
  gemma4-27b)
    port="${2:-$GEMMA4_27B_PORT}"
    if ! validate_port "$port" "gemma4-27b port"; then
      usage
      exit 1
    fi
    echo "Starting gemma4-27b-coding at http://$HOST:$port/v1 (NVIDIA CUDA)"
    start_gemma4_27b "$port"
    ;;
  gemma4-31b)
    port="${2:-$GEMMA4_31B_PORT}"
    if ! validate_port "$port" "gemma4-31b port"; then
      usage
      exit 1
    fi
    echo "Starting gemma4-31b-coding at http://$HOST:$port/v1 (NVIDIA CUDA)"
    start_gemma4_31b "$port"
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
  both-gemma4-27b)
    port32="${2:-$SUMMARY_BALANCED_PORT}"
    port27="${3:-$GEMMA4_27B_PORT}"

    if ! validate_port "$port32" "summary-balanced port"; then
      usage
      exit 1
    fi
    if ! validate_port "$port27" "gemma4-27b port"; then
      usage
      exit 1
    fi
    if ! validate_ports "$port32" "$port27" "gemma4-e4b port" "gemma4-27b port"; then
      usage
      exit 1
    fi

    start_both_gemma4_27b "$port32" "$port27"
    ;;
  *)
    usage
    exit 1
    ;;
esac
