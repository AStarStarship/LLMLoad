#!/usr/bin/env bash
set -euo pipefail

readonly LLAMA_ROOT="${LLAMA_ROOT:-$HOME/llama.cpp}"
readonly MODEL_BASE="${LLAMA_MODEL_BASE:-/mnt/llm}"

usage() {
  cat <<'EOF'
Usage:
  ./runllama.sh <port> <backend> <devices> <model> <batch> <parallel> <ctx/slot|auto> [threads] [split-mode] [tensor-split]
  ./runllama.sh UPDATE
  ./runllama.sh BUILD <backend>
  ./runllama.sh LS <backend>
  ./runllama.sh INSPECT <port>
  ./runllama.sh KILL <port|ALL>
  ./runllama.sh CLEAN <backend|ALL>
  ./runllama.sh CLONE

Backends:
  VULKAN | SYCL | ROCM | CUDA | HYBRID | CPU | NONE

Models:
  12b | 26b | 27b | 27bmtp | 35b | 35bmtp

Devices:
  A numeric backend index (0), a comma-separated list (0,1), "all", or full
  device names. HYBRID requires full names, for example Vulkan0,Vulkan1,CUDA0.
  Always confirm the names and ordering with: ./runllama.sh LS <backend>

Optional launch settings:
  threads       Defaults to LLAMA_THREADS or llama.cpp's automatic choice.
  split-mode    none, layer, row, or tensor. Defaults to none for one device
                and layer for multiple devices. tensor is experimental.
  tensor-split  Per-device proportions, for example 1,1 or 2,2,1.

Useful environment variables:
  LLAMA_HOST=127.0.0.1          Bind address; set 0.0.0.0 for LAN access.
  LLAMA_API_KEY=...             API key, strongly recommended on a LAN.
  LLAMA_FIT=on|off              Override fitting (auto context defaults on).
  LLAMA_FIT_TARGET=1024         Free-memory margin per device in MiB.
  LLAMA_FIT_CTX=4096            Minimum context accepted by the fitter.
  LLAMA_CACHE_K=f16             Override K-cache type.
  LLAMA_CACHE_V=q8_0            Override V-cache type.
  LLAMA_FLASH_ATTN=on           Set flash attention to on, off, or auto.
  LLAMA_DRY_MULTIPLIER=0        Enable DRY explicitly if desired.
  LLAMA_DRY_LAST_N=4096         Bound DRY work; never defaults to full context.
  LLAMA_MODEL=/path/model.gguf  Override the selected profile's model file.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

backend_build_dir() {
  case "${1^^}" in
    CPU|NONE) echo "build-cpu" ;;
    SYCL)     echo "build-sycl" ;;
    ROCM)     echo "build-rocm" ;;
    CUDA)     echo "build-cuda" ;;
    VULKAN)   echo "build-vulkan" ;;
    HYBRID)   echo "build-hybrid" ;;
    *) return 1 ;;
  esac
}

backend_device_prefix() {
  case "${1^^}" in
    SYCL)   echo "SYCL" ;;
    ROCM)   echo "ROCm" ;;
    CUDA)   echo "CUDA" ;;
    VULKAN) echo "Vulkan" ;;
    *) return 1 ;;
  esac
}

backend_binary() {
  local build_dir
  build_dir="$(backend_build_dir "$1")" || die "unknown backend: $1"
  echo "$LLAMA_ROOT/$build_dir/bin/llama-server"
}

require_binary() {
  local binary="$1"
  [[ -x "$binary" ]] || die "llama-server not found or not executable: $binary (run ./runllama.sh BUILD <backend>)"
}

build_backend() {
  local backend="${1^^}"
  local build_dir
  build_dir="$(backend_build_dir "$backend")" || die "unknown backend: $backend"
  local build_path="$LLAMA_ROOT/$build_dir"
  
  # Core optimization flags applied to all builds
  local cmake_args=(
    -S "$LLAMA_ROOT"
    -B "$build_path"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_FLAGS="-march=native -O3 -Wno-error"
    -DCMAKE_CXX_FLAGS="-march=native -O3 -Wno-error"
  )

  # Route specific hardware acceleration variables explicitly
  case "$backend" in
    CPU|NONE) 
      cmake_args+=( -DGGML_NATIVE=ON )
      ;;
    SYCL)
      echo "source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1" >> ~/.bashrc
      echo "export ZES_ENABLE_SYSMAN=1" >> ~/.bashrc
      source ~/.bashrc
      cmake_args+=(
        -DCMAKE_C_COMPILER=icx
        -DCMAKE_CXX_COMPILER=icpx
        -DGGML_SYCL=ON
        -DGGML_SYCL_F16=ON
        -DGGML_SYCL_DNN=ON
        -DGGML_SYCL_TARGET=INTEL
      )
      ;;
    ROCM)
      cmake_args+=( -DGGML_HIP=ON )
      ;;
    CUDA)
      cmake_args+=( -DGGML_CUDA=ON )
      ;;
    VULKAN)
      cmake_args+=( -DGGML_VULKAN=ON )
      ;;
    HYBRID)
      # Combines split-workloads across engines if supported by your setup
      cmake_args+=( -DGGML_VULKAN=ON -DGGML_CUDA=ON )
      ;;
  esac

  echo "Configuring $backend in $build_path"
  
  # Ensure the old build artifacts are cleared so CMake recalculates hardware backends 
  rm -rf "$build_path"
  
  cmake "${cmake_args[@]}"
  cmake --build "$build_path" --config Release --target llama-server llama-bench -j 2
}

update_source() {
  [[ -d "$LLAMA_ROOT/.git" ]] || die "not a Git checkout: $LLAMA_ROOT"
  git -C "$LLAMA_ROOT" pull --ff-only
}

clean_backend() {
  local backend="${1^^}"
  if [[ "$backend" == "ALL" ]]; then
    local item
    for item in CPU SYCL ROCM CUDA VULKAN HYBRID; do
      rm -rf -- "$LLAMA_ROOT/$(backend_build_dir "$item")"
    done
    return
  fi

  local build_dir
  build_dir="$(backend_build_dir "$backend")" || die "unknown backend: $backend"
  rm -rf -- "$LLAMA_ROOT/$build_dir"
}

clone() {
  cd ~/
  git clone https://github.com/ggml-org/llama.cpp
}

inspect_log() {
  local port="$1"
  is_uint "$port" || die "INSPECT requires a unsigned integer port"
  local log_file="${LLAMA_LOG_FILE:-/tmp/llama-$port.log}"
  [[ -f "$log_file" ]] || die "log file not found: $log_file"
  grep -Ei "offload|buffer|device|vram|memory|cpu|gpu|tensor|layer|slot|context|eval time|token" "$log_file" || true
}

kill_server() {
  local selector="${1^^}"
  if [[ "$selector" == "ALL" ]]; then
    pkill -TERM -x llama-server 2>/dev/null || true
    return
  fi

  is_uint "$selector" || die "KILL requires a port number or the explicit value ALL"
  local pid_file="${XDG_RUNTIME_DIR:-/tmp}/runllama-$selector.pid"
  [[ -f "$pid_file" ]] || die "PID file not found: $pid_file"

  local pid
  read -r pid < "$pid_file"
  is_uint "$pid" || die "invalid PID file: $pid_file"
  if [[ ! -r "/proc/$pid/cmdline" ]]; then
    rm -f -- "$pid_file"
    die "server process $pid is no longer running"
  fi

  local cmdline
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$cmdline" == *llama-server* && "$cmdline" == *"--port $selector"* ]] ||
    die "PID $pid is not the llama-server for port $selector"
  kill -TERM "$pid"
  rm -f -- "$pid_file"
}

if (($# == 0)); then
  usage
  exit 1
fi

case "${1^^}" in
  UPDATE)
    (($# == 1)) || die "usage: ./runllama.sh UPDATE"
    update_source
    exit 0
    ;;
  BUILD)
    (($# == 2)) || die "usage: ./runllama.sh BUILD <backend>"
    build_backend "$2"
    exit 0
    ;;
  LS)
    (($# == 2)) || die "usage: ./runllama.sh LS <backend>"
    LLAMA_BIN="$(backend_binary "$2")"
    require_binary "$LLAMA_BIN"
    exec "$LLAMA_BIN" --list-devices
    ;;
  INSPECT)
    (($# == 2)) || die "usage: ./runllama.sh INSPECT <port>"
    inspect_log "$2"
    exit 0
    ;;
  KILL)
    (($# == 2)) || die "usage: ./runllama.sh KILL <port|ALL>"
    kill_server "$2"
    exit 0
    ;;
  CLEAN)
    (($# == 2)) || die "usage: ./runllama.sh CLEAN <backend|ALL>"
    clean_backend "$2"
    exit 0
    ;;
  ClONE)
    (($# == 2)) || die "usage: ./runllama.sh CLONE"
    clone
    ;;
  HELP|-H|--HELP)
    usage
    exit 0
    ;;
esac

(($# >= 7 && $# <= 10)) || {
  usage >&2
  die "a launch requires 7 to 10 arguments"
}

PORT="$1"
TARGET="${2^^}"
DEVICE_SPEC="$3"
MODEL_SEL="${4,,}"
BATCH="$5"
PARALLEL="$6"
CONTEXT_PER_SLOT="${7,,}"
THREADS="${8:-${LLAMA_THREADS:-}}"
THREADS_BATCH="${LLAMA_THREADS_BATCH:-$THREADS}"
SPLIT_MODE="${9:-${LLAMA_SPLIT_MODE:-}}"
TENSOR_SPLIT="${10:-${LLAMA_TENSOR_SPLIT:-}}"

is_uint "$PORT" && ((PORT >= 1 && PORT <= 65535)) || die "port must be between 1 and 65535"
is_uint "$BATCH" && ((BATCH >= 1)) || die "batch size must be a positive integer"
is_uint "$PARALLEL" && ((PARALLEL >= 1)) || die "parallel must be a positive integer"
if [[ -n "$THREADS" ]]; then
  is_uint "$THREADS" && ((THREADS >= 1)) || die "threads must be a positive integer"
fi
if [[ -n "$THREADS_BATCH" ]]; then
  is_uint "$THREADS_BATCH" && ((THREADS_BATCH >= 1)) || die "LLAMA_THREADS_BATCH must be a positive integer"
fi
if [[ "$CONTEXT_PER_SLOT" != "auto" ]]; then
  is_uint "$CONTEXT_PER_SLOT" && ((CONTEXT_PER_SLOT >= 1)) || die "context per slot must be a positive integer or auto"
fi

BUILD_DIR="$(backend_build_dir "$TARGET")" || die "unknown backend: $TARGET"
LLAMA_BIN="${LLAMA_BIN:-$LLAMA_ROOT/$BUILD_DIR/bin/llama-server}"
require_binary "$LLAMA_BIN"

CUSTOM_FLAGS=()
DEVICE_ARGS=()
THREAD_ARGS=()

if [[ "$TARGET" == "SYCL" && -f /opt/intel/oneapi/setvars.sh ]]; then
  set +eu
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
  set -eu
  export ZES_ENABLE_PERSISTENT=1
  export ZES_ENABLE_SYSMAN=1
  unset SYCL_CACHE_PERSISTENT
fi

if [[ -n "$THREADS" ]]; then
  THREAD_ARGS+=( --threads "$THREADS" )
fi
if [[ -n "$THREADS_BATCH" ]]; then
  THREAD_ARGS+=( --threads-batch "$THREADS_BATCH" )
fi

if [[ "$TARGET" == "CPU" || "$TARGET" == "NONE" ]]; then
  DEVICE="none"
  DEVICE_ARGS=( --device none )
elif [[ "${DEVICE_SPEC,,}" == "all" ]]; then
  DEVICE="all devices exposed by the $TARGET build"
elif [[ "$TARGET" == "HYBRID" ]]; then
  [[ "$DEVICE_SPEC" == *[[:alpha:]]* ]] ||
    die "HYBRID devices must use full names, for example Vulkan0,Vulkan1,CUDA0"
  DEVICE="$DEVICE_SPEC"
  DEVICE_ARGS=( --device "$DEVICE" )
else
  PREFIX="$(backend_device_prefix "$TARGET")" || die "backend $TARGET does not accept GPU devices"
  IFS=',' read -r -a DEVICE_PARTS <<< "$DEVICE_SPEC"
  NORMALIZED_DEVICES=()
  for part in "${DEVICE_PARTS[@]}"; do
    [[ -n "$part" ]] || die "empty entry in device list: $DEVICE_SPEC"
    if is_uint "$part"; then
      NORMALIZED_DEVICES+=( "${PREFIX}${part}" )
    elif [[ "$part" == "${PREFIX}"* ]]; then
      NORMALIZED_DEVICES+=( "$part" )
    else
      die "device '$part' does not match backend $TARGET; use an index or a ${PREFIX} device name"
    fi
  done
  DEVICE="$(IFS=,; echo "${NORMALIZED_DEVICES[*]}")"
  DEVICE_ARGS=( --device "$DEVICE" )
fi

if [[ -z "$SPLIT_MODE" ]]; then
  if [[ "$DEVICE" == *,* || "${DEVICE_SPEC,,}" == "all" ]]; then
    SPLIT_MODE="layer"
  else
    SPLIT_MODE="none"
  fi
fi
case "$SPLIT_MODE" in
  none|layer|row|tensor) ;;
  *) die "split mode must be one of: none, layer, row, tensor" ;;
esac
CUSTOM_FLAGS+=( --split-mode "$SPLIT_MODE" )
if [[ -n "$TENSOR_SPLIT" ]]; then
  [[ "$TENSOR_SPLIT" =~ ^[0-9]+([.][0-9]+)?([,/][0-9]+([.][0-9]+)?)+$ ]] ||
    die "tensor split must contain at least two numeric proportions, for example 1,1"
  CUSTOM_FLAGS+=( --tensor-split "$TENSOR_SPLIT" )
fi

K_CACHE="f16"
V_CACHE="q8_0"
MODEL=""

case "$MODEL_SEL" in
  12b)
    MODEL="$MODEL_BASE/unsloth/gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
    K_CACHE="q8_0"
    CUSTOM_FLAGS+=(
        --jinja
    )
    ;;
  26b)
    MODEL="$MODEL_BASE/lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF/gemma-4-26B-A4B-it-QAT-Q4_0.gguf"
    CUSTOM_FLAGS+=(
        --jinja
        --n-cpu-moe 0
    )
    ;;
  27b)
    MODEL="$MODEL_BASE/unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf"
    CUSTOM_FLAGS+=(
      --jinja
      --chat-template-kwargs '{"preserve_thinking": true}'
      --n-cpu-moe 0
    )
    ;;
  27bmtp)
    MODEL="$MODEL_BASE/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf"
    CUSTOM_FLAGS+=(
      --jinja
      --chat-template-kwargs '{"preserve_thinking": true}'
      --n-cpu-moe 0
      --spec-type draft-mtp
      --spec-draft-n-max "${LLAMA_SPEC_DRAFT_N_MAX:-2}"
      --spec-draft-p-min "${LLAMA_SPEC_DRAFT_P_MIN:-0.0}"
    )
    ;;
  35b)
    MODEL="$MODEL_BASE/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    CUSTOM_FLAGS+=(
      --jinja
      --chat-template-kwargs '{"preserve_thinking": true}'
      -ngl 999
      #--flash-attn on
      #--chat-template raw
      #--no-mmap
      #--spec-draft-n-max 0
      #-fa on
      #--n-cpu-moe 0
    )
    ;;
  35bmtp)
    MODEL="$MODEL_BASE/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    CUSTOM_FLAGS+=(
      --jinja
      --n-cpu-moe 0
      --spec-type draft-mtp
      --spec-draft-n-max "${LLAMA_SPEC_DRAFT_N_MAX:-2}"
      --spec-draft-p-min "${LLAMA_SPEC_DRAFT_P_MIN:-0.0}"
    )
    ;;
  *) die "unknown model profile: $MODEL_SEL" ;;
esac

MODEL="${LLAMA_MODEL:-$MODEL}"
[[ -f "$MODEL" ]] || die "model file not found: $MODEL"
K_CACHE="${LLAMA_CACHE_K:-$K_CACHE}"
V_CACHE="${LLAMA_CACHE_V:-$V_CACHE}"

if [[ "$CONTEXT_PER_SLOT" == "auto" ]]; then
  CTX_TOTAL=0
  FIT_MODE="${LLAMA_FIT:-on}"
  CUSTOM_FLAGS+=(
    --gpu-layers auto
    --fit "$FIT_MODE"
    --fit-target "${LLAMA_FIT_TARGET:-1024}"
    --fit-ctx "${LLAMA_FIT_CTX:-4096}"
  )
else
  CTX_TOTAL=$((CONTEXT_PER_SLOT * PARALLEL))
  FIT_MODE="${LLAMA_FIT:-off}"
  CUSTOM_FLAGS+=( --gpu-layers all --fit "$FIT_MODE" )
  if [[ "$FIT_MODE" == "on" ]]; then
    CUSTOM_FLAGS+=(
      --fit-target "${LLAMA_FIT_TARGET:-1024}"
      --fit-ctx "${LLAMA_FIT_CTX:-4096}"
    )
  fi
fi
case "$FIT_MODE" in
  on|off) ;;
  *) die "LLAMA_FIT must be on or off" ;;
esac

UBATCH=$(((BATCH + 1) / 2))
MODEL_SIZE_MIB=$(( $(stat -c%s "$MODEL") / 1024 / 1024 ))
LLAMA_VERSION="$(git -C "$LLAMA_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
HOST="${LLAMA_HOST:-127.0.0.1}"
LOG_FILE="${LLAMA_LOG_FILE:-/tmp/llama-$PORT.log}"
PID_FILE="${XDG_RUNTIME_DIR:-/tmp}/runllama-$PORT.pid"
if [[ -z "${LLAMA_API_KEY:-}" && "$HOST" != "127.0.0.1" && "$HOST" != "localhost" && "$HOST" != "::1" ]]; then
  echo "Warning: server is listening beyond localhost without LLAMA_API_KEY." >&2
fi
case "${2^^}" in
  CPU|NONE)
    ;;
  SYCL)
    export ZES_ENABLE_SYSMAN=1
    ;;
  ROCM)
    ;;
  CUDA)
    ;;
  VULKAN)
    ;;
  HYBRID)
    ;;
esac

echo "Launching $LLAMA_BIN"
echo "  llama.cpp revision: $LLAMA_VERSION"
echo "  Model: $MODEL (${MODEL_SIZE_MIB} MiB)"
echo "  Backend/devices: $TARGET / $DEVICE"
echo "  Split: $SPLIT_MODE${TENSOR_SPLIT:+ ($TENSOR_SPLIT)}"
echo "  Port/host: $PORT / $HOST"
echo "  Batch/ubatch: $BATCH / $UBATCH"
echo "  Parallel slots: $PARALLEL"
echo "  CPU threads: ${THREADS:-automatic} / batch ${THREADS_BATCH:-automatic}"
if [[ "$CTX_TOTAL" == 0 ]]; then
  echo "  Context: model default, adjusted by fitter (minimum ${LLAMA_FIT_CTX:-4096})"
else
  echo "  Context: $CTX_TOTAL total ($CONTEXT_PER_SLOT per slot)"
fi
echo "  KV cache: K=$K_CACHE V=$V_CACHE"
echo "  Fit: $FIT_MODE"
echo "  Log: $LOG_FILE"

printf '%s\n' "$$" > "$PID_FILE"
exec "$LLAMA_BIN" \
  --model "$MODEL" \
  --port "$PORT" \
  --host "$HOST" \
  --ctx-size "$CTX_TOTAL" \
  "${DEVICE_ARGS[@]}" \
  --cache-type-k "$K_CACHE" \
  --cache-type-v "$V_CACHE" \
  --parallel "$PARALLEL" \
  --batch-size "$BATCH" \
  --ubatch-size "$UBATCH" \
  "${THREAD_ARGS[@]}" \
  --temp "${LLAMA_TEMP:-0.85}" \
  --min-p "${LLAMA_MIN_P:-0.05}" \
  --presence-penalty "${LLAMA_PRESENCE_PENALTY:-0.0}" \
  --dry-multiplier "${LLAMA_DRY_MULTIPLIER:-0.0}" \
  --dry-base "${LLAMA_DRY_BASE:-1.75}" \
  --dry-allowed-length "${LLAMA_DRY_ALLOWED_LENGTH:-2}" \
  --dry-penalty-last-n "${LLAMA_DRY_LAST_N:-4096}" \
  --samplers "penalties;dry;min_p;temperature" \
  --flash-attn "${LLAMA_FLASH_ATTN:-on}" \
  --log-file "$LOG_FILE" \
  --cont-batching \
  "${CUSTOM_FLAGS[@]}"
