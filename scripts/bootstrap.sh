#!/usr/bin/env bash
# One-shot setup for a fresh machine: venv, Python deps, Colibrí engine (CUDA build if possible), .env, hardware scan.
#
#   ./scripts/bootstrap.sh [--dev] [--hf-token TOKEN] [--no-engine] [--no-gpu-build] [--skip-torch] [--venv DIR]
#
# Safe to re-run; every step is skipped when already done.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$PWD
VENV=.venv
DEV=0 ENGINE=1 GPU_BUILD=1 TORCH=1 HF_TOKEN_ARG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --dev) DEV=1 ;;
    --no-engine) ENGINE=0 ;;
    --no-gpu-build) GPU_BUILD=0 ;;
    --skip-torch) TORCH=0 ;;
    --hf-token) HF_TOKEN_ARG=${2:?--hf-token needs a value}; shift ;;
    --venv) VENV=${2:?--venv needs a value}; shift ;;
    -h|--help) sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$*" >&2; }

# 1. Python >= 3.10
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then PY=$c; break; fi
done
[[ -n $PY ]] || { echo "Python >= 3.10 is required" >&2; exit 1; }
say "Using $($PY --version) ($PY)"

# 2. Virtual environment
if [[ ! -x $VENV/bin/python ]]; then
  say "Creating virtual environment in $VENV"
  "$PY" -m venv "$VENV"
fi
VPY=$ROOT/$VENV/bin/python
if command -v uv >/dev/null; then PIP=(uv pip install --python "$VPY"); else
  "$VPY" -m pip install --quiet --upgrade pip
  PIP=("$VPY" -m pip install --quiet)
fi

# 3. torch: only `coli convert` uses it, so the much smaller CPU wheel is enough on Linux x86_64.
if [[ $TORCH == 1 ]] && ! "$VPY" -c 'import torch' 2>/dev/null; then
  if [[ $(uname -s)-$(uname -m) == Linux-x86_64 ]]; then
    say "Installing CPU-only torch (used for model conversion)"
    "${PIP[@]}" torch --index-url "${COLIOLLAMA_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
  else
    say "Installing torch"
    "${PIP[@]}" torch
  fi
fi

# 4. ColiOllama itself
say "Installing ColiOllama and dependencies"
if [[ $DEV == 1 ]]; then "${PIP[@]}" -e ".[dev]"; else "${PIP[@]}" -e .; fi
COLI=("$VPY" -m coliollama)

# 5. .env for the Hugging Face token (gated repos such as meta-llama/*)
if [[ ! -f .env ]]; then
  cp .env.example .env
  say "Created .env"
fi
if [[ -n $HF_TOKEN_ARG ]]; then
  if grep -q '^HF_TOKEN=' .env; then sed -i.bak "s|^HF_TOKEN=.*|HF_TOKEN=$HF_TOKEN_ARG|" .env && rm -f .env.bak
  else echo "HF_TOKEN=$HF_TOKEN_ARG" >> .env; fi
  chmod 600 .env
  say "Stored HF token in .env"
fi

# 6. Colibrí engine: download the latest release, and build CUDA engines when an NVIDIA GPU + nvcc exist.
if [[ $ENGINE == 1 ]]; then
  if command -v nvidia-smi >/dev/null && ! { command -v nvcc >/dev/null || [[ -x /usr/local/cuda/bin/nvcc || -n ${CUDA_HOME:-} ]]; }; then
    warn "NVIDIA GPU found but no CUDA toolkit (nvcc): the engine will be CPU-only. Install the CUDA toolkit and re-run."
  fi
  if [[ $GPU_BUILD == 1 ]] && command -v nvidia-smi >/dev/null && ! command -v make >/dev/null; then
    warn "'make' and a C compiler are needed for the CUDA build (e.g. apt install build-essential)"
  fi
  say "Installing/refreshing the Colibrí engine"
  FLAG=--gpu-build; [[ $GPU_BUILD == 1 ]] || FLAG=--no-gpu-build
  if ! "${COLI[@]}" update "$FLAG"; then
    warn "engine install failed; set COLIBRI_HOME to an existing Colibrí checkout, or retry with: coliollama update"
  fi
fi

# 7. What can this machine run?
say "Hardware scan"
"${COLI[@]}" scan || true

cat <<MSG

Done. Activate the environment and go:

  source $VENV/bin/activate
  coliollama pull <model from the list above>
  coliollama run  <model>
MSG
