#!/usr/bin/env bash
# =============================================================================
# Shared Utilities Library for Stable Diffusion Multi-GPU Launchers
# =============================================================================
# This library contains common functions used by both:
#   - run_stablediffusion.sh (production, Ubuntu-only)
#   - run_stablediffusion_multidistro.sh (experimental, multi-distro)
#
# Include this file in launcher scripts:
#   source "$(dirname "$0")/lib/common.sh"

set -euo pipefail

# =============================================================================
# TERMINAL HELPERS & LOGGING
# =============================================================================
# Color codes for consistent visual output across all launchers

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Logging functions -- all output goes through these
info()    { echo -e "${CYAN}[INFO]${NC}      $*"; }
success() { echo -e "${GREEN}[OK]${NC}        $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}     $*"; }
error()   { echo -e "${RED}[ERROR]${NC}    $*" >&2; exit 1; }
section() { echo -e "\n${BOLD}+- $* ${NC}"; }
remark()  { echo -e "   ${CYAN}↳${NC} $*"; }

print_banner() {
  echo -e "${BOLD}"
  echo "  ╔══════════════════════════════════════════════════════════════════╗"
  echo "  ║   Stable Diffusion WebUI -- Universal Multi-GPU Launcher v3.0     ║"
  echo "  ║   NVIDIA * AMD * Intel * CPU  |  Smart Router  |  Multi-Distro   ║"
  echo "  ╚══════════════════════════════════════════════════════════════════╝"
  echo -e "${NC}"
}

# =============================================================================
# PYTHON DETECTION & INSTALLATION
# =============================================================================
# Stable Diffusion's dependency tree works best with Python 3.10.
# We try versions in preference order and fall back to system python3.

detect_python() {
  info "Detecting Python interpreter..."
  # Try preferred versions first; SD has known compatibility issues with 3.12+
  for ver in 3.10 3.11 3.9 3.12; do
    if command -v "python${ver}" &>/dev/null; then
      PYTHON_BIN="python${ver}"
      success "Found Python: $($PYTHON_BIN --version) at $(command -v $PYTHON_BIN)"
      return
    fi
  done
  # Last resort: whatever python3 points to
  if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
    warn "Using system python3: $($PYTHON_BIN --version) -- 3.10 is strongly recommended"
    return
  fi
  error "No Python 3 found.\n  Run: sudo apt install python3.10 python3.10-venv python3.10-dev"
}

ensure_python310_apt() {
  # If 3.10 is already present, nothing to do
  command -v python3.10 &>/dev/null && return

  info "Python 3.10 not found -- attempting to install via apt..."
  remark "Python 3.10 is required because some SD dependencies (clip, triton, xformers)"
  remark "are not yet published as wheels for Python 3.12, causing build-from-source failures."
  sudo apt update -qq
  sudo apt install -y python3.10 python3.10-venv python3.10-dev 2>/dev/null \
    || { warn "Could not install Python 3.10 -- continuing with $PYTHON_BIN"; return; }
  command -v python3.10 &>/dev/null && PYTHON_BIN="python3.10"
  success "Python 3.10 installed"
}

# =============================================================================
# NVIDIA VERSION RESOLUTION
# =============================================================================
# Maps NVIDIA driver version to compatible CUDA tag and PyTorch versions.
# This fixes the "torchaudio X requires torch==Y but you have Z" conflict.

resolve_nvidia_versions() {
  local cuda_tag="$1"

  # Each row is a verified compatible set:
  #  CUDA tag | torch  | torchvision | torchaudio
  #  ---------|--------|-------------|------------
  #  cu124    | 2.6.0  | 0.21.0      | 2.6.0       ← driver 560+
  #  cu121    | 2.5.1  | 0.20.1      | 2.5.1       ← driver 525-559
  #  cu118    | 2.3.1  | 0.18.1      | 2.3.1       ← driver 520-524
  #  cu117    | 2.0.1  | 0.15.2      | 2.0.2       ← driver 450-519
  #  cpu      | 2.0.1  | 0.15.2      | 2.0.2       ← no GPU / very old driver

  declare -A TORCH_VER=([cu124]="2.6.0" [cu121]="2.5.1" [cu118]="2.3.1" [cu117]="2.0.1" [cpu]="2.0.1")
  declare -A TV_VER=(   [cu124]="0.21.0" [cu121]="0.20.1" [cu118]="0.18.1" [cu117]="0.15.2" [cpu]="0.15.2")
  declare -A TA_VER=(   [cu124]="2.6.0" [cu121]="2.5.1" [cu118]="2.3.1" [cu117]="2.0.2" [cpu]="2.0.2")

  NVIDIA_PYTORCH_VERSION="${TORCH_VER[$cuda_tag]:-2.5.1}"
  NVIDIA_TORCHVISION_VERSION="${TV_VER[$cuda_tag]:-0.20.1}"
  NVIDIA_TORCHAUDIO_VERSION="${TA_VER[$cuda_tag]:-2.5.1}"
}

# =============================================================================
# VALIDATION HELPERS
# =============================================================================

validate_port() {
  local port=$1
  if ! [[ "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 1024 ] || [ "$port" -gt 65535 ]; then
    error "Invalid port number: $port (must be 1024-65535)"
  fi
}

validate_directory() {
  local path=$1
  if [ -z "$path" ]; then
    error "Directory path cannot be empty"
  fi
  if ! mkdir -p "$path" 2>/dev/null; then
    error "Cannot create or write to directory: $path"
  fi
}

# Export all functions so they're available to sourcing scripts
export -f info success warn error section remark print_banner
export -f detect_python ensure_python310_apt resolve_nvidia_versions
export -f validate_port validate_directory
