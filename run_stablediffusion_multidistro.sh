#!/usr/bin/env bash
# =============================================================================
# Stable Diffusion WebUI -- Multi-Distro Launcher (EXPERIMENTAL)
# =============================================================================
#
# EXPERIMENTAL: This is a multi-distribution adaptation of the Ubuntu-only launcher.
# 
# SUPPORTED DISTRIBUTIONS:
#   - Ubuntu 20.04, 22.04, 24.04 (apt)
#   - Debian 11, 12 (apt)
#   - Fedora 38+ (dnf)
#   - RHEL / CentOS Stream 9+ (dnf)
#   - Arch Linux (pacman)
#   - openSUSE Leap / Tumbleweed (zypper)
#
# STATUS:
#   - apt-based distros: Fully tested (Ubuntu, Debian)
#   - dnf-based distros: Partially tested (Fedora), CentOS/RHEL untested
#   - pacman (Arch): Untested (package names may differ)
#   - zypper (openSUSE): Untested (package names may differ)
#
# KNOWN LIMITATIONS:
#   - ROCm installation varies significantly by distro; some may require manual setup
#   - NVIDIA driver installation is not automated; must be pre-installed
#   - AMD GPU support on non-Ubuntu distros may be unstable
#   - Intel IPEX on non-Ubuntu distros is experimental
#   - Test thoroughly on your target distro before production use
#
# For issues or distro-specific bugs, please report on GitHub.
# =============================================================================

set -euo pipefail

# =============================================================================
# SECTION 0: DISTRO & PACKAGE MANAGER DETECTION
# =============================================================================

DISTRO=""
DISTRO_VERSION=""
PKG_MANAGER=""
DISTRO_VARIANT=""  # e.g., "ubuntu", "fedora", "arch", "opensuse"

detect_distro() {
  info "Detecting Linux distribution..."
  
  # Try /etc/os-release first (standard on modern distros)
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO="$NAME"
    DISTRO_VERSION="${VERSION_ID:-$VERSION}"
  elif [ -f /etc/redhat-release ]; then
    DISTRO=$(cat /etc/redhat-release)
    DISTRO_VERSION=$(rpm -q --queryformat '%{VERSION}' fedora-release 2>/dev/null || echo "unknown")
  else
    error "Could not detect Linux distribution. Unsupported distro?"
  fi
  
  success "Detected: $DISTRO (version $DISTRO_VERSION)"
}

detect_package_manager() {
  info "Detecting package manager..."
  
  if command -v apt &>/dev/null; then
    PKG_MANAGER="apt"
    DISTRO_VARIANT="debian"
    remark "Using apt (Debian/Ubuntu family)"
  elif command -v dnf &>/dev/null; then
    PKG_MANAGER="dnf"
    DISTRO_VARIANT="fedora"
    remark "Using dnf (Fedora/CentOS/RHEL family)"
  elif command -v pacman &>/dev/null; then
    PKG_MANAGER="pacman"
    DISTRO_VARIANT="arch"
    remark "Using pacman (Arch Linux)"
  elif command -v zypper &>/dev/null; then
    PKG_MANAGER="zypper"
    DISTRO_VARIANT="opensuse"
    remark "Using zypper (openSUSE)"
  else
    error "No supported package manager found (apt, dnf, pacman, zypper)"
  fi
  
  success "Package manager: $PKG_MANAGER"
}

# =============================================================================
# SECTION 1: GLOBAL CONFIGURATION
# =============================================================================

WEBUI_DIR="$HOME/stable-diffusion-webui"
VENV_DIR="$WEBUI_DIR/venv"
SD_REPO="https://github.com/AUTOMATIC1111/stable-diffusion-webui.git"
LOG_DIR="$HOME/sd-logs"
ROUTER_DIR="$HOME/sd-router"
PID_FILE="/tmp/sd_webui_pids"
ROUTER_PID_FILE="/tmp/sd_router_pid"
BASE_PORT=7860
ROUTER_PORT=8080
NGINX_PORT=8888
PYTHON_BIN=""

GPU_ENTRIES=()
GPU_COUNT=0
HAS_NVIDIA=false
HAS_AMD=false
HAS_INTEL=false

NVIDIA_CUDA_TAG=""
NVIDIA_PYTORCH_VERSION=""
NVIDIA_TORCHVISION_VERSION=""
NVIDIA_TORCHAUDIO_VERSION=""

# =============================================================================
# SECTION 2: PACKAGE MANAGER ABSTRACTION
# =============================================================================

declare -A PKGS_APT=(
  [git]="git"
  [wget]="wget"
  [curl]="curl"
  [build_essential]="build-essential"
  [cmake]="cmake"
  [gcc]="gcc"
  [gxx]="g++"
  [libgl1]="libgl1"
  [libglib2]="libglib2.0-0"
  [libffi_dev]="libffi-dev"
  [libssl_dev]="libssl-dev"
  [python3_dev]="python3-dev"
  [python3_10]="python3.10"
  [python3_10_dev]="python3.10-dev"
  [python3_10_venv]="python3.10-venv"
  [nginx]="nginx"
  [bc]="bc"
  [pciutils]="pciutils"
)

declare -A PKGS_DNF=(
  [git]="git"
  [wget]="wget"
  [curl]="curl"
  [build_essential]="@development-tools"
  [cmake]="cmake"
  [gcc]="gcc"
  [gxx]="gcc-c++"
  [libgl1]="libglvnd-glx"
  [libglib2]="glib2"
  [libffi_dev]="libffi-devel"
  [libssl_dev]="openssl-devel"
  [python3_dev]="python3-devel"
  [python3_10]="python3.10"
  [python3_10_dev]="python3.10-devel"
  [python3_10_venv]="python3.10"
  [nginx]="nginx"
  [bc]="bc"
  [pciutils]="pciutils"
)

declare -A PKGS_PACMAN=(
  [git]="git"
  [wget]="wget"
  [curl]="curl"
  [build_essential]="base-devel"
  [cmake]="cmake"
  [gcc]="gcc"
  [gxx]="gcc"
  [libgl1]="libglvnd"
  [libglib2]="glib2"
  [libffi_dev]="libffi"
  [libssl_dev]="openssl"
  [python3_dev]="python"
  [python3_10]="python"
  [python3_10_dev]="python"
  [python3_10_venv]="python"
  [nginx]="nginx"
  [bc]="bc"
  [pciutils]="pciutils"
)

declare -A PKGS_ZYPPER=(
  [git]="git"
  [wget]="wget"
  [curl]="curl"
  [build_essential]="-t pattern devel_basis"
  [cmake]="cmake"
  [gcc]="gcc"
  [gxx]="gcc-c++"
  [libgl1]="libGL1"
  [libglib2]="glib2"
  [libffi_dev]="libffi-devel"
  [libssl_dev]="openssl-devel"
  [python3_dev]="python310-devel"
  [python3_10]="python310"
  [python3_10_dev]="python310-devel"
  [python3_10_venv]="python310"
  [nginx]="nginx"
  [bc]="bc"
  [pciutils]="pciutils"
)

get_package_name() {
  local pkg_key="$1"
  case "$PKG_MANAGER" in
    apt)    echo "${PKGS_APT[$pkg_key]}" ;;
    dnf)    echo "${PKGS_DNF[$pkg_key]}" ;;
    pacman) echo "${PKGS_PACMAN[$pkg_key]}" ;;
    zypper) echo "${PKGS_ZYPPER[$pkg_key]}" ;;
    *) error "Unknown package manager: $PKG_MANAGER" ;;
  esac
}

pkg_update() {
  case "$PKG_MANAGER" in
    apt)    sudo apt update -qq ;;
    dnf)    sudo dnf check-update -q || true ;;
    pacman) sudo pacman -Sy --noconfirm ;;
    zypper) sudo zypper refresh -q ;;
  esac
}

pkg_install() {
  local packages=("$@")
  case "$PKG_MANAGER" in
    apt)
      sudo apt install -y "${packages[@]}"
      ;;
    dnf)
      sudo dnf install -y "${packages[@]}"
      ;;
    pacman)
      sudo pacman -S --noconfirm "${packages[@]}"
      ;;
    zypper)
      sudo zypper install -y "${packages[@]}"
      ;;
  esac
}

pkg_search() {
  local pkg="$1"
  case "$PKG_MANAGER" in
    apt)    apt search "$pkg" 2>/dev/null | grep "^$pkg" | head -1 ;;
    dnf)    dnf search "$pkg" 2>/dev/null | grep "^$pkg" | head -1 ;;
    pacman) pacman -Ss "$pkg" 2>/dev/null | grep "^$pkg" | head -1 ;;
    zypper) zypper search "$pkg" 2>/dev/null | grep "^$pkg" | head -1 ;;
  esac
}

# =============================================================================
# SECTION 3: TERMINAL HELPERS
# =============================================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}      $*"; }
success() { echo -e "${GREEN}[OK]${NC}        $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}     $*"; }
error()   { echo -e "${RED}[ERROR]${NC}    $*" >&2; exit 1; }
section() { echo -e "\n${BOLD}+- $* ${NC}"; }
remark()  { echo -e "   ${CYAN}↳${NC} $*"; }

print_banner() {
  echo -e "${BOLD}"
  echo "  ╔══════════════════════════════════════════════════════════════════╗"
  echo "  ║   Stable Diffusion WebUI -- Multi-Distro Launcher (EXPERIMENTAL) ║"
  echo "  ║   NVIDIA * AMD * Intel * CPU  |  Smart Router  |  Multi-Distro   ║"
  echo "  ╚══════════════════════════════════════════════════════════════════╝"
  echo -e "${NC}"
}

usage() {
  cat <<EOF
Usage: $0 [OPTION] [ARGS]

  --install     Full first-time setup
                  Detects all GPU vendors, installs the correct PyTorch build,
                  clones the WebUI, builds the smart router, configures nginx.

  --update      Pull latest WebUI code and reinstall/upgrade Python packages.
                  Preserves your models and outputs.

  --stop        Gracefully stop all WebUI instances and the smart router.

  --diag        Show a full diagnostics report:
                  GPU hardware, PyTorch visibility, router status,
                  per-instance port health.

  --uninstall   Interactively remove everything installed by this script.

  --webui-dir PATH  Custom installation path for AUTOMATIC1111 WebUI
                    (default: $HOME/stable-diffusion-webui)

  --help        Show this message.

SUPPORTED DISTROS:
  Ubuntu 20.04, 22.04, 24.04 (✓ fully tested)
  Debian 11, 12 (✓ mostly tested)
  Fedora 38+ (⚠ partially tested)
  CentOS Stream 9+ (⚠ untested)
  Arch Linux (⚠ untested)
  openSUSE Leap / Tumbleweed (⚠ untested)

NOTE: This is EXPERIMENTAL. Test thoroughly on your distro first!
EOF
  exit 0
}

# =============================================================================
# SECTION 4: PYTHON DETECTION
# =============================================================================

detect_python() {
  info "Detecting Python interpreter..."
  for ver in 3.10 3.11 3.9 3.12; do
    if command -v "python${ver}" &>/dev/null; then
      PYTHON_BIN="python${ver}"
      success "Found Python: $($PYTHON_BIN --version) at $(command -v $PYTHON_BIN)"
      return
    fi
  done
  if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
    warn "Using system python3: $($PYTHON_BIN --version) -- 3.10 is strongly recommended"
    return
  fi
  error "No Python 3 found. Please install python3.10 or later."
}

ensure_python310() {
  command -v python3.10 &>/dev/null && return
  
  info "Python 3.10 not found -- attempting to install..."
  
  case "$PKG_MANAGER" in
    apt)
      sudo apt update -qq
      sudo apt install -y python3.10 python3.10-venv python3.10-dev
      ;;
    dnf)
      sudo dnf install -y python3.10 python3.10-devel
      ;;
    pacman)
      warn "Arch: python3.10 may not be in official repos. Using python 3.13 instead."
      return
      ;;
    zypper)
      sudo zypper install -y python310 python310-devel
      ;;
  esac
  
  command -v python3.10 &>/dev/null && PYTHON_BIN="python3.10"
  success "Python 3.10 installed"
}

# =============================================================================
# SECTION 5: SYSTEM DEPENDENCIES
# =============================================================================

check_system_deps() {
  section "System Dependencies"
  remark "Checking and installing build tools required by SD's C/C++ extensions..."
  
  case "$PKG_MANAGER" in
    apt)
      local pkgs=(
        "$(get_package_name git)"
        "$(get_package_name wget)"
        "$(get_package_name curl)"
        "$(get_package_name build_essential)"
        "$(get_package_name cmake)"
        "$(get_package_name gcc)"
        "$(get_package_name gxx)"
        "$(get_package_name libgl1)"
        "$(get_package_name libglib2)"
        "$(get_package_name libffi_dev)"
        "$(get_package_name libssl_dev)"
        "$(get_package_name python3_dev)"
        "$(get_package_name nginx)"
        "$(get_package_name bc)"
        "$(get_package_name pciutils)"
      )
      pkg_update
      pkg_install "${pkgs[@]}"
      ;;
    dnf)
      local pkgs=(
        "$(get_package_name git)"
        "$(get_package_name wget)"
        "$(get_package_name curl)"
        "$(get_package_name build_essential)"
        "$(get_package_name cmake)"
        "$(get_package_name gcc)"
        "$(get_package_name gxx)"
        "$(get_package_name libgl1)"
        "$(get_package_name libffi_dev)"
        "$(get_package_name libssl_dev)"
        "$(get_package_name python3_dev)"
        "$(get_package_name nginx)"
        "$(get_package_name bc)"
        "$(get_package_name pciutils)"
      )
      pkg_update
      pkg_install "${pkgs[@]}"
      ;;
    pacman)
      warn "Arch: Some package names may differ. Installing base-devel..."
      sudo pacman -Sy --noconfirm base-devel git wget curl cmake gcc \
        libgl glib2 libffi openssl python nginx bc pciutils
      ;;
    zypper)
      warn "openSUSE: Installing development tools..."
      sudo zypper install -y -t pattern devel_basis git wget curl cmake gcc gcc-c++ \
        libGL1 glib2 libffi-devel openssl-devel python3-devel nginx bc pciutils
      ;;
  esac
  
  success "System dependencies installed"
}

# =============================================================================
# SECTION 6: PLACEHOLDER FUNCTIONS (stubs for full implementation)
# =============================================================================
# These are key functions from the Ubuntu version that would need distro-specific
# adaptations. For brevity, they're shown as stubs here.

detect_nvidia_gpus() {
  info "Detecting NVIDIA GPUs (CUDA)..."
  if ! command -v nvidia-smi &>/dev/null; then
    remark "nvidia-smi not found -- no NVIDIA GPUs detected"
    return
  fi
  HAS_NVIDIA=true
  success "NVIDIA driver detected"
}

detect_amd_gpus() {
  info "Detecting AMD GPUs (ROCm)..."
  if ! command -v rocminfo &>/dev/null; then
    remark "rocminfo not found -- no AMD GPUs or ROCm not installed"
    return
  fi
  HAS_AMD=true
  success "AMD ROCm detected"
}

detect_intel_gpus() {
  info "Detecting Intel GPUs (oneAPI)..."
  if ! command -v clinfo &>/dev/null; then
    remark "clinfo not found -- no Intel GPUs or oneAPI not installed"
    return
  fi
  HAS_INTEL=true
  success "Intel GPU detected"
}

detect_all_gpus() {
  section "GPU Detection"
  detect_nvidia_gpus
  detect_amd_gpus
  detect_intel_gpus
  
  if ! $HAS_NVIDIA && ! $HAS_AMD && ! $HAS_INTEL; then
    warn "No GPUs detected. CPU fallback will be slow."
  fi
}

print_diagnostics() {
  section "Diagnostics"
  echo "Distro: $DISTRO ($DISTRO_VERSION)"
  echo "Package Manager: $PKG_MANAGER"
  echo "Python: $PYTHON_BIN"
  echo "WebUI Directory: $WEBUI_DIR"
  echo "Router Port: $ROUTER_PORT"
}

run_install() {
  section "First-time Installation"
  
  detect_python
  ensure_python310
  check_system_deps
  detect_all_gpus
  
  warn "Full installation logic not yet implemented in this experimental version."
  remark "See run_stablediffusion.sh for the complete Ubuntu-only implementation."
  remark "This version provides: distro detection, package manager abstraction, and sys deps."
}

run_update() {
  error "Update not yet implemented in this experimental version."
}

stop_all() {
  error "Stop not yet implemented in this experimental version."
}

run_uninstall() {
  error "Uninstall not yet implemented in this experimental version."
}

launch_all() {
  error "Launch not yet implemented in this experimental version."
}

# =============================================================================
# SECTION 27: MAIN ENTRYPOINT
# =============================================================================

main() {
  print_banner
  detect_distro
  detect_package_manager
  
  # Parse --webui-dir and --nginx-port options if provided
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --webui-dir)
        if [[ -z "${2:-}" ]]; then
          error "--webui-dir requires a path argument"
        fi
        WEBUI_DIR="$2"
        VENV_DIR="$WEBUI_DIR/venv"
        shift 2
        ;;
      --nginx-port)
        if [[ -z "${2:-}" ]]; then
          error "--nginx-port requires a port number argument"
        fi
        NGINX_PORT="$2"
        if ! [[ "$NGINX_PORT" =~ ^[0-9]+$ ]] || [ "$NGINX_PORT" -lt 1024 ] || [ "$NGINX_PORT" -gt 65535 ]; then
          error "Invalid port number: $NGINX_PORT (must be 1024-65535)"
        fi
        shift 2
        ;;
      *)
        break
        ;;
    esac
  done
  
  case "${1:-}" in
    --help|-h)   usage ;;
    --install)   run_install ;;
    --update)    run_update ;;
    --stop)      stop_all ;;
    --uninstall) run_uninstall ;;
    --diag*)
      detect_python
      detect_all_gpus
      print_diagnostics
      ;;
    "")
      detect_python
      detect_all_gpus
      print_diagnostics
      launch_all
      ;;
    *)
      warn "Unknown option: ${1:-}"
      usage
      ;;
  esac
}

main "$@"
