#!/usr/bin/env bash
# =============================================================================
# Stable Diffusion WebUI -- Universal Multi-GPU Launcher v3.0
# =============================================================================
#
# OVERVIEW
# --------
# This script installs, configures, and launches AUTOMATIC1111 Stable Diffusion
# WebUI across any combination of GPUs in a single machine. It auto-detects
# every GPU present, selects the correct PyTorch build and launch flags for
# each GPU architecture, and starts one independent WebUI process per GPU.
#
# A smart Python router sits in front of all WebUI instances and routes each
# generation request to the most suitable GPU based on resolution, free VRAM,
# and current queue depth.
#
# SUPPORTED GPU VENDORS & USE CASES
# ----------------------------------
#
#   NVIDIA (CUDA)
#   -------------
#   * Volta    (V100, compute 7.0)
#       - FP16 is broken on Linux for V100 -> forced FP32 (--precision full --no-half)
#       - No xformers support
#       - Best for: large-batch jobs, SDXL at high resolution (32GB model)
#       - PyTorch: pinned matched set from download.pytorch.org/whl/cuXXX
#
#   * Turing   (RTX 2xxx, Quadro RTX, T4, compute 7.5)
#       - Full FP16 + xformers support -> fastest inference per watt
#       - Best for: SD 1.5, standard resolution, high throughput
#       - xformers reduces VRAM usage by ~30% and speeds up attention
#
#   * Ampere   (RTX 3xxx, A100, A10, compute 8.x)
#       - FP16 + xformers + BF16 capable
#       - Best for: SDXL, ControlNet, high-res, batched inference
#
#   * Ada      (RTX 4xxx, compute 8.9)
#       - FP16 + xformers + FP8 capable
#       - Best for: fastest consumer inference, SDXL at 2K+
#
#   * Hopper   (H100, compute 9.0)
#       - FP16 + FP8 + xformers + transformer engine
#       - Best for: datacenter batch workloads, fine-tuning
#
#   * Blackwell (RTX 5xxx, compute 10.x)
#       - FP16 + FP4 + xformers
#       - Best for: next-gen consumer, maximum throughput
#
#   * Pascal   (GTX 10xx, P100, compute 6.x)
#       - FP16 ok (no xformers), decent for SD 1.5
#       - Best for: older consumer cards, still usable for 512x512
#
#   * Maxwell  (GTX 9xx, compute 5.x) and older
#       - FP16 unreliable -> force FP32
#       - Best for: last resort, 512x512 only
#
#   AMD (ROCm)
#   ----------
#   * RDNA2/RDNA3 (RX 6xxx/7xxx) -- best ROCm support
#       - Uses PyTorch ROCm build from download.pytorch.org/whl/rocmX.Y
#       - HSA_OVERRIDE_GFX_VERSION may be needed for some cards
#       - Best for: SD 1.5 at standard resolution
#       - No xformers (ROCm xformers is experimental)
#
#   * Vega / older AMD
#       - ROCm support varies -- may need older ROCm version
#       - Use --precision full --no-half for stability
#
#   Intel (IPEX / XPU)
#   ------------------
#   * Arc (A770, A750, A380) -- best Intel support
#       - Uses Intel Extension for PyTorch (IPEX) + XPU torch build
#       - Level Zero driver required (ZE_AFFINITY_MASK for device selection)
#       - Best for: SD 1.5 at 512x512, still maturing
#
#   * Ponte Vecchio (Data Center GPU Max)
#       - Same IPEX stack, better FP16 support
#
#   CPU Fallback
#   ------------
#   * Used when no GPU is detected or all GPU installs fail
#   * Extremely slow (minutes per image) -- for testing only
#   * Uses --use-cpu all --precision full --no-half
#
# MIXED GPU COMBINATIONS
# ----------------------
#   NVIDIA only         -> CUDA PyTorch, xformers where supported, one instance per GPU
#   AMD only            -> ROCm PyTorch, one instance per AMD GPU
#   Intel only          -> IPEX PyTorch, one instance per Intel GPU
#   NVIDIA + AMD        -> NVIDIA CUDA PyTorch installed (shared venv limitation);
#                         AMD GPU launched with HIP env vars; ROCm PyTorch not
#                         installed in this case (two PyTorch builds can't share a venv)
#   NVIDIA + Intel      -> Same: NVIDIA CUDA PyTorch used; Intel GPU gets IPEX layer
#   AMD + Intel         -> ROCm PyTorch, Intel GPU falls back to CPU mode in same venv
#   All three vendors   -> NVIDIA takes priority; AMD and Intel get best-effort support
#
# SMART ROUTER
# ------------
# A FastAPI service (~/sd-router/router.py) listens on port 8080 and:
#   1. Inspects each /sdapi/v1/txt2img and /sdapi/v1/img2img request payload
#   2. Reads requested width, height, batch_size to estimate VRAM needed
#   3. Queries nvidia-smi for real-time free VRAM on each NVIDIA GPU
#   4. Tracks queue depth per WebUI instance
#   5. Routes high-resolution requests to high-VRAM GPUs (V100 > RTX 5000)
#   6. Routes standard requests to fastest GPU (RTX with xformers wins)
#   7. Falls back to least-busy GPU if best choice is unavailable
#
# PORTS
# -----
#   :7860  GPU 0 (direct WebUI access)
#   :7861  GPU 1 (direct WebUI access)
#   :7862  GPU 2 (direct WebUI access)
#   :8080  Smart router (use this for API calls and the browser)
#   :80    nginx (optional public-facing reverse proxy -> router)
#
# USAGE
# -----
#   chmod +x run_stablediffusion.sh
#   ./run_stablediffusion.sh --install     # first-time setup
#   ./run_stablediffusion.sh               # launch everything
#   ./run_stablediffusion.sh --stop        # stop all instances + router
#   ./run_stablediffusion.sh --update      # pull latest + reinstall deps
#   ./run_stablediffusion.sh --diag        # show GPU + PyTorch + router status
#   ./run_stablediffusion.sh --uninstall   # remove everything (with confirmation)
# =============================================================================

set -euo pipefail

# =============================================================================
# SECTION 1: GLOBAL CONFIGURATION
# =============================================================================
# All paths and ports are defined here. Change these if you want to install
# to a different location or use different ports.
# 
# Note: WEBUI_DIR can be overridden via --webui-dir command-line option:
#   ./run_stablediffusion.sh --webui-dir /custom/path --install

WEBUI_DIR="$HOME/stable-diffusion-webui"   # Where AUTOMATIC1111 is cloned
VENV_DIR="$WEBUI_DIR/venv"                 # Python virtualenv for WebUI deps
SD_REPO="https://github.com/AUTOMATIC1111/stable-diffusion-webui.git"
LOG_DIR="$HOME/sd-logs"                    # Per-GPU log files live here
ROUTER_DIR="$HOME/sd-router"              # Smart router script + its own venv
PID_FILE="/tmp/sd_webui_pids"             # Tracks WebUI process IDs for --stop
ROUTER_PID_FILE="/tmp/sd_router_pid"      # Tracks router process ID for --stop
BASE_PORT=7860                             # GPU 0 = 7860, GPU 1 = 7861, etc.
ROUTER_PORT=8080                           # Smart router listens here
PYTHON_BIN=""                              # Resolved by detect_python()

# GPU inventory arrays -- populated by detect_all_gpus()
# Each GPU_ENTRIES element is a pipe-delimited string:
#   "vendor|device_index|name|vram_mb|compute_cap_or_arch|extra"
GPU_ENTRIES=()
GPU_COUNT=0

# Vendor presence flags -- used to decide which PyTorch build to install
HAS_NVIDIA=false
HAS_AMD=false
HAS_INTEL=false

# NVIDIA-specific version strings -- resolved by resolve_nvidia_versions()
# Populated during detect_nvidia_gpus() based on the installed driver version
NVIDIA_CUDA_TAG=""           # e.g. "cu124" -- used as pip index suffix
NVIDIA_PYTORCH_VERSION=""    # e.g. "2.6.0"
NVIDIA_TORCHVISION_VERSION="" # must match torch exactly
NVIDIA_TORCHAUDIO_VERSION=""  # must match torch exactly

# =============================================================================
# SECTION 2: TERMINAL HELPERS
# =============================================================================
# Color codes and logging functions. All output goes through these so the
# visual style is consistent throughout the script.

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}      $*"; }
success() { echo -e "${GREEN}[OK]${NC}        $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}     $*"; }
error()   { echo -e "${RED}[ERROR]${NC}    $*" >&2; exit 1; }
section() { echo -e "\n${BOLD}+- $* ${NC}"; }
remark()  { echo -e "   ${CYAN}↳${NC} $*"; }   # Inline explanation of what just happened

print_banner() {
  echo -e "${BOLD}"
  echo "  ╔══════════════════════════════════════════════════════════════════╗"
  echo "  ║   Stable Diffusion WebUI -- Universal Multi-GPU Launcher v3.0     ║"
  echo "  ║   NVIDIA * AMD * Intel * CPU  |  Smart Router  |  Ubuntu 20-24   ║"
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
                  GPU hardware, PyTorch CUDA visibility, router status,
                  per-instance port health.

  --uninstall   Interactively remove everything installed by this script.
                  Prompts for confirmation before deleting. Does NOT remove
                  CUDA drivers, ROCm, or system Python.

  --webui-dir PATH  Custom installation path for AUTOMATIC1111 WebUI
                    (default: $HOME/stable-diffusion-webui)
                    Use with --install or other commands, e.g.:
                      $0 --webui-dir /mnt/gpu-storage/sd --install

  --help        Show this message.

GPU vendor support:
  NVIDIA  CUDA -- Volta / Turing / Ampere / Ada / Hopper / Blackwell / Pascal
  AMD     ROCm -- RDNA2, RDNA3, Vega
  Intel   IPEX -- Arc, Xe, Ponte Vecchio
  CPU          -- fallback if no GPU is found
EOF
  exit 0
}

# =============================================================================
# SECTION 3: PYTHON DETECTION
# =============================================================================
# Stable Diffusion's dependency tree works best with Python 3.10.
# We try versions in preference order and fall back to system python3.
# ensure_python310() attempts to install 3.10 via apt if it's missing.

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

ensure_python310() {
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
# SECTION 4: NVIDIA VERSION RESOLUTION
# =============================================================================
# This is the fix for the "torchaudio X requires torch==Y but you have Z" error.
#
# The problem: pip may resolve a newer torch than the cuda tag implies, but
# torchvision and torchaudio are published as pre-built wheels that are pinned
# to an EXACT torch version. If the versions don't match, pip's resolver either
# fails or installs a mismatched set that breaks at runtime.
#
# The solution: maintain a manually curated compatibility matrix that maps each
# CUDA tag to the exact (torch, torchvision, torchaudio) triplet that are all
# published together for that CUDA version. Install all three simultaneously
# with pinned versions so pip cannot resolve a different torch independently.
#
# Reference: https://github.com/pytorch/pytorch/wiki/PyTorch-Versions
#            https://download.pytorch.org/whl/

resolve_nvidia_versions() {
  local cuda_tag="$1"

  # Each row is a verified compatible set.
  # torch and torchaudio always share the same version number.
  # torchvision uses a separate versioning scheme but is published in sync.
  #
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
# SECTION 5: GPU DETECTION -- NVIDIA
# =============================================================================
# Uses nvidia-smi to enumerate all NVIDIA GPUs. nvidia-smi is reliable and
# always present when NVIDIA drivers are installed. We query:
#   - name         : human-readable GPU name
#   - memory.total : total VRAM in MiB (used for VRAM strategy flags)
#   - compute_cap  : CUDA compute capability (e.g. "7.0" for V100)
#                    This determines which flags and features are available
#   - driver_version: major version maps to the max supportable CUDA version
#
# CUDA driver version -> CUDA tag mapping:
#   Driver >= 560  -> CUDA 12.4 (cu124) -- supports torch 2.6.x
#   Driver >= 525  -> CUDA 12.1 (cu121) -- supports torch 2.5.x
#   Driver >= 520  -> CUDA 11.8 (cu118) -- supports torch 2.3.x
#   Driver >= 450  -> CUDA 11.7 (cu117) -- supports torch 2.0.x
#   Driver <  450 -> too old, use CPU PyTorch

detect_nvidia_gpus() {
  # Skip entirely if nvidia-smi is not installed
  command -v nvidia-smi &>/dev/null || return 0

  local count
  count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) || return 0
  [ "$count" -eq 0 ] && return 0

  HAS_NVIDIA=true
  info "Found ${count} NVIDIA GPU(s) via nvidia-smi"

  local names=() vrams=() computes=()
  mapfile -t names    < <(nvidia-smi --query-gpu=name         --format=csv,noheader | sed 's/^ //;s/ $//')
  mapfile -t vrams    < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader | grep -oE '[0-9]+')
  mapfile -t computes < <(nvidia-smi --query-gpu=compute_cap  --format=csv,noheader | tr -d ' ')

  for i in "${!names[@]}"; do
    # Store as pipe-delimited: vendor|nvidia_device_index|name|vram_mb|compute_cap|
    GPU_ENTRIES+=("nvidia|${i}|${names[$i]}|${vrams[$i]}|${computes[$i]}|")
    (( GPU_COUNT++ )) || true
  done

  # Determine CUDA tag from driver major version number
  local driver_major
  driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
    | head -1 | grep -oE '^[0-9]+')

  if   [ "$driver_major" -ge 560 ]; then NVIDIA_CUDA_TAG="cu124"
  elif [ "$driver_major" -ge 525 ]; then NVIDIA_CUDA_TAG="cu121"
  elif [ "$driver_major" -ge 520 ]; then NVIDIA_CUDA_TAG="cu118"
  elif [ "$driver_major" -ge 450 ]; then NVIDIA_CUDA_TAG="cu117"
  else
    warn "NVIDIA driver v${driver_major} is too old (need >= 450) -- using CPU PyTorch"
    NVIDIA_CUDA_TAG="cpu"
  fi

  # Resolve the exact compatible torch triplet for this CUDA tag
  resolve_nvidia_versions "$NVIDIA_CUDA_TAG"

  success "NVIDIA: driver=${driver_major} -> ${NVIDIA_CUDA_TAG} | torch=${NVIDIA_PYTORCH_VERSION}"
  remark "Will install: torch==${NVIDIA_PYTORCH_VERSION} torchvision==${NVIDIA_TORCHVISION_VERSION} torchaudio==${NVIDIA_TORCHAUDIO_VERSION}"
}

# =============================================================================
# SECTION 6: GPU DETECTION -- AMD
# =============================================================================
# Detection priority:
#   1. rocminfo -- the authoritative source; only present after ROCm is installed.
#      Parses GPU agents by looking for GFX architecture names (gfx1030, etc.)
#      and their marketing names (AMD Radeon RX 6800 XT, etc.)
#   2. lspci fallback -- used ONLY when rocminfo is not installed.
#      Very strict filtering to avoid false positives:
#        - NVIDIA cards must be excluded (lspci bus addresses can be ambiguous)
#        - ASPEED chips (server BMC/management) must be excluded
#        - Non-GPU AMD devices (audio, USB, SATA controllers) must be excluded
#
# ROCm GFX compatibility notes:
#   gfx1100/1101/1102 = RDNA3 (RX 7xxx)  -- best ROCm support
#   gfx1030/1031/1032 = RDNA2 (RX 6xxx)  -- good ROCm support
#   gfx906             = Vega20 (Radeon VII, MI50/60) -- older ROCm
#   gfx900             = Vega10 (RX Vega) -- limited, may need ROCm 5.x

detect_amd_gpus() {
  # Primary: rocminfo -- only fires if ROCm is installed
  if command -v rocminfo &>/dev/null; then
    info "rocminfo found -- using it as authoritative AMD GPU source"
    local gpu_name="" in_gpu=false vram=0

    while IFS= read -r line; do
      # Start of a GPU agent block (contains gfx architecture string)
      if echo "$line" | grep -q "^  Name:.*gfx"; then
        in_gpu=true
        gpu_name=$(echo "$line" | sed 's/.*Name: *//')
      fi
      # Replace gfxXXXX name with human-readable marketing name if available
      if $in_gpu && echo "$line" | grep -q "Marketing Name"; then
        gpu_name=$(echo "$line" | sed 's/.*Marketing Name: *//')
      fi
      # End of agent block -- save and reset
      if $in_gpu && echo "$line" | grep -q "^Agent [0-9]" && [ -n "$gpu_name" ]; then
        HAS_AMD=true
        GPU_ENTRIES+=("amd|${GPU_COUNT}|${gpu_name}|${vram}|rocm|")
        (( GPU_COUNT++ )) || true
        gpu_name=""; in_gpu=false; vram=0
      fi
    done < <(rocminfo 2>/dev/null)

    # rocminfo is authoritative -- never fall through to lspci
    return
  fi

  # Fallback: lspci -- only when rocminfo is absent (ROCm not yet installed)
  # This lets us detect AMD GPUs so we know to install ROCm during --install
  if [ "$HAS_AMD" = false ] && command -v lspci &>/dev/null; then
    info "rocminfo not found -- falling back to lspci for AMD GPU detection"
    remark "ROCm will be installed during --install; rocminfo will be available afterwards"

    while IFS= read -r line; do
      # CRITICAL: skip any line mentioning NVIDIA -- lspci PCI bus topology can
      # show NVIDIA chips under AMD-owned bus segments, causing false positives
      echo "$line" | grep -qi "NVIDIA"  && continue
      # ASPEED Technology makes BMC/management chips found in servers.
      # They appear as VGA devices in lspci but cannot run CUDA/ROCm workloads.
      echo "$line" | grep -qi "ASPEED"  && continue
      # Skip non-GPU AMD silicon: audio controllers, USB hubs, chipset bridges, etc.
      echo "$line" | grep -qiE "Audio|USB|SATA|NVMe|SMBus|Encryption|IOMMU|PCI bridge" && continue

      local name
      name=$(echo "$line" | sed 's/^[0-9a-f:.]* [^:]*: //' | sed 's/ (rev [0-9a-f]*)$//')
      HAS_AMD=true
      GPU_ENTRIES+=("amd|${GPU_COUNT}|${name}|0|rocm|")
      (( GPU_COUNT++ )) || true
    done < <(lspci 2>/dev/null \
      | grep -iE "VGA compatible controller|3D controller|Display controller" \
      | grep -iE "Advanced Micro Devices|AMD|ATI|Radeon" \
      | grep -iv "NVIDIA")
  fi
}

# =============================================================================
# SECTION 7: GPU DETECTION -- INTEL
# =============================================================================
# Detection priority:
#   1. xpu-smi -- Intel's management tool for Arc / Xe / Data Center GPUs.
#      Available after installing Intel GPU drivers and compute runtime.
#   2. lspci fallback -- detects Intel discrete GPUs (Arc, Xe) even before
#      Intel drivers are installed, so we know to install IPEX.
#      Strict filtering:
#        - Integrated graphics are excluded (UHD 6xx, Iris, HD Graphics NNN)
#          because they are not supported by IPEX for inference workloads
#        - Non-GPU Intel devices (Ethernet, Thunderbolt, Audio) are excluded
#        - NVIDIA and AMD cards that share PCI bus with Intel are excluded
#
# Intel GPU IPEX compatibility notes:
#   Arc A770, A750        -- best IPEX support (16GB, 8GB)
#   Arc A380              -- entry-level, 6GB
#   Ponte Vecchio (PVC)   -- datacenter, excellent FP16/BF16
#   Iris Xe (integrated)  -- NOT supported for SD inference

detect_intel_gpus() {
  # Primary: xpu-smi -- Intel's own tool
  if command -v xpu-smi &>/dev/null; then
    info "xpu-smi found -- using it as authoritative Intel GPU source"
    local output
    output=$(xpu-smi discovery 2>/dev/null || true)
    if echo "$output" | grep -qi "Device"; then
      while IFS= read -r line; do
        local name
        name=$(echo "$line" | grep -oE 'Intel.*GPU[^,]*' | head -1 || echo "Intel GPU")
        HAS_INTEL=true
        GPU_ENTRIES+=("intel|${GPU_COUNT}|${name}|0|xpu|")
        (( GPU_COUNT++ )) || true
      done < <(echo "$output" | grep -i "Device")
    fi
    return  # xpu-smi is authoritative
  fi

  # Fallback: lspci -- only for genuine Intel discrete GPUs
  if [ "$HAS_INTEL" = false ] && command -v lspci &>/dev/null; then
    info "xpu-smi not found -- falling back to lspci for Intel GPU detection"
    remark "IPEX drivers will be installed during --install"

    while IFS= read -r line; do
      # Exclude integrated Intel graphics -- not supported by IPEX for SD inference
      # UHD 6xx = Coffee/Ice/Tiger Lake integrated; Iris = same; HD Graphics NNN = older
      echo "$line" | grep -qiE "UHD Graphics [0-9]{3}|Iris (Xe|Plus|Pro)|HD Graphics [0-9]{3}[^0-9]" && continue
      # Exclude all non-GPU Intel PCI devices
      echo "$line" | grep -qiE "Ethernet|Audio|USB|SATA|NVMe|SMBus|Thunderbolt|Wi-Fi|Wireless|Serial|Management" && continue
      # Exclude NVIDIA or AMD devices that share the same PCI segment
      echo "$line" | grep -qiE "NVIDIA|AMD|ATI|Radeon" && continue

      local name
      name=$(echo "$line" | sed 's/^[0-9a-f:.]* [^:]*: //' | sed 's/ (rev [0-9a-f]*)$//')
      HAS_INTEL=true
      GPU_ENTRIES+=("intel|${GPU_COUNT}|${name}|0|xpu|")
      (( GPU_COUNT++ )) || true
    done < <(lspci 2>/dev/null \
      | grep -iE "VGA compatible controller|3D controller|Display controller" \
      | grep -i "Intel")
  fi
}

# =============================================================================
# SECTION 8: MASTER GPU DETECTION
# =============================================================================
# Runs all three vendor detectors in order and prints a summary table.
# NVIDIA is always detected first because it determines which PyTorch build
# will be installed into the shared venv.
# If nothing is detected at all, a CPU-only entry is added as a last resort.

detect_all_gpus() {
  section "GPU Detection"

  # Reset state
  GPU_ENTRIES=(); GPU_COUNT=0
  HAS_NVIDIA=false; HAS_AMD=false; HAS_INTEL=false

  detect_nvidia_gpus
  detect_amd_gpus
  detect_intel_gpus

  # Nothing found -- fall back to CPU
  if [ "$GPU_COUNT" -eq 0 ]; then
    warn "No GPUs detected -- adding CPU-only entry (image generation will be very slow)"
    remark "CPU inference: expect 2-10 minutes per 512x512 image"
    GPU_ENTRIES+=("cpu|0|CPU (no GPU)|0|cpu|")
    GPU_COUNT=1
  fi

  # Print detection summary
  echo ""
  echo -e "  ${BOLD}Detected ${GPU_COUNT} compute device(s):${NC}"
  local idx=0
  for entry in "${GPU_ENTRIES[@]}"; do
    local vendor name vram cc port arch flags
    IFS='|' read -r vendor _ name vram cc _ <<< "$entry"
    port=$((BASE_PORT + idx))
    arch=$(get_arch_label "$vendor" "$cc")
    flags=$(get_launch_flags "$vendor" "$cc" "$vram")

    local vc
    case "$vendor" in
      nvidia) vc="${GREEN}" ;;
      amd)    vc="${RED}" ;;
      intel)  vc="${CYAN}" ;;
      *)      vc="${NC}" ;;
    esac

    echo -e "  ${vc}[${vendor^^}]${NC} GPU ${idx} | ${name}"
    echo -e "              | VRAM: ${vram} MiB  |  Arch: ${arch}  |  Port: ${port}"
    echo -e "              | Launch flags: ${flags}"
    (( idx++ )) || true
  done
  echo ""

  # Explain mixed-vendor limitations if applicable
  local vendor_count=0
  $HAS_NVIDIA && (( vendor_count++ )) || true
  $HAS_AMD    && (( vendor_count++ )) || true
  $HAS_INTEL  && (( vendor_count++ )) || true

  if [ "$vendor_count" -gt 1 ]; then
    warn "Mixed GPU vendors detected:"
    remark "A single Python virtualenv cannot hold two different PyTorch builds simultaneously."
    remark "NVIDIA CUDA PyTorch takes priority. AMD/Intel GPUs will use CUDA PyTorch."
    remark "AMD GPUs will be launched with HIP env vars; performance may be suboptimal."
    remark "For best AMD or Intel performance, use a dedicated machine with one vendor."
  fi
}

# =============================================================================
# SECTION 9: GPU CAPABILITY HELPERS
# =============================================================================
# These functions translate compute capability numbers into human-readable
# architecture names, feature flags, and per-GPU WebUI launch arguments.

# Returns a human-readable architecture name from NVIDIA compute capability
get_arch_label() {
  local vendor="$1" cc="$2"

  # Non-NVIDIA vendors have simple labels
  case "$vendor" in
    amd)   echo "AMD ROCm"; return ;;
    intel) echo "Intel Xe/Arc"; return ;;
    cpu)   echo "CPU only"; return ;;
  esac

  # NVIDIA: parse major.minor compute capability
  local maj min
  maj=$(echo "$cc" | cut -d'.' -f1)
  min=$(echo "$cc" | cut -d'.' -f2)

  case "$maj" in
    10) echo "Blackwell (RTX 5xxx)" ;;     # 10.x -- consumer 2025
     9) echo "Hopper (H100)" ;;            # 9.0  -- datacenter
     8)
       if [ "$min" -ge 9 ]; then echo "Ada Lovelace (RTX 4xxx)"  # 8.9
       else                       echo "Ampere (RTX 3xxx / A100)" # 8.0/8.6
       fi ;;
     7)
       if [ "$min" -ge 5 ]; then echo "Turing (RTX 2xxx / T4)"   # 7.5
       else                       echo "Volta (V100)"             # 7.0
       fi ;;
     6)
       if [ "$min" -ge 1 ]; then echo "Pascal (GTX 10xx / P40)"  # 6.1
       else                       echo "Pascal GP100 (P100)"      # 6.0
       fi ;;
     5) echo "Maxwell (GTX 9xx)" ;;        # 5.x
     3) echo "Kepler (GTX 7xx)" ;;         # 3.x -- very old
     *) echo "NVIDIA (cc ${cc})" ;;
  esac
}

# Returns true (exit 0) if the GPU supports xformers memory-efficient attention
# xformers requires Turing or newer (compute 7.5+) on NVIDIA
# AMD and Intel do not use xformers in the standard AUTOMATIC1111 stack
gpu_supports_xformers() {
  local vendor="$1" cc="$2"
  [ "$vendor" != "nvidia" ] && return 1   # xformers is NVIDIA-only here
  local maj min
  maj=$(echo "$cc" | cut -d'.' -f1)
  min=$(echo "$cc" | cut -d'.' -f2)
  # Turing = 7.5; anything >= 7.5 supports xformers
  [ "$maj" -gt 7 ] || { [ "$maj" -eq 7 ] && [ "$min" -ge 5 ]; }
}

# Returns the optimal SD WebUI launch flags for a given GPU
# Flags are passed as COMMANDLINE_ARGS to webui.sh
#
# Precision flags:
#   --precision full --no-half --no-half-vae
#       Force FP32 everywhere. Required for Volta (V100) on Linux where FP16
#       produces black images. Also used for Maxwell and older.
#   (no precision flags)
#       Let WebUI use its default FP16 where available. Safe for Pascal+.
#
# Attention flags:
#   --xformers
#       Enable xformers memory-efficient attention. Reduces VRAM by ~30% and
#       speeds up generation on Turing+ (RTX 2xxx and newer).
#   --opt-split-attention
#       PyTorch-native split attention -- works on all GPUs including Volta.
#       Less effective than xformers but universal.
#
# VRAM strategy flags:
#   --lowvram  : For GPUs with < 4GB VRAM. Moves model layers to CPU RAM
#                between inference steps. Very slow but prevents OOM.
#   --medvram  : For 4-8 GB VRAM. Keeps model on GPU but streams some layers.
#   (none)     : 8GB+ -- entire model stays on GPU for maximum speed.

get_launch_flags() {
  local vendor="$1" cc="$2" vram_mb="$3"
  local flags=""

  case "$vendor" in
    nvidia)
      local maj min
      maj=$(echo "$cc" | cut -d'.' -f1)
      min=$(echo "$cc" | cut -d'.' -f2)

      if [ "$maj" -eq 7 ] && [ "$min" -eq 0 ]; then
        # Volta (V100, compute 7.0)
        # FP16 is known to produce black/corrupted images on V100 under Linux.
        # This is a driver-level issue, not a software one. Force full FP32.
        flags="--precision full --no-half --no-half-vae"

      elif [ "$maj" -lt 6 ]; then
        # Maxwell (5.x) and Kepler (3.x) and older
        # FP16 support is unreliable. Force FP32 for stability.
        flags="--precision full --no-half"

      elif [ "$maj" -eq 6 ]; then
        # Pascal (GTX 10xx, P100, P40)
        # FP16 works but xformers is not available. Use split attention instead.
        flags=""   # default precision is fine; --opt-split-attention added below

      elif [ "$maj" -gt 7 ] || { [ "$maj" -eq 7 ] && [ "$min" -ge 5 ]; }; then
        # Turing (7.5) and everything newer: Ampere, Ada, Hopper, Blackwell
        # xformers gives significant VRAM savings and speed improvements.
        flags="--xformers"
      fi
      ;;

    amd)
      # AMD ROCm: FP16 can be unstable depending on GPU and ROCm version.
      # Force FP32 for reliability. ROCm xformers is experimental -- not enabled.
      flags="--precision full --no-half"
      remark "(AMD) FP32 forced for stability -- ROCm FP16 can produce artifacts"
      ;;

    intel)
      # Intel IPEX: still maturing. FP32 is most reliable across Arc generations.
      flags="--precision full --no-half"
      remark "(Intel) FP32 forced -- IPEX FP16 support is hardware/driver dependent"
      ;;

    cpu)
      # CPU inference: always FP32, and explicitly disable GPU backends
      flags="--precision full --no-half --use-cpu all"
      ;;
  esac

  # VRAM strategy -- append appropriate memory flag based on reported VRAM
  # This applies to all vendors. AMD/Intel report 0 from lspci fallback,
  # so they skip the VRAM flags and use defaults.
  if [ "$vram_mb" -gt 0 ]; then
    if   [ "$vram_mb" -lt 4000 ]; then
      flags="$flags --lowvram --always-batch-cond-uncond"
      remark "< 4GB VRAM: using --lowvram (very slow; model layers offloaded to RAM)"
    elif [ "$vram_mb" -lt 8000 ]; then
      flags="$flags --medvram"
      remark "4-8GB VRAM: using --medvram (model partially on GPU)"
    fi
    # >= 8GB: no restriction flag needed
  fi

  # Split attention is a safe universal fallback -- appended for all GPUs
  # xformers already includes superior attention, so this is mainly for
  # non-xformers GPUs (Volta, Pascal, AMD, Intel, CPU)
  echo "$flags --opt-split-attention"
}

# =============================================================================
# SECTION 10: SYSTEM DEPENDENCIES
# =============================================================================
# Install apt packages required by the WebUI build process.
# These are needed at build time (not just runtime) for compiling C extensions.
#
#   build-essential, cmake, gcc, g++  -> compiling Python C extensions (triton, etc.)
#   libgl1, libglib2.0-0              -> OpenCV imports used by SD (even headlessly)
#   libffi-dev, libssl-dev            -> cryptography package deps
#   python3-dev                       -> Python headers for building native modules
#   nginx                             -> optional reverse proxy in front of router
#   pciutils                          -> provides lspci for GPU detection fallback
#   bc                                -> used by webui.sh for version comparisons

check_system_deps() {
  section "System Dependencies"
  remark "Installing build tools required by SD's C/C++ Python extensions"

  local pkgs=(
    git wget curl bc                          # version control and download tools
    build-essential cmake gcc g++            # C/C++ compiler toolchain
    libgl1 libglib2.0-0                      # OpenCV / image processing libs
    libffi-dev libssl-dev                    # cryptography build deps
    python3-dev                              # Python C headers
    nginx                                    # reverse proxy (optional but useful)
    pciutils                                 # provides lspci for GPU detection
  )

  local missing=()
  for pkg in "${pkgs[@]}"; do
    dpkg -s "$pkg" &>/dev/null || missing+=("$pkg")
  done

  if [ ${#missing[@]} -gt 0 ]; then
    info "Installing missing packages: ${missing[*]}"
    sudo apt update -qq
    sudo apt install -y "${missing[@]}"
  fi
  success "All system dependencies present"
}

# =============================================================================
# SECTION 11: PYTHON VIRTUAL ENVIRONMENT
# =============================================================================
# All SD Python dependencies are installed into an isolated virtualenv.
# This prevents conflicts with system Python packages and allows the SD
# stack to coexist with other Python projects on the same machine.
#
# setuptools is pinned to 68.0.0 to avoid the "AttributeError: install_layout"
# error that occurs with setuptools >= 69. That regression affects packages
# that use the legacy setup.py install path (notably the 'clip' package).

setup_venv() {
  section "Python Virtual Environment"
  remark "Creating isolated venv at: $VENV_DIR"
  remark "setuptools pinned to 68.0.0 -- avoids clip/install_layout build failure"

  if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "Virtualenv created"
  else
    info "Virtualenv already exists -- reusing"
  fi

  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip "setuptools==68.0.0" wheel \
    --quiet --disable-pip-version-check --no-cache-dir
  success "pip, setuptools==68.0.0, wheel installed"
}

# =============================================================================
# SECTION 12: PYTORCH INSTALLATION -- NVIDIA
# =============================================================================
# Installs torch, torchvision, and torchaudio as a pinned compatible set.
#
# WHY PINNED VERSIONS?
# pip's dependency resolver may independently resolve a newer torch than what
# torchvision/torchaudio expect, causing conflicts like:
#   "torchaudio 2.5.1+cu124 requires torch==2.5.1 but you have 2.6.0"
#
# The fix is to install all three simultaneously with explicit pinned versions
# from the same --index-url, preventing pip from resolving them independently.
# All three packages are then guaranteed to share the same CUDA ABI.
#
# xformers is installed separately and pinned to the same CUDA index URL so
# it gets a wheel compiled against the same torch ABI version.

install_pytorch_nvidia() {
  section "PyTorch for NVIDIA CUDA (${NVIDIA_CUDA_TAG})"
  remark "Installing pinned compatible set to prevent version conflicts:"
  remark "  torch==${NVIDIA_PYTORCH_VERSION} + torchvision==${NVIDIA_TORCHVISION_VERSION} + torchaudio==${NVIDIA_TORCHAUDIO_VERSION}"
  remark "  Source: https://download.pytorch.org/whl/${NVIDIA_CUDA_TAG}"

  source "$VENV_DIR/bin/activate"

  # Completely uninstall any existing torch stack first.
  # A partial or version-mismatched install is worse than a clean one.
  info "Removing any existing torch/torchvision/torchaudio/xformers..."
  pip uninstall torch torchvision torchaudio xformers -y 2>/dev/null || true

  info "Installing pinned torch stack..."
  pip install \
    "torch==${NVIDIA_PYTORCH_VERSION}" \
    "torchvision==${NVIDIA_TORCHVISION_VERSION}" \
    "torchaudio==${NVIDIA_TORCHAUDIO_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${NVIDIA_CUDA_TAG}" \
    --quiet --disable-pip-version-check

  # Verify that CUDA is actually usable after install
  local cuda_ok gpu_count
  cuda_ok=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
  gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

  if [ "$cuda_ok" = "True" ]; then
    success "PyTorch ${NVIDIA_PYTORCH_VERSION}+${NVIDIA_CUDA_TAG} installed -- ${gpu_count} GPU(s) visible"
  else
    warn "PyTorch installed but torch.cuda.is_available() returned False"
    remark "This usually means the driver CUDA version doesn't match the PyTorch CUDA version"
    remark "Run: $0 --diag  to see the full version report"
    remark "Try: nvidia-smi  to verify GPU is accessible"
  fi
}

# =============================================================================
# SECTION 13: PYTORCH INSTALLATION -- AMD ROCm
# =============================================================================
# AMD uses a ROCm-flavored PyTorch build published separately at:
#   https://download.pytorch.org/whl/rocmX.Y
#
# The ROCm version tag must match the ROCm stack installed on the system.
# We detect the installed ROCm version from rocminfo and pick the closest tag.
#
# ROCm version -> PyTorch index tag:
#   ROCm 6.x -> rocm6.2  (PyTorch 2.5.x)
#   ROCm 5.x -> rocm5.7  (PyTorch 2.3.x)
#
# Note: ROCm PyTorch and CUDA PyTorch cannot be installed in the same venv.
# If NVIDIA GPUs are also present, we skip this install and let NVIDIA CUDA
# PyTorch handle everything. AMD GPUs still get launched with HIP env vars.

install_pytorch_amd() {
  section "PyTorch for AMD ROCm"

  if [ "$HAS_NVIDIA" = true ]; then
    warn "Skipping ROCm PyTorch -- NVIDIA CUDA PyTorch already installed in this venv"
    remark "CUDA and ROCm PyTorch cannot coexist in the same virtualenv"
    remark "AMD GPUs will be launched with HIP env vars pointing at CUDA PyTorch"
    remark "For dedicated AMD performance, use a separate machine or separate venv"
    return
  fi

  source "$VENV_DIR/bin/activate"
  pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

  # Auto-detect installed ROCm version from rocminfo output
  local rocm_tag="rocm6.2"   # safe default
  if command -v rocminfo &>/dev/null; then
    local rv
    rv=$(rocminfo 2>/dev/null | grep -oE 'ROCm [0-9]+\.[0-9]+' | head -1 \
      | grep -oE '[0-9]+' | head -1 || echo "6")
    if [ "$rv" -ge 6 ]; then rocm_tag="rocm6.2"
    else                      rocm_tag="rocm5.7"
    fi
    remark "Detected ROCm major version: ${rv} -> using ${rocm_tag} PyTorch index"
  else
    remark "rocminfo not available -- defaulting to ${rocm_tag}"
  fi

  info "Installing PyTorch for ${rocm_tag}..."
  pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${rocm_tag}" \
    --quiet --disable-pip-version-check \
    || warn "ROCm PyTorch install failed -- check ROCm installation"

  local ok
  ok=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
  [ "$ok" = "True" ] \
    && success "ROCm PyTorch installed -- AMD GPU visible" \
    || warn "AMD GPU not yet visible -- may need to log out/in for group membership (render, video)"
}

# =============================================================================
# SECTION 14: PYTORCH INSTALLATION -- INTEL IPEX
# =============================================================================
# Intel Extension for PyTorch (IPEX) adds XPU device support for Arc/Xe GPUs.
# It must be installed alongside a matching PyTorch XPU build.
#
# Intel publishes PyTorch XPU wheels at:
#   https://download.pytorch.org/whl/xpu
# And IPEX wheels at:
#   https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
#
# The Level Zero driver (OneAPI) must be installed for XPU device enumeration.
# Without it, Intel GPUs fall back to CPU mode.
#
# Same venv limitation applies: Intel IPEX is only installed when there are
# no NVIDIA or AMD GPUs present.

install_pytorch_intel() {
  section "PyTorch for Intel IPEX/XPU"

  if [ "$HAS_NVIDIA" = true ] || [ "$HAS_AMD" = true ]; then
    warn "Skipping Intel IPEX -- another vendor's PyTorch already present in this venv"
    remark "Intel GPU will run in CPU fallback mode in this configuration"
    return
  fi

  source "$VENV_DIR/bin/activate"
  pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

  info "Installing Intel Extension for PyTorch (IPEX)..."
  remark "IPEX adds XPU (Intel GPU) device support to PyTorch"
  pip install intel-extension-for-pytorch \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ \
    --quiet --disable-pip-version-check \
    || warn "IPEX install failed -- Intel GPU will use CPU fallback"

  info "Installing XPU-flavored PyTorch..."
  pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/xpu \
    --quiet --disable-pip-version-check \
    || warn "XPU PyTorch install failed -- falling back to CPU PyTorch"

  success "Intel IPEX setup complete"
  remark "Requires Intel Level Zero driver to be installed for GPU to be visible"
}

# =============================================================================
# SECTION 15: ROCm SYSTEM INSTALL (AMD)
# =============================================================================
# ROCm is AMD's open-source GPU compute platform (equivalent of CUDA).
# It must be installed as a system package, not via pip.
# This section downloads and runs AMD's official installer script.
#
# After installation:
#   - The current user must be added to the 'render' and 'video' groups
#   - A logout/login is required for group membership to take effect
#   - rocminfo will then enumerate AMD GPUs correctly

install_rocm() {
  section "ROCm System Install (AMD)"
  remark "ROCm is AMD's compute platform -- the equivalent of NVIDIA CUDA"

  if command -v rocminfo &>/dev/null; then
    success "ROCm already installed -- skipping"
    return
  fi

  info "Downloading AMD GPU installer..."
  local rocm_ver="6.0"
  local codename
  codename=$(lsb_release -cs 2>/dev/null || echo "jammy")

  wget -q -O /tmp/amdgpu-install.deb \
    "https://repo.radeon.com/amdgpu-install/${rocm_ver}/ubuntu/${codename}/amdgpu-install_${rocm_ver}.50600-1_all.deb" \
    || { warn "Could not download ROCm installer -- AMD GPU will use CPU fallback"; return; }

  sudo apt install -y /tmp/amdgpu-install.deb -qq
  remark "Running amdgpu-install with ROCm use case (--no-dkms skips kernel module build)"
  sudo amdgpu-install -y --usecase=rocm --no-dkms 2>/dev/null \
    || sudo amdgpu-install -y --usecase=rocm 2>/dev/null \
    || warn "ROCm install encountered errors -- check manually with: sudo amdgpu-install -y --usecase=rocm"

  # Add current user to GPU access groups
  sudo usermod -aG render,video "$USER" 2>/dev/null || true
  success "ROCm installed"
  warn "ACTION REQUIRED: Log out and log back in for GPU group membership to take effect"
  remark "Then run: rocminfo  to verify AMD GPU is visible"
}

# =============================================================================
# SECTION 16: XFORMERS
# =============================================================================
# xformers provides memory-efficient attention for Transformer models.
# On SD it reduces VRAM usage by ~30% and speeds up inference by 10-30%.
#
# Compatibility:
#   - Requires Turing or newer NVIDIA GPU (compute capability >= 7.5)
#   - Must be installed from the same CUDA wheel index as torch to ensure
#     ABI compatibility (both must be compiled against the same libcuda)
#   - After install, we verify that xformers can actually import and that
#     its internal torch reference matches the installed torch version
#
# We skip xformers entirely if no compatible GPU is detected.

install_xformers_if_needed() {
  section "xformers"

  # Check if any GPU in the fleet actually supports xformers
  local needs=false
  for entry in "${GPU_ENTRIES[@]}"; do
    local vendor cc; IFS='|' read -r vendor _ _ _ cc _ <<< "$entry"
    gpu_supports_xformers "$vendor" "$cc" && needs=true && break
  done

  if [ "$needs" = false ]; then
    info "No GPU in this system supports xformers (requires Turing / compute 7.5+)"
    remark "Skipping xformers -- --opt-split-attention will be used instead"
    return
  fi

  source "$VENV_DIR/bin/activate"

  # Uninstall first to prevent ABI conflicts with a previously installed version
  info "Removing any existing xformers to prevent ABI mismatch..."
  pip uninstall xformers -y 2>/dev/null || true

  info "Installing xformers from ${NVIDIA_CUDA_TAG} wheel index..."
  remark "Using same CUDA index as torch ensures compatible CUDA ABI"
  pip install xformers \
    --index-url "https://download.pytorch.org/whl/${NVIDIA_CUDA_TAG}" \
    --quiet --disable-pip-version-check \
    || {
      warn "xformers from CUDA index failed -- trying PyPI..."
      pip install xformers --quiet --disable-pip-version-check \
        || { warn "xformers install failed -- affected GPUs run without it"; return; }
    }

  # Verify ABI compatibility: xformers must reference the same torch version
  local compat
  compat=$(python3 -c "
import torch, xformers
tv = torch.__version__.split('+')[0]
xv = xformers.__version__
print(f'torch={tv} xformers={xv}')
print('ok')
" 2>/dev/null || echo "mismatch")

  if echo "$compat" | grep -q "^ok$"; then
    local versions
    versions=$(echo "$compat" | head -1)
    success "xformers installed and ABI-compatible (${versions})"
  else
    warn "xformers version mismatch detected -- reinstalling from PyPI without pin"
    pip uninstall xformers -y 2>/dev/null || true
    pip install xformers --quiet --disable-pip-version-check \
      || warn "xformers could not be installed -- SD will run without it"
  fi
}

# =============================================================================
# SECTION 17: CLIP
# =============================================================================
# CLIP (Contrastive Language-Image Pretraining) by OpenAI is a required
# dependency of Stable Diffusion for text encoding.
#
# WHY NOT pip install clip?
# The 'clip' package on PyPI is an unofficial third-party upload that uses a
# broken setup.py which fails with newer setuptools due to the install_layout
# AttributeError. Installing from OpenAI's GitHub source avoids this entirely.
#
# --no-build-isolation allows the build to reuse the already-installed
# setuptools==68.0.0 from the venv rather than pulling in a newer version
# for the build step (which would trigger the same bug).

install_clip() {
  section "CLIP (OpenAI)"
  remark "Installing from GitHub source to avoid PyPI 'clip' package build failure"
  remark "Using --no-build-isolation to reuse venv's pinned setuptools==68.0.0"

  source "$VENV_DIR/bin/activate"
  pip install "git+https://github.com/openai/CLIP.git" \
    --quiet --no-build-isolation --disable-pip-version-check \
    || warn "CLIP install failed -- WebUI will attempt its own install at first launch"
  success "CLIP installed"
}

# =============================================================================
# SECTION 18: WEBUI + REQUIRED REPOSITORIES
# =============================================================================
# AUTOMATIC1111 WebUI is cloned from GitHub. It also requires several
# additional model/helper repositories that it clones itself on first launch.
# We pre-clone them here to avoid the GitHub credential prompt issue where
# some repos require authentication for anonymous HTTPS access.
#
# Pre-cloning strategy:
#   - Try the primary URL first with GIT_TERMINAL_PROMPT=0 (fail instead of prompt)
#   - On failure, try a known-public fallback URL if one exists
#   - If both fail, delete the partial clone and let WebUI retry at launch
#   - If the clone dir exists but has no .git, it's a broken partial clone -- remove it

install_webui() {
  section "AUTOMATIC1111 WebUI"
  remark "Cloning from: $SD_REPO"

  if [ -d "$WEBUI_DIR/.git" ]; then
    warn "WebUI already cloned at $WEBUI_DIR -- skipping"
    remark "Run --update to pull the latest commits"
    return
  fi

  GIT_TERMINAL_PROMPT=0 git clone "$SD_REPO" "$WEBUI_DIR" \
    || error "WebUI clone failed -- check your internet connection"
  success "WebUI cloned to $WEBUI_DIR"
}

clone_repositories() {
  section "Required Repositories"
  remark "Pre-cloning repos to avoid GitHub authentication prompts during launch"
  remark "Format: name|primary_url|fallback_url"
  remark "stable-diffusion-stability-ai uses CompVis as fallback (Stability-AI sometimes rate-limits)"

  local repo_dir="$WEBUI_DIR/repositories"
  mkdir -p "$repo_dir"

  # Each entry: "dir_name|primary_github_url|fallback_url_or_empty"
  local REPOS=(
    "stable-diffusion-stability-ai|https://github.com/Stability-AI/stablediffusion.git|https://github.com/CompVis/stable-diffusion.git"
    "CodeFormer|https://github.com/sczhou/CodeFormer.git|"
    "BLIP|https://github.com/salesforce/BLIP.git|"
    "k-diffusion|https://github.com/crowsonkb/k-diffusion.git|"
    "GFPGAN|https://github.com/TencentARC/GFPGAN.git|"
  )

  for repo_entry in "${REPOS[@]}"; do
    local name primary fallback target
    name=$(echo "$repo_entry"     | cut -d'|' -f1)
    primary=$(echo "$repo_entry"  | cut -d'|' -f2)
    fallback=$(echo "$repo_entry" | cut -d'|' -f3)
    target="$repo_dir/$name"

    # Clean up incomplete clones (directory exists but no .git inside)
    if [ -d "$target" ] && [ ! -d "$target/.git" ]; then
      warn "  Incomplete clone at $target -- removing and retrying"
      rm -rf "$target"
    fi

    if [ -d "$target/.git" ]; then
      info "  Already cloned: $name"; continue
    fi

    info "  Cloning $name..."
    if GIT_TERMINAL_PROMPT=0 git clone "$primary" "$target" 2>/dev/null; then
      success "  $name"
    elif [ -n "$fallback" ]; then
      warn "  Primary URL failed -- trying fallback for $name"
      rm -rf "$target"
      if GIT_TERMINAL_PROMPT=0 git clone "$fallback" "$target" 2>/dev/null; then
        success "  $name (via fallback)"
      else
        warn "  Both URLs failed for $name -- WebUI will retry at first launch"
        rm -rf "$target"
      fi
    else
      warn "  Failed to clone $name -- WebUI will retry at first launch"
      rm -rf "$target"
    fi
  done
  success "Repository pre-clone complete"
}

# =============================================================================
# SECTION 19: SMART ROUTER
# =============================================================================
# The smart router is a FastAPI application that acts as an intelligent proxy
# in front of all WebUI instances.
#
# HOW IT WORKS
# -------------
# 1. Every API request hits the router on port ROUTER_PORT (8080)
# 2. For generation endpoints (/sdapi/v1/txt2img, /sdapi/v1/img2img):
#    a. Parse the request body for width, height, batch_size
#    b. Estimate VRAM needed: base model overhead + pixels x bytes per pixel
#    c. Query nvidia-smi for real-time free VRAM on each NVIDIA GPU
#    d. Check each WebUI instance for liveness (GET /sdapi/v1/progress)
#    e. Filter out GPUs with insufficient free VRAM
#    f. Among remaining candidates, score by:
#         - Prefer high-VRAM GPU for high-res (>=1024px) requests
#         - Prefer GPU with shortest current queue
#         - Among ties, prefer GPU with most free VRAM
#    g. Forward the request to the winner; track queue depth
# 3. For all other endpoints (UI, model loading, etc.):
#    Round-robin to first available instance
# 4. /router/status returns real-time GPU fleet status as JSON
#
# GPU ROUTING EXAMPLES (your specific setup)
# -----------------------------------------
#   Request: 512x512 SD 1.5           -> RTX 5000 :7861 (fastest with xformers)
#   Request: 1024x1024 SDXL           -> V100 :7860 or :7862 (32GB, high-res preferred)
#   Request: 2048x2048 SDXL           -> V100 :7860 or :7862 (only GPUs with enough VRAM)
#   Request: batch_size=4, 512x512    -> whichever V100 has shorter queue
#   RTX 5000 busy, V100s idle         -> route standard request to a V100 instead

write_smart_router() {
  # ===========================================================================
  # SECTION 19: SMART ROUTER SETUP
  # ===========================================================================
  # Copies router_template.py (which must sit alongside this script) into
  # ~/sd-router/router.py and injects the GPU fleet JSON into it by replacing
  # the placeholder line: GPU_FLEET = []  # REPLACED_BY_LAUNCHER
  #
  # Using a separate template file avoids the nested-heredoc truncation problem
  # that occurs when embedding Python inside a bash heredoc inside a cat heredoc.

  section "Smart Router"
  remark "Setting up FastAPI GPU router at: $ROUTER_DIR/router.py"
  remark "Router listens on port ${ROUTER_PORT}"

  mkdir -p "$ROUTER_DIR"

  # Locate router_template.py next to this script
  local SCRIPT_DIR
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local TEMPLATE_SRC="$SCRIPT_DIR/router_template.py"

  if [ ! -f "$TEMPLATE_SRC" ]; then
    error "router_template.py not found at $TEMPLATE_SRC\nPlace it alongside run_stablediffusion.sh"
  fi

  # Build the GPU fleet JSON array
  local gpu_json="["
  local idx=0
  for entry in "${GPU_ENTRIES[@]}"; do
    local vendor name vram cc port arch xf nv_idx safe_name
    IFS="|" read -r vendor _ name vram cc _ <<< "$entry"
    port=$((BASE_PORT + idx))
    arch=$(get_arch_label "$vendor" "$cc")
    xf="false"; gpu_supports_xformers "$vendor" "$cc" && xf="true"
    nv_idx="null"; [ "$vendor" = "nvidia" ] && nv_idx="$idx"
    # Sanitise GPU name for JSON (escape any double quotes)
    safe_name=$(printf '%s' "$name" | sed 's/"/\\"/g')
    # Build entry using concatenation with single-quoted JSON keys
    gpu_json+='{"index":'
    gpu_json+=${idx}
    gpu_json+=',"vendor":"'${vendor}'","name":"'${safe_name}'"'
    gpu_json+=',"port":'${port}',"vram_mb":'${vram}
    gpu_json+=',"arch":"'${arch}'","xformers":'${xf}
    gpu_json+=',"nvidia_index":'${nv_idx}'},'
    (( idx++ )) || true
  done
  gpu_json="${gpu_json%,}]"

  # Inject fleet into the template by replacing the placeholder
  sed "s|GPU_FLEET = \[\]  # REPLACED_BY_LAUNCHER|GPU_FLEET = ${gpu_json}|" \
    "$TEMPLATE_SRC" > "$ROUTER_DIR/router.py"

  if grep -q "REPLACED_BY_LAUNCHER" "$ROUTER_DIR/router.py"; then
    error "GPU fleet injection failed -- placeholder still in router.py"
  fi
  success "Router written with ${GPU_COUNT} GPU(s)"
  remark "Fleet: $gpu_json"

  # Router uses its own venv to avoid dep conflicts with the WebUI venv
  if [ ! -d "$ROUTER_DIR/venv" ]; then
    info "Creating router virtualenv..."
    "$PYTHON_BIN" -m venv "$ROUTER_DIR/venv"
  fi
  info "Installing router deps (FastAPI, uvicorn, httpx)..."
  source "$ROUTER_DIR/venv/bin/activate"
  pip install fastapi "uvicorn[standard]" httpx \
    --quiet --disable-pip-version-check --no-cache-dir
  deactivate
  success "Router ready"
}

launch_router() {
  info "Starting smart router on port ${ROUTER_PORT}..."
  source "$ROUTER_DIR/venv/bin/activate"
  python3 "$ROUTER_DIR/router.py" "${ROUTER_PORT}" \
    > "$LOG_DIR/router.log" 2>&1 &
  echo $! > "$ROUTER_PID_FILE"
  deactivate
  sleep 2
  local router_pid
  router_pid=$(cat "$ROUTER_PID_FILE" 2>/dev/null || echo "")
  if [ -n "$router_pid" ] && kill -0 "$router_pid" 2>/dev/null; then
    success "Smart router running"
    remark "Main endpoint:  http://localhost:${ROUTER_PORT}"
    remark "Fleet status:   http://localhost:${ROUTER_PORT}/router/status"
    remark "Router log:     tail -f $LOG_DIR/router.log"
  else
    warn "Router failed to start -- check: tail -f $LOG_DIR/router.log"
  fi
}

stop_router() {
  if [ -f "$ROUTER_PID_FILE" ]; then
    local pid
    pid=$(cat "$ROUTER_PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      info "  Router stopped (PID $pid)"
    fi
    rm -f "$ROUTER_PID_FILE"
  fi
  pkill -f "router.py" 2>/dev/null || true
}

# =============================================================================
# SECTION 20: NGINX
# =============================================================================
# Optional public-facing reverse proxy. Handles SSL termination, large uploads,
# WebSocket upgrade headers (SD live preview), and long-generation timeouts.
# Traffic: client -> nginx:80 -> router:8080 -> WebUI:786x -> GPU

configure_nginx() {
  command -v nginx &>/dev/null || { info "nginx not installed -- skipping"; return; }
  remark "nginx -> router:${ROUTER_PORT} -> WebUI instances"

  local nginx_conf
  printf -v nginx_conf '%s
' \
    "upstream sd_router { server 127.0.0.1:${ROUTER_PORT}; }" \
    "server {" \
    "    listen 80; server_name localhost; client_max_body_size 50M;" \
    "    location / {" \
    "        proxy_pass http://sd_router; proxy_http_version 1.1;" \
    '        proxy_set_header Upgrade $http_upgrade;' \
    '        proxy_set_header Connection "upgrade";' \
    '        proxy_set_header Host $host;' \
    "        proxy_read_timeout 300s; proxy_connect_timeout 10s;" \
    "    }" \
    "}"

  printf '%s' "$nginx_conf" | sudo tee /etc/nginx/sites-available/stable-diffusion > /dev/null
  sudo ln -sf /etc/nginx/sites-available/stable-diffusion \
              /etc/nginx/sites-enabled/stable-diffusion 2>/dev/null || true
  sudo nginx -t -q && sudo systemctl reload nginx
  success "nginx configured -> smart router -> GPUs"
}

# =============================================================================
# SECTION 21: LAUNCH
# =============================================================================
# Starts one WebUI process per GPU pinned via vendor-specific env vars:
#   NVIDIA -> CUDA_VISIBLE_DEVICES=N
#   AMD    -> HIP_VISIBLE_DEVICES=N ROCR_VISIBLE_DEVICES=N HSA_OVERRIDE_GFX_VERSION=10.3.0
#   Intel  -> ZE_AFFINITY_MASK=N IPEX_XPU_ONEDNN_LAYOUT=1
# PIDs saved to PID_FILE for clean --stop. 2s stagger prevents port races.

launch_all() {
  mkdir -p "$LOG_DIR"
  > "$PID_FILE"

  section "Launching ${GPU_COUNT} WebUI Instance(s)"
  local idx=0
  for entry in "${GPU_ENTRIES[@]}"; do
    local vendor dev_idx name vram cc port log flags env_prefix
    IFS="|" read -r vendor dev_idx name vram cc _ <<< "$entry"
    port=$((BASE_PORT + idx))
    flags=$(get_launch_flags "$vendor" "$cc" "$vram")
    log="$LOG_DIR/gpu${idx}.log"

    info "GPU ${idx} [${vendor^^}] ${name} -> port ${port}"
    remark "Flags: ${flags}"

    case "$vendor" in
      nvidia) env_prefix="CUDA_VISIBLE_DEVICES=${dev_idx}" ;;
      amd)    env_prefix="HIP_VISIBLE_DEVICES=${dev_idx} ROCR_VISIBLE_DEVICES=${dev_idx} HSA_OVERRIDE_GFX_VERSION=10.3.0" ;;
      intel)  env_prefix="ZE_AFFINITY_MASK=${dev_idx} IPEX_XPU_ONEDNN_LAYOUT=1" ;;
      cpu)    env_prefix="" ;;
    esac

    eval "${env_prefix} COMMANDLINE_ARGS='--port ${port} ${flags}' \
      GIT_TERMINAL_PROMPT=0 \
      bash -c \"cd '${WEBUI_DIR}' && bash webui.sh\"" \
      > "$log" 2>&1 &

    echo $! >> "$PID_FILE"
    (( idx++ )) || true
    sleep 2
  done

  success "All ${GPU_COUNT} WebUI instance(s) launched"
  write_smart_router
  launch_router
  configure_nginx

  echo ""
  echo -e "  ${BOLD}Architecture:${NC}"
  echo -e "    Browser / API"
  echo -e "         |"
  echo -e "    ${CYAN}nginx :80${NC}        (public, optional)"
  echo -e "         |"
  echo -e "    ${GREEN}Router :${ROUTER_PORT}${NC}     (GPU selection, VRAM check, queue)"
  echo -e "         |"
  idx=0
  for entry in "${GPU_ENTRIES[@]}"; do
    local vendor name vram cc arch
    IFS="|" read -r vendor _ name vram cc _ <<< "$entry"
    arch=$(get_arch_label "$vendor" "$cc")
    local port=$((BASE_PORT + idx))
    local xf_note=""
    gpu_supports_xformers "$vendor" "$cc" && xf_note=" [xformers]"
    echo -e "    ${CYAN}:${port}${NC}  GPU${idx} [${vendor^^}] ${name} (${arch}, ${vram}MiB)${xf_note}"
    (( idx++ )) || true
  done
  echo ""
  echo -e "  ${BOLD}Endpoints:${NC}"
  echo -e "    Router  -> ${CYAN}http://localhost:${ROUTER_PORT}${NC}"
  echo -e "    Status  -> ${CYAN}http://localhost:${ROUTER_PORT}/router/status${NC}"
  echo -e "    nginx   -> ${CYAN}http://localhost:80${NC}"
  idx=0
  for entry in "${GPU_ENTRIES[@]}"; do
    local vendor name; IFS="|" read -r vendor _ name _ _ _ <<< "$entry"
    echo -e "    GPU${idx}  -> ${CYAN}http://localhost:$((BASE_PORT+idx))${NC} [${vendor^^}] ${name}"
    (( idx++ )) || true
  done
  echo ""
  echo -e "  ${BOLD}Logs:${NC}"
  echo -e "    Router: tail -f ${LOG_DIR}/router.log"
  echo -e "    All:    tail -f ${LOG_DIR}/gpu*.log ${LOG_DIR}/router.log"
  echo ""
  echo -e "  ${BOLD}Stop:${NC}   $0 --stop"
  echo ""
  tail -f "$LOG_DIR"/gpu*.log "$LOG_DIR/router.log"
}

# =============================================================================
# SECTION 22: STOP
# =============================================================================
stop_all() {
  section "Stopping All Processes"
  stop_router
  if [ -f "$PID_FILE" ]; then
    while read -r pid; do
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        info "  Stopped WebUI PID $pid"
      fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
  fi
  pkill -f "webui.sh"  2>/dev/null || true
  pkill -f "launch.py" 2>/dev/null || true
  success "All processes stopped"
}

# =============================================================================
# SECTION 23: UNINSTALL
# =============================================================================
# Removes all files created by this script after user types 'yes'.
# Does NOT remove: CUDA/ROCm/Intel drivers, system Python, nginx binary.
# WARNING: models inside WEBUI_DIR/models/ ARE deleted -- back them up first.

run_uninstall() {
  echo ""
  echo -e "${RED}${BOLD}WARNING -- The following will be permanently deleted:${NC}"
  echo ""
  echo -e "   WebUI + models : $WEBUI_DIR"
  echo -e "   Smart router   : $ROUTER_DIR"
  echo -e "   Log files      : $LOG_DIR"
  echo -e "   nginx config   : /etc/nginx/sites-*/stable-diffusion"
  echo -e "   PID files      : $PID_FILE  $ROUTER_PID_FILE"
  echo ""
  echo -e "   ${YELLOW}Models in $WEBUI_DIR/models/ WILL be deleted.${NC}"
  echo -e "   ${CYAN}NOT removed: CUDA/ROCm/Intel drivers, Python, nginx binary${NC}"
  echo -e "   ${CYAN}To remove ROCm: sudo amdgpu-uninstall${NC}"
  echo ""
  read -r -p "  Type 'yes' to confirm: " confirm
  [ "$confirm" != "yes" ] && { info "Cancelled -- nothing deleted"; exit 0; }

  echo ""
  stop_all 2>/dev/null || true
  info "Removing WebUI..."         ; rm -rf "$WEBUI_DIR"
  info "Removing smart router..."  ; rm -rf "$ROUTER_DIR"
  info "Removing logs..."          ; rm -rf "$LOG_DIR"
  info "Removing nginx config..."
  sudo rm -f /etc/nginx/sites-available/stable-diffusion
  sudo rm -f /etc/nginx/sites-enabled/stable-diffusion
  command -v nginx &>/dev/null && sudo nginx -t -q 2>/dev/null && sudo systemctl reload nginx 2>/dev/null || true
  info "Removing PID files..."     ; rm -f "$PID_FILE" "$ROUTER_PID_FILE"
  success "Uninstall complete"
  echo -e "  To reinstall: ${CYAN}$0 --install${NC}"
}

# =============================================================================
# SECTION 24: DIAGNOSTICS
# =============================================================================
print_diagnostics() {
  section "GPU Hardware"
  command -v nvidia-smi &>/dev/null && {
    echo -e "  ${GREEN}NVIDIA:${NC}"
    nvidia-smi \
      --query-gpu=index,name,compute_cap,memory.total,memory.free,driver_version \
      --format=csv,noheader | \
      while IFS="," read -r idx name cc mem_total mem_free drv; do
        local arch; arch=$(get_arch_label nvidia "$(echo "$cc"|tr -d ' ')")
        local xf; xf="no"; get_xformers_diag "$(echo "$cc"|tr -d ' ')" && xf="yes"
        echo -e "    GPU ${idx}: ${name}"
        echo -e "      Arch:     ${arch} (cc $(echo "$cc"|tr -d ' '))"
        echo -e "      VRAM:     $(echo "$mem_free"|tr -d ' ') free / $(echo "$mem_total"|tr -d ' ') total"
        echo -e "      Driver:   $(echo "$drv"|tr -d ' ')"
        echo -e "      xformers: ${xf}"
      done
  }
  command -v rocminfo &>/dev/null && {
    echo -e "
  ${RED}AMD ROCm:${NC}"
    rocminfo 2>/dev/null | grep -E "Marketing Name|gfx|Max Clock" | head -12 | sed "s/^/    /"
  }
  command -v xpu-smi &>/dev/null && {
    echo -e "
  ${CYAN}Intel:${NC}"
    xpu-smi discovery 2>/dev/null | grep -i "device\|name\|memory" | head -8 | sed "s/^/    /"
  }

  section "PyTorch"
  if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    python3 -c "
import torch
print(f'  Version  : {torch.__version__}')
print(f'  CUDA tag : {torch.version.cuda}')
print(f'  GPUs     : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f'  GPU {i}    : {p.name}  cc{p.major}.{p.minor}  {total//1024**2}MB total  {free//1024**2}MB free')
try:
    import xformers; print(f'  xformers : {xformers.__version__}')
except ImportError:
    print('  xformers : not installed')
try:
    import intel_extension_for_pytorch as ipex
    print(f'  IPEX     : {ipex.__version__}  XPU: {torch.xpu.device_count()}')
except ImportError:
    pass
" 2>/dev/null || warn "Could not load PyTorch -- run --install first"
  else
    warn "Venv not found at $VENV_DIR -- run: $0 --install"
  fi

  section "Smart Router"
  if curl -s --max-time 3 "http://localhost:${ROUTER_PORT}/router/status" &>/dev/null; then
    curl -s "http://localhost:${ROUTER_PORT}/router/status" | python3 -m json.tool 2>/dev/null \
      || echo "  Router running but status not parseable"
  else
    echo -e "  ${RED}Router OFFLINE${NC} (port ${ROUTER_PORT})"
    remark "Start with: $0"
    remark "Check log:  tail -f $LOG_DIR/router.log"
  fi

  section "WebUI Instance Health"
  for i in $(seq 0 $((GPU_COUNT > 0 ? GPU_COUNT - 1 : 2))); do
    local port=$((BASE_PORT + i))
    curl -s --max-time 2 "http://localhost:${port}/sdapi/v1/progress" &>/dev/null \
      && echo -e "  Port ${port} (GPU ${i}): ${GREEN}ONLINE${NC}" \
      || echo -e "  Port ${port} (GPU ${i}): ${RED}OFFLINE${NC}"
  done
  echo ""
}

get_xformers_diag() {
  local cc="$1"
  local maj min
  maj=$(echo "$cc" | cut -d'.' -f1)
  min=$(echo "$cc" | cut -d'.' -f2)
  [ "$maj" -gt 7 ] || { [ "$maj" -eq 7 ] && [ "$min" -ge 5 ]; }
}

# =============================================================================
# SECTION 25: UPDATE
# =============================================================================
run_update() {
  detect_all_gpus
  section "Updating WebUI"
  remark "git pull preserves models, outputs, and configs"
  cd "$WEBUI_DIR"
  GIT_TERMINAL_PROMPT=0 git pull
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip "setuptools==68.0.0" wheel \
    --quiet --disable-pip-version-check --no-cache-dir
  [ "$HAS_NVIDIA" = true ] && install_pytorch_nvidia && install_xformers_if_needed
  [ "$HAS_AMD"    = true ] && [ "$HAS_NVIDIA" = false ] && install_pytorch_amd
  [ "$HAS_INTEL"  = true ] && [ "$HAS_NVIDIA" = false ] && [ "$HAS_AMD" = false ] && install_pytorch_intel
  write_smart_router
  success "Update complete -- run $0 to launch"
}

# =============================================================================
# SECTION 26: MASTER INSTALL
# =============================================================================
# Order matters: GPU detect -> system deps -> Python -> ROCm -> PyTorch ->
#                xformers -> CLIP -> WebUI repos -> smart router
run_install() {
  detect_all_gpus
  check_system_deps
  detect_python
  ensure_python310
  install_webui
  clone_repositories
  setup_venv
  [ "$HAS_AMD"    = true ] && install_rocm
  [ "$HAS_NVIDIA" = true ] && install_pytorch_nvidia
  [ "$HAS_AMD"    = true ] && [ "$HAS_NVIDIA" = false ] && install_pytorch_amd
  [ "$HAS_INTEL"  = true ] && [ "$HAS_NVIDIA" = false ] && [ "$HAS_AMD" = false ] && install_pytorch_intel
  [ "$HAS_NVIDIA" = true ] && install_xformers_if_needed
  install_clip
  write_smart_router

  success "Installation complete!"
  echo ""
  echo -e "  ${BOLD}GPU Summary:${NC}"
  local idx=0
  for entry in "${GPU_ENTRIES[@]}"; do
    local vendor name vram cc flags arch
    IFS="|" read -r vendor _ name vram cc _ <<< "$entry"
    flags=$(get_launch_flags "$vendor" "$cc" "$vram")
    arch=$(get_arch_label "$vendor" "$cc")
    local xf_note=""
    gpu_supports_xformers "$vendor" "$cc" && xf_note=" + xformers"
    echo -e "    GPU ${idx} [${vendor^^}] ${name}"
    echo -e "            ${arch}${xf_note} | ${vram} MiB"
    echo -e "            Flags: ${flags}"
    (( idx++ )) || true
  done
  echo ""
  echo -e "  ${BOLD}Next steps:${NC}"
  echo -e "    1. Add models:  cp model.safetensors $WEBUI_DIR/models/Stable-diffusion/"
  echo -e "    2. Launch:      $0"
  echo -e "    3. Browser:     http://localhost:${ROUTER_PORT}"
  echo -e "    4. Status:      http://localhost:${ROUTER_PORT}/router/status"
  echo -e "    5. Remove all:  $0 --uninstall"
}

# =============================================================================
# SECTION 27: MAIN ENTRYPOINT
# =============================================================================
main() {
  print_banner
  
  # Parse --webui-dir option if provided
  if [[ "$1" == "--webui-dir" ]]; then
    if [[ -z "$2" ]]; then
      error "--webui-dir requires a path argument"
      usage
      exit 1
    fi
    WEBUI_DIR="$2"
    VENV_DIR="$WEBUI_DIR/venv"
    shift 2  # Remove both --webui-dir and the path from arguments
  fi
  
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
      warn "Unknown option: $1"
      usage
      ;;
  esac
}

main "$@"
