#!/usr/bin/env bash
# =============================================================================
# Post-Installation Verification Script
# =============================================================================
# Validates that a Stable Diffusion installation completed successfully.
# Checks: WebUI clone, venv, PyTorch, router, nginx (if enabled), logs.
#
# Usage:
#   chmod +x scripts/verify_install.sh
#   ./scripts/verify_install.sh [--webui-dir PATH] [--nginx-port NUM]
#
# Exit codes:
#   0 = All checks passed
#   1 = One or more checks failed (non-critical)
#   2 = Critical installation failure

set -euo pipefail

# Default paths and ports (match launcher defaults)
WEBUI_DIR="${HOME}/stable-diffusion-webui"
ROUTER_DIR="${HOME}/sd-router"
LOG_DIR="${HOME}/sd-logs"
NGINX_PORT=8888
ROUTER_PORT=8080
BASE_PORT=7860

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --webui-dir)
      WEBUI_DIR="$2"
      shift 2
      ;;
    --nginx-port)
      NGINX_PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 2
      ;;
  esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Helper functions
check_pass() {
  echo -e "${GREEN}✓${NC} $1"
  ((PASSED++)) || true
}

check_fail() {
  echo -e "${RED}✗${NC} $1"
  ((FAILED++)) || true
}

check_warn() {
  echo -e "${YELLOW}⚠${NC} $1"
  ((WARNINGS++)) || true
}

info() {
  echo -e "${CYAN}[INFO]${NC} $1"
}

# =============================================================================
# VERIFICATION CHECKS
# =============================================================================

echo ""
echo "====== Stable Diffusion Installation Verification ======"
echo ""

# SECTION 1: WebUI Clone
echo -e "${CYAN}WebUI Installation:${NC}"
if [ -d "$WEBUI_DIR" ] && [ -d "$WEBUI_DIR/.git" ]; then
  check_pass "WebUI directory exists: $WEBUI_DIR"
  local remote
  remote=$(cd "$WEBUI_DIR" && git remote -v | head -1 | awk '{print $2}' || echo "unknown")
  check_pass "Git repository valid (remote: $remote)"
else
  check_fail "WebUI not found at $WEBUI_DIR"
fi

# SECTION 2: Python Virtual Environment
echo ""
echo -e "${CYAN}Python Environment:${NC}"
local venv_path="$WEBUI_DIR/venv"
if [ -d "$venv_path/bin" ]; then
  check_pass "Virtual environment exists: $venv_path"
  if [ -f "$venv_path/bin/python" ]; then
    local py_version
    py_version=$("$venv_path/bin/python" --version 2>&1 || echo "unknown")
    check_pass "Python binary accessible: $py_version"
  else
    check_fail "Python binary not found in venv"
  fi
else
  check_fail "Virtual environment not found"
fi

# SECTION 3: PyTorch Installation
echo ""
echo -e "${CYAN}PyTorch & Dependencies:${NC}"
if [ -d "$venv_path/bin" ]; then
  if "$venv_path/bin/python" -c "import torch; print(f'PyTorch {torch.__version__}')" &>/dev/null; then
    local torch_ver
    torch_ver=$("$venv_path/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    check_pass "PyTorch installed: $torch_ver"
    
    if "$venv_path/bin/python" -c "import torch; print(torch.cuda.is_available())" &>/dev/null; then
      local has_cuda
      has_cuda=$("$venv_path/bin/python" -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')" 2>/dev/null || echo "unknown")
      [ "$has_cuda" = "Yes" ] && check_pass "CUDA available" || check_warn "CUDA not available (CPU-only mode)"
    fi
  else
    check_fail "PyTorch not found in venv"
  fi
else
  check_warn "Virtual environment not found - cannot verify PyTorch"
fi

# SECTION 4: Smart Router
echo ""
echo -e "${CYAN}Smart Router:${NC}"
if [ -f "$ROUTER_DIR/router.py" ]; then
  check_pass "Router script exists: $ROUTER_DIR/router.py"
  if python3 -m py_compile "$ROUTER_DIR/router.py" 2>/dev/null; then
    check_pass "Router Python syntax valid"
  else
    check_fail "Router has Python syntax errors"
  fi
else
  check_warn "Router not found - may not have been generated yet"
fi

if [ -d "$ROUTER_DIR/venv/bin" ]; then
  check_pass "Router venv exists"
else
  check_warn "Router venv not found"
fi

# SECTION 5: Logging
echo ""
echo -e "${CYAN}Logging:${NC}"
if [ -d "$LOG_DIR" ] && [ -w "$LOG_DIR" ]; then
  check_pass "Log directory exists and is writable: $LOG_DIR"
  local log_count
  log_count=$(find "$LOG_DIR" -type f -name "*.log" 2>/dev/null | wc -l || echo "0")
  [ "$log_count" -gt 0 ] && check_pass "Found $log_count log file(s)" || check_warn "No log files found yet (expected after first run)"
else
  check_warn "Log directory not writable"
fi

# SECTION 6: nginx Configuration (optional)
echo ""
echo -e "${CYAN}nginx (Optional):${NC}"
if command -v nginx &>/dev/null; then
  check_pass "nginx is installed"
  if [ -f /etc/nginx/sites-available/stable-diffusion ] || [ -f /etc/nginx/conf.d/stable-diffusion.conf ]; then
    check_pass "nginx stable-diffusion config found"
    if sudo nginx -t -q 2>/dev/null; then
      check_pass "nginx configuration valid"
    else
      check_warn "nginx configuration test failed - may need manual fixes"
    fi
  else
    check_warn "nginx not configured for stable-diffusion (may need to run --install)"
  fi
else
  check_warn "nginx not installed (optional - use Docker or manual reverse proxy)"
fi

# SECTION 7: Port Availability
echo ""
echo -e "${CYAN}Port Availability:${NC}"
if ! netstat -tuln 2>/dev/null | grep -q ":$ROUTER_PORT "; then
  check_pass "Router port $ROUTER_PORT is available"
else
  check_warn "Router port $ROUTER_PORT appears to be in use"
fi

if ! netstat -tuln 2>/dev/null | grep -q ":$NGINX_PORT "; then
  check_pass "nginx port $NGINX_PORT is available"
else
  check_warn "nginx port $NGINX_PORT appears to be in use"
fi

# SECTION 8: System Dependencies
echo ""
echo -e "${CYAN}System Dependencies:${NC}"
for cmd in git python3 gcc g++ make; do
  if command -v "$cmd" &>/dev/null; then
    check_pass "$cmd is installed"
  else
    check_fail "$cmd is NOT installed (required)"
  fi
done

# SECTION 9: GPU Detection
echo ""
echo -e "${CYAN}GPU Support:${NC}"
if command -v nvidia-smi &>/dev/null; then
  local gpu_count
  gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")
  check_pass "NVIDIA GPUs detected: $gpu_count"
elif command -v rocminfo &>/dev/null; then
  check_pass "AMD ROCm detected"
elif command -v clinfo &>/dev/null; then
  check_pass "Intel OneAPI/OpenCL detected"
else
  check_warn "No GPU vendor tools detected (CPU-only mode or drivers not installed)"
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "====== Verification Summary ======"
echo -e "  ${GREEN}Passed:${NC}  $PASSED"
echo -e "  ${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "  ${RED}Failed:${NC}  $FAILED"
echo ""

# Determine exit code
if [ "$FAILED" -gt 0 ]; then
  echo -e "${RED}Installation verification FAILED${NC} - critical issues detected."
  echo "See failures above. You may need to re-run: ./run_stablediffusion.sh --install"
  exit 2
elif [ "$WARNINGS" -gt 0 ]; then
  echo -e "${YELLOW}Installation verification PASSED with warnings${NC}"
  echo "Non-critical issues detected (see warnings above). Installation should work."
  exit 0
else
  echo -e "${GREEN}Installation verification PASSED${NC} - everything looks good!"
  echo "Ready to launch: ./run_stablediffusion.sh"
  exit 0
fi
