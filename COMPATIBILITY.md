# Compatibility Matrix

This document tracks tested and supported configurations for the Stable Diffusion Multi-GPU Launcher.

## Linux Distributions

| Distribution | Version | Package Manager | Status | Notes |
|---|---|---|---|---|
| **Ubuntu** | 20.04 LTS | apt | ✅ **Production** | Fully tested and supported |
| **Ubuntu** | 22.04 LTS | apt | ✅ **Production** | Recommended for new installs |
| **Ubuntu** | 24.04 LTS | apt | ✅ **Production** | Latest, fully compatible |
| **Debian** | 11 (Bullseye) | apt | ✅ **Supported** | Mostly compatible, same as Ubuntu |
| **Debian** | 12 (Bookworm) | apt | ✅ **Supported** | Fully compatible |
| **Fedora** | 38+ | dnf | ⚠️ **Experimental** | Partially tested; multi-distro launcher in progress |
| **CentOS Stream** | 9+ | dnf | ⚠️ **Experimental** | Untested; contributions welcome |
| **RHEL** | 9+ | dnf | ⚠️ **Experimental** | Untested; contributions welcome |
| **Arch Linux** | Latest | pacman | ⚠️ **Experimental** | Untested; rolling release may have compatibility issues |
| **openSUSE** | Leap/Tumbleweed | zypper | ⚠️ **Experimental** | Untested; contributions welcome |

**Legend:**
- ✅ **Production**: Tested and recommended for production workloads
- ⚠️ **Experimental**: Use `run_stablediffusion_multidistro.sh` (may require manual fixes)
- ❌ **Unsupported**: Not tested or known to have issues

## Python Versions

| Python Version | Status | Notes |
|---|---|---|
| 3.9 | ✅ Supported | Functional but not recommended |
| 3.10 | ✅ **Preferred** | Best compatibility with SD dependencies |
| 3.11 | ✅ Supported | Works well; some packages build from source |
| 3.12 | ⚠️ Partial | Some dependencies not yet published as wheels; slower install |
| 3.13+ | ❌ Not tested | Too new; likely conflicts |

## PyTorch & CUDA Versions

### NVIDIA GPU Stack

| CUDA Version | PyTorch | Driver Min | Status | GPU Examples |
|---|---|---|---|---|
| **cu124** | 2.6.0 | 560+ | ✅ Recommended | RTX 50xx (Blackwell) |
| **cu121** | 2.5.1 | 525+ | ✅ Current stable | RTX 40xx (Ada), A100 |
| **cu118** | 2.3.1 | 520+ | ✅ Supported | RTX 30xx (Ampere), A10 |
| **cu117** | 2.0.1 | 450+ | ✅ Legacy | RTX 20xx (Turing), V100 |
| **CPU** | 2.0.1 | - | ✅ Fallback | CPU-only (very slow) |

**Matching Strategy:**
- Launcher auto-detects driver version and selects compatible CUDA tag
- PyTorch, torchvision, and torchaudio versions are pinned together (no conflicts)
- All three packages installed simultaneously in one pip command

### AMD GPU Stack

| ROCm Version | PyTorch | GPU Family | Status | Notes |
|---|---|---|---|---|
| 6.x | 2.5.x | RDNA2/3 | ✅ Recommended | RX 6xxx, 7xxx - best support |
| 5.x | 2.4.x | RDNA2/RDNA1 | ✅ Supported | RX 5700 XT may work |
| Older | 2.0.x | Vega | ⚠️ Partial | May require `HSA_OVERRIDE_GFX_VERSION` |

**Known Issues:**
- Multi-distro launcher ROCm setup not yet implemented (Ubuntu only)
- Some RDNA2 cards need environment variable tweaks
- No xformers support on ROCm (uses slower attention)

### Intel GPU Stack

| IPEX Version | PyTorch | GPU | Status | Notes |
|---|---|---|---|---|
| 2.x | 2.5.x | Arc A770 | ⚠️ Experimental | Level Zero driver required |
| 2.x | 2.5.x | Arc A750 | ⚠️ Experimental | Slower than NVIDIA equivalent |
| 2.x | 2.5.x | Arc A380 | ⚠️ Experimental | Entry-level, limited VRAM |

**Known Issues:**
- IPEX/XPU significantly slower than CUDA/ROCm
- Multi-distro launcher not yet implemented
- Driver/package installation not automated

## GPU Support Details

### NVIDIA Architectures

| Compute Capability | Architecture | Models | xformers | FP16 | Status |
|---|---|---|---|---|---|
| 9.0 | Hopper | H100, H20 | ✅ | ✅ | ✅ Full |
| 8.9 | Ada | RTX 4090, RTX 4080 | ✅ | ✅ FP8 | ✅ Full |
| 8.0 | Ampere | RTX 3090, RTX 3080, A100 | ✅ | ✅ BF16 | ✅ Full |
| 7.5 | Turing | RTX 2080, T4 | ✅ | ✅ | ✅ Full |
| 7.0 | Volta | V100, Tesla V100 | ❌ | ❌ FP32 only | ⚠️ Degraded |
| 6.1 | Pascal | GTX 1080, P100 | ❌ | ⚠️ Unreliable | ⚠️ Limited |
| 5.0 | Maxwell | GTX 950, GTX 960 | ❌ | ❌ | ❌ Unsupported |

**Notes:**
- V100 (Volta) forces FP32 due to Linux driver bug with FP16; no xformers
- Turing+ fully supported with xformers (30%+ speed boost)
- Maxwell and older: not recommended for production

## Dependency Versions

### System Packages (apt)

| Package | Min Version | Current | Status |
|---|---|---|---|
| build-essential | any | latest | ✅ Required |
| python3.10 | 3.10+ | 3.10.12+ | ✅ Required |
| gcc/g++ | 9+ | 11+ | ✅ Required |
| libssl-dev | 1.1+ | 3.x | ✅ Required |
| libffi-dev | any | 3.4+ | ✅ Required |
| nginx | 1.18+ | 1.24+ | ⚠️ Optional |

### Python Packages (pip)

| Package | Min Version | Current | Status | Notes |
|---|---|---|---|---|
| torch | 2.0 | 2.6.0 | ✅ Pinned | Version determined by CUDA tag |
| torchvision | 0.15 | 0.21.0 | ✅ Pinned | Must match torch version exactly |
| torchaudio | 2.0 | 2.6.0 | ✅ Pinned | Must match torch version exactly |
| xformers | 0.0.20+ | 0.0.27+ | ✅ Recommended | 30%+ speed boost; NVIDIA only |
| fastapi | 0.95+ | 0.110+ | ✅ Router | Smart GPU routing |
| uvicorn | 0.20+ | 0.27+ | ✅ Router | ASGI server |
| requests | 2.28+ | 2.31+ | ✅ Utils | HTTP client |

## Browser & Client Compatibility

| Client | Min Version | Status | Notes |
|---|---|---|---|
| Chrome / Chromium | 90+ | ✅ Full | WebSocket support required |
| Firefox | 88+ | ✅ Full | WebSocket support required |
| Safari | 15.1+ | ✅ Full | macOS 12+, iOS 15.1+ |
| Edge | 90+ | ✅ Full | Based on Chromium |
| curl | 7.64+ | ✅ Full | CLI API calls work well |
| Python requests | 2.25+ | ✅ Full | Programmatic access |

## Storage Requirements

| Component | Size | Speed Requirement |
|---|---|---|
| SD 1.5 model | 4-7 GB | HDD acceptable |
| SDXL model | 15-20 GB | HDD acceptable |
| WebUI + deps | 8-10 GB | SSD recommended |
| Python venv | 2-3 GB | SSD recommended |
| pip wheels cache | 3-5 GB | HDD acceptable |
| **Total** | **40-50 GB** | **SSD for /tmp + venv** |

**Recommendation:** 
- Install WebUI and venv on fast NVMe SSD for 50%+ faster startup
- Models can be on HDD/network storage (first load slower, then cached in VRAM)

## Network & Performance

| Metric | Min | Recommended | Status |
|---|---|---|---|
| Bandwidth (model download) | 10 Mbps | 100+ Mbps | For first install |
| Router latency (per request) | <50ms | <30ms | ✅ Typical: 10-15ms |
| WebUI response time | <5s | <2s | Depends on GPU |
| Generation (512×512) | <60s | <10s | Depends on GPU tier |

## Testing & Validation

### How to Report Compatibility Issues

1. Run diagnostics: `./run_stablediffusion.sh --diag`
2. Check logs: `tail -50 ~/sd-logs/*.log`
3. Run verification: `./scripts/verify_install.sh`
4. Open GitHub issue with:
   - OS/distro version
   - GPU model and driver version
   - Python version
   - Output of `--diag`
   - Error messages from logs

### How to Test a New Configuration

```bash
# Fresh install on new distro
./run_stablediffusion_multidistro.sh --diag
./run_stablediffusion_multidistro.sh --install 2>&1 | tee install.log
./scripts/verify_install.sh
./run_stablediffusion_multidistro.sh  # Launch and test

# If issues occur
tail -100 ~/sd-logs/*.log
./run_stablediffusion_multidistro.sh --diag
```

## Update Frequency

This compatibility matrix is updated:
- **Weekly** during active development
- **Per release** before major version bumps
- **Ad hoc** when critical compatibility issues are discovered

Last updated: 2026-06-06
