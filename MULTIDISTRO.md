# Multi-Distro Support (Experimental)

This document describes the **experimental** multi-distro launcher script.

## Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Distro Detection** | ✅ Complete | Detects /etc/os-release, Fedora-specific files |
| **Package Manager Detection** | ✅ Complete | Detects apt, dnf, pacman, zypper |
| **Package Abstraction Layer** | ✅ Complete | Maps package names across 4 distros |
| **System Dependencies** | ✅ Implemented | Installs build tools for each distro |
| **Python Detection** | ✅ Complete | Finds python3.9-3.12 |
| **GPU Detection** | ✅ Partial | NVIDIA/AMD/Intel detection (distro-agnostic) |
| **PyTorch Installation** | ⏳ TODO | Needs per-distro PyTorch setup |
| **ROCm Setup** | ⏳ TODO | Requires distro-specific repo URLs |
| **WebUI Cloning** | ⏳ TODO | Should work on any distro (git-based) |
| **Router Installation** | ⏳ TODO | Python-only (should be distro-agnostic) |
| **Full Install/Update/Uninstall** | ⏳ TODO | Not yet ported |

## Usage

```bash
chmod +x run_stablediffusion_multidistro.sh

# Detect distro and run diagnostics
./run_stablediffusion_multidistro.sh --diag

# Install (partially implemented)
./run_stablediffusion_multidistro.sh --install

# Custom WebUI path
./run_stablediffusion_multidistro.sh --webui-dir /mnt/nvme/sd --install
```

## Supported Distros

### Tested ✅

- **Ubuntu 20.04, 22.04, 24.04** — Fully compatible with original script
- **Debian 11, 12** — Mostly compatible (apt-based, similar packages)

### Partially Tested ⚠️

- **Fedora 38+** — dnf package manager works; ROCm/CUDA setup untested
- **CentOS Stream 9+** — Should work like Fedora; not tested
- **Arch Linux** — pacman setup works; Python/ROCm untested
- **openSUSE Leap/Tumbleweed** — zypper setup works; package names may differ

### Not Tested ❌

- **Alpine** (musl libc, different package layout)
- **Clear Linux** (swupd package manager)
- **NixOS** (declarative package management)

## Architecture

### Package Manager Abstraction

The script defines package name mappings as associative arrays:

```bash
declare -A PKGS_APT=( [gcc]="gcc" [build_essential]="build-essential" ... )
declare -A PKGS_DNF=( [gcc]="gcc" [build_essential]="@development-tools" ... )
declare -A PKGS_PACMAN=( [gcc]="gcc" [build_essential]="base-devel" ... )
declare -A PKGS_ZYPPER=( [gcc]="gcc" [build_essential]="-t pattern devel_basis" ... )
```

Functions like `get_package_name()`, `pkg_install()`, and `pkg_update()` handle distro differences.

### Conditional Logic

Key functions detect the package manager and branch accordingly:

```bash
check_system_deps() {
  case "$PKG_MANAGER" in
    apt)    # Ubuntu/Debian logic
    dnf)    # Fedora/CentOS logic
    pacman) # Arch logic
    zypper) # openSUSE logic
  esac
}
```

## What Still Needs Work

### 1. PyTorch Installation

Different distros have different conventions:
- **apt**: `pip install torch==2.x.x+cuXXX` from index URL
- **dnf**: May need to install CUDA toolkit from distro repos first
- **pacman**: CUDA/ROCm may be in AUR instead of official repos
- **zypper**: Similar to apt; check openSUSE repos

### 2. ROCm Setup

ROCm repositories vary by distro:
- **Ubuntu**: `repo.radeon.com/amdgpu-install/ubuntu/$VERSION`
- **Fedora**: `repo.radeon.com/amdgpu-install/fedora/rhel/$VERSION`
- **RHEL/CentOS**: `repo.radeon.com/amdgpu-install/rhel/rhel$VERSION`
- **Arch**: AUR packages (rocm, hip)
- **openSUSE**: Community repos

The current Ubuntu-hardcoded ROCm installation (`SECTION 13` in original) needs distro-aware URLs.

### 3. NVIDIA CUDA Setup

- **Ubuntu/Debian**: APT packages from NVIDIA repos
- **Fedora**: RPM packages from NVIDIA repos
- **Arch**: `cuda` package from AUR
- **openSUSE**: Community repos

Current script assumes NVIDIA driver is pre-installed; CUDA toolkit installation is not automated.

### 4. nginx Configuration

All distros have nginx; placement differs:
- **apt**: `/etc/nginx/sites-available/`
- **dnf**: `/etc/nginx/conf.d/`
- **pacman**: `/etc/nginx/`
- **zypper**: `/etc/nginx/`

### 5. Systemd vs Init Systems

Some older distro versions may use different init systems. Assuming systemd is widely available on modern distros, but script doesn't handle `service` fallback.

### 6. Python Package Install

Some distros separate `-dev` packages; others bundle them. Script already tries version-specific packages but may need refinement for distros without python3.10 in repos (e.g., Arch).

## Contributing

To test or improve support for a specific distro:

1. Test the diagnostic output:
   ```bash
   ./run_stablediffusion_multidistro.sh --diag
   ```

2. Try install (will fail at unimplemented steps, but tests early logic):
   ```bash
   ./run_stablediffusion_multidistro.sh --install 2>&1 | tee install.log
   ```

3. Debug package mapping by running:
   ```bash
   source run_stablediffusion_multidistro.sh
   detect_distro
   detect_package_manager
   echo "GCC package: $(get_package_name gcc)"
   echo "Python3.10 dev: $(get_package_name python3_10_dev)"
   ```

4. Report findings (package names, missing packages, failing steps) on GitHub.

## Recommended Path Forward

For production use, recommend staying on **Ubuntu 20.04+** with the original `run_stablediffusion.sh` until multi-distro support is fully tested.

To achieve full multi-distro support:

1. **Port remaining install functions** (PyTorch, ROCm, router setup)
2. **Test on each supported distro** with various GPU combinations
3. **Document distro-specific workarounds** (e.g., AUR for Arch, NVidia driver installation)
4. **Create CI/CD testing** with Docker images for each distro
5. **Maintain separate docs** for distro-specific setup (drivers, repos, etc.)

## Quick Start (Ubuntu/Debian Only)

For now, Ubuntu/Debian users should use the original script:
```bash
./run_stablediffusion.sh --install
```

Multi-distro support is experimental and not production-ready.
