# Local Wheels for Docker Builds

This guide explains how to pre-download Python wheel files (particularly large packages like PyTorch and torchvision) to speed up and stabilize your Docker builds.

## Table of Contents

- [Why Use Local Wheels?](#why-use-local-wheels)
- [Quick Start](#quick-start)
- [Detailed Setup Guide](#detailed-setup-guide)
- [Troubleshooting](#troubleshooting)
- [Additional Notes](#additional-notes)

## Why Use Local Wheels?

**Benefits:**

- **Reliability**: Avoid network timeout failures during Docker builds
- **Speed**: Significantly faster builds by eliminating large downloads
- **Reproducibility**: Guaranteed consistent package versions
- **Offline builds**: Build Docker images without internet access

Large packages like PyTorch (500+ MB) can cause build failures when downloaded over unreliable networks. Pre-downloading wheels eliminates this risk.

## Quick Start

For most users working with CUDA 11.7 and Python 3.8:

```bash
# 1. Create wheels directory
mkdir -p environment/wheels

# 2. Download PyTorch wheels for CUDA 11.7
wget -P environment/wheels \
  "https://download.pytorch.org/whl/cu117/torch-1.13.1%2Bcu117-cp38-cp38-linux_x86_64.whl" \
  "https://download.pytorch.org/whl/cu117/torchvision-0.14.1%2Bcu117-cp38-cp38-linux_x86_64.whl"

# 3. Verify downloads
ls -lh environment/wheels

# 4. Build Docker image
docker build -f environment/Dockerfile -t my-image:latest .
```

## Detailed Setup Guide

### Prerequisites

**Important**: Commands must be run from a Linux environment (native Linux or WSL on Windows) to download compatible wheels for the Docker Linux container.

### Step 1: Prepare Your Environment (WSL Users)

If you're using WSL on Windows, ensure you're using the system Python:

```bash
# Remove Windows paths temporarily (this session only)
export PATH=$(echo $PATH | tr ':' '\n' | grep -v '^/mnt/c' | paste -sd: -)

# Verify you're using WSL Python
which python3  # Should show /usr/bin/python3
python3 -V
```

### Step 2: Install/Update pip

```bash
# Install pip (Ubuntu/Debian)
sudo apt update
sudo apt install -y python3-pip ca-certificates

# Upgrade pip and tools (recommended)
python3 -m pip install --upgrade pip setuptools wheel
```

### Step 3: Create Wheels Directory

```bash
mkdir -p environment/wheels
```

### Step 4: Download Wheels

Choose the method that works best for you:

#### Option A: Direct Download with wget (Recommended)

**For CUDA 11.7 + Python 3.8:**

```bash
wget -P environment/wheels \
  "https://download.pytorch.org/whl/cu117/torch-1.13.1%2Bcu117-cp38-cp38-linux_x86_64.whl" \
  "https://download.pytorch.org/whl/cu117/torchvision-0.14.1%2Bcu117-cp38-cp38-linux_x86_64.whl"

# Optional: Add torchaudio
wget -P environment/wheels \
  "https://download.pytorch.org/whl/cu117/torchaudio-0.13.1%2Bcu117-cp38-cp38-linux_x86_64.whl"
```

**For CPU-only builds:**

```bash
python3 -m pip download --dest environment/wheels \
  torch==1.13.1 \
  torchvision==0.14.1
```

#### Option B: Using pip download

```bash
python3 -m pip download --no-deps --dest environment/wheels \
  "torch==1.13.1+cu117" \
  "torchvision==0.14.1+cu117" \
  -f https://download.pytorch.org/whl/cu117 \
  --timeout 120 \
  --retries 5
```

#### Option C: Using curl

```bash
curl -L -o environment/wheels/torch-1.13.1+cu117-cp38-cp38-linux_x86_64.whl \
  "https://download.pytorch.org/whl/cu117/torch-1.13.1%2Bcu117-cp38-cp38-linux_x86_64.whl"

curl -L -o environment/wheels/torchvision-0.14.1+cu117-cp38-cp38-linux_x86_64.whl \
  "https://download.pytorch.org/whl/cu117/torchvision-0.14.1%2Bcu117-cp38-cp38-linux_x86_64.whl"
```

### Step 5: Verify Downloads

```bash
ls -lh environment/wheels
```

You should see wheel files with names like:

- `torch-1.13.1+cu117-cp38-cp38-linux_x86_64.whl` (~800 MB)
- `torchvision-0.14.1+cu117-cp38-cp38-linux_x86_64.whl` (~40 MB)

### Step 6: Build Docker Image

```bash
# From project root
docker build -f environment/Dockerfile -t my-image:latest .
```

The Dockerfile automatically detects and installs any `.whl` files in the `environment/wheels/` directory.

## Troubleshooting

### Finding Available Wheels

If you need a different Python version or CUDA version, list available wheels:

```bash
# List all Python 3.8 wheels for CUDA 11.7
curl -s https://download.pytorch.org/whl/cu117/torch_stable.html | \
  grep -oP 'href="[^"]+"' | \
  sed -e 's/href="//' -e 's/"$//' | \
  grep -i 'cp38' | \
  sort -u
```

Replace `cp38` with your Python version:

- Python 3.7: `cp37`
- Python 3.8: `cp38`
- Python 3.9: `cp39`
- Python 3.10: `cp310`

### Common Issues

**Problem**: Wrong platform wheels downloaded

- **Solution**: Always download from a Linux environment (WSL or Linux VM), not from macOS or native Windows

**Problem**: pip download fails with resolution errors

- **Solution**: Use the direct wget method (Option A) instead

**Problem**: Docker build still downloads packages

- **Solution**: Verify wheels are in `environment/wheels/` and have `.whl` extension

## Additional Notes

### Version Compatibility

Make sure your wheel versions match your requirements:

| CUDA Version | PyTorch Suffix | Example |
|--------------|----------------|---------|
| CPU only | (none) | `torch==1.13.1` |
| CUDA 11.7 | `+cu117` | `torch==1.13.1+cu117` |
| CUDA 11.8 | `+cu118` | `torch==1.13.1+cu118` |

### How It Works

The `environment/Dockerfile` includes logic to:

1. Copy the `environment/wheels/` directory into the build context
2. Install any `.whl` files found before other dependencies
3. Fall back to conda/pip install if no wheels are present

This means the wheels directory can be empty and the build will still work—it just won't benefit from the speed improvements.

### Additional Packages

To include other packages as local wheels:

```bash
# Download any package as a wheel
python3 -m pip download --dest environment/wheels numpy==1.24.0

# Download with all dependencies
python3 -m pip download --dest environment/wheels scikit-learn==1.2.0
```

### Cleaning Up

To remove downloaded wheels and start fresh:

```bash
rm -rf environment/wheels/*.whl
```

---

**Questions or issues?** Check that:

1. You're running commands from a Linux environment
2. Wheel filenames match your Python version (e.g., `cp38` for Python 3.8)
3. The `environment/wheels/` directory exists and contains `.whl` files
4. Your Docker build is running from the project root directory
