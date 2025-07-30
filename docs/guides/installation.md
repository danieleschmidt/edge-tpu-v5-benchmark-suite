# Installation Guide

This guide covers installation of the Edge TPU v5 Benchmark Suite.

## Requirements

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8 or higher
- **Hardware**: Google TPU v5 edge device

### TPU v5 Edge Hardware
- **Model**: TPU v5 Edge
- **Performance**: 8 TOPS peak, 50 TOPS/W efficiency
- **Memory**: 4GB on-device memory
- **Interface**: USB 3.0 or PCIe

## Installation Methods

### Method 1: PyPI (Recommended)

```bash
# Install latest stable version
pip install edge-tpu-v5-benchmark

# Verify installation
edge-tpu-v5-benchmark --version
```

### Method 2: From Source

```bash
# Clone repository
git clone https://github.com/yourusername/edge-tpu-v5-benchmark-suite.git
cd edge-tpu-v5-benchmark-suite

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev,test,docs]"
```

### Method 3: Docker

```bash
# Pull image
docker pull edge-tpu-v5-benchmark:latest

# Or build from source
git clone https://github.com/yourusername/edge-tpu-v5-benchmark-suite.git
cd edge-tpu-v5-benchmark-suite
docker build -t edge-tpu-v5-benchmark .
```

## TPU Runtime Installation

### Ubuntu/Debian

```bash
# Add Google repository
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
  sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

# Add signing key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Update and install
sudo apt update
sudo apt install edgetpu-runtime-v5
```

### CentOS/RHEL/Fedora

```bash
# Add Google repository
sudo tee /etc/yum.repos.d/coral-edgetpu.repo << 'EOF'
[coral-edgetpu-stable]
name=Coral Edge TPU Stable
baseurl=https://packages.cloud.google.com/yum/repos/coral-edgetpu-stable-el8-x86_64
enabled=1
gpgcheck=1
gpgkey=https://packages.cloud.google.com/yum/doc/yum-key.gpg
       https://packages.cloud.google.com/yum/doc/rpm-package-key.gpg
EOF

# Install
sudo yum install edgetpu-runtime-v5
```

## Verification

### Check TPU Detection

```bash
# Detect TPU devices
edge-tpu-v5-benchmark detect

# Expected output:
# ✓ Found 1 TPU v5 edge device at /dev/apex_0
# Device info:
#   - Version: TPU v5 Edge
#   - Runtime: 2.15.0
#   - Compiler: 3.0
```

### Run Test Benchmark

```bash
# Quick test
edge-tpu-v5-benchmark run --model mobilenet_v3 --iterations 10

# Expected output:
# Running benchmark: mobilenet_v3
# Iterations: 10
# ✓ Benchmark completed
# Results:
#   - Throughput: 892.3 FPS
#   - Latency p99: 1.12 ms
#   - Power: 0.85 W
```

## Troubleshooting

### TPU Not Detected

**Check device permissions:**
```bash
ls -la /dev/apex_*
# Should show: crw-rw-rw- 1 root root 120, 0 /dev/apex_0

# Fix permissions if needed
sudo chmod 666 /dev/apex_0
```

**Check runtime installation:**
```bash
# Verify runtime files
ls /usr/lib/x86_64-linux-gnu/libedgetpu*

# Check for udev rules
ls /etc/udev/rules.d/*edgetpu*
```

**Restart udev service:**
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Python Import Errors

**Install missing dependencies:**
```bash
pip install --upgrade pip
pip install edge-tpu-v5-benchmark --force-reinstall
```

**Check Python path:**
```python
import sys
print(sys.path)

# Verify package installation
import edge_tpu_v5_benchmark
print(edge_tpu_v5_benchmark.__version__)
```

### Docker Issues

**Device access in container:**
```bash
# Run with device access
docker run --device=/dev/apex_0 edge-tpu-v5-benchmark detect

# Or use privileged mode
docker run --privileged edge-tpu-v5-benchmark detect
```

**Build issues:**
```bash
# Clean build
docker system prune -f
docker build --no-cache -t edge-tpu-v5-benchmark .
```

## Advanced Installation

### Development Environment

```bash
# Full development setup
git clone https://github.com/yourusername/edge-tpu-v5-benchmark-suite.git
cd edge-tpu-v5-benchmark-suite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with all development tools
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install

# Verify development setup
make test
```

### Custom Compiler

```bash
# Install specific TPU compiler version
export TPU_COMPILER_VERSION=3.0.1
pip install edge-tpu-v5-benchmark[compiler-${TPU_COMPILER_VERSION}]
```

### Multiple TPU Devices

```bash
# For systems with multiple TPUs
export TPU_DEVICES="/dev/apex_0,/dev/apex_1"
edge-tpu-v5-benchmark detect --all-devices
```

## Performance Optimization

### System Tuning

```bash
# Increase file descriptor limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU frequency scaling
sudo systemctl disable ondemand
```

### Memory Configuration

```bash
# Increase shared memory for large models
echo "kernel.shmmax = 17179869184" | sudo tee -a /etc/sysctl.conf
echo "kernel.shmall = 4194304" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Uninstallation

### Remove Package

```bash
pip uninstall edge-tpu-v5-benchmark
```

### Remove TPU Runtime

```bash
# Ubuntu/Debian
sudo apt remove edgetpu-runtime-v5

# CentOS/RHEL/Fedora  
sudo yum remove edgetpu-runtime-v5
```

### Clean Configuration

```bash
# Remove configuration files
rm -rf ~/.edge-tpu-v5-benchmark/
sudo rm -f /etc/udev/rules.d/*edgetpu*
```

## Getting Help

If you encounter issues:

1. **Check logs**: `edge-tpu-v5-benchmark --verbose detect`
2. **Search issues**: [GitHub Issues](https://github.com/yourusername/edge-tpu-v5-benchmark-suite/issues)
3. **Ask questions**: [GitHub Discussions](https://github.com/yourusername/edge-tpu-v5-benchmark-suite/discussions)
4. **Email support**: daniel@terragonlabs.com