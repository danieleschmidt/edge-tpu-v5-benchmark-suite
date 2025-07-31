# Supply Chain Security Guide

This document outlines the supply chain security measures implemented for the Edge TPU v5 Benchmark Suite.

## Overview

Supply chain security protects against malicious code injection through dependencies, build processes, and distribution channels.

## Dependency Management

### Automated Dependency Updates
- **Dependabot**: Automated dependency updates with security patches
- **Weekly Schedule**: Regular updates for Python, Docker, and GitHub Actions
- **Version Pinning**: Critical dependencies pinned to specific versions
- **Review Process**: All dependency updates require code review

### Dependency Scanning
- **pip-audit**: Python package vulnerability scanning
- **Safety**: Additional Python security checks
- **SBOM Generation**: Software Bill of Materials for transparency
- **License Compliance**: Automated license compatibility checking

## Build Security

### Secure Build Environment
- **GitHub Actions**: Isolated, ephemeral build environments
- **Pinned Actions**: All GitHub Actions pinned to specific commit hashes
- **Multi-Stage Builds**: Docker builds use multi-stage pattern
- **Minimal Base Images**: Distroless/minimal container images

### Code Integrity
- **Signed Commits**: Encourage GPG-signed commits
- **CodeQL Analysis**: Semantic code analysis for vulnerabilities
- **Secret Scanning**: Automated detection of committed secrets
- **Pre-commit Hooks**: Client-side security checks

## Distribution Security

### Package Publishing
- **Trusted Publishers**: Use GitHub OIDC for PyPI publishing
- **Package Signing**: Sign packages with GPG keys
- **Checksums**: Provide SHA256 checksums for all releases
- **Provenance**: Generate SLSA provenance attestations

### Container Security
- **Image Scanning**: Trivy vulnerability scanning
- **Distroless Images**: Minimal attack surface
- **Non-root User**: Containers run as non-privileged user
- **Registry Security**: Signed container images

## Runtime Security

### Model Security
```python
# Example: Secure model loading
def load_model_securely(model_path: str) -> TPUModel:
    # Validate file exists and size
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Check file size (prevent DoS)
    size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    if size_mb > MAX_MODEL_SIZE_MB:
        raise ValueError(f"Model too large: {size_mb}MB > {MAX_MODEL_SIZE_MB}MB")
    
    # Verify checksum if available
    if (checksum_file := Path(f"{model_path}.sha256")).exists():
        verify_checksum(model_path, checksum_file.read_text().strip())
    
    return TPUModel.load(model_path)
```

### Input Validation
```python
# Example: Validate benchmark parameters
def validate_benchmark_params(params: BenchmarkParams) -> None:
    if not (1 <= params.iterations <= 100000):
        raise ValueError("Invalid iteration count")
    
    if not (1 <= params.batch_size <= 256):
        raise ValueError("Invalid batch size")
    
    if params.input_shape and any(dim <= 0 for dim in params.input_shape):
        raise ValueError("Invalid input shape")
```

## Monitoring and Response

### Security Monitoring
- **Vulnerability Alerts**: GitHub Security Advisories
- **CVE Tracking**: Monitor CVE databases for dependencies
- **License Changes**: Track license changes in dependencies
- **Incident Response**: Documented process for security incidents

### Security Updates
- **Priority Levels**: Critical, High, Medium, Low
- **Response Times**: 
  - Critical: 24 hours
  - High: 72 hours
  - Medium: 1 week
  - Low: Next release cycle

## SLSA Compliance

### Level 1: Source Requirements
✅ Version controlled source  
✅ Generated build steps  
✅ Build service

### Level 2: Build Requirements  
✅ Hosted build service  
✅ Source and build integrity  
✅ Generated provenance

### Level 3: Hardened Build
🔄 Source and build integrity (in progress)  
🔄 Isolated build environment  
🔄 Provenance non-falsifiable

### Level 4: Maximum Trust
🔄 Two-person reviewed source  
🔄 Reproducible builds  
🔄 Sealed provenance

## Security Tools

### Development Tools
- **bandit**: Python security linting
- **safety**: Python dependency vulnerability scanner
- **pip-audit**: Python package auditing
- **semgrep**: Static analysis security scanner

### CI/CD Tools
- **CodeQL**: Semantic code analysis
- **Trivy**: Container vulnerability scanner
- **TruffleHog**: Secret scanning
- **Dependabot**: Automated dependency updates

### Runtime Tools
- **SBOM**: Software Bill of Materials generation
- **Provenance**: Build provenance attestations
- **Checksums**: File integrity verification
- **Signatures**: Cryptographic signatures

## Implementation Checklist

### Repository Setup
- [ ] Enable Dependabot
- [ ] Configure security workflows
- [ ] Set up CodeQL analysis
- [ ] Enable secret scanning
- [ ] Configure branch protection

### Build Pipeline
- [ ] Pin all action versions
- [ ] Add security scanning steps
- [ ] Generate SBOM
- [ ] Create provenance attestations
- [ ] Sign artifacts

### Distribution
- [ ] Use trusted publishers
- [ ] Sign packages/containers
- [ ] Provide checksums
- [ ] Document security measures

### Monitoring
- [ ] Set up vulnerability alerts
- [ ] Monitor security advisories
- [ ] Track dependency licenses
- [ ] Document incident response

## Resources

- [SLSA Framework](https://slsa.dev/)
- [NIST SSDF](https://csrc.nist.gov/Projects/ssdf)
- [OpenSSF Scorecard](https://github.com/ossf/scorecard)
- [Python Security Guide](https://python-security.readthedocs.io/)
- [Container Security Guide](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)