# GitHub Actions Workflows

This directory contains documentation for GitHub Actions workflows that should be implemented for the Edge TPU v5 Benchmark Suite.

## Required Workflows

### 1. Main CI/CD Pipeline (ci.yml)
- **Location**: `.github/workflows/ci.yml`
- **Triggers**: Pull requests, pushes to main
- **Purpose**: Comprehensive testing and quality checks

### 2. Security Scanning (security.yml)
- **Location**: `.github/workflows/security.yml` 
- **Triggers**: Push, pull request, scheduled
- **Purpose**: Dependency scanning, SBOM generation

### 3. Release Automation (release.yml)
- **Location**: `.github/workflows/release.yml`
- **Triggers**: Tag creation, release creation
- **Purpose**: Automated package building and publishing

### 4. Container Build (docker.yml)
- **Location**: `.github/workflows/docker.yml`
- **Triggers**: Push to main, releases
- **Purpose**: Docker image building and registry push

## Implementation Steps

1. Create `.github/workflows/` directory in repository root
2. Copy workflow files from templates in this directory
3. Configure repository secrets as documented
4. Test workflows with draft pull request

## Workflow Templates

See individual workflow template files:
- [ci.yml.template](ci.yml.template) - Main CI/CD pipeline
- [security.yml.template](security.yml.template) - Security scanning
- [release.yml.template](release.yml.template) - Release automation
- [docker.yml.template](docker.yml.template) - Container workflows

## Repository Configuration

Required repository settings and secrets are documented in [repo-setup.md](repo-setup.md).