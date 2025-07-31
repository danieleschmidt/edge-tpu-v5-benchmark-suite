# Repository Setup Guide

This guide covers the configuration needed to implement the GitHub Actions workflows for the Edge TPU v5 Benchmark Suite.

## Repository Settings

### Branch Protection Rules

Configure branch protection for `main` branch:

1. Go to **Settings** → **Branches**
2. Add rule for `main` branch:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (1 minimum)
   - ✅ Dismiss stale PR approvals when new commits are pushed
   - ✅ Require review from code owners
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - Required status checks:
     - `Code Quality`
     - `Type Checking`
     - `Tests (ubuntu-latest, 3.11)`
     - `Build Package`
     - `Integration Tests`
   - ✅ Require conversation resolution before merging
   - ✅ Include administrators

### Repository Secrets

Configure the following secrets in **Settings** → **Secrets and variables** → **Actions**:

#### Required Secrets
- `CODECOV_TOKEN` - Token for code coverage reporting
- `PYPI_API_TOKEN` - Token for PyPI package publishing
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_TOKEN` - Docker Hub access token

#### Optional Secrets (for enhanced features)
- `SLACK_WEBHOOK_URL` - Slack notifications for builds
- `TEAMS_WEBHOOK_URL` - Microsoft Teams notifications

### Environment Variables

Configure environment variables:
- `PYTHON_VERSION` - Default Python version (3.11)
- `NODE_VERSION` - Node.js version for documentation (18)

## External Service Setup

### Codecov Integration

1. Visit [codecov.io](https://codecov.io)
2. Sign in with GitHub
3. Add your repository
4. Copy the upload token to `CODECOV_TOKEN` secret

### PyPI Publishing

1. Create account at [pypi.org](https://pypi.org)
2. Generate API token in account settings
3. Add token to `PYPI_API_TOKEN` secret

### Docker Hub

1. Create account at [hub.docker.com](https://hub.docker.com)
2. Create access token in security settings
3. Add username to `DOCKER_USERNAME` secret
4. Add token to `DOCKER_TOKEN` secret

## Security Configuration

### Dependabot

The Dependabot configuration will be automatically created as part of the workflow setup.

### Code Scanning

GitHub's CodeQL analysis will run automatically with the security workflow.

### Secret Scanning

GitHub's secret scanning is enabled by default for public repositories.

## Workflow Implementation Steps

1. **Create workflow directory**:
   ```bash
   mkdir -p .github/workflows
   ```

2. **Copy workflow templates**:
   ```bash
   cp docs/workflows/ci.yml.template .github/workflows/ci.yml
   cp docs/workflows/security.yml.template .github/workflows/security.yml
   cp docs/workflows/release.yml.template .github/workflows/release.yml
   cp docs/workflows/docker.yml.template .github/workflows/docker.yml
   ```

3. **Copy Dependabot config**:
   ```bash
   cp docs/workflows/dependabot.yml.template .github/dependabot.yml
   ```

4. **Create issue templates**:
   ```bash
   mkdir -p .github/ISSUE_TEMPLATE
   cp docs/templates/*.md .github/ISSUE_TEMPLATE/
   ```

5. **Add CODEOWNERS**:
   ```bash
   cp docs/templates/CODEOWNERS.template .github/CODEOWNERS
   ```

6. **Test workflows**:
   - Create a draft pull request to test CI pipeline
   - Check that all status checks appear and pass
   - Verify security scans run successfully

## Monitoring and Maintenance

### Weekly Tasks
- Review Dependabot PRs
- Check security scan results
- Update workflow dependencies

### Monthly Tasks
- Review and update pinned action versions
- Audit security settings
- Update documentation

### Quarterly Tasks
- Review branch protection rules
- Audit access permissions
- Update runner configurations

## Troubleshooting

### Common Issues

**Workflow not triggering**:
- Check YAML syntax
- Verify branch names in triggers
- Ensure proper permissions

**Secret not found**:
- Verify secret name matches exactly
- Check secret is set at repository level
- Ensure workflow has proper permissions

**Test failures**:
- Check Python version compatibility
- Verify all dependencies are installed
- Review test isolation

### Getting Help

- GitHub Actions Documentation: https://docs.github.com/actions
- Workflow examples: https://github.com/actions/starter-workflows
- Community discussions: https://github.community

## Security Best Practices

1. **Minimal Permissions**: Each workflow uses least-privilege principle
2. **Secret Management**: Secrets are scoped to specific environments
3. **Dependency Pinning**: Action versions are pinned to specific commits
4. **Audit Logging**: All workflow runs are logged and auditable
5. **Regular Updates**: Dependencies are automatically updated via Dependabot