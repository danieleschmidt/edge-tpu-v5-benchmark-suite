# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. Please help us maintain the security of the Edge TPU v5 Benchmark Suite.

### How to Report

**DO NOT** create public GitHub issues for security vulnerabilities.

Instead, please report security issues privately:

1. **Email**: Send details to [daniel@terragonlabs.com](mailto:daniel@terragonlabs.com)
2. **Subject**: Include "SECURITY:" prefix in email subject
3. **Encrypt**: Use PGP key if handling sensitive information

### What to Include

Please provide as much information as possible:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if available)
- Your contact information

### Response Process

1. **Acknowledgment**: We'll acknowledge receipt within 2 business days
2. **Assessment**: We'll assess the vulnerability within 5 business days
3. **Fix**: We'll work on a fix and coordinate disclosure
4. **Release**: Security fixes are released as soon as possible
5. **Credit**: We'll credit reporters in release notes (if desired)

## Security Best Practices

### For Users

**Model Security**
- Only load models from trusted sources
- Verify model checksums when possible
- Be cautious with user-provided model files

**Data Protection**
- Ensure input data doesn't contain sensitive information
- Review benchmark results before sharing publicly
- Use appropriate data anonymization

**System Security**
- Run benchmarks in isolated environments when possible
- Keep TPU drivers and runtime updated
- Monitor system resources during benchmarks

### For Developers

**Code Security**
- Never commit API keys, passwords, or secrets
- Use environment variables for sensitive configuration
- Validate all user inputs
- Follow secure coding practices

**Dependencies**
- Regularly update dependencies
- Use `pip-audit` to check for vulnerabilities
- Pin dependency versions in production

**Testing**
- Include security-focused tests
- Test with malformed inputs
- Validate error handling

## Common Security Considerations

### Model Loading
```python
# ✅ Good: Validate model files
def load_model_safely(model_path: str):
    if not Path(model_path).is_file():
        raise FileNotFoundError("Model file not found")
    if Path(model_path).stat().st_size > MAX_MODEL_SIZE:
        raise ValueError("Model file too large")
    return load_model(model_path)

# ❌ Bad: No validation
def load_model_unsafely(model_path: str):
    return load_model(model_path)  # Could load malicious files
```

### Input Validation
```python
# ✅ Good: Validate inputs
def benchmark_model(iterations: int):
    if not 1 <= iterations <= 100000:
        raise ValueError("Invalid iteration count")
    return run_benchmark(iterations)

# ❌ Bad: No validation
def benchmark_model_unsafe(iterations: int):
    return run_benchmark(iterations)  # Could cause DoS
```

### Resource Management
```python
# ✅ Good: Limit resource usage
with resource_limit(memory_mb=1024, timeout_sec=300):
    results = run_benchmark()

# ❌ Bad: Unlimited resources
results = run_benchmark()  # Could consume all system resources
```

## Vulnerability Disclosure Policy

### Coordinated Disclosure

We follow responsible disclosure practices:

1. **Private reporting** to our security team
2. **Investigation** and fix development
3. **Coordinated public disclosure** after fix is available
4. **Credit** to security researchers (with permission)

### Timeline

- **Day 0**: Vulnerability reported
- **Day 1-2**: Acknowledgment sent
- **Day 3-7**: Initial assessment completed  
- **Day 8-30**: Fix developed and tested
- **Day 31**: Security advisory published
- **Day 32+**: Fix released publicly

### Severity Guidelines

**Critical** - Remote code execution, privilege escalation
- Fix target: 7 days
- Immediate security advisory

**High** - Information disclosure, DoS attacks
- Fix target: 14 days
- Security advisory within 30 days

**Medium** - Limited information disclosure
- Fix target: 30 days
- Security advisory with next release

**Low** - Best practice improvements
- Fix target: Next minor release
- Mentioned in release notes

## Security Updates

Security updates are distributed through:

- **PyPI releases** with security tags
- **GitHub Security Advisories**
- **Release notes** with security sections
- **Email notifications** to security mailing list

## Compliance

This project follows:

- **OWASP Top 10** security practices
- **Python security best practices**
- **Dependency vulnerability monitoring**
- **Regular security audits**

## Security Tools

We use the following tools for security:

- **bandit** - Python security linting
- **safety** - Dependency vulnerability scanning  
- **pip-audit** - Python package auditing
- **CodeQL** - Semantic code analysis
- **Dependabot** - Automated dependency updates

## Contact

For security-related questions:
- **Email**: daniel@terragonlabs.com
- **PGP Key**: Available on request
- **Response time**: 2 business days

Thank you for helping keep Edge TPU v5 Benchmark Suite secure! 🔒