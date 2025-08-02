# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the Edge TPU v5 Benchmark Suite project.

## What are ADRs?

Architecture Decision Records are short text documents that capture an important architectural decision made along with its context and consequences.

## ADR Template

Use the following template for new ADRs:

```markdown
# ADR-XXXX: [Title]

## Status

[Proposed | Accepted | Deprecated | Superseded]

## Context

[Describe the forces at play, including technological, political, social, and project local forces]

## Decision

[State the architecture decision and full justification]

## Consequences

[Describe the resulting context, after applying the decision. All consequences should be listed here, not just the "positive" ones]

## Alternatives Considered

[List other options that were considered and why they were rejected]

## Related Decisions

[List any related ADRs or decisions that influenced this one]
```

## Creating New ADRs

1. Copy the template above
2. Number sequentially (ADR-0001, ADR-0002, etc.)
3. Use descriptive titles
4. Follow the format consistently
5. Update this README with links to new ADRs

## Current ADRs

- [ADR-0001: Python as Primary Language](./adr-0001-python-primary-language.md)
- [ADR-0002: TensorFlow Lite for Model Runtime](./adr-0002-tensorflow-lite-runtime.md)
- [ADR-0003: Modular Plugin Architecture](./adr-0003-modular-plugin-architecture.md)

## Guidelines

- Keep ADRs short and focused
- One decision per ADR
- Write in present tense
- Include rationale, not just the decision
- Update status when decisions change
- Reference relevant ADRs when making new decisions