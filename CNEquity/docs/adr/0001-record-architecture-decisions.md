# ADR 0001: Record architecture decisions

- Status: Accepted
- Date: 2026-06-28

## Context

CNEquity has several non-obvious architecture choices (storage format,
orchestration model, multi-source policy). New contributors need to understand
*why*, not just *what*.

## Decision

Use lightweight Architecture Decision Records (ADRs) stored in `docs/adr/`,
one Markdown file per decision, numbered sequentially. Copy `0000-template.md`
for new records.

## Consequences

- Decisions are discoverable and versioned alongside code.
- Superseding a decision is explicit (status link), preserving history.

## Alternatives considered

- Wiki/Confluence: drifts from the codebase, not version-controlled with PRs.
- Comments in code: too local, no cross-cutting narrative.
