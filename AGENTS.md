# Agent Configuration

## Agent skills

### Issue tracker

Issues are tracked as Local Markdown files in `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` at repo root + `docs/adr/`. See `docs/agents/domain.md`.

## Installed skills

The following skills have been loaded and are available for use by the agent:

- **writing-plans** — Create comprehensive implementation plans with bite-sized tasks. Use when planning new features or stages.
- **executing-plans** — Execute implementation plans systematically with verification. Use with writing-plans.
- **verification-before-completion** — Verify work before claiming completion. Use before any completion claim.
- **improve-codebase-architecture** — Surface architectural friction, propose deepening opportunities. Use for codebase refactoring.
- **systematic-debugging** — Root cause investigation before fixes. Use for bugs, test failures, unexpected behavior.
