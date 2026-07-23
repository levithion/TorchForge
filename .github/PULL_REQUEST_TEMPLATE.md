## Summary

Describe what changed and why.

## User impact

Explain the behavior visible to users or developers.

## Validation

- [ ] `uv run --extra dev ruff check src tests main.py`
- [ ] `uv run --extra dev pyright`
- [ ] `uv run --extra dev --extra reference pytest -q`
- [ ] `cd frontend && npm run lint && npm test`
- [ ] Documentation was updated when behavior or configuration changed

List any additional manual checks:

## Security and generated-code considerations

Describe changes involving generated-code execution, file paths, external
processes, network access, or sensitive data. Write `None` when not applicable.

## Screenshots

Include before/after screenshots for visible Studio changes, if applicable.
