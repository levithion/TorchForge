# Contributing to TorchForge

Thank you for helping improve TorchForge. Focused pull requests with tests and a
clear explanation of user impact are welcome.

## Development setup

Install the backend and reference-test dependencies:

```bash
uv python install 3.11
uv sync --extra dev --extra reference
```

Install the frontend dependencies:

```bash
cd frontend
npm ci
```

TorchForge uses local Ollama models for the vision and code-generation stages.
Most automated tests replace those integrations with deterministic fakes, so
Ollama is not required for the standard test suite.

## Making a change

1. Fork and clone the repository.
2. Create a branch from `main`.
3. Keep each pull request focused on one behavior or concern.
4. Add or update tests for behavioral changes.
5. Keep generated PDFs, artifacts, model outputs, credentials, and virtual
   environments out of Git.
6. Document user-facing behavior and configuration changes.

## Quality checks

Run the same checks used by CI:

```bash
uv run --extra dev ruff check src tests main.py
uv run --extra dev pyright
uv run --extra dev --extra reference pytest -q
cd frontend
npm run lint
npm test
```

## Pull requests

Complete the pull request template, including validation evidence and any
security implications. Generated Python is executed during Phase 4, so changes
to validation, code generation, path handling, or process execution deserve
extra scrutiny.

By contributing, you agree that your contribution will be distributed under
the repository's license.
