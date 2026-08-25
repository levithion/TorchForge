# Changelog

All notable changes to TorchForge are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-25

First open-source release under the Apache License 2.0.

### Added

- Apache-2.0 `LICENSE`, package metadata, classifiers, and contribution terms.
- `CHANGELOG.md` and an automated GitHub Release workflow on version tags.
- Persistent job storage: pipeline jobs are recorded in a local SQLite database
  (`torchforge.db`) and survive backend restarts, replacing the in-memory job
  list and its 200-job eviction cap.
- Server-Sent Events endpoint `GET /api/jobs/stream` for live job progress;
  the frontend subscribes via `EventSource` with polling as a fallback.
- Runtime performance report in `validation.json`: forward-pass latency
  (mean/p50/p95), throughput, peak device memory, and estimated FLOPs per
  selected device, surfaced in TorchForge Studio.
- GPT-2 Small certified architecture profile: canonical topology,
  deterministic reference implementation, structural conformance checks, and
  Hugging Face numerical parity tests alongside the existing BERT Base suite.
- TorchScript export endpoint `GET /api/papers/{paper_id}/exports/torchscript`
  with a Studio download button and health capability flag.
- Optional Docker sandbox for generated-code execution
  (`TORCHFORGE_SANDBOX=docker`): runs Phase 4 inside a network-isolated,
  read-only, resource-limited container; off by default with graceful
  fallback to in-process validation.

### Changed

- Frontend dependencies upgraded to clear all fixable `npm audit` advisories
  (`vite` 8.2.2, `@cloudflare/vite-plugin` 1.53.x, React 19.2.8, patched
  transitive pins); one unpatched build-time advisory is documented in
  `SECURITY.md`.

## [0.5.0]

Initial public-ready pipeline: PDF extraction, certified BERT Base profile,
Ollama vision parsing and code generation, guarded compilation, runtime and
conformance validation, CLI, FastAPI service, and TorchForge Studio.

[1.0.0]: https://github.com/levithion/TorchForge/releases/tag/v1.0.0
[0.5.0]: https://github.com/levithion/TorchForge/releases/tag/v0.5.0
