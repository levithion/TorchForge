# TorchForge

TorchForge is a local pipeline for turning machine-learning research papers into
structured artifacts that later stages can use to generate and validate PyTorch
implementations. Phases 1–4 provide deterministic PDF ingestion, strict local
vision parsing, local PyTorch source generation, and portable PyTorch runtime
validation for transformer and NLP papers.

TorchForge Studio adds a responsive browser workspace for uploading papers,
running all four phases, monitoring the local runtime, and inspecting generated
artifacts without invoking each phase manually.

## Phase 1 capabilities

- Extracts page text and document metadata with PyMuPDF.
- Preserves embedded images without recompressing them.
- Renders likely figure pages at 150 DPI, including pages whose diagrams are
  vector graphics.
- Optionally runs Nougat to preserve equations and tables as `.mmd` markup.
- Watches `input_papers/` for stable, newly created PDFs and deduplicates
  repeated filesystem events by content hash.
- Writes a manifest that later pipeline stages can consume without guessing
  filenames.

## Phase 2 capabilities

- Sends extracted figure-page images to a configurable local Ollama vision model.
- Uses Ollama structured outputs and a strict Pydantic JSON Schema.
- Validates unique layer IDs, layer inputs, connection endpoints, confidence
  bounds, collection-size limits, and unknown fields before accepting model output.
- Uses an unambiguous paper title to select a certified topology profile when one
  exists. BERT Base is normalized to a complete encoder graph with inputs, outputs,
  embedding components, connections, shapes, and paper-reported hyperparameters.
- Caps confidence for incomplete unknown topologies instead of preserving an
  unsupported high-confidence model score.
- Explicitly prevents prediction heads from being mislabeled as decoder stacks.
- Records layers, tensor shapes, parameters, sequential edges, residual/skip
  connections, assumptions, and confidence values in `topology.json`.
- Updates the Phase 1 manifest with the model, source images, image count, and
  schema version used for the vision run.

## Phase 3 capabilities

- Combines `topology.json` with Nougat markup or PyMuPDF text and sends it to a
  configurable local Ollama coding model.
- Uses a structured response while saving only importable Python source.
- Requires exactly one `nn.Module` with `__init__` and `forward`.
- Statically rejects syntax errors, executable top-level statements, CUDA usage,
  class-name mismatches, and module attributes used before initialization.
- Writes generated source to `output_code/` and records its class, model,
  assumptions, and SHA-256 digest in the manifest.

## Phase 4 capabilities

- Dynamically imports the Phase 3 module and verifies its generated class is an
  `nn.Module`.
- Infers common required constructor arguments and dummy input tensors from
  `topology.json`.
- Automatically selects NVIDIA CUDA, Apple Silicon MPS, or the universally
  available CPU fallback, executes `forward` under `torch.no_grad()`, synchronizes
  accelerators, and records input/output shapes.
- Captures complete runtime tracebacks and gives them to the Phase 3 coding model
  for up to two repair attempts.
- Applies strict architecture profiles when a paper is recognized. The BERT Base
  profile verifies its three-part embeddings, 12 encoder layers, 12 attention
  heads, 3072-wide GELU feed-forward blocks, batch semantics, attention-mask
  interface, LayerNorm configuration, exact parameter count, pooler outputs,
  finite values, gradient flow, batch independence, and checkpoint adapter.
- The optional reference suite loads identical Hugging Face `BertModel` weights
  into TorchForge and checks numerical parity for masked and segmented batches.
- Treats architecture-conformance failures exactly like runtime failures and sends
  the failed checks to the repair model. A toy Transformer cannot pass as BERT.
- Writes `validation.json` and a validation summary into the manifest. A clean
  first run reports `completed`; a corrected run reports `repaired`; exhausted
  attempts report `failed`.

## Requirements and setup

- Windows, macOS, or Linux on a platform supported by Python, PyMuPDF, and PyTorch
- [`uv`](https://docs.astral.sh/uv/)
- Python 3.11 (installed automatically by `uv` when needed)
- [Ollama](https://ollama.com/) with a vision model for Phase 2
- An Ollama coding model for Phase 3
- A GPU is optional. Phase 4 falls back to CPU automatically.

```bash
uv python install 3.11
uv sync --extra dev
```

For independent BERT parity verification:

```bash
uv sync --extra dev --extra reference
uv run --extra reference pytest tests/test_bert_reference.py
```

Nougat is optional and deliberately excluded from the base environment because
of its model size and ML dependencies. To enable equation OCR, install
`nougat-ocr` in a compatible separate environment or add its `nougat` executable
to `PATH`. TorchForge invokes the upstream interface:

```bash
nougat paper.pdf -o output_directory
```

Without Nougat, extraction still succeeds and the manifest status is
`completed_with_warnings`.

For diagram parsing, start Ollama and install the default README-compatible model:

```bash
ollama serve
ollama pull llava
```

Another vision-capable model can be selected with `--model`.

Install the default coding model selected for a 16 GB Apple Silicon Mac:

```bash
ollama pull qwen2.5-coder:3b
```

## Usage

### Browser workspace

Start the local TorchForge engine:

```bash
uv run torchforge serve
```

In a second terminal, start the frontend:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000`. The workspace supports PDF upload, phase-by-phase
execution, paper history, environment health, and text/topology/code/validation
artifact viewing. The local API listens on `http://127.0.0.1:8000` by default.
Set `NEXT_PUBLIC_TORCHFORGE_API_URL` in `frontend/.env.local` when the engine is
hosted elsewhere, and list the frontend origin in
`TORCHFORGE_ALLOWED_ORIGINS` for cross-origin access.

### Command line

Extract one PDF:

```bash
uv run torchforge extract path/to/paper.pdf
```

Skip the optional Nougat check:

```bash
uv run torchforge extract path/to/paper.pdf --no-nougat
```

Watch the default input directory:

```bash
uv run torchforge watch
```

Parse diagrams from a completed Phase 1 artifact directory:

```bash
uv run torchforge parse temp_assets/<paper-name-and-hash>
```

Compile a completed Phase 2 artifact into PyTorch source:

```bash
uv run torchforge compile temp_assets/<paper-name-and-hash>
```

Validate a completed Phase 3 artifact on the best available device:

```bash
uv run torchforge validate temp_assets/<paper-name-and-hash>
```

Override automatic selection when needed:

```bash
uv run torchforge validate temp_assets/<paper-name-and-hash> --device cpu
# Other explicit choices: --device cuda or --device mps
```

Use a different local model or endpoint:

```bash
uv run torchforge parse temp_assets/<paper-name-and-hash> \
  --model gemma3 \
  --ollama-url http://127.0.0.1:11434
```

Useful options:

```text
--assets-root PATH         artifact root (default: temp_assets)
--nougat-timeout SECONDS  Nougat timeout (default: 1200)
--input-dir PATH           watched directory (default: input_papers)
--stability-timeout SEC    maximum wait for a copied PDF to stabilize
--model NAME               Ollama vision model (default: llava)
--ollama-url URL           Ollama server (default: http://127.0.0.1:11434)
--timeout SECONDS          vision request timeout (default: 300)
--max-images COUNT         maximum candidate pages sent (default: 8)
--output-dir PATH          generated source directory (default: output_code)
--max-text-chars COUNT     maximum paper text supplied (default: 6000)
--context-window TOKENS    Ollama context size (default: 8192)
--max-output-tokens COUNT  maximum generated code response (default: 4096)
--device DEVICE            auto, cuda, mps, or cpu (default: auto)
--max-repairs COUNT        runtime-driven recompilations (default: 2)
```

Press `Ctrl-C` to stop the watcher cleanly. `python main.py` accepts the same
subcommands after the project has been installed with `uv sync`.

## Artifacts

Each paper is stored under a collision-safe path such as:

```text
temp_assets/attention-is-all-you-need-a1b2c3d4e5f6/
├── manifest.json
├── pymupdf.md
├── nougat.mmd             # only when Nougat succeeds
├── topology.json           # after successful Phase 2 parsing
├── validation.json         # after a Phase 4 validation attempt
├── images/                # embedded raster images
└── pages/                 # likely figure pages rendered as PNG
```

`manifest.json` records the source SHA-256, extraction status, page count,
metadata, OCR provider, relative artifact paths, warnings, and errors. The three
terminal statuses are `completed`, `completed_with_warnings`, and `failed`.
After Phase 2 succeeds, its `vision` section identifies the Ollama model and
exact source images used. The original extraction status is left unchanged.
After Phase 3 succeeds, `artifacts.generated_code` points to the generated `.py`
file and `compilation` records the coding model, class, assumptions, and digest.
After Phase 4 runs, `artifacts.validation_report` points to the detailed attempt
history and the manifest's `validation` section summarizes device, status, attempt
count, and output shapes.

## Tests

```bash
uv run pytest
```

The suite covers text/image/page extraction, hash-based artifact isolation,
Nougat fallbacks, watcher behavior, the strict topology schema, Ollama request
construction and error handling, path containment, manifest updates, and CLI
exit codes. Phase 3 tests also cover structured compiler requests, Python AST
validation, device independence, output writing, and compilation metadata. Phase
4 tests cover constructor inference, dummy forward execution, tensor output
shapes, traceback-driven repair, report persistence, failure status, and CLI exit
codes.

## Current limitations

- Password-protected PDFs are rejected.
- Figure-page detection uses embedded-image presence and `Figure`/`Fig.` caption
  text; unusual caption styles may be missed.
- Watcher deduplication is in memory and resets when the process restarts.
- Nougat can be slow on Apple Silicon and may download model weights on its first
  run. Its failures never discard valid PyMuPDF artifacts.
- Phase 2 prioritizes rendered figure pages because they preserve vector diagrams
  and caption context; embedded images are used when no rendered pages exist.
- Only the first eight candidate pages are sent by default to bound local model
  cost. Increase `--max-images` when an architecture spans more pages.
- Vision output is schema-valid but remains a model interpretation. Low-confidence
  fields and recorded assumptions should be reviewed before code generation.
- Phase 4 performs strict structural and semantic checks for recognized profiles
  (currently BERT Base). Other architectures receive portable runtime validation
  but are not claimed to exactly reproduce the paper.
- Constructor and input inference recognizes common transformer/NLP parameter
  names. Unusual required arguments are rejected explicitly instead of guessed.
- Automatic device selection prefers CUDA, then MPS, and otherwise uses CPU.
  Explicitly requesting an unavailable accelerator produces a clear error.
- CPU execution makes runtime validation portable, but vision parsing and code
  generation through Ollama can be substantially slower without a GPU and still
  require enough memory for the selected models.
- A repair is still model-generated code. Static checks, strict known-architecture
  profiles, and runtime checks substantially reduce obvious failures but do not
  prove pretrained-weight equivalence or downstream task quality.
- The BERT reference suite proves forward-output equivalence after importing the
  same Hugging Face weights. It does not bundle copyrighted checkpoints, a
  tokenizer, pretraining data, or downstream fine-tuning.
