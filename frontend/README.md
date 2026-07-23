# TorchForge Studio

The browser workspace for the local TorchForge paper-to-PyTorch pipeline.

## Run locally

Start the Python engine from the repository root:

```bash
uv run torchforge serve
```

Then start this frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The application connects to
`http://127.0.0.1:8000` by default.

To use another engine URL, copy `.env.example` to `.env.local` and change
`NEXT_PUBLIC_TORCHFORGE_API_URL`.

## Available workflows

- Upload and extract PDFs.
- Run architecture parsing with the local Ollama vision model.
- Generate PyTorch source with the configured coding model.
- Validate and repair generated modules on CUDA, MPS, or CPU.
- Inspect text, topology, source code, manifests, and validation reports.
- Monitor the local device and Ollama connection.

## Verification

```bash
npm test
```
