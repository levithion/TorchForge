"""Command-line interface for the TorchForge pipeline and local web API."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from torchforge.extractor import extract_pdf
from torchforge.compiler import (
    DEFAULT_CODE_CONTEXT,
    DEFAULT_CODE_MODEL,
    DEFAULT_CODE_OUTPUT_TOKENS,
    DEFAULT_MAX_TEXT_CHARS,
    CompilerError,
    OllamaCodeCompiler,
    compile_artifact_directory,
)
from torchforge.vision_parser import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_VISION_CONTEXT,
    DEFAULT_VISION_MODEL,
    OllamaVisionClient,
    VisionParserError,
    parse_artifact_directory,
)
from torchforge.validator import RuntimeValidationError, validate_artifact_directory
from torchforge.watcher import watch_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchforge", description="Extract and parse research-paper PDFs"
    )
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="extract one PDF")
    extract.add_argument("pdf", type=Path)
    extract.add_argument("--assets-root", type=Path, default=Path("temp_assets"))
    extract.add_argument("--no-nougat", action="store_true")
    extract.add_argument("--nougat-timeout", type=float, default=1200)

    watch = subparsers.add_parser("watch", help="watch a directory for new PDFs")
    watch.add_argument("--input-dir", type=Path, default=Path("input_papers"))
    watch.add_argument("--assets-root", type=Path, default=Path("temp_assets"))
    watch.add_argument("--no-nougat", action="store_true")
    watch.add_argument("--nougat-timeout", type=float, default=1200)
    watch.add_argument("--stability-timeout", type=float, default=60)

    parse = subparsers.add_parser("parse", help="parse extracted diagrams with Ollama")
    parse.add_argument("artifact_dir", type=Path)
    parse.add_argument("--model", default=DEFAULT_VISION_MODEL)
    parse.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parse.add_argument("--timeout", type=float, default=300)
    parse.add_argument("--max-images", type=int, default=8)
    parse.add_argument("--context-window", type=int, default=DEFAULT_VISION_CONTEXT)

    compile_command = subparsers.add_parser("compile", help="generate PyTorch code with Ollama")
    compile_command.add_argument("artifact_dir", type=Path)
    compile_command.add_argument("--output-dir", type=Path, default=Path("output_code"))
    compile_command.add_argument("--model", default=DEFAULT_CODE_MODEL)
    compile_command.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    compile_command.add_argument("--timeout", type=float, default=600)
    compile_command.add_argument("--max-text-chars", type=int, default=DEFAULT_MAX_TEXT_CHARS)
    compile_command.add_argument("--context-window", type=int, default=DEFAULT_CODE_CONTEXT)
    compile_command.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_CODE_OUTPUT_TOKENS
    )

    validate = subparsers.add_parser(
        "validate", help="run generated PyTorch code and repair runtime failures"
    )
    validate.add_argument("artifact_dir", type=Path)
    validate.add_argument("--output-dir", type=Path, default=Path("output_code"))
    validate.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="runtime device (default: auto; CUDA, then MPS, then CPU)",
    )
    validate.add_argument("--max-repairs", type=int, default=2)
    validate.add_argument("--model", default=DEFAULT_CODE_MODEL)
    validate.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    validate.add_argument("--timeout", type=float, default=600)
    validate.add_argument("--context-window", type=int, default=DEFAULT_CODE_CONTEXT)
    validate.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_CODE_OUTPUT_TOKENS
    )

    serve = subparsers.add_parser("serve", help="run the local frontend API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "extract":
        result = extract_pdf(
            args.pdf,
            args.assets_root,
            not args.no_nougat,
            nougat_timeout=args.nougat_timeout,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.succeeded else 1

    if args.command == "parse":
        client = OllamaVisionClient(
            base_url=args.ollama_url,
            model=args.model,
            timeout=args.timeout,
            context_window=args.context_window,
        )
        try:
            topology = parse_artifact_directory(
                args.artifact_dir,
                client=client,
                max_images=args.max_images,
            )
        except VisionParserError as exc:
            print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
            return 1
        print(topology.model_dump_json(indent=2))
        return 0

    if args.command == "compile":
        engine = OllamaCodeCompiler(
            base_url=args.ollama_url,
            model=args.model,
            timeout=args.timeout,
            context_window=args.context_window,
            max_output_tokens=args.max_output_tokens,
        )
        try:
            output = compile_artifact_directory(
                args.artifact_dir,
                args.output_dir,
                compiler=engine,
                max_text_chars=args.max_text_chars,
            )
        except CompilerError as exc:
            print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
            return 1
        print(json.dumps({"status": "completed", "output": str(output)}, indent=2))
        return 0

    if args.command == "validate":
        engine = OllamaCodeCompiler(
            base_url=args.ollama_url,
            model=args.model,
            timeout=args.timeout,
            context_window=args.context_window,
            max_output_tokens=args.max_output_tokens,
        )
        try:
            report = validate_artifact_directory(
                args.artifact_dir,
                args.output_dir,
                device_name=args.device,
                max_repairs=args.max_repairs,
                compiler=engine,
            )
        except RuntimeValidationError as exc:
            print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
            return 1
        print(report.model_dump_json(indent=2))
        return 0 if report.succeeded else 1

    if args.command == "serve":
        import uvicorn

        uvicorn.run("torchforge.api:app", host=args.host, port=args.port)
        return 0

    watch_directory(
        args.input_dir,
        args.assets_root,
        nougat_enabled=not args.no_nougat,
        nougat_timeout=args.nougat_timeout,
        stability_timeout=args.stability_timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
