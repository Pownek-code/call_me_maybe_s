"""Entry point: `uv run python -m src [--functions_definition ...] [--input ...]
[--output ...]`.

Wires the pieces together (adapter -> vocab -> decoder), runs every prompt, and
writes the results. All domain errors are caught at this single top level and
reported as a clean message on stderr -- the program never dumps a traceback.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from .decoder import ConstrainedDecoder
from .errors import CallMeMaybeError
from .io_adapter import load_functions, load_prompts, write_results
from .llm_adapter import LLMAdapter
from .schemas import FunctionCallResult
from .vocab import Vocabulary

_DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
_DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
_DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="src", description="Constrained function-calling generator.")
    parser.add_argument("--functions_definition", type=Path, default=_DEFAULT_FUNCTIONS)
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> None:
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    adapter = LLMAdapter()
    vocab = Vocabulary(
        vocab_json_path=adapter.vocab_path(),
        tokenizer_json_path=adapter.tokenizer_path(),
        logits_length=adapter.logits_length(),
    )
    decoder = ConstrainedDecoder(adapter, vocab)

    results: List[FunctionCallResult] = []
    for entry in prompts:
        result = decoder.generate(entry.prompt, functions)
        results.append(result)
        print(f"  {entry.prompt!r} -> {result.name}({result.parameters})", file=sys.stderr)

    write_results(args.output, results)
    print(f"Wrote {len(results)} result(s) to {args.output}", file=sys.stderr)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        _run(args)
    except CallMeMaybeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
