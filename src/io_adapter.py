
"""File I/O behind an InputError fault domain.

Loads and validates the two input files into pydantic models, and writes the
output array. Every low-level failure -- missing file, malformed JSON, schema
violation -- is caught here and re-raised as InputError with a clear message, so
the engine never sees a raw FileNotFoundError / JSONDecodeError / ValidationError.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from pydantic import ValidationError

from .errors import InputError
from .schemas import FunctionCallResult, FunctionDefinition, TestPrompt


def load_functions(path: Path) -> List[FunctionDefinition]:
    """Parse functions_definition.json into validated FunctionDefinition models."""
    raw = read_json_array(path)
    try:
        return [FunctionDefinition.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise InputError(f"invalid function definition in {path}: {exc}") from exc


def load_prompts(path: Path) -> List[TestPrompt]:
    """Parse function_calling_tests.json into validated TestPrompt models."""
    raw = read_json_array(path)
    try:
        return [TestPrompt.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise InputError(f"invalid prompt entry in {path}: {exc}") from exc


def write_results(path: Path, results: List[FunctionCallResult]) -> None:
    """Serialize results to the output JSON array, creating parent dirs as needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [r.model_dump() for r in results]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise InputError(f"could not write output to {path}: {exc}") from exc


def read_json_array(path: Path) -> List[object]:
    """Read a file expected to contain a JSON array. Wrap every failure mode."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise InputError(f"input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"malformed JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise InputError(f"could not read {path}: {exc}") from exc

    if not isinstance(data, list):
        raise InputError(f"expected a JSON array in {path}, got {type(data).__name__}")
    return data
