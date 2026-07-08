"""The constrained-decoding engine.

Core idea: we build the JSON skeleton ourselves and invoke the model only at the
holes. Every structural byte -- `{`, `"`, `:`, `,`, `}` -- is appended directly to
the token history. The model is called only to choose a tool NAME and each
argument VALUE, and every such call is filtered through a mask from masks.py so an
invalid token can never be selected. The output is therefore valid-by-construction.

The loop is autoregressive: current token history -> logits for the next token ->
mask -> argmax over survivors -> append -> repeat. History is appended to only
AFTER a token clears the mask (never a masked-out token enters history).
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np

from . import masks
from .errors import GenerationError
from .llm_adapter import LLMAdapter
from .schemas import FunctionCallResult, FunctionDefinition, ParamValue
from .vocab import Vocabulary

# Hard ceiling on tokens per field, so a pathological state can never loop forever.
_MAX_FIELD_TOKENS = 64
_QUOTE = '"'


class ConstrainedDecoder:
    """Drives generation for one prompt against one set of function definitions."""

    def __init__(self, adapter: LLMAdapter, vocab: Vocabulary) -> None:
        self._adapter = adapter
        self._vocab = vocab

    # -- public entry point --------------------------------------------------

    def generate(
        self,
        prompt: str,
        functions: List[FunctionDefinition],
    ) -> FunctionCallResult:
        """Produce a validated FunctionCallResult for `prompt`."""
        history: List[int] = self._adapter.encode(self.build_preamble(prompt))
        name = self._decode_tool_name(history, functions)
        chosen = next(f for f in functions if f.name == name)
        parameters = self._decode_parameters(history, chosen)

        return FunctionCallResult(prompt=prompt, name=name, parameters=parameters)


    def build_preamble(self, prompt: str) -> str:
        """The instruction + the opening skeleton up to the first hole (the name).

        We commit the structure `{"name": "` ourselves; the model never decides
        whether a brace or quote belongs here.
        """
        return (
            "Select the function and arguments for this request.\n"
            f"Request: {prompt}\n"
            'Answer: {"name": "'
        )

    def _append_literal(self, history: List[int], text: str) -> None:
        """Append the token ids for a skeleton literal we control (not the model)."""
        history.extend(self._adapter.encode(text))

    # -- the tool-name field (the load-bearing logic) ------------------------

    def _decode_tool_name(
        self,
        history: List[int],
        functions: List[FunctionDefinition],
    ) -> str:
        """Generate the tool name under the prefix+quote-commit constraint.

        clean_str accumulates the committed name characters. Each step we mask to
        (a) tokens continuing toward some allowed name, plus (b) the closing quote
        IF clean_str is already an exact name. Termination is the model selecting
        the closing quote while clean_str is exact -- NOT clean_str merely being a
        name. That deferral keeps fn_add and fn_add_numbers both reachable.
        """
        allowed = [f.name for f in functions]
        clean_str = ""

        for _ in range(_MAX_FIELD_TOKENS):
            logits = self._adapter.logits(history)
            mask = masks.tool_name_mask(self._vocab, allowed, clean_str)
            token_id = self._select(logits, mask)
            tok = self._vocab.clean_string(token_id)

            if tok == _QUOTE and clean_str in allowed:
                return clean_str  # the quote commits this name

            history.append(token_id)
            clean_str += tok

        raise GenerationError(
            f"tool name did not terminate; got prefix {clean_str!r}"
        )

    # -- the parameters object -----------------------------------------------

    def _decode_parameters(
        self,
        history: List[int],
        function: FunctionDefinition,
    ) -> Dict[str, ParamValue]:
        """Walk the function's declared parameters, generating each value under a
        type-specific mask. Object braces, keys, colons and commas are skeleton we
        write; only the values come from the model."""
        result: Dict[str, ParamValue] = {}
        self._append_literal(history, '", "parameters": {')

        items = list(function.parameters.items())
        for index, (arg_name, spec) in enumerate(items):
            self._append_literal(history, f'"{arg_name}": ')
            value = self._decode_value(history, spec.type)
            result[arg_name] = value
            if index < len(items) - 1:
                self._append_literal(history, ", ")

        self._append_literal(history, "}}")
        return result

    def _decode_value(self, history: List[int], param_type: str) -> ParamValue:
        """Dispatch on the declared type and generate one value."""
        if param_type == "string":
            return self._decode_string(history)
        if param_type == "number":
            return self._decode_number(history)
        if param_type == "boolean":
            return self._decode_boolean(history)
        raise GenerationError(f"unknown parameter type {param_type!r}")

    def _decode_string(self, history: List[int]) -> str:
        self._append_literal(history, _QUOTE)
        committed = ""
        for _ in range(_MAX_FIELD_TOKENS):
            logits = self._adapter.logits(history)
            mask = masks.string_value_mask(self._vocab, committed)
            token_id = self._select(logits, mask)
            tok = self._vocab.clean_string(token_id)
            if tok == _QUOTE:
                self._append_literal(history, _QUOTE)
                return committed
            history.append(token_id)
            committed += tok
        raise GenerationError("string value did not terminate")

    def _decode_number(self, history: List[int]) -> float:
        """Generate a numeric value, stopping when the model selects a structural
        terminator (',' or '}') -- NOT when the value merely parses. This mirrors
        the tool-name rule: the model picks the end, we never guess it. Without
        this, '144' truncates to '1' the instant the first digit parses as float.
        """
        committed = ""
        for _ in range(_MAX_FIELD_TOKENS):
            logits = self._adapter.logits(history)
            mask = masks.number_value_mask(self._vocab, committed)
            token_id = self._select(logits, mask)
            tok = self._vocab.clean_string(token_id)
            # print(f"  num step: committed={committed!r} tok={tok!r}", file=sys.stderr)
 
            # The model chose the terminator: the number is done. Do NOT append it
            # -- the skeleton (_decode_parameters) writes the comma/brace itself.
            if tok in (",", "}") and masks._is_complete_number(committed):
                break
 
            history.append(token_id)
            committed += tok
 
        try:
            return float(committed)
        except ValueError as exc:
            raise GenerationError(f"invalid number {committed!r}") from exc

    def _number_complete(self, committed: str, history: List[int]) -> bool:
        """A number is 'complete enough' when it parses and the model prefers to
        emit a structural terminator next. We peek: build a number mask and see if
        any digit still dominates; simplest safe rule is to stop when committed is
        parseable and non-empty (the skeleton comma/brace follows)."""
        if committed in ("", "-"):
            return False
        try:
            float(committed)
            return True
        except ValueError:
            return False

    def _decode_boolean(self, history: List[int]) -> bool:
        committed = ""
        for _ in range(_MAX_FIELD_TOKENS):
            logits = self._adapter.logits(history)
            mask = masks.boolean_value_mask(self._vocab, committed)
            token_id = self._select(logits, mask)
            tok = self._vocab.clean_string(token_id)
            history.append(token_id)
            committed += tok
            if committed in ("true", "false"):
                return committed == "true"
        raise GenerationError(f"boolean did not resolve; got {committed!r}")

    # -- selection -----------------------------------------------------------

    def _select(self, logits: List[float], mask: np.ndarray) -> int:
        """Apply the mask in-place and return the argmax survivor.

        Masked positions are set to -inf so they can never be the argmax. If the
        whole vector is masked, the constraint is unsatisfiable -- a logic fault.
        """
        arr = np.asarray(logits, dtype=np.float64)
        arr[~mask] = -np.inf
        if not np.isfinite(arr).any():
            raise GenerationError("every candidate token was masked")
        return int(np.argmax(arr))
