"""Domain exception hierarchy — the fault domains for the project.

The rule (Defensive Exception Isolation): low-level failures must never leak into
the core engine as raw `FileNotFoundError`, `KeyError`, or SDK runtime panics.
They are caught at the adapter boundary and re-raised as one of these, each
carrying a clear, user-facing message. `__main__` catches these at the top level
and prints the message to stderr instead of dumping a traceback -- satisfying the
subject's "never crash unexpectedly, always provide clear error messages".

They share a common base so the top-level handler can catch the whole family with
a single `except CallMeMaybeError`.
"""
from __future__ import annotations


class CallMeMaybeError(Exception):
    """Base for every domain error. Catch this at the top level to handle all."""


class InputError(CallMeMaybeError):
    """A problem with an input file: missing, unreadable, invalid JSON, or a
    payload that fails schema validation.

    Raised by io_adapter after catching the low-level cause (FileNotFoundError,
    json.JSONDecodeError, pydantic.ValidationError) so the engine only ever sees
    a clean InputError with a human-readable message.
    """


class SDKError(CallMeMaybeError):
    """A failure originating in the LLM SDK: model load, tokenisation, or a logits
    call that raised or returned something unusable.

    Raised by llm_adapter (the Anti-Corruption Layer) so no raw SDK exception
    reaches the engine.
    """


class GenerationError(CallMeMaybeError):
    """The constrained-decoding loop reached an impossible state: every candidate
    token was masked, no valid continuation exists, or a generation-length guard
    tripped. Signals a logic/constraint fault rather than an I/O or SDK fault.
    """
