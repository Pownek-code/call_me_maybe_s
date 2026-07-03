"""Constraint masks: given the current generation state and what has been
committed so far, produce a boolean array over the full logit width marking which
token ids are legal as the next token. The engine sets illegal logits to -inf and
argmaxes what survives.

Every mask starts from the vocab's phantom set already forbidden, then layers the
state-specific rule on top. All masks are built once as a NumPy boolean array and
applied in a single vectorized assignment -- no Python loop walks the 151936-wide
logit vector.

THE LOAD-BEARING RULE (tool-name termination):
Names can be prefixes of each other -- `fn_add` vs `fn_add_numbers`. When the
committed clean string is exactly `fn_add` and both remain viable, we do NOT
decide. We permit BOTH the continuation char (`_`, heading to fn_add_numbers) AND
the closing quote `"` (which commits fn_add), and let the model's logits choose.
Emitting `"` is the model's signal "I meant this name"; emitting `_` means "keep
going". Termination is therefore: the model selected the closing quote while the
committed string is an exact name -- NOT merely "the committed string is a name".
That deferral is what makes both fn_add and fn_add_numbers reachable.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .vocab import Vocabulary

# The closing quote that ends a JSON string value / the tool name.
_QUOTE = '"'


def _base_mask(vocab: Vocabulary) -> np.ndarray:
    """A boolean array, logits-wide, True = allowed. Phantoms start forbidden."""
    mask = np.ones(vocab.logits_length, dtype=bool)
    if vocab.phantom_ids():
        mask[np.fromiter(vocab.phantom_ids(), dtype=np.int64)] = False
    return mask


def tool_name_mask(
    vocab: Vocabulary,
    allowed_names: List[str],
    clean_str: str,
) -> np.ndarray:
    """Mask for the tool-name field, given the name chars committed so far.

    A candidate id is allowed when EITHER:
      (a) it continues toward some allowed name: name.startswith(clean_str + tok),
          i.e. clean_str + tok is still a prefix of at least one allowed name; OR
      (b) clean_str is ALREADY an exact allowed name AND the candidate is the
          closing quote -- this is the commit path for a prefix-name like fn_add.

    Both may be true at once (fn_add complete, fn_add_numbers still viable): then
    `_` survives via (a) and `"` survives via (b), and the model picks. That is
    the termination rule described in the module docstring.
    """
    mask = _base_mask(vocab)
    clean_is_exact_name = clean_str in allowed_names

    for token_id in range(vocab.logits_length):
        if not mask[token_id]:
            continue  # already forbidden (phantom)
        tok = vocab.clean_string(token_id)

        continues = any(name.startswith(clean_str + tok) for name in allowed_names)
        commits = clean_is_exact_name and tok == _QUOTE

        mask[token_id] = continues or commits
    return mask


def string_value_mask(vocab: Vocabulary, committed: str) -> np.ndarray:
    """Mask for a string parameter value. Any token is allowed except one that
    would inject an unescaped quote mid-value; the closing quote itself is allowed
    (it terminates the value). committed is the value chars so far (unused for the
    permissive rule, kept for symmetry / future escape handling)."""
    mask = _base_mask(vocab)
    for token_id in range(vocab.logits_length):
        if not mask[token_id]:
            continue
        tok = vocab.clean_string(token_id)
        # Allow the lone closing quote (terminator) but forbid tokens that embed a
        # quote inside other characters, which would break JSON structure.
        if _QUOTE in tok and tok != _QUOTE:
            mask[token_id] = False
    return mask


def number_value_mask(vocab: Vocabulary, committed: str) -> np.ndarray:
    """Mask for a numeric parameter value. A candidate is allowed iff
    committed + tok stays a parseable numeric prefix: digits, at most one leading
    '-', at most one '.'. Additionally, once `committed` is already a valid
    non-empty number, the structural terminators (',' and '}') are permitted --
    this is the number's commit signal, exactly as the closing quote commits a
    tool name. The engine stops when the model SELECTS a terminator; it does not
    guess where the number ends."""
    mask = _base_mask(vocab)
    committed_is_number = _is_complete_number(committed)
    for token_id in range(vocab.logits_length):
        if not mask[token_id]:
            continue
        tok = vocab.clean_string(token_id)
        continues = _is_number_prefix(committed + tok)
        commits = committed_is_number and tok in (",", "}")
        mask[token_id] = continues or commits
    return mask


def _is_complete_number(text: str) -> bool:
    """True iff text already parses as a JSON number (non-empty, not just '-')."""
    if text in ("", "-"):
        return False
    try:
        float(text)
        return True
    except ValueError:
        return False


def boolean_value_mask(vocab: Vocabulary, committed: str) -> np.ndarray:
    """Mask for a boolean value: only tokens that keep committed a prefix of
    'true' or 'false'."""
    mask = _base_mask(vocab)
    for token_id in range(vocab.logits_length):
        if not mask[token_id]:
            continue
        tok = vocab.clean_string(token_id)
        candidate = committed + tok
        mask[token_id] = "true".startswith(candidate) or "false".startswith(candidate)
    return mask


def _is_number_prefix(text: str) -> bool:
    """True if text could be extended into a valid JSON number (int or float)."""
    if text in ("", "-"):
        return True
    if text.count("-") > 1 or ("-" in text and not text.startswith("-")):
        return False
    if text.count(".") > 1:
        return False
    body = text.lstrip("-").replace(".", "", 1)
    return body.isdigit() or body == ""


# The dispatch dict: generation state -> the mask builder for that state. A small,
# honest mapping -- four states -- not a general parser. The engine looks up the
# active state here each step rather than branching inline.
MaskBuilder = Callable[..., np.ndarray]

STATE_HANDLERS: Dict[str, MaskBuilder] = {
    "tool_name": tool_name_mask,
    "string": string_value_mask,
    "number": number_value_mask,
    "boolean": boolean_value_mask,
}
