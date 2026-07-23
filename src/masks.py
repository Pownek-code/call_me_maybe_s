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
      (a) it continues toward some allowed name: name.startswith(clean_str + tok); OR
      (b) clean_str is ALREADY an exact allowed name AND the candidate is the
          closing quote -- the commit path for a prefix-name like fn_add.

    Both may hold at once (fn_add complete, fn_add_numbers still viable): `_`
    survives via (a) and `"` via (b), and the model's logits pick. That deferral
    is the termination rule.

    VECTORIZATION NOTE: the naive form tests all 151,936 ids against every allowed
    name, every generated token. Instead we derive the SET of legal continuation
    strings from the (few) names still matching clean_str -- a handful of strings
    -- then flip those ids on via NumPy fancy indexing. Work becomes proportional
    to the number of allowed names, not to the vocabulary size.
    """
    mask = np.zeros(vocab.logits_length, dtype=bool)

    # Only names still consistent with what we have committed can contribute.
    viable = [n for n in allowed_names if n.startswith(clean_str)]

    legal_continuations: set[str] = set()
    for name in viable:
        remainder = name[len(clean_str):]
        # Every non-empty prefix of the remainder is a legal next-token string.
        for end in range(1, len(remainder) + 1):
            legal_continuations.add(remainder[:end])

    for text in legal_continuations:
        ids = vocab.ids_for(text)
        if ids:
            mask[np.asarray(ids, dtype=np.int64)] = True

    # (b) the commit path: permit the closing quote once clean_str is exact.
    if clean_str in allowed_names:
        for token_id in vocab.ids_for(_QUOTE):
            mask[token_id] = True

    return mask


def string_value_mask(vocab: Vocabulary, committed: str) -> np.ndarray:
    """Mask for a string parameter value: any token except one embedding an
    unescaped quote inside other characters. The lone closing quote is allowed --
    it terminates the value.

    VECTORIZATION NOTE: the set of ids containing a quote never changes, so it is
    computed once and cached on the vocab object rather than rescanned per token.
    """
    forbidden = _quote_bearing_ids(vocab)
    mask = np.ones(vocab.logits_length, dtype=bool)
    if vocab.phantom_ids():
        mask[np.fromiter(vocab.phantom_ids(), dtype=np.int64)] = False
    if forbidden.size:
        mask[forbidden] = False
    return mask


def _quote_bearing_ids(vocab: Vocabulary) -> np.ndarray:
    """Ids whose clean string contains a quote among other characters. Cached: the
    answer depends only on the vocabulary, never on generation state."""
    cached = getattr(vocab, "_quote_bearing_cache", None)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    ids = [
        i
        for i, text in enumerate(vocab.lookup)
        if _QUOTE in text and text != _QUOTE
    ]
    arr = np.asarray(ids, dtype=np.int64)
    setattr(vocab, "_quote_bearing_cache", arr)
    return arr


def number_value_mask(vocab: Vocabulary, committed: str) -> np.ndarray:
    """Mask for a numeric parameter value: tokens keeping committed+tok a valid
    numeric prefix, plus the structural terminators (',' '}') once committed is
    already a valid number -- the number's commit signal.

    VECTORIZATION NOTE: numeric tokens are a small, fixed subset of the vocabulary
    (digits, '.', '-'), so the candidate set is computed once and cached; per step
    we only re-test that small set instead of all 151,936 ids.
    """
    candidates = _numeric_candidate_ids(vocab)
    mask = np.zeros(vocab.logits_length, dtype=bool)

    lookup = vocab.lookup
    for token_id in candidates:
        if _is_number_prefix(committed + lookup[token_id]):
            mask[token_id] = True

    if _is_complete_number(committed):
        for terminator in (",", "}"):
            for token_id in vocab.ids_for(terminator):
                mask[token_id] = True
    return mask


def _numeric_candidate_ids(vocab: Vocabulary) -> List[int]:
    """Ids whose clean string is composed only of digits, '.', '-'. Cached: this
    depends only on the vocabulary, so the 152k scan happens once, not per token."""
    cached = getattr(vocab, "_numeric_candidate_cache", None)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    allowed_chars = set("0123456789.-")
    ids = [
        i
        for i, text in enumerate(vocab.lookup)
        if text and set(text) <= allowed_chars
    ]
    setattr(vocab, "_numeric_candidate_cache", ids)
    return ids


def boolean_value_mask(vocab: Vocabulary, committed: str) -> np.ndarray:
    """Mask for a boolean value: only tokens keeping committed a prefix of 'true'
    or 'false'. The legal set is tiny, so it is derived directly rather than by
    scanning the vocabulary."""
    mask = np.zeros(vocab.logits_length, dtype=bool)
    for word in ("true", "false"):
        if not word.startswith(committed):
            continue
        remainder = word[len(committed):]
        for end in range(1, len(remainder) + 1):
            for token_id in vocab.ids_for(remainder[:end]):
                mask[token_id] = True
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


def _is_complete_number(text: str) -> bool:
    """True iff text already parses as a JSON number (non-empty, not just '-').

    Used by both the number mask (to decide when terminators become legal) and the
    decoder (to confirm a selected terminator really ends a valid value).
    """
    if text in ("", "-"):
        return False
    try:
        float(text)
        return True
    except ValueError:
        return False


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
