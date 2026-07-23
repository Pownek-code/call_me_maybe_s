"""Token id <-> string resolution, and the ONLY place that knows about byte-level
BPE markers (`Ġ` = leading space, `Ċ` = newline).

The rest of the codebase works in clean string space. Every mask function asks
this layer "what clean string does id N decode to?" and never sees a raw marker.

Why this module is careful about coverage (verified live against Qwen/Qwen3-0.6B):

    vocab.json entries : 151643   -> ids 0 .. 151642
    added_tokens       :     26   -> ids 151643 .. 151668
    logits length      : 151936   -> ids 0 .. 151935

Ids 151669 .. 151935 exist in the logit vector but decode to NOTHING -- they are
embedding-padding phantoms (the model pads its embedding matrix past the real
vocabulary). They MUST be permanently masked. If they leaked through as the empty
string, `name.startswith("")` is always True, so a phantom id would survive every
prefix mask and get selected, emitting a garbage token. Guarding them here is the
single most important correctness property of this file.
"""
from __future__ import annotations

import json
from typing import Dict, List, Set

# Byte-level BPE markers used by the Qwen / GPT-2 tokenizer family.
_SPACE_MARKER = "\u0120"  # 'Ġ'  -> a single leading space
_NEWLINE_MARKER = "\u010a"  # 'Ċ' -> a newline


def _demarker(raw_token: str) -> str:
    """Translate a raw byte-level-BPE token string into clean text.

    `Ġthe` -> ` the`,  `Ċ` -> `\\n`. Ordinary tokens pass through unchanged.
    """
    return raw_token.replace(_SPACE_MARKER, " ").replace(_NEWLINE_MARKER, "\n")


class Vocabulary:
    """Resolves every logit index to a clean string, or marks it as a phantom.

    The engine holds one instance and consults it at every generation step.
    """

    def __init__(
        self,
        vocab_json_path: str,
        tokenizer_json_path: str,
        logits_length: int,
    ) -> None:
        # id -> clean string, for every id that decodes to real text.
        self._id_to_str: Dict[int, str] = {}
        # ids that exist in the logit vector but decode to nothing.
        self._phantom_ids: Set[int] = set()
        # the true width of the logit vector -- the engine builds masks this wide.
        self._logits_length: int = logits_length

        self._load_base_vocab(vocab_json_path)
        self._load_added_tokens(tokenizer_json_path)
        self._mark_phantoms()
        self._build_lookup_array()

    def _build_lookup_array(self) -> None:
        """Precompute an id-indexed list of clean strings, built ONCE at startup.

        Mask construction previously did a dict lookup per id per generated token
        (151,936 lookups every step). Indexing a flat list instead removes that
        cost entirely. Phantom ids get "" as a placeholder -- they are separately
        forbidden by the base mask, so the placeholder is never selectable.
        """
        self._lookup: List[str] = [
            self._id_to_str.get(i, "") for i in range(self._logits_length)
        ]
        # A reverse index: clean string -> the ids producing it. Lets a mask turn
        # "which tokens are legal" into set membership instead of a 152k scan.
        self._str_to_ids: Dict[str, List[int]] = {}
        for token_id, text in enumerate(self._lookup):
            if text and token_id not in self._phantom_ids:
                self._str_to_ids.setdefault(text, []).append(token_id)

    # -- construction steps --------------------------------------------------

    def _load_base_vocab(self, path: str) -> None:
        """Source 1: vocab.json is {token_string: id}. Invert to {id: clean_str}."""
        with open(path, "r", encoding="utf-8") as handle:
            raw: Dict[str, int] = json.load(handle)
        for token_str, token_id in raw.items():
            self._id_to_str[token_id] = _demarker(token_str)

    def _load_added_tokens(self, path: str) -> None:
        """Source 2: tokenizer.json 'added_tokens' -> the <|...|> specials.

        These are literal, byte-for-byte; they carry no space marker, so they are
        stored WITHOUT demarkering. They occupy ids just past the base vocab.
        """
        with open(path, "r", encoding="utf-8") as handle:
            tokenizer: Dict[str, object] = json.load(handle)
        added = tokenizer.get("added_tokens", [])
        if isinstance(added, list):
            for entry in added:
                if isinstance(entry, dict) and "id" in entry and "content" in entry:
                    self._id_to_str[int(entry["id"])] = str(entry["content"])

    def _mark_phantoms(self) -> None:
        """Source 3: every id in [0, logits_length) with no string is a phantom."""
        for token_id in range(self._logits_length):
            if token_id not in self._id_to_str:
                self._phantom_ids.add(token_id)

    # -- query interface (what the engine and masks call) --------------------

    @property
    def logits_length(self) -> int:
        """Width of the logit vector; masks are built to exactly this size."""
        return self._logits_length

    def clean_string(self, token_id: int) -> str:
        """Clean text for a real id. Raises KeyError for phantom/out-of-range ids.

        Callers that might pass a phantom should gate on `is_selectable` first;
        the raise here is deliberate so a phantom NEVER silently returns "".
        """
        return self._id_to_str[token_id]

    def is_selectable(self, token_id: int) -> bool:
        """True iff the id decodes to a real token (i.e. is not a phantom)."""
        return token_id not in self._phantom_ids

    def phantom_ids(self) -> Set[int]:
        """The permanently-masked id set. The engine seeds every mask with these
        already forbidden, then applies state-specific constraints on top."""
        return self._phantom_ids

    @property
    def lookup(self) -> List[str]:
        """Id-indexed clean strings, precomputed once. lookup[i] is id i's text."""
        return self._lookup

    def ids_for(self, text: str) -> List[int]:
        """All selectable token ids whose clean string is exactly `text`."""
        return self._str_to_ids.get(text, [])

    def all_strings(self) -> Dict[str, List[int]]:
        """The full clean-string -> ids index."""
        return self._str_to_ids
