"""The Anti-Corruption Layer around the LLM SDK.

This is the ONLY module that imports Small_LLM_Model. Every SDK detail -- its
tensor return types, its method names, its runtime failures -- is isolated
here so the core engine works in native Python collections and never sees
the third party.
Boundary contract exposed to the engine:
    encode(text) -> list[int]          (native, not a tensor)
    logits(ids)  -> list[float]        (native, next-token logits)
    vocab_path() / tokenizer_path()    (real SDK method names)

Any SDK exception is caught and re-raised as SDKError so no
raw SDK panic leaks in.
"""
from __future__ import annotations
from typing import List
from llm_sdk import Small_LLM_Model
from .errors import SDKError


class LLMAdapter:
    """Wraps Small_LLM_Model behind a native-Python interface."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        try:
            self._model = Small_LLM_Model(model_name=model_name)
        except Exception as exc:
            raise SDKError(
                f"failed to load model {model_name!r}: {exc}"
            ) from exc

    def encode(self, text: str) -> List[int]:
        """Encode text to a NATIVE list of token ids.

        The SDK returns a 2-D tensor; we flatten it to list[int] here so the
        engine never handles a tensor.
        """
        try:
            tensor = self._model.encode(text)
            return [int(x) for x in tensor[0].tolist()]
        except Exception as exc:  # noqa: BLE001
            raise SDKError(f"encode failed for {text!r}: {exc}") from exc

    def logits(self, input_ids: List[int]) -> List[float]:
        """Next-token logits for a native list of ids. Native list in,
        native out."""
        try:
            return self._model.get_logits_from_input_ids(input_ids)
        except Exception as exc:  # noqa: BLE001
            raise SDKError(f"logits call failed: {exc}") from exc

    def vocab_path(self) -> str:
        """Path to vocab.json
        (the real SDK method is get_path_to_vocab_file)."""
        try:
            return self._model.get_path_to_vocab_file()
        except Exception as exc:  # noqa: BLE001
            raise SDKError(f"could not locate vocab file: {exc}") from exc

    def tokenizer_path(self) -> str:
        """Path to tokenizer.json
        (for the added_tokens overlay in Vocabulary)."""
        try:
            return self._model.get_path_to_tokenizer_file()
        except Exception as exc:  # noqa: BLE001
            raise SDKError(f"could not locate tokenizer file: {exc}") from exc

    def logits_length(self) -> int:
        """The true width of the logit vector, discovered by one probe call.

        Vocabulary needs this to size masks and to compute the phantom-id set.
        """
        try:
            probe = self._model.encode("hi")
            ids = [int(x) for x in probe[0].tolist()]
            return len(self._model.get_logits_from_input_ids(ids))
        except Exception as exc:  # noqa: BLE001
            raise SDKError(
                f"could not determine logits length: {exc}"
            ) from exc
