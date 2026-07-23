"""Domain contracts for the function-calling project.
These pydantic models define what "valid" means for every piece of data that
crosses a boundary: the two input files and the one output file. Because the
LLM engine is downstream of these, a malformed input fails HERE (loudly, in the
I/O fault domain) rather than deep in the generation loop.
"""
from __future__ import annotations
from typing import Dict, Union
from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal

# The type field is a CLOSED set. Typing it as Literal (not str) makes pydantic
# raise a ValidationError at parse time on anything else
# -- e.g. {"type": "foo"}.
# Note this is the key reason pydantic beats a frozen dataclass here:
# a dataclass
# treats Literal as a hint mypy reads but the running program IGNORES, so a bad
# value constructs silently and detonates later in the mask dispatch. Pydantic
# checks at construction, so the failure lands at load,
# in the I/O fault domain.
ParamType = Literal["number", "string", "boolean"]
# frozen=True: a schema loaded from disk must never be mutated afterwards.
# Freezing turns an accidental write into an immediate error instead of silent
# corruption -- the same defensive immutability the original
# frozen dataclass had.
_FROZEN = ConfigDict(frozen=True)


class ParameterProperty(BaseModel):
    """Defines strict primitive data type signatures for active tool keys."""

    model_config = _FROZEN
    type: ParamType


class ReturnProperty(BaseModel):
    """Enforces type validation on execution return signatures.

    Kept separate from ParameterProperty (rather than merged) as a deliberate
    choice: returns and parameters are semantically distinct roles even when
    structurally identical. Naming the concept is worth one extra class.
    """

    model_config = _FROZEN
    type: ParamType


class FunctionDefinition(BaseModel):
    """Maps the complete tool schema registry matching the tool
    configuration data.
    Note where the dict lives: `parameters` maps an arbitrary, not-known-ahead
    argument name (a, b, name, s...) to that argument's ParameterProperty. The
    dict is a field of the FUNCTION, not of a parameter -- the function is the
    thing that HAS parameters. This is the container;
    ParameterProperty is the item.
    """
    model_config = _FROZEN
    name: str
    description: str
    parameters: Dict[str, ParameterProperty]
    returns: ReturnProperty


class TestPrompt(BaseModel):
    """Encapsulates an isolated query item matching the
    sequential batch dataset."""

    model_config = _FROZEN
    prompt: str


# A parameter VALUE at output time is a concrete number, string, or boolean --
# not a type-spec. bool is listed before int deliberately: in Python bool is a
# subclass of int, and pydantic's union resolution can coerce True -> 1 if int
# is tried first. Ordering bool ahead keeps a real boolean a boolean.
ParamValue = Union[bool, float, str]


class FunctionCallResult(BaseModel):
    """Enforces the strict structural envelope required
    for output serialization.
    The category distinction that makes this a SEPARATE model from
    FunctionDefinition: here `parameters` holds VALUES
    ({"a": 2.0, "s": "hello"}), whereas in FunctionDefinition it holds
    TYPES ({"a": {"type": "number"}}).
    One describes an instance, the other a schema.
    Not frozen: unlike the schema
    models, an output object is assembled as generation proceeds.
    """

    prompt: str
    name: str
    parameters: Dict[str, ParamValue]
