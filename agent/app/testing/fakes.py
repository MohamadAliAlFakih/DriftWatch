"""FakeChatModel — deterministic stand-in for ChatGroq in tests.

D-20: canned responses indexed by call order. Each fixture supplies a list of
expected responses; FakeChatModel returns them in order. Calling more times
than the list provided is a hard error. Plain `.ainvoke` returns AIMessage-shaped
objects (with .content); `.with_structured_output(Schema)` returns a runnable
whose .ainvoke validates the next response as `Schema` and returns the parsed model.

Tests pass `responses=[...]` mixing strings (for plain ainvoke) and dicts (for
structured-output calls — dicts are model_validate'd into the requested schema).
"""

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel


# minimal AIMessage-shaped wrapper exposing .content for plain ainvoke
class _Plain:
    # store the canned content string as the .content attribute LangChain consumers expect
    def __init__(self, content: str) -> None:
        self.content = content


# runnable returned by with_structured_output — pops next response and validates against schema
class _StructuredFakeRunnable:
    # remember which schema we're expected to validate against and which parent owns the queue
    def __init__(self, schema: type[BaseModel], parent: "FakeChatModel") -> None:
        self._schema = schema
        self._parent = parent

    # pop the next canned response and coerce it into a `schema` instance
    async def ainvoke(self, messages: Any, **_kwargs: Any) -> BaseModel:
        # exhausting the queue means the test under-seeded responses; raise loudly so it gets fixed
        if not self._parent._responses:
            raise AssertionError(
                f"FakeChatModel exhausted (schema={self._schema.__name__})"
            )
        value = self._parent._responses.pop(0)
        # record what was returned for assertions in tests
        self._parent.call_history.append(
            {"schema": self._schema.__name__, "value": value}
        )
        # if the test passed an already-built schema instance, return it directly
        if isinstance(value, self._schema):
            return value
        # otherwise expect a dict that we can validate into the schema
        if isinstance(value, dict):
            return self._schema.model_validate(value)
        raise AssertionError(
            f"FakeChatModel: expected {self._schema.__name__} or dict, got {type(value).__name__}"
        )


# ordered-canned-response fake — D-20
class FakeChatModel:
    # accept an ordered iterable of responses; copy into a mutable list we can pop from
    def __init__(self, *, responses: Iterable[Any] | None = None) -> None:
        self._responses: list[Any] = list(responses or [])
        self.call_history: list[dict[str, Any]] = []

    # mimic langchain BaseChatModel.with_structured_output by returning a runnable wrapper
    def with_structured_output(
        self, schema: type[BaseModel]
    ) -> _StructuredFakeRunnable:
        return _StructuredFakeRunnable(schema, self)

    # mimic langchain BaseChatModel.bind — return self unchanged; tests don't care about kwargs
    def bind(self, **_kwargs: Any) -> "FakeChatModel":
        return self

    # plain ainvoke — pops next response and wraps as AIMessage-shape
    async def ainvoke(self, messages: Any, **_kwargs: Any) -> _Plain:
        # exhausted queue means test under-seeded responses; raise loudly
        if not self._responses:
            raise AssertionError("FakeChatModel exhausted (plain ainvoke)")
        value = self._responses.pop(0)
        # record the call for assertions
        self.call_history.append({"schema": None, "value": value})
        # plain calls expect strings; wrap into AIMessage-shaped object
        if isinstance(value, str):
            return _Plain(value)
        # tolerance: if a non-string snuck in for a plain call, stringify it
        return _Plain(str(value))
