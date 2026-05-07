"""Structured API error helpers.

File summary:
- Defines a reusable error body shape for API responses.
- Wraps errors under a top-level `error` key.
- Keeps error response structure consistent across endpoints.
"""

from typing import Any

from pydantic import BaseModel


class ErrorBody(BaseModel):
    """Represent the inner error object returned by API failures."""

    code: str
    message: str
    details: dict[str, Any] | list[Any] | None = None


class ErrorResponse(BaseModel):
    """Represent the top-level API error response wrapper."""

    error: ErrorBody
