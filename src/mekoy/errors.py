"""Typed compiler errors."""

from typing import override


class CompileError(Exception):
    """Boundary error for CLI and API."""

    message: str

    def __init__(self, *, message: str) -> None:
        """Store a message Click can attach a traceback to."""
        super().__init__(message)
        self.message = message

    @override
    def __str__(self) -> str:
        """Return the error message."""
        return self.message


class ModelUnreachableError(CompileError):
    """Local OpenAI-compatible server did not answer."""
