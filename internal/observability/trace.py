"""Distributed tracing for observability.

Provides span-based tracing to track operation timing and hierarchy.
"""

from __future__ import annotations

import json
import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# trace_key is the context key for storing Span
TRACE_KEY = "span_context"


@dataclass
class Span:
    """Represents a time span and operation node in distributed tracing.

    Attributes:
        name: Operation name.
        start_time: When the span started.
        end_time: When the span ended.
        duration_ms: Duration in milliseconds.
        attributes: Metadata (e.g., tokens consumed, command executed).
        children: Child spans.
    """

    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    duration_ms: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def end_span(self) -> None:
        """End the span and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)

    def add_attribute(self, key: str, value: Any) -> None:
        """Add metadata to the span.

        Args:
            key: Attribute name.
            value: Attribute value.
        """
        with self._lock:
            self.attributes[key] = value

    def add_child(self, child: Span) -> None:
        """Add a child span.

        Args:
            child: Child span to add.
        """
        with self._lock:
            self.children.append(child)

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
        }

        if self.attributes:
            result["attributes"] = self.attributes

        if self.children:
            result["children"] = [child.to_dict() for child in self.children]

        return result


def start_span(ctx: dict[str, Any], name: str) -> tuple[dict[str, Any], Span]:
    """Start a new tracing span and cascade it into context.

    Args:
        ctx: Context dictionary containing potential parent span.
        name: Span name for the operation.

    Returns:
        Tuple of (new context with span, new span instance).
    """
    span = Span(
        name=name,
        start_time=time.time(),
        attributes={},
    )

    # Try to get parent span from context
    parent = ctx.get(TRACE_KEY)
    if parent is not None and isinstance(parent, Span):
        parent.add_child(span)

    # Create new context with current span as the latest parent
    new_ctx = ctx.copy()
    new_ctx[TRACE_KEY] = span

    return new_ctx, span


def export_trace_to_file(root_span: Span, work_dir: str, session_id: str) -> str:
    """Export trace to JSON file when root span ends.

    Args:
        root_span: Root span of the trace.
        work_dir: Working directory.
        session_id: Session ID for filename.

    Returns:
        Path to the saved trace file.

    Raises:
        OSError: If file write fails.
    """
    trace_dir = Path(work_dir) / ".claw" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    filename = trace_dir / f"trace_{session_id}_{int(time.time())}.json"

    # Pretty print JSON for human and tool readability
    data = json.dumps(root_span.to_dict(), indent=2, ensure_ascii=False)

    filename.write_text(data, encoding="utf-8")

    return str(filename)