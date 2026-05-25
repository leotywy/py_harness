"""Message schema for LLM communication."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Role defines the message role, the foundation of LLM communication."""

    SYSTEM = "system"      # System prompt: establishes Agent's personality and boundaries
    USER = "user"          # User input / tool execution result (Observation)
    ASSISTANT = "assistant"  # Model output: contains reasoning or tool calls


@dataclass
class ToolCall:
    """Represents a model request to call a specific tool."""

    id: str                          # Unique ID for the tool call
    name: str                        # Tool name to call (e.g., "bash")
    arguments: dict[str, Any]        # JSON arguments for the tool


@dataclass
class ToolResult:
    """Represents the physical result after a tool executes locally."""

    tool_call_id: str                # ID of the corresponding tool call
    output: str                      # Tool execution console output or error stack
    is_error: bool = False           # Mark if failed, for error self-healing


@dataclass
class Message:
    """Represents a single message in the context."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary for JSON serialization."""
        result: dict[str, Any] = {"role": self.role.value}

        if self.content:
            result["content"] = self.content

        if self.tool_calls:
            result["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in self.tool_calls
            ]

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        return result


@dataclass
class ToolDefinition:
    """Describes tool metadata for the model to understand its purpose."""

    name: str
    description: str
    input_schema: dict[str, Any]      # Corresponds to JSON Schema

    def to_dict(self) -> dict[str, Any]:
        """Convert tool definition to dictionary for JSON serialization."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }