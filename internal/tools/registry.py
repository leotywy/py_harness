"""Tool registry for registration and execution dispatch."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from internal.schema import ToolCall, ToolDefinition, ToolResult


@runtime_checkable
class Tool(Protocol):
    """Protocol defining the contract for a tool implementation."""

    def get_definition(self) -> ToolDefinition:
        """Return the tool's schema definition for the LLM."""
        ...

    def execute(self, arguments: dict) -> str:
        """Execute the tool with given arguments.

        Args:
            arguments: Tool arguments from the LLM.

        Returns:
            Tool execution output as string.
        """
        ...


class BaseTool(ABC):
    """Abstract base class for tool implementations."""

    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Return the tool's schema definition for the LLM."""
        pass

    @abstractmethod
    def execute(self, arguments: dict) -> str:
        """Execute the tool with given arguments.

        Args:
            arguments: Tool arguments from the LLM.

        Returns:
            Tool execution output as string.
        """
        pass


@runtime_checkable
class Registry(Protocol):
    """Protocol defining tool registration and execution dispatch."""

    def get_available_tools(self) -> list[ToolDefinition]:
        """Return all registered tools' schema definitions."""
        ...

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute the requested tool and return the result.

        Args:
            call: Tool call request from the LLM.

        Returns:
            Tool execution result.
        """
        ...


class BaseRegistry(ABC):
    """Abstract base class for tool registry."""

    @abstractmethod
    def get_available_tools(self) -> list[ToolDefinition]:
        """Return all registered tools' schema definitions."""
        pass

    @abstractmethod
    def execute(self, call: ToolCall) -> ToolResult:
        """Execute the requested tool and return the result.

        Args:
            call: Tool call request from the LLM.

        Returns:
            Tool execution result.
        """
        pass


class ToolRegistry(BaseRegistry):
    """Concrete implementation of tool registry."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool implementation to register.
        """
        definition = tool.get_definition()
        self._tools[definition.name] = tool

    def get_available_tools(self) -> list[ToolDefinition]:
        """Return all registered tools' schema definitions."""
        return [tool.get_definition() for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute the requested tool and return the result.

        Args:
            call: Tool call request from the LLM.

        Returns:
            Tool execution result.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                output=f"Unknown tool: {call.name}",
                is_error=True,
            )

        try:
            output = tool.execute(call.arguments)
            return ToolResult(
                tool_call_id=call.id,
                output=output,
                is_error=False,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=call.id,
                output=str(e),
                is_error=True,
            )