"""Tool registry for registration and execution dispatch."""

import logging
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from internal.schema import ToolCall, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


# ==========================================
# BaseTool Interface
# ==========================================
@runtime_checkable
class BaseTool(Protocol):
    """Protocol defining the contract for a tool implementation.

    All concrete tools must implement this interface.
    """

    def name(self) -> str:
        """Return the tool's globally unique name (LLM calls by this name)."""
        ...

    def definition(self) -> ToolDefinition:
        """Return the tool's metadata and JSON Schema for the LLM."""
        ...

    def execute(self, args: dict) -> str:
        """Execute the tool with arguments from the LLM.

        Args:
            args: Tool arguments (deserialized JSON from the LLM).

        Returns:
            Tool execution output as string.
        """
        ...


class Tool(ABC):
    """Abstract base class for tool implementations.

    Inherit from this class for explicit inheritance over Protocol.
    """

    @abstractmethod
    def name(self) -> str:
        """Return the tool's globally unique name."""
        pass

    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool's metadata and JSON Schema for the LLM."""
        pass

    @abstractmethod
    def execute(self, args: dict) -> str:
        """Execute the tool with arguments from the LLM.

        Args:
            args: Tool arguments (deserialized JSON from the LLM).

        Returns:
            Tool execution output as string.
        """
        pass


# ==========================================
# Registry Interface
# ==========================================
@runtime_checkable
class Registry(Protocol):
    """Protocol defining tool registration and execution dispatch."""

    def register(self, tool: BaseTool) -> None:
        """Register a new tool in the system."""
        ...

    def get_available_tools(self) -> list[ToolDefinition]:
        """Return all registered tools' schema definitions for the Main Loop."""
        ...

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute the requested tool and return the result."""
        ...


class BaseRegistry(ABC):
    """Abstract base class for tool registry."""

    @abstractmethod
    def register(self, tool: BaseTool) -> None:
        """Register a new tool in the system."""
        pass

    @abstractmethod
    def get_available_tools(self) -> list[ToolDefinition]:
        """Return all registered tools' schema definitions."""
        pass

    @abstractmethod
    def execute(self, call: ToolCall) -> ToolResult:
        """Execute the requested tool and return the result."""
        pass


# ==========================================
# Registry Implementation
# ==========================================
class ToolRegistry(BaseRegistry):
    """Default implementation of Registry interface.

    Uses a dict with tool Name as key for O(1) routing lookup.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool implementation to register.
        """
        name = tool.name()
        if name in self._tools:
            logger.warning(f"[Warning] 工具 '{name}' 已经被注册，将被覆盖。")

        self._tools[name] = tool
        logger.info(f"[Registry] 成功挂载工具: {name}")

    def unregister(self, name: str) -> None:
        """Unregister a tool from the registry.

        Args:
            name: Name of the tool to remove.
        """
        if name in self._tools:
            del self._tools[name]
            logger.info(f"[Registry] 已卸载工具: {name}")

    def get_available_tools(self) -> list[ToolDefinition]:
        """Return all registered tools' schema definitions."""
        return [tool.definition() for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute the requested tool and return the result.

        Args:
            call: Tool call request from the LLM.

        Returns:
            Tool execution result.
        """
        # 1. Route lookup: if tool not found, model hallucinated
        tool = self._tools.get(call.name)
        if tool is None:
            err_msg = f"Error: 系统中不存在名为 '{call.name}' 的工具。"
            logger.error(err_msg)
            return ToolResult(
                tool_call_id=call.id,
                output=err_msg,
                is_error=True,  # Mark as error, model will attempt correction
            )

        # 2. Execute tool logic
        try:
            output = tool.execute(call.arguments)
            logger.info(f"[Registry] 工具 '{call.name}' 执行成功")
            return ToolResult(
                tool_call_id=call.id,
                output=output,
                is_error=False,
            )

        # 3. Handle execution error
        except Exception as e:
            err_msg = f"Error executing {call.name}: {e}"
            logger.error(err_msg)
            return ToolResult(
                tool_call_id=call.id,
                output=err_msg,
                is_error=True,
            )


# ==========================================
# Factory Function
# ==========================================
def new_registry() -> Registry:
    """Create a new Registry instance."""
    return ToolRegistry()