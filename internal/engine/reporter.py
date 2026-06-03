"""Reporter interface for Agent engine output."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class Reporter(Protocol):
    """Protocol defining how Agent engine outputs information.

    This allows seamless switching between CLI, Feishu, DingTalk, WebUI,
    and other presentation layers.
    """

    def on_thinking(self) -> None:
        """Called when model starts slow thinking (Reasoning)."""
        ...

    def on_tool_call(self, tool_name: str, args: str) -> None:
        """Called when model decides to invoke a tool.

        Args:
            tool_name: Name of the tool being called.
            args: Tool arguments as string.
        """
        ...

    def on_tool_result(self, tool_name: str, result: str, is_error: bool) -> None:
        """Called when tool execution completes.

        Args:
            tool_name: Name of the tool that was executed.
            result: Tool execution result.
            is_error: Whether the execution failed.
        """
        ...

    def on_message(self, content: str) -> None:
        """Called when model outputs final text response to user.

        Args:
            content: Final message content.
        """
        ...


class BaseReporter(ABC):
    """Abstract base class for Reporter implementations.

    Inherit from this class for explicit inheritance over Protocol.
    """

    @abstractmethod
    def on_thinking(self) -> None:
        """Called when model starts slow thinking (Reasoning)."""
        pass

    @abstractmethod
    def on_tool_call(self, tool_name: str, args: str) -> None:
        """Called when model decides to invoke a tool.

        Args:
            tool_name: Name of the tool being called.
            args: Tool arguments as string.
        """
        pass

    @abstractmethod
    def on_tool_result(self, tool_name: str, result: str, is_error: bool) -> None:
        """Called when tool execution completes.

        Args:
            tool_name: Name of the tool that was executed.
            result: Tool execution result.
            is_error: Whether the execution failed.
        """
        pass

    @abstractmethod
    def on_message(self, content: str) -> None:
        """Called when model outputs final text response to user.

        Args:
            content: Final message content.
        """
        pass


class CLIReporter(BaseReporter):
    """Default CLI reporter with colored output."""

    def on_thinking(self) -> None:
        """Print thinking indicator."""
        print("\n🧠 [Thinking] 模型正在思考...")

    def on_tool_call(self, tool_name: str, args: str) -> None:
        """Print tool call info."""
        print(f"🔧 [ToolCall] {tool_name}")
        print(f"   参数: {args[:100]}{'...' if len(args) > 100 else ''}")

    def on_tool_result(self, tool_name: str, result: str, is_error: bool) -> None:
        """Print tool result."""
        status = "❌" if is_error else "✅"
        print(f"{status} [ToolResult] {tool_name}")
        print(f"   结果: {result[:100]}{'...' if len(result) > 100 else ''}")

    def on_message(self, content: str) -> None:
        """Print final message."""
        print(f"\n🤖 [Message] {content}")


class SilentReporter(BaseReporter):
    """Silent reporter that does nothing."""

    def on_thinking(self) -> None:
        pass

    def on_tool_call(self, tool_name: str, args: str) -> None:
        pass

    def on_tool_result(self, tool_name: str, result: str, is_error: bool) -> None:
        pass

    def on_message(self, content: str) -> None:
        pass