"""Terminal reporter for Agent engine status output."""

from internal.engine.reporter import BaseReporter


class TerminalReporter(BaseReporter):
    """Reporter implementation for terminal output.

    Prints Agent status in a user-friendly format.
    """

    def on_thinking(self) -> None:
        """Print thinking indicator."""
        print("\n[🤔 思考中] 模型正在推理...")

    def on_tool_call(self, tool_name: str, args: str) -> None:
        """Print tool call info.

        Args:
            tool_name: Name of the tool being called.
            args: Tool arguments as string.
        """
        print(f"[🛠️ 调用工具] {tool_name}")

        # Truncate long args display to keep terminal clean
        display_args = args.replace("\n", "\\n").replace("\r", "\\r")
        if len(display_args) > 150:
            display_args = display_args[:150] + "... (已截断)"

        print(f"   参数: {display_args}")

    def on_tool_result(self, tool_name: str, result: str, is_error: bool) -> None:
        """Print tool execution result.

        Args:
            tool_name: Name of the tool that was executed.
            result: Tool execution result.
            is_error: Whether the execution failed.
        """
        if is_error:
            print(f"[❌ 执行失败] {tool_name}")
            if result:
                # Truncate error message
                display_result = result
                if len(display_result) > 200:
                    display_result = display_result[:200] + "... (已截断)"
                print(f"   错误: {display_result}")
        else:
            print(f"[✅ 执行成功] {tool_name}")

    def on_message(self, content: str) -> None:
        """Print final message.

        Args:
            content: Final message content.
        """
        if not content:
            return

        print(f"\n🤖 Agent 回复:\n{content}\n")