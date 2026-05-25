#!/usr/bin/env python3
"""Main entry point for py-tiny-claw engine with mock components."""

import logging
import os

from internal.engine import AgentEngine
from internal.provider import LLMProvider
from internal.schema import Message, Role, ToolCall, ToolDefinition, ToolResult
from internal.tools import Registry

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ==========================================
# 1. Mock LLM Provider
# ==========================================
class MockProvider:
    """Mock LLM provider that simulates model responses."""

    def __init__(self) -> None:
        self.turn = 0

    def generate(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition],
    ) -> Message:
        """Simulate LLM response: first turn requests bash, second turn outputs final result."""
        self.turn += 1

        if self.turn == 1:
            return Message(
                role=Role.ASSISTANT,
                content="让我来看看当前目录下有什么文件。",
                tool_calls=[
                    ToolCall(
                        id="call_123",
                        name="bash",
                        arguments={"command": "ls -la"},
                    ),
                ],
            )

        return Message(
            role=Role.ASSISTANT,
            content="我看到了文件列表，里面包含 main.py，任务完成！",
        )


# ==========================================
# 2. Mock Tool Registry
# ==========================================
class MockRegistry:
    """Mock tool registry that returns fake terminal output."""

    def get_available_tools(self) -> list[ToolDefinition]:
        """Return empty tool definitions for mock."""
        return []

    def execute(self, call: ToolCall) -> ToolResult:
        """Return fake terminal output."""
        return ToolResult(
            tool_call_id=call.id,
            output="-rw-r--r--  1 user group  234 Oct 24 10:00 main.py\n",
            is_error=False,
        )


# ==========================================
# 3. Assemble and Run
# ==========================================
def main() -> None:
    """Main entry point."""
    logging.info("🚀 欢迎来到 py-tiny-claw 引擎启动序列")

    # Get current directory as WorkDir physical boundary
    work_dir = os.getcwd()

    p = MockProvider()
    r = MockRegistry()

    # Instantiate core engine
    eng = AgentEngine(p, r, work_dir)

    # Execute task
    logging.info("开始执行任务...")
    try:
        eng.run("帮我检查当前目录的文件")
    except RuntimeError as e:
        logging.error(f"引擎崩溃: {e}")
        raise


if __name__ == "__main__":
    main()