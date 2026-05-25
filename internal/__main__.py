#!/usr/bin/env python3
"""Main entry point for py-tiny-claw engine with real DashScope provider."""

import logging
import os

from internal.engine import AgentEngine
from internal.provider import OpenAIProvider, ProviderError
from internal.schema import ToolCall, ToolDefinition, ToolResult
from internal.tools import Registry

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ==========================================
# Mock Tool Registry (for testing Provider tool extraction)
# ==========================================
class MockRegistry:
    """Mock tool registry with get_weather tool."""

    def get_available_tools(self) -> list[ToolDefinition]:
        """Return tool definitions."""
        return [
            ToolDefinition(
                name="get_weather",
                description="获取指定城市的当前天气情况。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            ),
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute mock tool and return fake weather result."""
        logging.info(f"  -> [Mock 工具执行] 获取 {call.arguments.get('city', '未知')} 的天气中...")
        return ToolResult(
            tool_call_id=call.id,
            output="API 返回：今天是晴天，气温 25 度。",
            is_error=False,
        )


# ==========================================
# Main Entry Point
# ==========================================
def main() -> None:
    """Main entry point."""
    logging.info("🚀 欢迎来到 py-tiny-claw 引擎启动序列")

    # Ensure DASHSCOPE_API_KEY is set
    if not os.getenv("DASHSCOPE_API_KEY"):
        logging.error("请先设置 DASHSCOPE_API_KEY 环境变量")
        return

    work_dir = os.getcwd()

    # 1. Initialize real Provider (DashScope with glm-5)
    try:
        llm_provider = OpenAIProvider.new_dashscope_provider("glm-5")
        logging.info(f"Provider initialized: {llm_provider.model}")
    except ProviderError as e:
        logging.error(f"Provider 初始化失败: {e}")
        return

    # 2. Inject mock tool registry
    registry = MockRegistry()

    # 3. Instantiate and run engine with thinking mode enabled
    eng = AgentEngine(llm_provider, registry, work_dir, enable_thinking=False)

    # Test prompt
    prompt = "我想去北京跑步，帮我查查天气适合吗？"

    logging.info("开始执行任务...")
    try:
        eng.run(prompt)
    except RuntimeError as e:
        logging.error(f"引擎运行崩溃: {e}")
        raise


if __name__ == "__main__":
    main()