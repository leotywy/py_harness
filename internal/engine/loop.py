"""Agent engine core loop implementation."""

import logging

from internal.provider import LLMProvider
from internal.schema import Message, Role, ToolCall
from internal.tools import Registry

logger = logging.getLogger(__name__)


class AgentEngine:
    """Core engine of the mini OS, driving the agent lifecycle."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: Registry,
        work_dir: str,
    ) -> None:
        """Initialize the agent engine.

        Args:
            provider: LLM provider for reasoning.
            registry: Tool registry for execution dispatch.
            work_dir: Working directory boundary for the agent.
        """
        self.provider = provider
        self.registry = registry
        self.work_dir = work_dir

    def run(self, user_prompt: str) -> None:
        """Start the agent lifecycle with ReAct loop.

        Args:
            user_prompt: Initial user request to process.

        Raises:
            RuntimeError: If LLM generation fails.
        """
        logger.info(f"[Engine] 引擎启动，锁定工作区: {self.work_dir}")

        # 1. Initialize session context (memory)
        # In production, this would load AGENTS.md via dynamic prompt assembler.
        # Currently hardcoded for simplicity.
        context_history: list[Message] = [
            Message(
                role=Role.SYSTEM,
                content="You are py-tiny-claw, an expert coding assistant. You have full access to tools in the workspace.",
            ),
            Message(
                role=Role.USER,
                content=user_prompt,
            ),
        ]

        turn_count = 0

        # 2. The Main Loop: heartbeat starts (standard ReAct cycle)
        while True:
            turn_count += 1
            logger.info(f"========== [Turn {turn_count}] 开始 ==========")

            # Get all registered tool definitions
            available_tools = self.registry.get_available_tools()

            # Request LLM inference (including Reasoning)
            logger.info("[Engine] 正在思考 (Reasoning)...")
            try:
                response_msg = self.provider.generate(context_history, available_tools)
            except Exception as e:
                raise RuntimeError(f"模型生成失败: {e}") from e

            # Append model response to context history
            context_history.append(response_msg)

            # Print model's text response (thinking process or final result)
            if response_msg.content:
                print(f"🤖 模型: {response_msg.content}")

            # 3. Exit condition check
            # If no tool calls, model considers task complete, break loop.
            if not response_msg.tool_calls:
                logger.info("[Engine] 任务完成，退出循环。")
                break

            # 4. Execute Actions and get Observations
            logger.info(f"[Engine] 模型请求调用 {len(response_msg.tool_calls)} 个工具...")

            for tool_call in response_msg.tool_calls:
                logger.info(f"  -> 🛠️ 执行工具: {tool_call.name}, 参数: {tool_call.arguments}")

                # Dispatch and execute tool via Registry
                result = self.registry.execute(tool_call)

                if result.is_error:
                    logger.info(f"  -> ❌ 工具执行报错: {result.output}")
                else:
                    logger.info(f"  -> ✅ 工具执行成功 (返回 {len(result.output)} 字节)")

                # Wrap observation result as User Message and append to context
                # Note: ToolCallID must be included! This maintains reasoning chain.
                observation_msg = Message(
                    role=Role.USER,
                    content=result.output,
                    tool_call_id=tool_call.id,
                )
                context_history.append(observation_msg)

            # Loop back, model will think again with new observations...