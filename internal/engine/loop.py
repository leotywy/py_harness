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
        enable_thinking: bool = False,
    ) -> None:
        """Initialize the agent engine.

        Args:
            provider: LLM provider for reasoning.
            registry: Tool registry for execution dispatch.
            work_dir: Working directory boundary for the agent.
            enable_thinking: Enable slow thinking mode (extended reasoning).
        """
        self.provider = provider
        self.registry = registry
        self.work_dir = work_dir
        self.enable_thinking = enable_thinking

    def run(self, user_prompt: str) -> None:
        """Start the agent lifecycle with ReAct loop.

        Args:
            user_prompt: Initial user request to process.

        Raises:
            RuntimeError: If LLM generation fails.
        """
        logger.info(f"[Engine] 引擎启动，锁定工作区: {self.work_dir}")
        logger.info(f"[Engine] 慢思考模式 (Thinking Phase): {self.enable_thinking}")

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

        while True:
            turn_count += 1
            logger.info(f"\n========== [Turn {turn_count}] 开始 ==========")

            available_tools = self.registry.get_available_tools()

            # ====================================================================
            # Phase 1: 慢思考阶段 (Thinking) - 剥夺工具，强制规划
            # ====================================================================
            if self.enable_thinking:
                logger.info("[Engine][Phase 1] 剥夺工具访问权，强制进入慢思考与规划阶段...")

                # 核心机制：传入的 available_tools 为 None！
                # 大模型看不到任何 JSON Schema，被迫只能输出纯文本的思考过程。
                try:
                    think_resp = self.provider.generate(context_history, None)
                except Exception as e:
                    raise RuntimeError(f"Thinking 阶段生成失败: {e}") from e

                # 如果模型输出了思考过程，将其作为 Assistant 消息追加到上下文中
                if think_resp.content:
                    print(f"🧠 [内部思考 Trace]: {think_resp.content}")
                    context_history.append(think_resp)

            # ====================================================================
            # Phase 2: 行动阶段 (Action) - 恢复工具，顺着规划执行
            # ====================================================================
            logger.info("[Engine][Phase 2] 恢复工具挂载，等待模型采取行动...")

            # 此时的 context_history 中已经包含了上一阶段模型自己的 Thinking Trace。
            # 模型会顺着自己的逻辑，结合恢复的 available_tools 发起精准的工具调用。
            try:
                action_resp = self.provider.generate(context_history, available_tools)
            except Exception as e:
                raise RuntimeError(f"Action 阶段生成失败: {e}") from e

            context_history.append(action_resp)

            if action_resp.content:
                print(f"🤖 [对外回复]: {action_resp.content}")

            # ====================================================================
            # 退出与执行逻辑
            # ====================================================================
            if not action_resp.tool_calls:
                logger.info("[Engine] 模型未请求调用工具，任务宣告完成。")
                break

            logger.info(f"[Engine] 模型请求调用 {len(action_resp.tool_calls)} 个工具...")

            for tool_call in action_resp.tool_calls:
                logger.info(f"  -> 🛠️ 执行工具: {tool_call.name}, 参数: {tool_call.arguments}")

                result = self.registry.execute(tool_call)

                if result.is_error:
                    logger.info(f"  -> ❌ 工具执行报错: {result.output}")
                else:
                    logger.info(f"  -> ✅ 工具执行成功 (返回 {len(result.output)} 字节)")

                observation_msg = Message(
                    role=Role.USER,
                    content=result.output,
                    tool_call_id=tool_call.id,
                )
                context_history.append(observation_msg)