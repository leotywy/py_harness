"""Agent engine core loop implementation."""

import json
import logging
import threading

from internal.context import Compactor, PromptComposer
from internal.engine.reporter import Reporter
from internal.engine.session import Session
from internal.provider import LLMProvider
from internal.schema import Message, Role, ToolCall
from internal.tools import Registry

logger = logging.getLogger(__name__)


class WaitGroup:
    """Python implementation of Go's sync.WaitGroup."""

    def __init__(self) -> None:
        self._count = 0
        self._condition = threading.Condition()

    def add(self, delta: int = 1) -> None:
        """Add delta to the counter."""
        with self._condition:
            self._count += delta

    def done(self) -> None:
        """Decrement counter by one."""
        with self._condition:
            self._count -= 1
            if self._count == 0:
                self._condition.notify_all()

    def wait(self) -> None:
        """Block until counter reaches zero."""
        with self._condition:
            while self._count > 0:
                self._condition.wait()


class AgentEngine:
    """Core engine of the mini OS, driving the agent lifecycle."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: Registry,
        enable_thinking: bool = False,
        plan_mode: bool = False,
    ) -> None:
        """Initialize the agent engine.

        Args:
            provider: LLM provider for reasoning.
            registry: Tool registry for execution dispatch.
            enable_thinking: Enable slow thinking mode (extended reasoning).
            plan_mode: Enable plan mode for long-running tasks with state externalization.

        Note:
            WorkDir is now attached to Session, not Engine level!
        """
        self.provider = provider
        self.registry = registry
        self.enable_thinking = enable_thinking
        # 【新增】暴露给外部的计划模式开关
        self.plan_mode = plan_mode

        # 【初始化压缩器】: Higher threshold (20000 chars) for production,
        # protecting last 6 messages (about 2 turns)
        self.compactor = Compactor(max_chars=20000, retain_last_msgs=6)

    def run(self, session: Session, reporter: Reporter | None = None) -> None:
        """Start the agent lifecycle with ReAct loop.

        Args:
            session: Session instance with work_dir and history.
            reporter: Reporter for output notifications (optional).

        Raises:
            RuntimeError: If LLM generation fails.
        """
        logger.info(
            f"[Engine] 唤醒会话 [{session.id}], 锁定工作区: {session.work_dir} "
            f"(PlanMode: {self.plan_mode})"
        )

        # Dynamically generate composer with current PlanMode state
        composer = PromptComposer(session.work_dir, plan_mode=self.plan_mode)
        system_msg = composer.build()

        while True:
            available_tools = self.registry.get_available_tools()

            # 1. Extract recent Working Memory (last 20 messages)
            working_memory = session.get_working_memory(limit=20)

            context_history: list[Message] = [system_msg]
            context_history.extend(working_memory)

            # 2. 【核心注入点】: Pass through memory compactor before Provider call!
            compacted_context = self.compactor.compact(context_history)

            # 3. ================= Phase 1: Thinking =================
            if self.enable_thinking:
                if reporter:
                    reporter.on_thinking()

                logger.info("[Engine][Phase 1] 剥夺工具访问权，强制进入慢思考与规划阶段...")

                try:
                    think_resp = self.provider.generate(compacted_context, None)
                except Exception as e:
                    raise RuntimeError(f"Thinking 阶段生成失败: {e}") from e

                if think_resp.content:
                    # 【驾驭精髓】: Persist full response to Session (not affected by Compact!)
                    session.append(think_resp)
                    compacted_context.append(think_resp)
                    print(f"🧠 [内部思考 Trace]: {think_resp.content}")

            # 4. ================= Phase 2: Action =================
            logger.info("[Engine][Phase 2] 恢复工具挂载，等待模型采取行动...")

            try:
                action_resp = self.provider.generate(compacted_context, available_tools)
            except Exception as e:
                raise RuntimeError(f"Action 阶段生成失败: {e}") from e

            # 【驾驭精髓】: Persist full response to Session (not affected by Compact!)
            session.append(action_resp)
            compacted_context.append(action_resp)

            if action_resp.content and reporter:
                reporter.on_message(action_resp.content)
            elif action_resp.content:
                print(f"🤖 [对外回复]: {action_resp.content}")

            # Exit if no tool calls
            if not action_resp.tool_calls:
                logger.info("[Engine] 模型未请求调用工具，任务宣告完成。")
                break

            logger.info(f"[Engine] 模型请求并发调用 {len(action_resp.tool_calls)} 个工具...")

            # 5. ================= 并发执行底层工具 =================
            observation_msgs: list[Message | None] = [None] * len(action_resp.tool_calls)
            wg = WaitGroup()

            def worker(idx: int, call: ToolCall) -> None:
                try:
                    args_str = json.dumps(call.arguments)

                    if reporter:
                        reporter.on_tool_call(call.name, args_str)

                    logger.info(f"  -> [Thread-{idx}] 🛠️ 触发并行执行: {call.name}")

                    result = self.registry.execute(call)

                    if result.is_error:
                        logger.info(f"  -> [Thread-{idx}] ❌ 工具执行报错: {result.output[:100]}...")
                    else:
                        logger.info(f"  -> [Thread-{idx}] ✅ 工具执行成功 (返回 {len(result.output)} 字节)")

                    # Truncate display output
                    display_output = result.output
                    if len(display_output) > 200:
                        display_output = display_output[:200] + "... (已截断)"

                    if reporter:
                        reporter.on_tool_result(call.name, display_output, result.is_error)

                    # Wrap as User message
                    obs_msg = Message(
                        role=Role.USER,
                        content=result.output,
                        tool_call_id=call.id,
                    )
                    observation_msgs[idx] = obs_msg

                finally:
                    wg.done()

            for i, tool_call in enumerate(action_resp.tool_calls):
                wg.add(1)
                thread = threading.Thread(target=worker, args=(i, tool_call))
                thread.start()

            wg.wait()
            logger.info("[Engine] 所有并发工具执行完毕，开始聚合观察结果 (Observation)...")

            # Persist full observations to Session
            for obs in observation_msgs:
                if obs is not None:
                    session.append(obs)