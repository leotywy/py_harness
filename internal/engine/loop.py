"""Agent engine core loop implementation."""

import json
import logging
import threading

from internal.context import PromptComposer
from internal.engine.reporter import Reporter
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
        # 【新增】引擎持有 Composer 实例
        self.composer = PromptComposer(work_dir)

    def run(self, user_prompt: str, reporter: Reporter | None = None) -> None:
        """Start the agent lifecycle with ReAct loop.

        Args:
            user_prompt: Initial user request to process.
            reporter: Reporter for output notifications (optional).

        Raises:
            RuntimeError: If LLM generation fails.
        """
        logger.info(f"[Engine] 引擎启动，锁定工作区: {self.work_dir}")
        logger.info(f"[Engine] 慢思考模式 (Thinking Phase): {self.enable_thinking}")

        # 【核心修改】动态组装 System Prompt
        # 彻底替换掉以前硬编码的提示词！
        system_msg = self.composer.build()

        # 注入动态组装的内核、AGENTS.md 与 Skills
        context_history: list[Message] = [
            system_msg,
            Message(role=Role.USER, content=user_prompt),
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

                # 【触发 Reporter】: 开始慢思考
                if reporter:
                    reporter.on_thinking()

                try:
                    think_resp = self.provider.generate(context_history, None)
                except Exception as e:
                    raise RuntimeError(f"Thinking 阶段生成失败: {e}") from e

                if think_resp.content:
                    print(f"🧠 [内部思考 Trace]: {think_resp.content}")
                    context_history.append(think_resp)

            # ====================================================================
            # Phase 2: 行动阶段 (Action) - 恢复工具，顺着规划执行
            # ====================================================================
            logger.info("[Engine][Phase 2] 恢复工具挂载，等待模型采取行动...")

            try:
                action_resp = self.provider.generate(context_history, available_tools)
            except Exception as e:
                raise RuntimeError(f"Action 阶段生成失败: {e}") from e

            context_history.append(action_resp)

            # 【触发 Reporter】: 输出阶段性总结或最终回复
            if action_resp.content and reporter:
                reporter.on_message(action_resp.content)
            elif action_resp.content:
                print(f"🤖 [对外回复]: {action_resp.content}")

            # ====================================================================
            # 退出判断
            # ====================================================================
            if not action_resp.tool_calls:
                logger.info("[Engine] 模型未请求调用工具，任务宣告完成。")
                break

            logger.info(f"[Engine] 模型请求并发调用 {len(action_resp.tool_calls)} 个工具...")

            # ====================================================================
            # 【核心改造】: 从串行演进为并行
            # ====================================================================

            # 1. 预分配固定长度列表，安全存放各并发工具的执行结果
            observation_msgs: list[Message | None] = [None] * len(action_resp.tool_calls)

            # 2. 声明 WaitGroup 用于阻塞等待所有线程完成
            wg = WaitGroup()

            # 3. 遍历所有工具，为每个工具 Fork 出独立线程
            for i, tool_call in enumerate(action_resp.tool_calls):
                wg.add(1)

                def worker(idx: int, call: ToolCall) -> None:
                    try:
                        args_str = json.dumps(call.arguments)

                        # 【触发 Reporter】: 报告即将执行的工具
                        if reporter:
                            reporter.on_tool_call(call.name, args_str)

                        logger.info(f"  -> [Thread-{idx}] 🛠️ 触发并行执行: {call.name}")

                        # 调用底层 Registry 执行工具
                        result = self.registry.execute(call)

                        if result.is_error:
                            logger.info(f"  -> [Thread-{idx}] ❌ 工具执行报错: {result.output[:100]}...")
                        else:
                            logger.info(f"  -> [Thread-{idx}] ✅ 工具执行成功 (返回 {len(result.output)} 字节)")

                        # Truncate display output for Reporter
                        display_output = result.output
                        if len(display_output) > 200:
                            display_output = display_output[:200] + "... (已截断)"

                        # 【触发 Reporter】: 汇报工具执行结果
                        if reporter:
                            reporter.on_tool_result(call.name, display_output, result.is_error)

                        # 将执行结果封装为用户消息
                        obs_msg = Message(
                            role=Role.USER,
                            content=result.output,
                            tool_call_id=call.id,
                        )

                        # 【线程安全】: 每个线程操作预分配列表的不同索引
                        observation_msgs[idx] = obs_msg

                    finally:
                        wg.done()

                # 启动线程，传入参数避免闭包陷阱
                thread = threading.Thread(target=worker, args=(i, tool_call))
                thread.start()

            # 4. Join 阻塞等待
            wg.wait()
            logger.info("[Engine] 所有并发工具执行完毕，开始聚合观察结果 (Observation)...")

            # 5. 聚合装填：将并行结果追加到上下文时间线
            for obs in observation_msgs:
                if obs is not None:
                    context_history.append(obs)