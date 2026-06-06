"""Agent engine core loop implementation."""

import json
import logging
import threading

from internal.context import Compactor, PromptComposer, RecoveryManager
from internal.engine.reminder import ReminderInjector
from internal.engine.reporter import Reporter
from internal.engine.session import Session
from internal.observability import Span, export_trace_to_file, start_span
from internal.provider import LLMProvider
from internal.schema import Message, Role, ToolCall, ToolResult
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
        self.plan_mode = plan_mode

        # Initialize compactor
        self.compactor = Compactor(max_chars=20000, retain_last_msgs=6)

        # 【新增】自愈管理器
        self.recovery = RecoveryManager()

        # 【新增】死循环探测与干预注入器
        self.injector = ReminderInjector()

    def run(
        self,
        session: Session,
        reporter: Reporter | None = None,
        ctx: dict | None = None,
    ) -> None:
        """Start the agent lifecycle with ReAct loop.

        Args:
            session: Session instance with work_dir and history.
            reporter: Reporter for output notifications (optional).
            ctx: Context for tracing (optional).

        Raises:
            RuntimeError: If LLM generation fails.
        """
        logger.info(
            f"[Engine] 唤醒会话 [{session.id}], 锁定工作区: {session.work_dir} "
            f"(PlanMode: {self.plan_mode})"
        )

        # 【埋点 1】：开启 Root Span，记录整个任务的生命周期
        if ctx is None:
            ctx = {}
        ctx, root_span = start_span(ctx, "Agent.Run")
        root_span.add_attribute("SessionID", session.id)
        root_span.add_attribute("WorkDir", session.work_dir)

        composer = PromptComposer(session.work_dir, plan_mode=self.plan_mode)
        system_msg = composer.build()

        turn_count = 0
        try:
            while True:
                turn_count += 1

                # 【埋点 2】：记录单次 Turn 循环
                ctx, turn_span = start_span(ctx, f"Turn-{turn_count}")

                try:
                    available_tools = self.registry.get_available_tools()
                    working_memory = session.get_working_memory(limit=20)

                    context_history: list[Message] = [system_msg]
                    context_history.extend(working_memory)

                    compacted_context = self.compactor.compact(context_history)

                    # 记录发给模型的实际上下文大小，非常有助于排查幻觉
                    turn_span.add_attribute("context_message_count", len(compacted_context))

                    current_turn_thinking_content = ""

                    # ================= Phase 1: Thinking =================
                    if self.enable_thinking:
                        if reporter:
                            reporter.on_thinking()

                        logger.info("[Engine][Phase 1] 剥夺工具访问权，强制进入慢思考与规划阶段...")

                        # 【埋点 3】：记录 Thinking 调用
                        ctx, think_span = start_span(ctx, "LLM.Thinking")
                        try:
                            think_resp = self.provider.generate(compacted_context, None)
                        except Exception as e:
                            raise RuntimeError(f"Thinking 阶段生成失败: {e}") from e
                        finally:
                            think_span.end_span()

                        if think_resp.content:
                            current_turn_thinking_content = think_resp.content
                            compacted_context.append(think_resp)
                            print(f"🧠 [内部思考 Trace]: {think_resp.content}")

                    # ================= Phase 2: Action =================
                    logger.info("[Engine][Phase 2] 恢复工具挂载，等待模型采取行动...")

                    # 【埋点 4】：记录 Action 调用
                    ctx, act_span = start_span(ctx, "LLM.Action")
                    try:
                        action_resp = self.provider.generate(compacted_context, available_tools)
                    except Exception as e:
                        raise RuntimeError(f"Action 阶段生成失败: {e}") from e
                    finally:
                        act_span.end_span()

                    # (Key fix from previous session: merge into single valid Assistant message)
                    final_content = (current_turn_thinking_content + "\n" + action_resp.content).strip()
                    final_assistant_msg = Message(
                        role=Role.ASSISTANT,
                        content=final_content,
                        tool_calls=action_resp.tool_calls,
                    )
                    session.append(final_assistant_msg)

                    if action_resp.content and reporter:
                        reporter.on_message(action_resp.content)
                    elif action_resp.content:
                        print(f"🤖 [对外回复]: {action_resp.content}")

                    # Exit if no tool calls
                    if not action_resp.tool_calls:
                        logger.info("[Engine] 模型未请求调用工具，任务宣告完成。")
                        break

                    logger.info(f"[Engine] 模型请求并发调用 {len(action_resp.tool_calls)} 个工具...")

                    # ================= Execute tools and inject recovery =================
                    observation_msgs: list[Message | None] = [None] * len(action_resp.tool_calls)
                    wg = WaitGroup()

                    # 用于收集本轮执行的最后一个工具，供 Reminder 探测器分析
                    # (简化为取第一个工具调用)
                    last_tool_call: ToolCall | None = None
                    last_tool_result: ToolResult | None = None
                    last_lock = threading.Lock()

                    def worker(idx: int, call: ToolCall, turn_ctx: dict) -> None:
                        nonlocal last_tool_call, last_tool_result
                        try:
                            args_str = json.dumps(call.arguments)

                            if reporter:
                                reporter.on_tool_call(call.name, args_str)

                            logger.info(f"  -> [Thread-{idx}] 🛠️ 触发并行执行: {call.name}")

                            # Physical tool execution (passing turn_ctx for tracing)
                            result = self.registry.execute(call)

                            # 【核心拦截与注入】
                            final_output = result.output
                            if result.is_error:
                                # Error occurred, let RecoveryManager diagnose and inject hints
                                final_output = self.recovery.analyze_and_inject(call.name, result.output)
                                logger.info(f"  -> [Thread-{idx}] ❌ 注入救援指南: {final_output[:100]}...")
                            else:
                                logger.info(f"  -> [Thread-{idx}] ✅ 工具执行成功 (返回 {len(result.output)} 字节)")

                            # Truncate display output for reporter
                            display_output = final_output
                            if len(display_output) > 200:
                                display_output = display_output[:200] + "... (已截断)"

                            if reporter:
                                reporter.on_tool_result(call.name, display_output, result.is_error)

                            # Write recovery-injected result to context history
                            observation_msgs[idx] = Message(
                                role=Role.USER,
                                content=final_output,
                                tool_call_id=call.id,
                            )

                            # 捕获状态供探测器使用 (取第一个工具调用)
                            with last_lock:
                                if idx == 0:
                                    last_tool_call = call
                                    last_tool_result = result

                        finally:
                            wg.done()

                    for i, tool_call in enumerate(action_resp.tool_calls):
                        wg.add(1)
                        # Pass turn_ctx to worker for potential future tracing
                        thread = threading.Thread(target=worker, args=(i, tool_call, ctx))
                        thread.start()

                    wg.wait()
                    logger.info("[Engine] 所有并发工具执行完毕，开始聚合观察结果 (Observation)...")

                    # Persist all observations to Session
                    for obs in observation_msgs:
                        if obs is not None:
                            session.append(obs)

                    # 【核心防线】：在准备进入下一轮之前，进行死循环探测！
                    if last_tool_call is not None and last_tool_result is not None:
                        reminder_msg = self.injector.check_and_inject(last_tool_call, last_tool_result)
                        if reminder_msg is not None:
                            # 如果触发了干预规则，将这条严厉的提醒作为 User 消息追加到 Session 最末尾
                            # 大模型在下一轮被唤醒时，第一眼就会看到这句话，从而打破局部执念
                            session.append(reminder_msg)
                            logger.warning("[Engine] ⚠️ 死循环干预触发，已注入修正指令")

                finally:
                    # 结束本轮 Turn 的 Span
                    turn_span.end_span()

        finally:
            # 【埋点 5】：defer 保证在引擎退出时，无论成功失败，都能结束根 Span 并导出 Trace 报告
            root_span.end_span()
            export_trace_to_file(root_span, session.work_dir, session.id)
            logger.info("📊 [Tracing] 本次任务的执行回放链路已保存至工作区的 .claw/traces 目录下")

    def run_sub(
        self,
        task_prompt: str,
        read_only_registry: Registry,
        reporter: object | None = None,
    ) -> str:
        """Run a one-time isolated sub-agent loop for exploration tasks.

       专为 Subagent 拉起的一次性受限循环。
        不依赖外部 Session，打完就跑。
        Reporter：为了让用户在终端看到子智能体的工作轨迹，我们将主线程的 Reporter 透传进来。

        Args:
            task_prompt: Clear instruction for the sub-agent.
            read_only_registry: Restricted read-only registry for sub-agent.
            reporter: Reporter for output notifications (optional).

        Returns:
            Pure text summary from sub-agent's exploration.

        Raises:
            RuntimeError: If sub-agent exceeds max turns or LLM fails.
        """
        # 【核心优化】：子智能体极其容易偷懒。我们必须在 System Prompt 中严厉警告它必须使用工具！
        subagent_system_prompt = """你是一个专门负责深度探索的探路者 (Explorer Subagent)。
你的任务是根据主架构师的指令，在当前工作区内仔细阅读代码、查阅日志，搜集足够的信息。

【核心纪律】
1. 你必须、且只能依靠内置工具（如 bash 的 find/grep，或 read_file）去寻找答案。绝对不允许凭空捏造或猜测！
2. 如果你没有找到确切的答案，你必须继续使用工具深入搜索。
3. 当且仅当你找到了确切的线索后，停止调用工具，直接输出一段纯文本作为你的终极汇报。主架构师会根据你的汇报来做下一步决策。"""

        context_history: list[Message] = [
            Message(role=Role.SYSTEM, content=subagent_system_prompt),
            Message(role=Role.USER, content=task_prompt),
        ]

        # 限制子智能体最多只能跑 10 个 Turn，防止它自己卡死
        max_sub_turns = 10
        turn_count = 0

        while True:
            turn_count += 1
            if turn_count > max_sub_turns:
                raise RuntimeError(
                    f"子智能体探索过于深入，超过 {max_sub_turns} 轮被强制召回，"
                    "请主 Agent 给它更明确的指令"
                )

            # 【驾驭底线】：子智能体仅能获取传入的只读工具注册表
            available_tools = read_only_registry.get_available_tools()
            compacted_context = self.compactor.compact(context_history)

            # 子任务要求急速响应，强制关闭主体的慢思考，直接预测行动
            try:
                action_resp = self.provider.generate(compacted_context, available_tools)
            except Exception as e:
                raise RuntimeError(f"子智能体推理失败: {e}") from e

            context_history.append(action_resp)

            # 【核心退出条件】：子智能体一旦不调用工具了，说明它做好了总结汇报
            if not action_resp.tool_calls:
                # 直接将它的这段汇报内容剥离出来返回给上层
                return action_resp.content or ""

            # 执行只读工具的并发循环
            observation_msgs: list[Message | None] = [None] * len(action_resp.tool_calls)
            wg = WaitGroup()

            def sub_worker(idx: int, call: ToolCall) -> None:
                try:
                    # 【可视化的关键】：让终端用户看到 Subagent 正在干嘛
                    args_str = json.dumps(call.arguments)
                    if reporter is not None:
                        r = reporter
                        if hasattr(r, "on_tool_call"):
                            r.on_tool_call(f"[Subagent] {call.name}", args_str)

                    result = read_only_registry.execute(call)

                    final_output = result.output
                    if result.is_error:
                        final_output = self.recovery.analyze_and_inject(call.name, result.output)

                    if reporter is not None:
                        r = reporter
                        if hasattr(r, "on_tool_result"):
                            display = final_output
                            if len(display) > 200:
                                display = display[:200] + "... (已截断)"
                            r.on_tool_result(f"[Subagent] {call.name}", display, result.is_error)

                    observation_msgs[idx] = Message(
                        role=Role.USER,
                        content=final_output,
                        tool_call_id=call.id,
                    )

                finally:
                    wg.done()

            for i, tool_call in enumerate(action_resp.tool_calls):
                wg.add(1)
                thread = threading.Thread(target=sub_worker, args=(i, tool_call))
                thread.start()

            wg.wait()
            context_history.extend([msg for msg in observation_msgs if msg is not None])
