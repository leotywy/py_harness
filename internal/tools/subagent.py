"""Subagent tool for spawning child agents for deep exploration tasks.

Breaks circular dependency between tools and engine packages via AgentRunner protocol.
"""

import logging
from typing import Protocol, runtime_checkable

from internal.schema import ToolDefinition
from internal.tools import Registry, Tool

logger = logging.getLogger(__name__)


@runtime_checkable
class AgentRunner(Protocol):
    """Abstract interface to break circular dependency.

    Since SubagentTool lives in tools package, while AgentEngine lives in engine package,
    we define this interface for external injection to allow Tool to invoke Engine.
    """

    def run_sub(
        self,
        task_prompt: str,
        read_only_registry: Registry,
        reporter: object,
    ) -> str:
        """Spawn an anonymous, one-time sub-agent task.

        Args:
            task_prompt: Clear instruction for the sub-agent.
            read_only_registry: Restricted read-only registry for sub-agent.
            reporter: Reporter for output notifications.

        Returns:
            Pure text summary from sub-agent's exploration.
        """
        ...


class SubagentTool(Tool):
    """Tool for spawning sub-agents for deep exploration tasks.

    When you need to read large amounts of code or cross-file logic search,
    call this tool. It returns an extremely refined summary report.
    """

    def __init__(
        self,
        runner: AgentRunner,
        read_only_registry: Registry,
        reporter: object,
    ) -> None:
        """Initialize subagent tool.

        Args:
            runner: AgentRunner interface for spawning sub-agent.
            read_only_registry: Restricted read-only registry for sub-agent.
            reporter: Reporter for output notifications (use object to avoid circular import).
        """
        self._runner = runner
        self._read_only_registry = read_only_registry
        self._reporter = reporter

    def name(self) -> str:
        """Return tool's globally unique name."""
        return "spawn_subagent"

    def definition(self) -> ToolDefinition:
        """Return tool's metadata and JSON Schema for the LLM."""
        return ToolDefinition(
            name=self.name(),
            description=(
                "派出一个专门用于深度探索（Exploration）的子智能体。"
                "当你需要阅读大量代码、跨文件查找逻辑时请调用此工具。"
                "它在探索完毕后，会给你返回一份极度精炼的摘要报告。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "task_prompt": {
                        "type": "string",
                        "description": "给子智能体下达的明确指令。",
                    },
                },
                "required": ["task_prompt"],
            },
        )

    def execute(self, args: dict) -> str:
        """Execute sub-agent spawning with arguments from the LLM.

        Args:
            args: Tool arguments containing 'task_prompt'.

        Returns:
            Summary report from sub-agent's exploration.
        """
        # 1. Parse arguments
        task_prompt = args.get("task_prompt")
        if not task_prompt:
            return "参数解析失败: 缺少 'task_prompt' 参数"

        logger.info(f"[Subagent] 🚀 主 Agent 发起委派！正在拉起探路者: [{task_prompt[:50]}...]...")

        # 2. 【核心降维打击】：拉起一个完全物理隔离的子循环
        # 我们把针对该任务的专项指令传给子智能体，并仅提供 readOnlyRegistry。
        # (子智能体只能读文件或执行只读的 bash，不能搞破坏)
        try:
            summary = self._runner.run_sub(
                task_prompt=task_prompt,
                read_only_registry=self._read_only_registry,
                reporter=self._reporter,
            )
        except Exception as e:
            return f"子智能体执行失败: {e}"

        logger.info("[Subagent] ✅ 子智能体任务结束。报告返回给主干...")

        # 最终，几万字的代码探索，化作了这一段轻量级的 Summary，
        # 就像一次普通的 API 调用一样，返回给了始终保持清醒的主 Agent。
        return f"【子智能体探索报告】:\n{summary}"