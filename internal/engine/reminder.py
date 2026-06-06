"""Reminder injector for detecting and breaking agent loops.

Monitors tool call failures and injects corrective messages when the agent
gets stuck in repetitive failed attempts.
"""

import hashlib
import json
import logging

from internal.schema import Message, Role, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ReminderInjector:
    """Monitors context and injects reminders when model gets obsessed.

    Tracks consecutive failures of tool calls with the same fingerprint
    (tool name + arguments hash) and injects a strong intervention message
    when the agent is stuck in a loop.
    """

    def __init__(self) -> None:
        """Initialize the reminder injector."""
        # Maps fingerprint -> consecutive failure count
        self._consecutive_failures: dict[str, int] = {}

    def _generate_fingerprint(self, tool_name: str, arguments: dict) -> str:
        """Generate a unique fingerprint for a tool call.

        Used to determine if the model is repeating the same action.

        Args:
            tool_name: Name of the tool being called.
            arguments: Tool arguments dictionary.

        Returns:
            MD5 hash fingerprint of the tool call.
        """
        hasher = hashlib.md5()
        hasher.update(tool_name.encode())
        hasher.update(json.dumps(arguments, sort_keys=True).encode())
        return hasher.hexdigest()

    def check_and_inject(
        self, last_tool_call: ToolCall, last_result: ToolResult
    ) -> Message | None:
        """Analyze execution result and decide whether to inject a reminder.

        If the tool executed successfully, clears all failure counters.
        If it failed, increments the counter for that fingerprint and
        injects a corrective message after 3 consecutive failures.

        Args:
            last_tool_call: The most recent tool call made.
            last_result: The result of that tool call.

        Returns:
            A Message to inject if intervention is needed, None otherwise.
        """
        fingerprint = self._generate_fingerprint(
            last_tool_call.name, last_tool_call.arguments
        )

        # Success on this path - clear all failure counters
        if not last_result.is_error:
            self._consecutive_failures.clear()
            return None

        # Failure - increment counter for this fingerprint
        self._consecutive_failures[fingerprint] = (
            self._consecutive_failures.get(fingerprint, 0) + 1
        )
        fail_count = self._consecutive_failures[fingerprint]

        logger.info(
            f"[Reminder] Tool {last_tool_call.name} failed, "
            f"consecutive failures for this pattern: {fail_count}"
        )

        # Intervention threshold: 3 consecutive identical failures
        if fail_count >= 3:
            logger.warning(
                "[Reminder] Loop detected! Injecting corrective instruction."
            )

            nudge_content = f"""[SYSTEM REMINDER 警告]
你似乎陷入了死循环。你刚刚连续 {fail_count} 次使用相同的参数调用了 '{last_tool_call.name}' 工具，并且都失败了。
请立即停止这种无效的重试！你的注意力被当前的报错过度吸引了。
你需要：
1. 停止猜测参数。跳出当前的局部思维。
2. 彻底改变你的策略。
3. 如果你确实无法通过系统工具解决当前问题，请直接结束任务并向用户说明你需要什么人工帮助，而不是继续盲目消耗 API 资源尝试。"""

            return Message(
                role=Role.USER,  # Must be USER for highest recency weight
                content=nudge_content,
            )

        return None