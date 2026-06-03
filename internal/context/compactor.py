"""Context compactor for preventing LLM OOM."""

import logging

from internal.schema import Message, Role

logger = logging.getLogger(__name__)


class Compactor:
    """Monitors and compresses context memory to prevent LLM OOM.

    Implements dual degradation defense:
    - Full Masking for distant history
    - Head-Tail Truncation for short-term memory
    """

    def __init__(
        self,
        max_chars: int = 8000,
        retain_last_msgs: int = 6,
    ) -> None:
        """Initialize compactor.

        Args:
            max_chars: Max character threshold to trigger compression.
            retain_last_msgs: Working Memory protection zone (recent N messages).
        """
        self.max_chars = max_chars
        self.retain_last_msgs = retain_last_msgs

    def estimate_length(self, msgs: list[Message]) -> int:
        """Roughly calculate total context length.

        Args:
            msgs: List of messages.

        Returns:
            Total character count.
        """
        length = 0
        for msg in msgs:
            length += len(msg.content)
            for tc in msg.tool_calls:
                length += len(tc.name)
                # Arguments is dict, convert to str for estimation
                import json
                length += len(json.dumps(tc.arguments))
        return length

    def compact(self, msgs: list[Message]) -> list[Message]:
        """Compress messages if total length exceeds threshold.

        Args:
            msgs: Messages to potentially compress.

        Returns:
            Compressed or original messages.
        """
        current_length = self.estimate_length(msgs)

        # If under threshold, return original (normal path)
        if current_length < self.max_chars:
            return msgs

        logger.warning(
            f"[Compactor] ⚠️ 内存告警：当前上下文长度 ({current_length} 字符) "
            f"超过阈值 ({self.max_chars})，触发压缩清理..."
        )

        compacted: list[Message] = []
        msg_count = len(msgs)

        # Calculate Working Memory protection start index
        protect_start_index = max(0, msg_count - self.retain_last_msgs)

        for i, msg in enumerate(msgs):
            # 1. System Prompt is sacred - never modify
            if msg.role == Role.SYSTEM:
                compacted.append(msg)
                continue

            # Create a copy to avoid modifying original in concurrent environment
            new_msg = Message(
                role=msg.role,
                content=msg.content,
                tool_calls=msg.tool_calls.copy() if msg.tool_calls else [],
                tool_call_id=msg.tool_call_id,
            )

            is_in_working_memory = i >= protect_start_index

            # 【核心驾驭逻辑】: Dual degradation defense
            # Check if this is a Tool Result (User message with ToolCallID)
            if msg.role == Role.USER and msg.tool_call_id:
                # Tool Result (Observation)
                if not is_in_working_memory:
                    # 【第一道防线：远期历史】Full Masking - ruthless replacement
                    if len(msg.content) > 200:
                        new_msg.content = (
                            f"...[为了节省内存，早期的工具输出已被系统强制清理。"
                            f"原始长度: {len(msg.content)} 字节]..."
                        )
                else:
                    # 【第二道防线：短期记忆】Head-Tail Truncation
                    # Keep first 500 + last 500 chars (error message + summary)
                    max_keep = 1000
                    if len(msg.content) > max_keep:
                        head = msg.content[:500]
                        tail = msg.content[-500:]
                        truncated_middle = len(msg.content) - max_keep
                        new_msg.content = (
                            f"{head}\n\n"
                            f"...[内容过长，中间 {truncated_middle} 字节已被系统截断]...\n\n"
                            f"{tail}"
                        )

            elif msg.role == Role.ASSISTANT and msg.content:
                # Assistant's verbose thinking trace
                if not is_in_working_memory and len(msg.content) > 200:
                    new_msg.content = "...[早期的推理思考过程已折叠]..."

            # IMPORTANT: We NEVER modify ToolCalls!
            # ToolCalls are evidence of model actions, critical for logic chain!

            compacted.append(new_msg)

        new_length = self.estimate_length(compacted)
        logger.info(
            f"[Compactor] ✅ 压缩完成。上下文长度从 {current_length} 降至 {new_length} 字符。"
        )

        return compacted