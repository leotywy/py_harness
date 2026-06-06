"""Session management for multi-user/multi-terminal isolation."""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from internal.schema import Message, Role


@dataclass
class Session:
    """Represents a continuous human-machine interaction session.

    Maintains complete history for this session.
    """

    id: str
    work_dir: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history: list[Message] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # 【新增】用于统计该 Session 累计消耗的资源
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_cny: float = 0.0

    def append(self, *msgs: Message) -> None:
        """Thread-safe append messages to session history.

        Args:
            msgs: Messages to append.
        """
        with self._lock:
            self.history.extend(msgs)
            self.updated_at = time.time()

            # 【持久化预留点】In production, we would save history to disk:
            # work_dir/.claw/sessions/{id}.jsonl

    def record_usage(self, prompt_tokens: int, completion_tokens: int, cost: float) -> None:
        """Record usage for billing accumulation.

        给外部 Tracker 调用的辅助方法，用于累加账单。

        Args:
            prompt_tokens: Number of input tokens consumed.
            completion_tokens: Number of output tokens generated.
            cost: Cost in CNY for this API call.
        """
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cost_cny += cost

    def get_working_memory(self, limit: int = 0) -> list[Message]:
        """Get recent messages as working memory.

        This is the core of steering engineering!
        Returns the most recent N messages, not full history.

        Args:
            limit: Maximum number of messages to return.
                   0 or negative means return all.

        Returns:
            List of recent messages (deep copy).
        """
        with self._lock:
            total = len(self.history)

            if total <= limit or limit <= 0:
                # Return full history (deep copy to prevent external modification)
                return [msg for msg in self.history]

            # Take the most recent limit messages
            result = self.history[total - limit : total]

            # 【驾驭防线】: Ensure message continuity!
            # If the first message in our slice is an "orphan" ToolResult
            # (RoleUser with ToolCallID), but the ToolCall that requested it
            # was truncated away, the LLM API will return 400 Bad Request.
            # So we must skip orphan tool responses.
            while result:
                first_msg = result[0]
                # Check if first message is orphan tool result
                if first_msg.role == Role.USER and first_msg.tool_call_id:
                    # Skip this orphan tool result
                    result = result[1:]
                else:
                    break

            return list(result)


class SessionManager:
    """Global session manager for multi-user/multi-terminal isolation."""

    def __init__(self) -> None:
        """Initialize session manager."""
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def get_or_create(self, id: str, work_dir: str) -> Session:
        """Get existing session or create new one.

        Args:
            id: Session ID.
            work_dir: Working directory for the session.

        Returns:
            Session instance.
        """
        with self._lock:
            if id in self._sessions:
                return self._sessions[id]

            session = Session(id=id, work_dir=work_dir)
            self._sessions[id] = session
            return session

    def get(self, id: str) -> Optional[Session]:
        """Get session by ID if exists.

        Args:
            id: Session ID.

        Returns:
            Session instance or None.
        """
        with self._lock:
            return self._sessions.get(id)

    def delete(self, id: str) -> bool:
        """Delete session by ID.

        Args:
            id: Session ID.

        Returns:
            True if session was deleted, False if not found.
        """
        with self._lock:
            if id in self._sessions:
                del self._sessions[id]
                return True
            return False

    def list_sessions(self) -> list[str]:
        """List all session IDs.

        Returns:
            List of session IDs.
        """
        with self._lock:
            return list(self._sessions.keys())


# Global session manager instance
GlobalSessionMgr = SessionManager()