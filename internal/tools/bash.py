"""Bash tool implementation."""

import subprocess
import os
from pathlib import Path

from internal.schema import ToolDefinition
from internal.tools import Tool


class BashTool(Tool):
    """Tool for executing bash commands in work directory.

    Supports chained commands (&&), pipes, and complex shell syntax.
    """

    # Timeout in seconds to prevent hanging processes
    TIMEOUT = 30

    # Max output length to prevent OOM
    MAX_LEN = 8000

    def __init__(self, work_dir: str) -> None:
        """Initialize with work directory boundary.

        Args:
            work_dir: Working directory for command execution.
        """
        self.work_dir = Path(work_dir).resolve()

    def name(self) -> str:
        """Return tool's globally unique name."""
        return "bash"

    def definition(self) -> ToolDefinition:
        """Return tool's metadata and JSON Schema for the LLM."""
        return ToolDefinition(
            name=self.name(),
            description="在当前工作区执行任意的 bash 命令。支持链式命令(如 &&)。返回标准输出(stdout)和标准错误(stderr)。",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 bash 命令，例如: ls -la 或 pytest ./...",
                    },
                },
                "required": ["command"],
            },
        )

    def execute(self, args: dict) -> str:
        """Execute bash command with arguments from the LLM.

        Args:
            args: Tool arguments containing 'command'.

        Returns:
            Command output (stdout + stderr combined).

        Note:
            Errors are NOT raised as exceptions. Instead, they are
            returned as strings so the model can self-correct.
        """
        # 1. Parse arguments
        command = args.get("command")
        if not command:
            return "参数解析失败: 缺少 'command' 参数"

        # 2. Execute command with timeout and work directory
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT,
            )

            # Combine stdout and stderr
            output = result.stdout + result.stderr

        except subprocess.TimeoutExpired:
            return f"命令执行超时({self.TIMEOUT}s)，已被系统强制终止。如果是启动常驻服务，请尝试将其转入后台。"

        except Exception as e:
            return f"执行异常: {e}"

        # 3. Handle command error (Self-Correction mechanism)
        # When bash fails, we DON'T raise exception!
        # We return error + output so model can analyze and self-correct.
        if result.returncode != 0:
            return f"执行报错 (exit code: {result.returncode})\n输出:\n{output}"

        # 4. Handle empty output
        if not output.strip():
            return "命令执行成功，无终端输出。"

        # 5. Length truncation protection (prevent OOM)
        if len(output) > self.MAX_LEN:
            return output[:self.MAX_LEN] + f"\n\n...[终端输出过长，已截断至前 {self.MAX_LEN} 字节]..."

        return output