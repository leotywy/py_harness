"""Read file tool implementation."""

import os
from pathlib import Path

from internal.schema import ToolDefinition
from internal.tools import Tool


class ReadFileTool(Tool):
    """Tool for reading local file contents.

    Restricted to work directory and its subdirectories.
    """

    # Max length to prevent OOM from large files
    MAX_LEN = 8000

    def __init__(self, work_dir: str) -> None:
        """Initialize with work directory boundary.

        Args:
            work_dir: Working directory to restrict file access.
        """
        self.work_dir = Path(work_dir).resolve()

    def name(self) -> str:
        """Return tool's globally unique name."""
        return "read_file"

    def definition(self) -> ToolDefinition:
        """Return tool's metadata and JSON Schema for the LLM."""
        return ToolDefinition(
            name=self.name(),
            description="读取指定路径的文件内容。请提供相对工作区的路径。",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要读取的文件路径，如 cmd/claw/main.py",
                    },
                },
                "required": ["path"],
            },
        )

    def execute(self, args: dict) -> str:
        """Execute file reading with arguments from the LLM.

        Args:
            args: Tool arguments containing 'path'.

        Returns:
            File content as string.

        Raises:
            ValueError: If path argument is missing.
            FileNotFoundError: If file does not exist.
            PermissionError: If path traversal detected or access denied.
        """
        # 1. Parse arguments
        path = args.get("path")
        if not path:
            raise ValueError("参数解析失败: 缺少 'path' 参数")

        # 2. Build absolute path with security check
        full_path = (self.work_dir / path).resolve()

        # Security: prevent path traversal (../../etc/passwd)
        if not str(full_path).startswith(str(self.work_dir)):
            raise PermissionError(f"路径穿越检测: '{path}' 超出工作目录边界")

        # 3. Read file content
        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        if not full_path.is_file():
            raise ValueError(f"路径不是文件: {path}")

        try:
            content = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with different encoding for binary-ish files
            content = full_path.read_text(encoding="latin-1")

        # 4. Length truncation protection
        # Prevent OOM from reading large files (logs, etc.)
        if len(content) > self.MAX_LEN:
            truncated_msg = (
                content[:self.MAX_LEN]
                + f"\n\n...[由于内容过长，已被系统截断至前 {self.MAX_LEN} 字节]..."
            )
            return truncated_msg

        return content