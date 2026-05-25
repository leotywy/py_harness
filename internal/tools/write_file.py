"""Write file tool implementation."""

import os
from pathlib import Path

from internal.schema import ToolDefinition
from internal.tools import Tool


class WriteFileTool(Tool):
    """Tool for creating or overwriting files.

    Restricted to work directory and its subdirectories.
    Auto-creates parent directories if needed.
    """

    def __init__(self, work_dir: str) -> None:
        """Initialize with work directory boundary.

        Args:
            work_dir: Working directory to restrict file access.
        """
        self.work_dir = Path(work_dir).resolve()

    def name(self) -> str:
        """Return tool's globally unique name."""
        return "write_file"

    def definition(self) -> ToolDefinition:
        """Return tool's metadata and JSON Schema for the LLM."""
        return ToolDefinition(
            name=self.name(),
            description="创建或覆盖写入一个文件。如果目录不存在会自动创建。请提供相对于工作区的相对路径。",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要写入的文件路径，如 src/main.py",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的完整文件内容",
                    },
                },
                "required": ["path", "content"],
            },
        )

    def execute(self, args: dict) -> str:
        """Execute file writing with arguments from the LLM.

        Args:
            args: Tool arguments containing 'path' and 'content'.

        Returns:
            Success message string.

        Raises:
            ValueError: If path or content argument is missing.
            PermissionError: If path traversal detected.
        """
        # 1. Parse arguments
        path = args.get("path")
        content = args.get("content")

        if not path:
            raise ValueError("参数解析失败: 缺少 'path' 参数")
        if content is None:
            raise ValueError("参数解析失败: 缺少 'content' 参数")

        # 2. Build absolute path with security check
        full_path = (self.work_dir / path).resolve()

        # Security: prevent path traversal (../../etc/passwd)
        if not str(full_path).startswith(str(self.work_dir)):
            raise PermissionError(f"路径穿越检测: '{path}' 超出工作目录边界")

        # 3. Auto-create parent directories
        parent_dir = full_path.parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, mode=0o755)

        # 4. Write file content
        full_path.write_text(content, encoding="utf-8")

        return f"成功将内容写入到文件: {path}"