"""Edit file tool implementation with fuzzy matching."""

import os
from pathlib import Path

from internal.schema import ToolDefinition
from internal.tools import Tool


def _line_by_line_replace(content: str, old_text: str, new_text: str) -> tuple[str, str | None]:
    """L4: Line-by-line matching with indentation ignored.

    Uses sliding window to find matching blocks.
    Returns (new_content, error_message).
    """
    content_lines = content.split("\n")
    old_lines = old_text.strip().split("\n")

    if len(old_lines) == 0 or len(content_lines) < len(old_lines):
        return "", "找不到该代码片段"

    # Clean old_lines: strip each line
    stripped_old_lines = [line.strip() for line in old_lines]

    match_count = 0
    match_start_index = -1
    match_end_index = -1

    # Sliding window to find matching block
    for i in range(len(content_lines) - len(old_lines) + 1):
        is_match = True
        for j in range(len(old_lines)):
            if content_lines[i + j].strip() != stripped_old_lines[j]:
                is_match = False
                break

        if is_match:
            match_count += 1
            match_start_index = i
            match_end_index = i + len(old_lines)

    if match_count == 0:
        return "", "在文件中未找到 old_text，请先调用 read_file 确认文件内容和缩进"
    if match_count > 1:
        return "", f"模糊匹配到了 {match_count} 处相似代码，请提供更多上下行代码以精确定位"

    # Build new content
    new_content_lines = content_lines[:match_start_index] + [new_text] + content_lines[match_end_index:]
    return "\n".join(new_content_lines), None


def _fuzzy_replace(original_content: str, old_text: str, new_text: str) -> tuple[str, str | None]:
    """Four-level fuzzy replacement algorithm.

    L1: Exact match
    L2: Newline normalization (\r\n → \n)
    L3: TrimSpace match (ignore leading/trailing whitespace)
    L4: Line-by-line match (ignore all indentation)

    Returns (new_content, error_message).
    """
    # L1: Exact match
    count = original_content.count(old_text)
    if count == 1:
        return original_content.replace(old_text, new_text, 1), None
    if count > 1:
        return "", f"old_text 匹配到了 {count} 处，请提供更多的上下文代码以确保唯一性"

    # L2: Newline normalization
    normalized_content = original_content.replace("\r\n", "\n")
    normalized_old = old_text.replace("\r\n", "\n")

    count = normalized_content.count(normalized_old)
    if count == 1:
        return normalized_content.replace(normalized_old, new_text, 1), None

    # L3: TrimSpace match
    trimmed_old = normalized_old.strip()
    if trimmed_old:
        count = normalized_content.count(trimmed_old)
        if count == 1:
            return normalized_content.replace(trimmed_old, new_text, 1), None

    # L4: Line-by-line match (strongest fuzzy matching)
    return _line_by_line_replace(normalized_content, normalized_old, new_text)


class EditFileTool(Tool):
    """Tool for making localized string replacements in existing files.

    Safer and faster than rewriting the entire file.
    Uses four-level fuzzy matching for robustness.
    """

    def __init__(self, work_dir: str) -> None:
        """Initialize with work directory boundary.

        Args:
            work_dir: Working directory to restrict file access.
        """
        self.work_dir = Path(work_dir).resolve()

    def name(self) -> str:
        """Return tool's globally unique name."""
        return "edit_file"

    def definition(self) -> ToolDefinition:
        """Return tool's metadata and JSON Schema for the LLM."""
        return ToolDefinition(
            name=self.name(),
            description="对现有文件进行局部的字符串替换。这比重写整个文件更安全、更快速。请提供足够的 old_text 上下文以确保匹配的唯一性。",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要修改的文件路径",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "文件中原有的文本。必须包含足够的上下文（建议上下各多包含几行），以确保在文件中的唯一性。",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "要替换成的新文本",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        )

    def execute(self, args: dict) -> str:
        """Execute file editing with arguments from the LLM.

        Args:
            args: Tool arguments containing 'path', 'old_text', 'new_text'.

        Returns:
            Success message string.

        Raises:
            ValueError: If arguments are missing or fuzzy match fails.
            FileNotFoundError: If file does not exist.
            PermissionError: If path traversal detected.
        """
        # 1. Parse arguments
        path = args.get("path")
        old_text = args.get("old_text")
        new_text = args.get("new_text")

        if not path:
            raise ValueError("参数解析失败: 缺少 'path' 参数")
        if old_text is None:
            raise ValueError("参数解析失败: 缺少 'old_text' 参数")
        if new_text is None:
            raise ValueError("参数解析失败: 缺少 'new_text' 参数")

        # 2. Build absolute path with security check
        full_path = (self.work_dir / path).resolve()

        # Security: prevent path traversal
        if not str(full_path).startswith(str(self.work_dir)):
            raise PermissionError(f"路径穿越检测: '{path}' 超出工作目录边界")

        # 3. Read original file content
        try:
            original_content = full_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"读取文件失败，请确认路径是否正确: {path}")

        # 4. Apply multi-level fuzzy replacement
        new_content, error = _fuzzy_replace(original_content, old_text, new_text)

        if error:
            # 【驾驭哲学】Return specific error to let model self-correct
            raise ValueError(error)

        # 5. Safely write back to disk
        full_path.write_text(new_content, encoding="utf-8")

        return f"✅ 成功修改文件: {path}"