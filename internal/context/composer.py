"""Prompt composer for dynamic System Prompt generation."""

import os
from pathlib import Path

from internal.context.skill import SkillLoader
from internal.schema import Message, Role


class PromptComposer:
    """Composes dynamic System Prompt based on workspace environment."""

    # Minimal core prompt (establishes identity and critical rules)
    CORE_PROMPT = """# 核心身份
你名叫 py-tiny-claw，一个由驾驭工程驱动的骨灰级研发助手。
你具备极简主义哲学，拒绝废话。你能通过系统提供的内置工具，创建、读取、修改和执行工作区中的代码。

# 核心纪律 (CRITICAL)
1. 如需检查文件是否存在，请使用 bash 的 ls 或 test -f，而不是对目录使用 read_file。
2. 创建新文件时，务必使用 write_file，并同时提供 path 和 content 参数。
3. 编辑文件前务必先读取现有文件，以理解上下文。
4. 无论何时你需要写代码或创建文件，都要直接使用 write_file 工具。
5. 遇到工具执行报错时，仔细阅读 stderr，尝试自己修正命令并重试。
6. 始终用中文回复，以便传达你的进展和想法。
"""

    def __init__(self, work_dir: str) -> None:
        """Initialize with work directory.

        Args:
            work_dir: Working directory for loading AGENTS.md and skills.
        """
        self.work_dir = Path(work_dir).resolve()
        self.skill_loader = SkillLoader(work_dir)

    def build(self) -> Message:
        """Build complete System Prompt message.

        Returns:
            RoleSystem message with composed prompt content.
        """
        prompt_parts = [self.CORE_PROMPT]

        # 2. Externalize state: load project-specific rules (AGENTS.md)
        agents_md_path = self.work_dir / "AGENTS.md"
        if agents_md_path.exists():
            try:
                content = agents_md_path.read_text(encoding="utf-8")
                prompt_parts.append("\n# 项目专属指南 (来自 AGENTS.md)")
                prompt_parts.append("以下是当前工作区特有的架构规范与注意事项，你的行为必须绝对符合以下要求：")
                prompt_parts.append("```markdown")
                prompt_parts.append(content)
                prompt_parts.append("\n```")
            except Exception:
                # Silently skip if file can't be read
                pass

        # 3. Dynamically load skills
        skills_content = self.skill_loader.load_all()
        if skills_content:
            prompt_parts.append(skills_content)

        # Join all parts
        full_prompt = "\n".join(prompt_parts)

        return Message(
            role=Role.SYSTEM,
            content=full_prompt,
        )