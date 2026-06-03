"""Skill loader for parsing SKILL.md files from .claw/skills directory."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    """Standardized skill structure parsed from SKILL.md."""

    name: str
    description: str
    body: str  # Markdown body content


def parse_skill_md(content: str) -> Skill:
    """Parse SKILL.md content with YAML Frontmatter.

    Args:
        content: Raw file content.

    Returns:
        Parsed Skill object.
    """
    skill = Skill(
        name="Unknown Skill",
        description="No description provided.",
        body=content,  # Default: use full content as body
    )

    # Simple YAML Frontmatter parsing (wrapped by ---)
    if content.startswith("---\n") or content.startswith("---\r\n"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            skill.body = parts[2].strip()

            # Extract metadata line by line
            for line in frontmatter.split("\n"):
                line = line.strip()
                if line.startswith("name:"):
                    skill.name = line[len("name:"):].strip()
                elif line.startswith("description:"):
                    skill.description = line[len("description:"):].strip()

    return skill


class SkillLoader:
    """Loads and parses skill templates from local filesystem."""

    def __init__(self, work_dir: str) -> None:
        """Initialize with work directory.

        Args:
            work_dir: Working directory containing .claw/skills.
        """
        self.work_dir = Path(work_dir).resolve()

    def load_all(self) -> str:
        """Scan .claw/skills directory, parse all SKILL.md files.

        Returns:
            Formatted string ready for Context injection.
            Returns empty string if no skills configured.
        """
        skill_base_dir = self.work_dir / ".claw" / "skills"

        # If directory doesn't exist, silently return
        if not skill_base_dir.exists():
            return ""

        # Build skills string
        lines = [
            "\n### 可用专业技能 (Agent Skills)",
            "以下是你拥有的标准化外挂技能，请在符合 description 描述的场景下严格遵循其正文指令：\n\n",
        ]

        # Walk directory to find SKILL.md files
        skill_count = 0
        for root, _, files in os.walk(skill_base_dir):
            for filename in files:
                if filename == "SKILL.md":
                    filepath = Path(root) / filename
                    try:
                        content = filepath.read_text(encoding="utf-8")
                        skill = parse_skill_md(content)

                        # Inject parsed skill
                        lines.append(f"#### 技能名称: {skill.name}")
                        lines.append(f"**触发条件**: {skill.description}\n")
                        lines.append("**执行指南**:")
                        lines.append(skill.body)
                        lines.append("\n\n---\n")

                        skill_count += 1
                    except Exception:
                        # Silently skip files that can't be read
                        pass

        # Return empty if no skills found or content too short
        if skill_count == 0:
            return ""

        result = "\n".join(lines)
        if len(result) < 100:
            return ""

        return result