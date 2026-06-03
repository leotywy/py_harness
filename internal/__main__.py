#!/usr/bin/env python3
"""Main entry point for py-tiny-claw engine with full tool set."""

import logging
import os

from internal.engine import AgentEngine
from internal.provider import OpenAIProvider, ProviderError
from internal.tools import BashTool, ReadFileTool, WriteFileTool, new_registry

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    """Main entry point."""
    logging.info("🚀 欢迎来到 py-tiny-claw 引擎启动序列")

    # Ensure DASHSCOPE_API_KEY is set
    if not os.getenv("DASHSCOPE_API_KEY"):
        logging.error("请先设置 DASHSCOPE_API_KEY 环境变量")
        return

    # 1. Get work directory physical boundary
    work_dir = os.getcwd()

    # 2. Initialize real Provider (DashScope with glm-5)
    try:
        llm_provider = OpenAIProvider.new_dashscope_provider("glm-5")
        logging.info(f"Provider initialized: {llm_provider.model}")
    except ProviderError as e:
        logging.error(f"Provider 初始化失败: {e}")
        return

    # 3. Initialize Tool Registry
    registry = new_registry()

    # 4. Mount minimal tool set
    registry.register(ReadFileTool(work_dir))
    registry.register(WriteFileTool(work_dir))
    registry.register(BashTool(work_dir))

    # 5. Instantiate engine with thinking disabled (YOLO fast mode)
    eng = AgentEngine(llm_provider, registry, work_dir, enable_thinking=False)

    # 6. Create test files for parallel read test
    test_files = {
        "a.txt": "人工智能（AI）是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。\n主要研究领域包括机器学习、自然语言处理、计算机视觉等。\n深度学习是近年来最热门的技术方向。",
        "b.txt": "量子力学是现代物理学的基础理论，描述了微观粒子的运动规律。\n薛定谔方程是量子力学的核心方程。\n量子计算是量子力学在计算机科学中的应用。",
        "c.txt": "生物学是研究生命现象和生命活动规律的科学。\n分子生物学研究生命现象的分子基础。\n基因工程是现代生物技术的重要组成部分。",
    }

    for filename, content in test_files.items():
        filepath = os.path.join(work_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    # 7. Execute a task requiring multi-source information collection (parallel test)
    prompt = """
    我当前目录下有 a.txt, b.txt, c.txt 三个文件。
    为了节省时间，请你同时一次性读取这三个文件，并将它们的内容综合起来，
    告诉我它们分别记录了什么领域的信息。
    """

    logging.info("开始执行任务...")
    try:
        eng.run(prompt)
    except RuntimeError as e:
        logging.error(f"引擎运行崩溃: {e}")
        raise
    finally:
        # Cleanup test files
        for filename in test_files.keys():
            filepath = os.path.join(work_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == "__main__":
    main()