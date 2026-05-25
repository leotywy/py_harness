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

    # 6. Execute a task requiring sequential physical actions
    prompt = """
    请帮我执行以下操作：
    1. 用 bash 查看一下我当前电脑的 Python 版本。
    2. 帮我写一个简单的 helloworld.py 文件，输出 "Hello, py-tiny-claw!"。
    3. 用 bash 运行这个 Python 文件，确认它能正常工作。
    """

    logging.info("开始执行任务...")
    try:
        eng.run(prompt)
    except RuntimeError as e:
        logging.error(f"引擎运行崩溃: {e}")
        raise


if __name__ == "__main__":
    main()