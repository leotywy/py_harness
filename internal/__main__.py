#!/usr/bin/env python3
"""Main entry point for py-tiny-claw engine."""

import logging
import os

from internal.engine import AgentEngine, TerminalReporter
from internal.provider import OpenAIProvider, ProviderError
from internal.tools import BashTool, EditFileTool, ReadFileTool, WriteFileTool, new_registry

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    """Main entry point."""
    logging.info("🚀 欢迎来到 py-tiny-claw 引擎启动序列")

    # Ensure DASHSCOPE_API_KEY is set
    if not os.getenv("DASHSCOPE_API_KEY"):
        logging.error("请先设置 DASHSCOPE_API_KEY 环境变量")
        return

    # 1. Get work directory with workspace subdirectory
    work_dir = os.path.join(os.getcwd(), "workspace")

    # Create workspace directory if not exists
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # 2. Initialize Provider (DashScope with glm-5)
    try:
        llm_provider = OpenAIProvider.new_dashscope_provider("glm-5")
        logging.info(f"Provider initialized: {llm_provider.model}")
    except ProviderError as e:
        logging.error(f"Provider 初始化失败: {e}")
        return

    # 3. Initialize Tool Registry
    registry = new_registry()

    # 4. Mount full tool set
    registry.register(ReadFileTool(work_dir))
    registry.register(WriteFileTool(work_dir))
    registry.register(BashTool(work_dir))
    registry.register(EditFileTool(work_dir))

    # 5. Instantiate engine with thinking enabled
    eng = AgentEngine(llm_provider, registry, work_dir, enable_thinking=True)

    # 6. Inject TerminalReporter
    reporter = TerminalReporter()

    # 7. Execute task
    prompt = """
    我需要在当前目录下新建一个 ping.py，提供一个简单的 HTTP ping 接口。
    写完之后，帮我把代码用 git 提交一下。
    """

    logging.info("开始执行任务...")
    try:
        eng.run(prompt, reporter=reporter)
    except RuntimeError as e:
        logging.error(f"引擎运行崩溃: {e}")
        raise


if __name__ == "__main__":
    main()