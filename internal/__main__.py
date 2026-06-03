#!/usr/bin/env python3
"""Main entry point for py-tiny-claw engine with CLI support."""

import argparse
import logging
import os

from internal.engine import AgentEngine, GlobalSessionMgr, TerminalReporter
from internal.provider import OpenAIProvider, ProviderError
from internal.schema import Message, Role
from internal.tools import BashTool, EditFileTool, ReadFileTool, WriteFileTool, new_registry

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="py-tiny-claw Agent Engine")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="要交给 Agent 执行的任务描述",
    )
    args = parser.parse_args()

    # Ensure DASHSCOPE_API_KEY is set
    if not os.getenv("DASHSCOPE_API_KEY"):
        logging.error("请先设置 DASHSCOPE_API_KEY 环境变量")
        return

    # 1. Get work directory
    work_dir = os.path.join(os.getcwd(), "workspace")
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # 2. Initialize Provider
    try:
        llm_provider = OpenAIProvider.new_dashscope_provider("glm-5")
        logging.info(f"Provider initialized: {llm_provider.model}")
    except ProviderError as e:
        logging.error(f"Provider 初始化失败: {e}")
        return

    # 3. Mount 4 basic tools
    registry = new_registry()
    registry.register(ReadFileTool(work_dir))
    registry.register(WriteFileTool(work_dir))
    registry.register(BashTool(work_dir))
    registry.register(EditFileTool(work_dir))

    # 4. Instantiate engine with PlanMode enabled
    eng = AgentEngine(llm_provider, registry, enable_thinking=False, plan_mode=True)

    # 5. Terminal reporter
    reporter = TerminalReporter()

    # 6. Use fixed SessionID to share memory-based working memory across runs
    # (In real CLI, if process restarts, Session's memory history is lost.
    # But that's the point we want to demonstrate: even if short-term memory is lost,
    # as long as TODO.md exists, the task can continue!)
    session_id = "task_web_server_01"
    session = GlobalSessionMgr.get_or_create(session_id, work_dir)

    logging.info(f"\n>>> 🚀 收到指令: {args.prompt}")

    # 7. Push user prompt to Session
    session.append(Message(role=Role.USER, content=args.prompt))

    # 8. Run engine
    try:
        eng.run(session, reporter=reporter)
    except RuntimeError as e:
        logging.error(f"引擎运行崩溃: {e}")
        raise


if __name__ == "__main__":
    main()