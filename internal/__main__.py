#!/usr/bin/env python3
"""Main entry point for py-tiny-claw engine with tracing."""

import logging
import os

from internal.engine import AgentEngine, GlobalSessionMgr, TerminalReporter
from internal.provider import OpenAIProvider, ProviderError
from internal.schema import Message, Role
from internal.tools import BashTool, WriteFileTool, new_registry

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    """Main entry point - tracing test with parallel tool execution."""
    # 确保环境变量已设置
    #if not os.getenv("ZHIPU_API_KEY"):
    #    logging.error("请先设置 ZHIPU_API_KEY 环境变量")
    #    return

    # 1. Get work directory
    work_dir = os.path.join(os.getcwd(), "workspace")
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # 2. 初始化 LLM Provider (保持不变，不包裹 CostTracker)
    model_name = "glm-5"
    try:
        llm_provider = OpenAIProvider.new_dashscope_provider(model_name)
        logging.info(f"Provider initialized: {llm_provider.model}")
    except ProviderError as e:
        logging.error(f"Provider 初始化失败: {e}")
        return

    # 3. 准备 Registry
    registry = new_registry()
    registry.register(BashTool(work_dir))
    registry.register(WriteFileTool(work_dir))

    # 4. 初始化引擎
    eng = AgentEngine(llm_provider, registry, enable_thinking=False, plan_mode=False)

    # 5. Terminal reporter
    reporter = TerminalReporter()

    # 6. 创建 Session
    session_id = "test_trace_001"
    session = GlobalSessionMgr.get_or_create(session_id, work_dir)

    # 7. 触发一个跨工具类型的并发任务
    prompt = """
    为了加快执行速度，请你在一轮回复中，【同时并行】完成以下两件事：
    1. 使用 bash 工具执行 'sleep 2 && echo "系统环境检查完毕"'
    2. 使用 write_file 工具，在当前目录下创建一个 'trace_test.md'，内容写上 "测试并发的写入"。
    请确保你是分别调用两个不同的工具，不要试图把它们合并成一个命令！
    """

    logging.info("\n>>> 🚀 启动带 Tracing 链路追踪的测试...")

    # 8. Push user prompt to Session
    session.append(Message(role=Role.USER, content=prompt))

    # 9. Run engine
    try:
        eng.run(session, reporter=reporter)
    except RuntimeError as e:
        logging.error(f"引擎运行崩溃: {e}")
        raise


if __name__ == "__main__":
    main()