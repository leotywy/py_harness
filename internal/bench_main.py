#!/usr/bin/env python3
"""Benchmark entry point for automated agent evaluation."""

import logging
import os

from internal.eval import BenchmarkRunner, TestCase

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    """Run benchmark test suite for agent evaluation."""
    #if not os.getenv("ZHIPU_API_KEY"):
    #   logging.error("请先导出 ZHIPU_API_KEY 环境变量进行跑分测试")
    #    return

    # 构建一套微型评测集
    testcases: list[TestCase] = [
        TestCase(
            id="test_001_edit",
            name="测试模糊替换工具的准确性",
            # 准备靶机：生成一个有错误的 json 文件
            setup_script='echo \'{"name": "tiny-claw", "version": "v1.0.0"}\' > config.json',
            # 考题：要求修改版本号
            task_prompt=(
                "当前目录下有一个 config.json。"
                "请你使用 edit_file 工具，将其中的 version 从 v1.0.0 改为 v2.0.0。"
                "不要做其他多余操作。"
            ),
            # 判卷脚本：使用 grep 检查文件是否包含 v2.0.0
            validate_script='grep \'"version": "v2.0.0"\' config.json',
        ),
        TestCase(
            id="test_002_code_gen",
            name="测试代码阅读与创建新文件的综合能力",
            # 准备靶机：生成一个简单的 Python 函数 (用 calc.py 避免 shadowing 内置 math)
            setup_script="""echo 'def multiply(a, b):
    return a * b' > calc.py""",
            # 考题：要求 Agent 根据刚才的代码，自己去写一份单元测试
            task_prompt=(
                "当前目录下有一个 calc.py。"
                "请你仔细阅读它，然后在同级目录下，"
                "帮我写一个规范的单元测试文件 test_calc.py，用来测试 multiply 函数。"
                "请务必包含正常的测试用例。"
            ),
            # 判卷脚本：直接运行 pytest！如果不通过则直接 0 分。
            validate_script="python3 -m pytest test_calc.py -v",
        ),
    ]

    # 启动跑分执行器！
    # 我们选用国内极其廉价但能力不错的 glm-5 跑分，省点钱。
    runner = BenchmarkRunner("glm-5")
    report = runner.run_suite(testcases)

    # 打印详细报告
    print(report.to_string())


if __name__ == "__main__":
    main()