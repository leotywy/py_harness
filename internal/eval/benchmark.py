"""Benchmark module for evaluating agent performance.

Provides test case definitions and execution framework for automated validation.
"""

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Defines an independent task for the agent to complete and validate.

    Attributes:
        id: Unique test case identifier.
        name: Test case name for display.
        setup_script: Optional bash script to run before agent execution
                      (used to initialize target code/files).
        task_prompt: Task instruction sent to the agent.
        validate_script: Core bash validation script run after agent execution.
                        exit 0 = success, non-zero = failure.
        max_turns: Maximum turns allowed for agent attempts (timeout = failure).
    """

    id: str
    name: str
    setup_script: str = ""
    task_prompt: str = ""
    validate_script: str = ""
    max_turns: int = 10


@dataclass
class TestResult:
    """Stores the result of a single benchmark run.

    Attributes:
        test_case_id: ID of the test case that was run.
        passed: Whether the validation passed (True/False).
        total_cost_usd: Total API cost in USD for this test.
        duration_ms: Total execution duration in milliseconds.
        error_msg: Error message if test failed.
        total_prompt_tokens: Total input tokens consumed.
        total_completion_tokens: Total output tokens generated.
        trace_file: Path to the trace file generated.
    """

    test_case_id: str
    passed: bool = False
    total_cost_usd: float = 0.0
    duration_ms: int = 0
    error_msg: str = ""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    trace_file: str = ""


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report for all test cases.

    Attributes:
        results: List of individual test results.
        total_passed: Number of passed tests.
        total_failed: Number of failed tests.
        total_cost_usd: Total cost across all tests.
        avg_duration_ms: Average duration per test.
    """

    results: list[TestResult] = field(default_factory=list)
    total_passed: int = 0
    total_failed: int = 0
    total_cost_usd: float = 0.0
    avg_duration_ms: float = 0.0

    def add_result(self, result: TestResult) -> None:
        """Add a test result to the report and update aggregates."""
        self.results.append(result)
        if result.passed:
            self.total_passed += 1
        else:
            self.total_failed += 1
        self.total_cost_usd += result.total_cost_usd

    def compute_summary(self) -> None:
        """Compute summary statistics from all results."""
        if self.results:
            total_duration = sum(r.duration_ms for r in self.results)
            self.avg_duration_ms = total_duration / len(self.results)

    def to_string(self) -> str:
        """Generate a formatted report string."""
        self.compute_summary()

        lines = [
            "\n================ Benchmark Report ================\n",
            f"Total Tests: {len(self.results)}",
            f"Passed: {self.total_passed}",
            f"Failed: {self.total_failed}",
            f"Pass Rate: {self.total_passed / len(self.results) * 100:.1f}%",
            f"Total Cost: ${self.total_cost_usd:.4f}",
            f"Avg Duration: {self.avg_duration_ms:.0f}ms",
            "",
            "Details:",
        ]

        for r in self.results:
            status = "✅ PASSED" if r.passed else "❌ FAILED"
            lines.append(
                f"  [{r.test_case_id}] {status} | "
                f"Cost: ${r.total_cost_usd:.4f} | "
                f"Duration: {r.duration_ms}ms"
            )
            if r.error_msg:
                lines.append(f"    Error: {r.error_msg}")

        lines.append("\n==================================================\n")

        return "\n".join(lines)


def run_setup_script(work_dir: str, script: str) -> tuple[bool, str]:
    """Run setup script in work directory.

    Args:
        work_dir: Working directory for script execution.
        script: Bash script content to execute.

    Returns:
        Tuple of (success, output/error message).
    """
    if not script:
        return True, ""

    try:
        result = subprocess.run(
            ["bash", "-c", script],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr or result.stdout
    except subprocess.TimeoutExpired:
        return False, "Setup script timed out after 30s"
    except Exception as e:
        return False, str(e)


def run_validate_script(work_dir: str, script: str) -> tuple[bool, str]:
    """Run validation script in work directory.

    Args:
        work_dir: Working directory for script execution.
        script: Bash script content to execute.

    Returns:
        Tuple of (passed, output/error message).
    """
    if not script:
        return True, "No validation script provided"

    try:
        result = subprocess.run(
            ["bash", "-c", script],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, result.stdout or "Validation passed"
        else:
            return False, result.stderr or result.stdout or "Validation failed"
    except subprocess.TimeoutExpired:
        return False, "Validation script timed out after 30s"
    except Exception as e:
        return False, str(e)


class BenchmarkRunner:
    """Runner for executing benchmark test suites.

    Creates isolated sandbox environments for each test case,
    runs the agent, and validates results.
    """

    def __init__(self, model_name: str) -> None:
        """Initialize benchmark runner.

        Args:
            model_name: Model name to use for agent inference.
        """
        self._model_name = model_name

    def run_suite(self, testcases: list[TestCase]) -> BenchmarkReport:
        """Execute a suite of test cases and return benchmark report.

        Args:
            testcases: List of test cases to run.

        Returns:
            BenchmarkReport with aggregated results.
        """
        logger.info("==================================================")
        logger.info(f"🚀 启动自动化 Harness Benchmark 评估... | 模型: {self._model_name}")
        logger.info("==================================================")

        report = BenchmarkReport()

        for tc in testcases:
            logger.info(f"\n>>> ⏳ 正在执行用例 [{tc.id}]: {tc.name}")

            result = self._run_single_test(tc)
            report.add_result(result)

            if result.passed:
                logger.info(
                    f">>> ✅ 用例 [{tc.id}] 测试通过! | "
                    f"耗时: {result.duration_ms}ms | "
                    f"花费: ${result.total_cost_usd:.6f}"
                )
            else:
                logger.info(f">>> ❌ 用例 [{tc.id}] 测试失败! | 错误: {result.error_msg}")

        # 打印终极报表
        report.compute_summary()
        logger.info("\n================ 🏆 跑分终极报告 ================\n")
        logger.info(
            f"总用例数: {len(testcases)} | "
            f"成功数: {report.total_passed} | "
            f"成功率: {report.total_passed / len(testcases) * 100:.2f}%"
        )
        logger.info(f"总消耗成本: ${report.total_cost_usd:.6f}")
        logger.info("==================================================\n")

        return report

    def _run_single_test(self, tc: TestCase) -> TestResult:
        """Run a single test case in isolated sandbox.

        Args:
            tc: Test case to run.

        Returns:
            TestResult with pass/fail status and metrics.
        """
        start_time = time.time()

        # 1. 为每个用例创建一个绝对干净的沙箱目录 (物理隔离)
        base_dir = os.getcwd()
        work_dir = os.path.join(base_dir, "workspace", f"{tc.id}_{int(time.time())}")
        os.makedirs(work_dir, exist_ok=True)

        # 2. (可选) 执行 Setup 脚本准备靶机代码
        if tc.setup_script:
            success, output = run_setup_script(work_dir, tc.setup_script)
            if not success:
                return TestResult(
                    test_case_id=tc.id,
                    passed=False,
                    error_msg=f"靶机 Setup 失败: {output}",
                )

        # 3. 组装具备打点能力 (Tracker) 的引擎
        # 延迟导入避免循环依赖
        from internal.engine import AgentEngine, GlobalSessionMgr
        from internal.observability import CostTracker
        from internal.provider import OpenAIProvider
        from internal.schema import Message, Role
        from internal.tools import BashTool, EditFileTool, ReadFileTool, WriteFileTool, new_registry

        try:
            real_provider = OpenAIProvider.new_dashscope_provider(self._model_name)
        except Exception as e:
            return TestResult(
                test_case_id=tc.id,
                passed=False,
                error_msg=f"Provider 初始化失败: {e}",
            )

        # 为本次跑分单独建一个 Session 记账
        session = GlobalSessionMgr.get_or_create(tc.id, work_dir)

        # 用 CostTracker 包裹 Provider
        tracked_provider = CostTracker(
            next_provider=real_provider,
            model_name=self._model_name,
            session=session,
        )

        # 注册工具
        registry = new_registry()
        registry.register(ReadFileTool(work_dir))
        registry.register(WriteFileTool(work_dir))
        registry.register(BashTool(work_dir))
        registry.register(EditFileTool(work_dir))

        # 创建引擎 (关闭 thinking 和 plan_mode)
        eng = AgentEngine(tracked_provider, registry, enable_thinking=False, plan_mode=False)

        # 4. 让 Agent 开始干活
        session.append(Message(role=Role.USER, content=tc.task_prompt))

        # 传入空的 reporter (None) 屏蔽普通日志，防止刷屏
        try:
            eng.run(session, reporter=None)
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return TestResult(
                test_case_id=tc.id,
                passed=False,
                duration_ms=duration_ms,
                error_msg=f"Agent 崩溃: {e}",
            )

        # 5. 【核心断言】Agent 跑完了，我们来验收成果！
        passed, output = run_validate_script(work_dir, tc.validate_script)

        duration_ms = int((time.time() - start_time) * 1000)

        if not passed:
            return TestResult(
                test_case_id=tc.id,
                passed=False,
                total_cost_usd=session.total_cost_cny,  # Note: using CNY field as USD for simplicity
                duration_ms=duration_ms,
                total_prompt_tokens=session.total_prompt_tokens,
                total_completion_tokens=session.total_completion_tokens,
                error_msg=f"验证脚本执行失败: {output}",
            )

        return TestResult(
            test_case_id=tc.id,
            passed=True,
            total_cost_usd=session.total_cost_cny,
            duration_ms=duration_ms,
            total_prompt_tokens=session.total_prompt_tokens,
            total_completion_tokens=session.total_completion_tokens,
            trace_file=os.path.join(work_dir, ".claw", "traces", f"trace_{tc.id}_*.json"),
        )