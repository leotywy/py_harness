"""Cost tracker for monitoring LLM API usage and billing."""

import logging
import time

from internal.provider import LLMProvider
from internal.schema import Message, ToolDefinition

logger = logging.getLogger(__name__)


# PricingModel 定义了不同大模型的计费标准 (单位: 美元/1M Tokens)
# 为了演示，这里硬编码了当前市面上几个主流模型的官方大致定价。
PRICING_MODEL: dict[str, dict[str, float]] = {
    "glm-4-flash": {"input_price": 0.1, "output_price": 0.1},
    "glm-4.5-air": {"input_price": 0.15, "output_price": 0.15},
    "glm-5": {"input_price": 0.5, "output_price": 0.5},
    # 可以继续添加其他模型价格...
}


class CostTracker:
    """装饰器中间件，包装真实 LLMProvider 进行成本监控.

    实现了 LLMProvider 接口，可以无缝注入到 Main Loop 中。
    """

    def __init__(
        self,
        next_provider: LLMProvider,
        model_name: str,
        session: object | None = None,
    ) -> None:
        """初始化成本追踪器.

        Args:
            next_provider: 被包装的真实 LLM Provider.
            model_name: 模型名称，用于查找计费标准.
            session: 当前所属的会话，用于累加总成本 (需有 record_usage 方法).
        """
        self._next_provider = next_provider
        self._model_name = model_name
        self._session = session

    def generate(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition] | None,
    ) -> Message:
        """实现 LLMProvider 接口，带成本追踪.

        Args:
            messages: 当前上下文历史.
            available_tools: 可用工具列表.

        Returns:
            模型响应消息.

        Raises:
            ProviderError: 如果底层 Provider 调用失败.
        """
        # 1. 记录请求发起的时刻
        start_time = time.time()

        # 2. 调用真实的底层大模型去执行耗时的网络请求
        try:
            resp_msg = self._next_provider.generate(messages, available_tools)
        except Exception as e:
            # 如果报错了，只打印报错时间，不计费
            latency = time.time() - start_time
            logger.error(f"[Tracker] ❌ API 调用失败，耗时: {latency:.2f}s")
            raise

        # 3. 计算耗时
        latency = time.time() - start_time

        # 4. 解析 Token 并计算成本
        if resp_msg.usage is not None:
            prompt_tokens = resp_msg.usage.prompt_tokens
            completion_tokens = resp_msg.usage.completion_tokens

            cost = 0.0
            if self._model_name in PRICING_MODEL:
                price = PRICING_MODEL[self._model_name]
                # 计算花费 = (输入Tokens * 输入单价 + 输出Tokens * 输出单价) / 1000000
                cost = (
                    prompt_tokens * price["input_price"]
                    + completion_tokens * price["output_price"]
                ) / 1_000_000.0

            # 5. 打印精美的仪表盘日志
            logger.info(
                f"[Tracker] 📊 API 调用完成 | 耗时: {latency:.2f}s | "
                f"输入: {prompt_tokens} tk | 输出: {completion_tokens} tk | "
                f"花费: ¥{cost:.6f}"
            )

            # 6. 将账单累加到当前的 Session 中，供人类后续随时查询
            if self._session is not None and hasattr(self._session, "record_usage"):
                self._session.record_usage(prompt_tokens, completion_tokens, cost)
                total_cost = getattr(self._session, "total_cost_cny", 0.0)
                session_id = getattr(self._session, "id", "unknown")
                logger.info(f"[Tracker] 💰 当前会话 ({session_id}) 累计花费: ¥{total_cost:.6f}")

        else:
            logger.warning(f"[Tracker] ⚠️ API 调用完成，但未返回 Usage 数据 | 耗时: {latency:.2f}s")

        return resp_msg