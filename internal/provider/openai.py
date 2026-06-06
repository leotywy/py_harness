"""OpenAI-compatible provider implementation."""

import json
import os

from openai import OpenAI

from internal.provider.interface import BaseProvider, ProviderError
from internal.schema import Message, Role, ToolCall, ToolDefinition, Usage


class OpenAIProvider(BaseProvider):
    """OpenAI-compatible provider for LLM communication.

    Supports OpenAI API and compatible endpoints (e.g., Zhipu, DeepSeek).
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None) -> None:
        """Initialize OpenAI provider.

        Args:
            model: Model name to use.
            base_url: Custom API endpoint (optional).
            api_key: API key (optional, defaults to environment variable).

        Raises:
            ProviderError: If API key is not provided.
        """
        if api_key is None:
            # Determine which API key to use based on base_url
            if base_url and "bigmodel.cn" in base_url:
                api_key = os.getenv("ZHIPU_API_KEY")
                if not api_key:
                    raise ProviderError("请设置 ZHIPU_API_KEY 环境变量")
            elif base_url and "dashscope.aliyuncs.com" in base_url:
                api_key = os.getenv("DASHSCOPE_API_KEY")
                if not api_key:
                    raise ProviderError("请设置 DASHSCOPE_API_KEY 环境变量")
            else:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ProviderError("请设置 OPENAI_API_KEY 环境变量")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @classmethod
    def new_zhipu_provider(cls, model: str = "glm-4-flash") -> "OpenAIProvider":
        """Create provider for Zhipu (智谱) API.

        Args:
            model: Zhipu model name (default: glm-4-flash).

        Returns:
            OpenAIProvider configured for Zhipu endpoint.
        """
        base_url = "https://open.bigmodel.cn/api/paas/v4/"
        return cls(model=model, base_url=base_url)

    @classmethod
    def new_deepseek_provider(cls, model: str = "deepseek-chat") -> "OpenAIProvider":
        """Create provider for DeepSeek API.

        Args:
            model: DeepSeek model name (default: deepseek-chat).

        Returns:
            OpenAIProvider configured for DeepSeek endpoint.
        """
        base_url = "https://api.deepseek.com/v1"
        return cls(model=model, base_url=base_url)

    @classmethod
    def new_dashscope_provider(cls, model: str = "glm-5") -> "OpenAIProvider":
        """Create provider for Alibaba Cloud DashScope API.

        Args:
            model: DashScope model name (default: qwen-coder-plus-latest).

        Returns:
            OpenAIProvider configured for DashScope endpoint.
        """
        base_url = "https://coding.dashscope.aliyuncs.com/v1"
        return cls(model=model, base_url=base_url)

    def generate(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition] | None,
    ) -> Message:
        """Generate response from LLM.

        Args:
            messages: Context history.
            available_tools: Available tools (None for thinking phase).

        Returns:
            Model response message.

        Raises:
            ProviderError: If API request fails.
        """
        # 1. Translate context messages
        openai_msgs = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                openai_msgs.append({"role": "system", "content": msg.content})

            elif msg.role == Role.USER:
                if msg.tool_call_id:
                    # Tool result message
                    openai_msgs.append({
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id,
                    })
                else:
                    openai_msgs.append({"role": "user", "content": msg.content})

            elif msg.role == Role.ASSISTANT:
                assistant_msg: dict = {"role": "assistant"}

                if msg.content:
                    assistant_msg["content"] = msg.content

                # Important: If history contains ToolCalls, must restore them
                # to maintain the model's logical chain
                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]

                openai_msgs.append(assistant_msg)

        # 2. Translate tool definitions
        openai_tools = None
        if available_tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "parameters": tool_def.input_schema,
                    },
                }
                for tool_def in available_tools
            ]

        # 3. Build request and send
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_msgs,
                tools=openai_tools,
            )
        except Exception as e:
            raise ProviderError(f"OpenAI/Zhipu API 请求失败: {e}") from e

        if not response.choices:
            raise ProviderError("API 返回了空的 Choices")

        # 4. Translate API response to internal schema.Message
        choice = response.choices[0].message
        result_msg = Message(
            role=Role.ASSISTANT,
            content=choice.content or "",
        )

        # 【新增】提取 Usage 信息
        if response.usage and (response.usage.prompt_tokens > 0 or response.usage.completion_tokens > 0):
            result_msg.usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        if choice.tool_calls:
            for tc in choice.tool_calls:
                if tc.type == "function":
                    # Parse arguments from JSON string
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": tc.function.arguments}

                    result_msg.tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=arguments,
                        )
                    )

        return result_msg