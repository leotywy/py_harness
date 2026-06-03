"""Feishu bot implementation for Agent engine."""

import json
import logging
import os
import threading
from typing import Any

import requests

from internal.engine import AgentEngine, BaseReporter

logger = logging.getLogger(__name__)


class FeishuReporter(BaseReporter):
    """Reporter that sends engine output to Feishu chat."""

    def __init__(self, client: "FeishuClient", chat_id: str) -> None:
        """Initialize with Feishu client and chat ID.

        Args:
            client: Feishu API client.
            chat_id: Target chat ID for messages.
        """
        self.client = client
        self.chat_id = chat_id

    def send_msg(self, text: str) -> None:
        """Send text message to Feishu chat.

        Args:
            text: Message content to send.
        """
        self.client.send_message(self.chat_id, text)

    def on_thinking(self) -> None:
        """Send lightweight thinking indicator."""
        self.send_msg("🤔 模型正在慢思考 (Thinking)...")

    def on_tool_call(self, tool_name: str, args: str) -> None:
        """Send tool call notification."""
        self.send_msg(f"🛠️ **正在执行工具**：`{tool_name}`\n参数：`{args}`")

    def on_tool_result(self, tool_name: str, result: str, is_error: bool) -> None:
        """Send tool result notification."""
        if is_error:
            self.send_msg(f"⚠️ **执行报错** ({tool_name})：\n{result}")
        else:
            # Only report success, not full logs
            self.send_msg(f"✅ **执行成功** ({tool_name})")

    def on_message(self, content: str) -> None:
        """Send final message to user."""
        self.send_msg(content)


class FeishuClient:
    """Feishu API client for message operations."""

    BASE_URL = "https://open.feishu.cn/open-apis"

    def __init__(self, app_id: str, app_secret: str) -> None:
        """Initialize with app credentials.

        Args:
            app_id: Feishu app ID.
            app_secret: Feishu app secret.
        """
        self.app_id = app_id
        self.app_secret = app_secret
        self._tenant_access_token: str | None = None

    def get_tenant_access_token(self) -> str:
        """Get tenant access token from Feishu.

        Returns:
            Tenant access token string.

        Raises:
            RuntimeError: If token request fails.
        """
        if self._tenant_access_token:
            return self._tenant_access_token

        url = f"{self.BASE_URL}/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self.app_id,
            "app_secret": self.app_secret,
        }

        response = requests.post(url, json=payload)
        data = response.json()

        if data.get("code") != 0:
            raise RuntimeError(f"获取 token 失败: {data.get('msg')}")

        self._tenant_access_token = data.get("tenant_access_token")
        return self._tenant_access_token

    def send_message(self, chat_id: str, text: str) -> dict[str, Any]:
        """Send text message to Feishu chat.

        Args:
            chat_id: Target chat ID.
            text: Message content.

        Returns:
            API response data.

        Raises:
            RuntimeError: If message send fails.
        """
        token = self.get_tenant_access_token()

        url = f"{self.BASE_URL}/im/v1/messages"
        headers = {
            "Authorization": f"Bearer {token}",
        }
        params = {
            "receive_id_type": "chat_id",
        }
        content = json.dumps({"text": text})
        payload = {
            "receive_id": chat_id,
            "msg_type": "text",
            "content": content,
        }

        response = requests.post(url, headers=headers, params=params, json=payload)
        data = response.json()

        if data.get("code") != 0:
            logger.error(f"发送消息失败: {data.get('msg')}")
            raise RuntimeError(f"发送消息失败: {data.get('msg')}")

        return data


class FeishuBot:
    """Feishu bot encapsulating configuration and core business flow."""

    def __init__(self, engine: AgentEngine) -> None:
        """Initialize with Agent engine.

        Args:
            engine: Agent engine instance.

        Raises:
            RuntimeError: If Feishu credentials not set.
        """
        app_id = os.getenv("FEISHU_APP_ID")
        app_secret = os.getenv("FEISHU_APP_SECRET")

        if not app_id or not app_secret:
            raise RuntimeError("请设置 FEISHU_APP_ID 和 FEISHU_APP_SECRET 环境变量")

        self.client = FeishuClient(app_id, app_secret)
        self.app_id = app_id
        self.app_secret = app_secret
        self.engine = engine

        logger.info(f"[FeishuBot] 初始化完成, app_id: {app_id}")

    def handle_message(self, chat_id: str, content: str) -> None:
        """Handle incoming message from Feishu.

        Args:
            chat_id: Source chat ID.
            content: Message content.
        """
        # 【驾驭并发】: Handle each request in separate thread
        # to avoid blocking HTTP callback
        thread = threading.Thread(
            target=self._handle_agent_run,
            args=(chat_id, content),
        )
        thread.start()

    def _handle_agent_run(self, chat_id: str, prompt: str) -> None:
        """Run agent engine for Feishu message.

        Args:
            chat_id: Target chat ID.
            prompt: User prompt from Feishu.
        """
        logger.info(f"[FeishuBot] 收到会话 {chat_id} 消息: {prompt}")

        # Create reporter for this chat
        reporter = FeishuReporter(self.client, chat_id)

        # Run engine!
        try:
            self.engine.run(prompt)
        except Exception as e:
            reporter.send_msg(f"❌ Agent 运行崩溃: {e}")

    def parse_event_content(self, content: str) -> str:
        """Parse message content from Feishu event.

        Args:
            content: Raw content string from Feishu.

        Returns:
            Cleaned text content.
        """
        # Strip JSON wrapper: {"text":"..."}
        content = content.strip()
        if content.startswith("{\"text\":\""):
            content = content[9:]  # Remove {"text":"
        if content.endswith("\"}"):
            content = content[:-2]  # Remove "}
        return content

    def create_flask_handler(self) -> callable:
        """Create Flask handler for Feishu webhook.

        Returns:
            Flask route handler function.
        """
        def handler(request_body: dict) -> dict:
            """Handle Feishu webhook event.

            Args:
                request_body: Parsed JSON body from Feishu.

            Returns:
                Response dict.
            """
            event = request_body.get("event", {})
            message = event.get("message", {})

            chat_id = message.get("chat_id")
            content = message.get("content", "")

            if chat_id and content:
                clean_content = self.parse_event_content(content)
                self.handle_message(chat_id, clean_content)

            return {"code": 0, "msg": "success"}

        return handler


# Create __init__.py for feishu module