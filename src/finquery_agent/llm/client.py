"""公共 LLM HTTP 客户端。

统一 answer/rag/analysis/意图识别各处对 OpenAI 兼容 /chat/completions
端点的调用,避免重复的请求拼装与错误处理代码。

设计说明:
- 使用标准库 urllib 而非 requests/httpx/SDK:保持零额外依赖,
  且测试可通过 monkeypatch urllib.request.urlopen 统一 mock。
- 失败统一返回 None 而非抛异常:上层降级策略一致(LLM 失败即回退
  确定性答案/规则引擎),调用方只需判空,无需 try/except。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from finquery_agent.config import LLMSettings

# 限流(429)与服务端抖动(5xx)是瞬时性错误,退避后重试大概率成功;
# 其他错误(401/400/网络不通)重试无意义,直接返回 None 让调用方降级。
# 评测脚本并发跑大 prompt 时容易触发 TPM 限流,没有重试会成批失败。
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 2
_BACKOFF_SECONDS = 2.0


class LLMClient:
    """对 OpenAI 兼容 /chat/completions 端点的最小封装(DeepSeek 使用此协议)。"""

    def __init__(self, settings: LLMSettings):
        self.settings = settings

    def is_available(self) -> bool:
        """配置完整才认为可用;调用方以此决定是否走 LLM 分支。"""
        return bool(self.settings.enabled and self.settings.api_key and self.settings.model and self.settings.base_url)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
    ) -> str | None:
        """发送 chat 请求,返回首个 choice 的文本;任何失败返回 None。

        response_format 用于意图识别场景传 {"type": "json_object"},
        DeepSeek 支持该参数强制输出合法 JSON,减少解析失败率。
        """
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        message = self._post_chat(payload)
        if message is None:
            return None
        return str(message.get("content") or "").strip()

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.1,
    ) -> dict[str, Any] | None:
        """function-calling 请求,返回完整 assistant message(含 tool_calls)。

        与 chat() 分开而不是加参数:两者返回类型不同(文本 vs message dict),
        合并会让所有现有调用方为不需要的场景做类型分支。
        """
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
        }
        return self._post_chat(payload)

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.is_available():
            return None
        endpoint = f"{str(self.settings.base_url).rstrip('/')}/chat/completions"
        for attempt in range(_MAX_RETRIES + 1):
            try:
                request = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    if response.status != 200:
                        return None
                    data = json.loads(response.read().decode("utf-8"))
                message = data["choices"][0]["message"]
                return message if isinstance(message, dict) else None
            except urllib.error.HTTPError as exc:
                if exc.code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                    time.sleep(_BACKOFF_SECONDS * (attempt + 1))
                    continue
                return None
            except Exception:
                # 网络失败/超时/结构异常都视为"LLM 本次不可用",由调用方降级。
                return None
        return None
