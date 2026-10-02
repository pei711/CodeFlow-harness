"""实现符合 API Version 2024-10-21 的 Azure OpenAI Direct Provider。

该实现绕过 LiteLLM，直接用 HTTP 调用 Azure Deployment Chat Completions；Model Field 作为 URL
中的 Deployment Name，鉴权使用 ``api-key`` Header，生成上限使用 ``max_completion_tokens``。
Request/Response 仍归一为 LLMProvider Contract，无真实 Stream 时使用 Base Single-delta Fallback。
"""

from __future__ import annotations

import uuid
from typing import Any
from urllib.parse import urljoin

import httpx
import json_repair

from codeflow.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_AZURE_MSG_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name"})


class AzureOpenAIProvider(LLMProvider):
    """提供 Azure OpenAI API 2024-10-21 Compliance 的 LLMProvider。

    API Version 固定 ``2024-10-21``；``model`` 是 Azure Deployment Name 并进入 URL Path；Header
    使用 ``api-key`` 而非 Authorization Bearer；Payload 使用 ``max_completion_tokens`` 而非
    ``max_tokens``。所有请求是 Direct HTTP，bypasses LiteLLM。

    Constructor 要求非空 api_key/api_base，并规范 Base 尾部 Slash。GPT-5/O-series 或显式 Reasoning
    Effort 时不发送 Temperature，避免 Azure 拒绝不支持参数。HTTP/Parse Error 返回 Structured
    LLMResponse Error，不抛给 AgentLoop。
    """

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "",
        default_model: str = "gpt-5.2-chat",
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.api_version = "2024-10-21"

        # 校验必需参数。
        if not api_key:
            raise ValueError("Azure OpenAI api_key is required")
        if not api_base:
            raise ValueError("Azure OpenAI api_base is required")

        # 确保 api_base 以 / 结尾。
        if not api_base.endswith("/"):
            api_base += "/"
        self.api_base = api_base

    def _build_chat_url(self, deployment_name: str) -> str:
        """根据 ``deployment_name`` 构建 Azure OpenAI Chat Completions URL。

        Base URL 确保尾 Slash，再用 urljoin 追加 ``openai/deployments/<name>/chat/completions``，
        Query 固定当前 ``api-version``。函数不 URL encode Deployment Name，也不发 Network Request；
        Caller 应传 Config 中真实 Azure Deployment Identifier。
        """
        # Azure OpenAI URL 格式：
        # 端点格式：https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={version}
        base_url = self.api_base
        if not base_url.endswith("/"):
            base_url += "/"

        url = urljoin(base_url, f"openai/deployments/{deployment_name}/chat/completions")
        return f"{url}?api-version={self.api_version}"

    def _build_headers(self) -> dict[str, str]:
        """构建 Azure OpenAI API Request Headers，使用 ``api-key`` 而非 Bearer Auth。

        Content-Type 固定 JSON；每次调用生成随机 ``x-session-affinity`` 改善 Backend Cache Locality。
        返回 Dict 含 Secret，Caller 只能用于 HTTP Request，不能写入 Log 或 Error Body。
        """
        return {
            "Content-Type": "application/json",
            "api-key": self.api_key,  # Azure OpenAI 使用 api-key header，而非 Authorization
            "x-session-affinity": uuid.uuid4().hex,  # 保持缓存局部性
        }

    @staticmethod
    def _supports_temperature(
        deployment_name: str,
        reasoning_effort: str | None = None,
    ) -> bool:
        """判断目标 Deployment 是否可能支持 ``temperature`` Parameter。

        显式 ``reasoning_effort`` 时返回 False；否则 Deployment Name 转 Lowercase，包含 gpt-5、o1、
        o3、o4 任一 Token 也 False，其余 True。该判断避免已知 API 400，不声称探测 Azure 实际
        Capability；新 Model Family 需要更新规则。
        """
        if reasoning_effort:
            return False
        name = deployment_name.lower()
        return not any(token in name for token in ("gpt-5", "o1", "o3", "o4"))

    def _prepare_request_payload(
        self,
        deployment_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """按 Azure OpenAI 2024-10-21 Contract 准备 Provider-safe Request Payload。

        Messages 先清 Empty Content，再投影 `_AZURE_MSG_KEYS`；``max_tokens`` 映射为至少 1 的
        ``max_completion_tokens``。Temperature 只在 `_supports_temperature` 允许时加入，Reasoning
        Effort 按需保留；有 Tools 才加入 Definitions 与 tool_choice（Default auto）。返回新 Dict，
        不修改 Caller Message。
        """
        payload: dict[str, Any] = {
            "messages": self._sanitize_request_messages(
                self._sanitize_empty_content(messages),
                _AZURE_MSG_KEYS,
            ),
            "max_completion_tokens": max(1, max_tokens),  # Azure API 2024-10-21 使用该字段
        }

        if self._supports_temperature(deployment_name, reasoning_effort):
            payload["temperature"] = temperature

        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        return payload

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """向 Azure OpenAI 发送一次 Chat Completion，并返回统一 `LLMResponse`。

        ``messages`` 是带 ``role``/``content`` 的 Dict List，与可选 OpenAI-format ``tools`` 经 Payload
        Gate；``model`` 用作 Deployment Name，
        缺失时使用 Default；``max_tokens`` 映射 ``max_completion_tokens``，Temperature 与
        reasoning_effort 按 Capability 处理。HTTP Client Timeout 60 秒且 Verify TLS。

        Status 非 200 时返回包含 Azure Status/Text 的 finish_reason=error；成功 JSON 交
        `_parse_response`。Network/Decode 等 Exception 也转 Error Response。Same-model Retry 由 Base
        ``chat_with_retry`` 调用本 Single Attempt 完成。
        """
        deployment_name = model or self.default_model
        url = self._build_chat_url(deployment_name)
        headers = self._build_headers()
        payload = self._prepare_request_payload(
            deployment_name,
            messages,
            tools,
            max_tokens,
            temperature,
            reasoning_effort,
            tool_choice=tool_choice,
        )

        try:
            async with httpx.AsyncClient(timeout=60.0, verify=True) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    return LLMResponse(
                        content=f"Azure OpenAI API Error {response.status_code}: {response.text}",
                        finish_reason="error",
                    )

                response_data = response.json()
                return self._parse_response(response_data)

        except Exception as e:
            return LLMResponse(
                content=f"Error calling Azure OpenAI: {repr(e)}",
                finish_reason="error",
            )

    def _parse_response(self, response: dict[str, Any]) -> LLMResponse:
        """把 Azure OpenAI Response Dict 解析为 CodeFlow Standard LLMResponse。

        读取首个 Choice/Message；Tool Function Arguments 若是 JSON String 用 json_repair 解析，Usage
        缺 Field 时补 0，Reasoning Content 可选。Key/Index 缺失返回 finish_reason=error 的 Parse
        Response，不抛异常；其他 JSON Shape Error 仍由上层 Chat Exception Boundary 捕获。
        """
        try:
            choice = response["choices"][0]
            message = choice["message"]

            tool_calls = []
            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    # 必要时从 JSON 字符串解析参数。
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        args = json_repair.loads(args)

                    tool_calls.append(
                        ToolCallRequest(
                            id=tc["id"],
                            name=tc["function"]["name"],
                            arguments=args,
                        )
                    )

            usage = {}
            if response.get("usage"):
                usage_data = response["usage"]
                usage = {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                }

            reasoning_content = message.get("reasoning_content") or None

            return LLMResponse(
                content=message.get("content"),
                tool_calls=tool_calls,
                finish_reason=choice.get("finish_reason", "stop"),
                usage=usage,
                reasoning_content=reasoning_content,
            )

        except (KeyError, IndexError) as e:
            return LLMResponse(
                content=f"Error parsing Azure OpenAI response: {str(e)}",
                finish_reason="error",
            )

    def get_default_model(self) -> str:
        """返回默认 Model Identifier，它同时是 Default Azure Deployment Name。

        值来自 Constructor Config，不向 Azure 查询 Deployment List，也不验证当前可用性；Call Error
        仍经 Structured Response 与 Retry Layer 处理。
        """
        return self.default_model
