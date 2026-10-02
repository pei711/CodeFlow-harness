"""提供外部 ``web_search`` 与受 URL 安全门禁保护的 ``web_fetch`` Tool。

`WebSearchTool` 调用 Serper 返回标题、URL 与 snippet；`WebFetchTool` 调用 Jina Reader 抽取可读
正文，并在请求前验证目标 URL、返回后再次验证 final URL，防止 redirect 绕过 SSRF 边界。
两者都是 EXTERNAL effect，支持显式 Proxy，API key 在调用时解析以接纳运行中配置变化。
"""

import json
import os
from typing import Any

import httpx
from loguru import logger

from codeflow.agent.tools.base import Tool, ToolResult
from codeflow.agent.tools.execution import ToolCapability, ToolEffect
from codeflow.security.network import validate_resolved_url, validate_url_target


class WebSearchTool(Tool):
    """通过 Serper 搜索公开 Web，并返回有限数量的结构化文本结果。

    ``query`` 必填，``count`` 在 1–10 内并受实例 max_results 默认值约束。Tool 读取 Serper answer
    box、knowledge graph 与 organic results，保持标题、link、snippet 供模型判断；无命中返回
    正常文本。缺 API key、HTTP 或 Proxy failure 返回 Error/failed ToolResult，不缓存搜索结果。
    """

    capability = ToolCapability(effect=ToolEffect.EXTERNAL)
    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str | None = None, max_results: int = 5, proxy: str | None = None):
        self._init_api_key = api_key
        self.max_results = max_results
        self.proxy = proxy

    @property
    def api_key(self) -> str:
        """在每次调用时解析 Serper API key，使 Environment/Config 变化立即生效。

        Constructor 显式 key 优先，否则读取 ``SERPER_API_KEY``；两者都没有时返回空字符串，
        execute 再给出配置指引。属性不记录或展示 key，也不把它写入 Tool Result。
        """
        return self._init_api_key or os.environ.get("SERPER_API_KEY", "")

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        if not self.api_key:
            return (
                "Error: Serper API key not configured. Set it in "
                "~/.codeflow/config.json under tools.web.search.apiKey "
                "(or export SERPER_API_KEY), then restart the gateway."
            )

        try:
            n = min(max(count or self.max_results, 1), 10)
            logger.debug("WebSearch: {}", "proxy enabled" if self.proxy else "direct connection")
            async with httpx.AsyncClient(proxy=self.proxy) as client:
                r = await client.post(
                    "https://google.serper.dev/search",
                    json={"q": query, "num": n},
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "X-API-KEY": self.api_key,
                    },
                    timeout=10.0,
                )
                r.raise_for_status()

            data = r.json()
            results = data.get("organic", [])[:n]
            if not results:
                return f"No results for: {query}"

            lines = [f"Results for: {query}\n"]
            if answer := data.get("answerBox"):
                snippet = answer.get("answer") or answer.get("snippet")
                if snippet:
                    lines.append(f"Answer: {snippet}\n")
            if knowledge := data.get("knowledgeGraph"):
                title = knowledge.get("title")
                description = knowledge.get("description")
                if title or description:
                    lines.append(f"Knowledge: {title or ''}")
                    if description:
                        lines.append(f"   {description}")
            for i, item in enumerate(results, 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('link', '')}")
                if desc := item.get("snippet"):
                    lines.append(f"   {desc}")
            return "\n".join(lines)
        except httpx.ProxyError as e:
            logger.error("WebSearch proxy error: {}", e)
            return ToolResult(f"Proxy error: {e}", failed=True)
        except Exception as e:
            logger.error("WebSearch error: {}", e)
            return f"Error: {e}"


class WebFetchTool(Tool):
    """通过 Jina Reader 抓取 URL，并返回经过双重目标验证的可读内容。

    输入先由 `validate_url_target` 拒绝内网、危险 scheme 等目标；Reader 返回 JSON 后，必须存在
    data object、final URL 与 content，final URL 再经 `validate_resolved_url`，防止安全地址重定向
    到禁止网络。正文按 ``maxChars`` 截断，并连同 original/final URL、status、extractMode、长度
    和 truncated flag 编码为 JSON。

    API key 可选，Authorization 使用 Bearer；Proxy/HTTP/响应形状错误都返回 failed ToolResult。
    Tool 不执行页面内指令，获取内容仍应在 Context 中作为外部不可信数据处理。
    """

    capability = ToolCapability(effect=ToolEffect.EXTERNAL)
    name = "web_fetch"
    description = "Fetch URL and extract readable content via Jina Reader."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100},
        },
        "required": ["url"],
    }

    def __init__(self, api_key: str | None = None, max_chars: int = 50000, proxy: str | None = None):
        self._init_api_key = api_key
        self.max_chars = max_chars
        self.proxy = proxy

    @property
    def api_key(self) -> str:
        """在每次 Fetch 时解析 Jina API key，以接纳运行中 Environment/Config 更新。

        Constructor key 优先，否则读取 ``JINA_API_KEY``；空值表示使用 Reader 未鉴权路径，而不是
        配置错误。属性不发请求、不验证 key，也不把敏感值写入日志或返回 JSON。
        """
        return self._init_api_key or os.environ.get("JINA_API_KEY", "")

    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:  # noqa: N803  （LLM 工具模式使用驼峰命名）
        max_chars = maxChars or self.max_chars
        is_valid, error_msg = validate_url_target(url)
        if not is_valid:
            return ToolResult(
                json.dumps(
                    {"error": f"URL validation failed: {error_msg}", "url": url},
                    ensure_ascii=False,
                ),
                failed=True,
            )

        try:
            logger.debug("WebFetch: {}", "proxy enabled" if self.proxy else "direct connection")
            # Reader 当前的 X-Base 合约在 JSON data.url 中返回快照链接。Reader 已完成抓取，
            # CodeFlow 只将验证用作返回内容的门禁。
            headers = {"Accept": "application/json", "X-Base": "final"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            async with httpx.AsyncClient(timeout=30.0, proxy=self.proxy) as client:
                r = await client.get(f"https://r.jina.ai/{url}", headers=headers)
                r.raise_for_status()

            payload = r.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, dict):
                return ToolResult(
                    json.dumps(
                        {"error": "Invalid Jina Reader response: missing data object", "url": url},
                        ensure_ascii=False,
                    ),
                    failed=True,
                )

            final_url = data.get("url")
            if not isinstance(final_url, str) or not final_url:
                return ToolResult(
                    json.dumps(
                        {"error": "Invalid Jina Reader response: missing final URL", "url": url},
                        ensure_ascii=False,
                    ),
                    failed=True,
                )

            is_valid, error_msg = validate_resolved_url(final_url)
            if not is_valid:
                return ToolResult(
                    json.dumps(
                        {"error": f"Final URL validation failed: {error_msg}", "url": url},
                        ensure_ascii=False,
                    ),
                    failed=True,
                )

            text = data.get("content")
            if not isinstance(text, str):
                return ToolResult(
                    json.dumps(
                        {"error": "Invalid Jina Reader response: missing content", "url": url},
                        ensure_ascii=False,
                    ),
                    failed=True,
                )

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]

            return json.dumps(
                {
                    "url": url,
                    "finalUrl": final_url,
                    "status": r.status_code,
                    "extractor": "jina-reader",
                    "extractMode": extractMode,
                    "truncated": truncated,
                    "length": len(text),
                    "text": text,
                },
                ensure_ascii=False,
            )
        except httpx.ProxyError as e:
            logger.error("WebFetch proxy error for {}: {}", url, e)
            return ToolResult(
                json.dumps({"error": f"Proxy error: {e}", "url": url}, ensure_ascii=False),
                failed=True,
            )
        except Exception as e:
            logger.error("WebFetch error for {}: {}", url, e)
            return ToolResult(
                json.dumps({"error": str(e), "url": url}, ensure_ascii=False),
                failed=True,
            )
