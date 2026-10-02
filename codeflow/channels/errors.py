"""Channel Adapters 共用的 Outbound Send-error Classifiers。

Classifier 返回 True 表示 Failure 值得通过 `DeliveryHub` Retry Path Raise，例如 Connection Drop/Timeout/5xx。
Platform Business Rejection 应走 Terminal Failure Path，不能无意义重试。这里仅分类异常，不执行 Retry、
Backoff 或 Delivery Outcome 记录。
"""

from __future__ import annotations

import httpx

_transient_bases: list[type[BaseException]] = [TimeoutError, ConnectionError]
try:
    from websockets.exceptions import WebSocketException

    _transient_bases.append(WebSocketException)
except ImportError:
    pass
try:
    import aiohttp

    _transient_bases.append(aiohttp.ClientError)
except ImportError:
    pass
try:
    import requests

    _transient_bases.append(requests.exceptions.ConnectionError)
    _transient_bases.append(requests.exceptions.Timeout)
except ImportError:
    pass

TRANSIENT_NETWORK_ERRORS: tuple[type[BaseException], ...] = tuple(_transient_bases)


def transient_network(err: BaseException) -> bool:
    """Connection Drop、Timeout、WebSocket Close 等 Network-ish Failure 返回 `True`。

    可选 SDK Exception Types 只在 Dependency Installed 时加入。SDK Business Errors 刻意排除，让 Manager
    不把永久拒绝当 Transient Retry。
    """
    return isinstance(err, TRANSIENT_NETWORK_ERRORS)


def retryable_http(err: Exception) -> bool:
    """判断 Httpx Error 是否 Retryable：Timeout/Transport Error 与 5xx Response 为 True。

    4xx 通常表示 Request/Auth/Policy Error，返回 False；非 Httpx Exception 也返回 False。
    """
    if isinstance(err, httpx.TimeoutException | httpx.TransportError):
        return True
    if isinstance(err, httpx.HTTPStatusError):
        return bool(err.response is not None and err.response.status_code >= 500)
    return False
