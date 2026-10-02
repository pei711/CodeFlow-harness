"""Network Security Utilities，为 Outbound URL Fetches 提供 SSRF Protection。

代码从 ``nanobot/security/network.py``（MIT）Ported，并移除了 Module-level CIDR Allowlist；如果未来
需要 Tailscale-style Whitelist，应重新评估而不是绕过当前检查。模块只允许 HTTP/HTTPS，解析目标
Hostname 的全部地址，并拒绝 Loopback、Private、Link-local 与其他 Internal Networks。

检查应同时用于 Original Target 和 Redirect 后的 Final URL，防止公开地址把请求重定向到内网。
DNS 在验证后到真实连接前仍可能变化，因此这是一层 Fail-closed Target Validation，不是完整的网络
Sandbox。
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _is_private(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return any(addr in net for net in _BLOCKED_NETWORKS)


def validate_url_target(url: str) -> tuple[bool, str]:
    """通过 Scheme、Hostname 与 Resolved IPs 验证 URL 是否可安全 Fetch。

    只接受 `http` / `https`，要求存在 Domain 与 Hostname，并用 `getaddrinfo` 解析 IPv4/IPv6。任一可
    解析地址落入 Blocked Networks 就拒绝整个 Target，避免多地址域名混入 Internal Address。

    Returns:
        ``(ok, error_message)``。`ok` 为 `True` 时 Error Message 为空；解析失败、不支持的 Scheme、
        DNS Failure 或 Private Address 都返回 `False` 与可诊断原因。成功只证明验证瞬间的目标地址
        通过规则，不代表远端内容可信。
    """
    try:
        p = urlparse(url)
    except Exception as e:
        return False, str(e)

    if p.scheme not in ("http", "https"):
        return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
    if not p.netloc:
        return False, "Missing domain"

    hostname = p.hostname
    if not hostname:
        return False, "Missing hostname"

    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        return False, f"Cannot resolve hostname: {hostname}"

    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if _is_private(addr):
            return False, f"Blocked: {hostname} resolves to private/internal address {addr}"

    return True, ""


def validate_resolved_url(url: str) -> tuple[bool, str]:
    """以与 Original Target 相同的 Fail-closed Policy 验证 Reported Final URL。

    下载器在跟随 Redirect 后应调用此入口，确保 Final Destination 没有转向 Private/Internal Address。
    当前实现直接复用 `validate_url_target`，返回值与错误语义完全相同；它不比较原域名与最终域名是否
    一致，只重新验证最终目标本身。
    """
    return validate_url_target(url)
