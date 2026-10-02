"""Capability Tokens，是 Multi-agent Coordination 的 Scaffold。

目前 **仅是 Scaffolding**：模块定义 `CapabilityToken` Dataclass 与 Deterministic HMAC-based Issue /
Verify Pair，为未来工作保留 Stable Seam。现在尚无 Callers，`AgentLoop` 与 Subagent Spawning 都不会
查询 Token。等具体 Multi-agent Flow 需要强制“这个 Subagent 只能调用这些 Tools”时，才会通过这些
Primitives 接入 Enforcement。

Design Choices：

- **Tokens 使用 JSON + HMAC，不用 JWT**：当前不需要 JWT Algorithm-agility Surface，JSON 也让
  Payload 在 Logs / Debug Dumps 中保持可读；
- **Tokens 按 ID 绑定 Issuer，不按 Rotating Signing Keys**：Secret 来自 Workspace-local Config；
  Rotation Policy 属于 Operator Concern，延后到真实 Deployment 需要时决定；
- **Verification Fails Closed**：Structure、Signature、Expiry 任一不匹配都返回 `None`。Callers 必须
  把 `None` 解释为 ``no capability``。

由于尚未 Wire-up，成功 Issue 或 Verify 只证明令牌格式和签名有效，**不证明任何运行时权限已经被
执行**。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any

_SIG_ALGO = hashlib.sha256


@dataclass
class CapabilityToken:
    """一个 Token 为一个 Agent Identity 描述一组 Capabilities。

    `agent_id` 标识主体，`capabilities` 保存允许能力的字符串列表，`issued_at` / `expires_at` 描述时间
    边界，`metadata` 携带不参与专门类型约束的扩展信息。Dataclass 是可序列化 Carrier，本身不会把
    能力应用到 Tool Registry；真正授权仍需未来 Caller 在执行入口验证并解释这些字段。
    """

    agent_id: str
    capabilities: list[str] = field(default_factory=list)
    issued_at: int = field(default_factory=lambda: int(time.time()))
    expires_at: int | None = None  # None 表示永不过期
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "capabilities": list(self.capabilities),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CapabilityToken":
        return cls(
            agent_id=str(payload["agent_id"]),
            capabilities=list(payload.get("capabilities") or []),
            issued_at=int(payload.get("issued_at", 0)),
            expires_at=(int(payload["expires_at"]) if payload.get("expires_at") is not None else None),
            metadata=dict(payload.get("metadata") or {}),
        )

    def is_expired(self, now: int | None = None) -> bool:
        if self.expires_at is None:
            return False
        ref = int(now if now is not None else time.time())
        return ref >= self.expires_at


def issue_token(token: CapabilityToken, secret: str) -> str:
    """序列化并用 HMAC 签署 Token，返回 ``payload.signature``。

    Payload JSON 使用稳定 Key Order 与紧凑分隔符，之后编码为 URL-safe Base64；Signature 对编码后的
    Payload 使用 Workspace Secret 和 SHA-256 计算，再做同样编码。两个 Half 都可安全放入 URL 风格
    字符串，但没有加密 Payload，持有者仍可读取内容；Secret 为空也不会在此被拒绝，配置层必须保证
    密钥质量。
    """
    raw = json.dumps(token.to_payload(), sort_keys=True, separators=(",", ":"))
    payload_b64 = _b64(raw.encode("utf-8"))
    sig = hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), _SIG_ALGO)
    sig_b64 = _b64(sig.digest())
    return f"{payload_b64}.{sig_b64}"


def verify_token(token_str: str, secret: str) -> CapabilityToken | None:
    """执行 :func:`issue_token` 的 Reverse，验证成功后返回 `CapabilityToken`。

    Malformed、Bad Signature、Expired 或 Decode Error 等任意 Failure Mode 都返回 `None`。Signature
    使用 `hmac.compare_digest` 做 Constant-time Comparison，验证通过后才解析 JSON 并检查过期时间。
    返回对象只证明 Token 在当前 Secret 下结构、签名与时间有效；在 Enforcement 尚未接入前，它不会
    自动限制任何 Agent 或 Tool。
    """
    if not isinstance(token_str, str) or token_str.count(".") != 1:
        return None
    payload_b64, sig_b64 = token_str.split(".", 1)
    expected = hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), _SIG_ALGO)
    if not hmac.compare_digest(_b64(expected.digest()), sig_b64):
        return None
    try:
        raw = _unb64(payload_b64)
        payload = json.loads(raw.decode("utf-8"))
        token = CapabilityToken.from_payload(payload)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if token.is_expired():
        return None
    return token


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(s: str) -> bytes:
    padding = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + padding).encode("ascii"))


__all__ = ["CapabilityToken", "issue_token", "verify_token"]
