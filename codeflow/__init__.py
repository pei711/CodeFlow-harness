"""CodeFlow 是紧凑、可组合、受 Benchmark Gate 约束的 Agent Harness，并提供 Opt-in Evolution Path。

Agent Harness 负责把一次模型调用扩展成可持续工作的 Agent Runtime：组织 Context、执行 Provider 与
Tools、记录调用证据，并在明确门禁下复用或演化能力。CodeFlow 的 Three Feature Pillars 是：

1. Context Management：``context_engine/``，核心是 Curator Engine；
2. Call Efficiency：``call_efficiency/``；
3. Skill Self-Evolution：``memory_engine/skill_forge/``。

Public 与 Internal Python Namespace 都是 ``codeflow``。License 与 Third-party Attribution Details 见
``NOTICES.md`` 和 ``LICENSES/``。这里的 “Benchmark-gated” 表示能力主张要通过评测门禁，不代表每次
真实任务都会自动获得相同结果。
"""

import logging as _logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from codeflow.product import PRODUCT_LOGO


class _LiteLLMBotocorePreloadFilter(_logging.Filter):
    """过滤 LiteLLM 因缺少 ``botocore`` 产生的 Bedrock/SageMaker Pre-load Warnings。

    未安装 ``botocore`` 时，LiteLLM 会在 Import-time 输出这些日志；而 ``boto3`` 在 CodeFlow 中只是
    Optional Extra，不属于 Default Dependencies。对于不使用 AWS Bedrock / SageMaker 的用户，这些
    Warning 只是 Noise。

    Filter 只屏蔽两条已知 Pre-load Pattern，不隐藏其他 LiteLLM 日志。如果用户真的调用上述
    Providers，Downstream Call 仍会暴露比 Pre-load Chatter 更清晰的依赖错误，因此这里不会把实际
    Provider Failure 伪装成成功。
    """

    _patterns = (
        "could not pre-load bedrock-runtime response stream shape",
        "could not pre-load sagemaker-runtime response stream shape",
    )

    def filter(self, record: _logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(p in msg for p in self._patterns)


_logging.getLogger("LiteLLM").addFilter(_LiteLLMBotocorePreloadFilter())

try:
    __version__ = _pkg_version("codeflow-harness")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
__logo__ = PRODUCT_LOGO
