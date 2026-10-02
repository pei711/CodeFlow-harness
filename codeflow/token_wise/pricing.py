"""Historical TokenWise Pricing Imports 的 Compatibility Alias。

旧扩展从 `codeflow.token_wise.pricing` 导入定价函数时，本模块把请求透明转交给当前的
`codeflow.call_efficiency.pricing`。通过共享同一个 Module Object，旧路径看到的新费率、缓存与函数身份
都与主实现一致；这里不是独立 Pricing Source，也不应新增只存在于兼容层的业务逻辑。
"""

from __future__ import annotations

import sys

from codeflow.call_efficiency import pricing as _pricing

sys.modules[__name__] = _pricing
