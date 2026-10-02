"""Historical TokenWise Catalog Module 的 Compatibility Alias。

旧代码仍可能导入 `codeflow.token_wise.model_catalog_cache`，而当前真实实现已经迁移到
`codeflow.call_efficiency.model_catalog_cache`。本模块通过替换 `sys.modules` 中的当前条目，让旧路径与
新路径得到同一个 Module Object；它不复制状态，也不维护第二套缓存实现。
"""

from __future__ import annotations

import sys

from codeflow.call_efficiency import model_catalog_cache as _model_catalog_cache

sys.modules[__name__] = _model_catalog_cache
