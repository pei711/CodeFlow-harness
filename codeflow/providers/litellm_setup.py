"""在抑制 Import-time Terminal Noise 的边界内加载 litellm。

litellm 会打印受 ``suppress_debug_info`` 控制的 "Provider List" Banner，还在 ``LiteLLM*`` Logger 上
安装 stderr ``StreamHandler``，导致 Import 期间 DEBUG 直达 Terminal。本模块临时把三个 Logger
提高到 WARNING，Import 后设置 suppress flag，再恢复原 Level。

整段 Session 的 Handler Strip 仍由 CodeFlow ``_strip_tty_stream_handlers`` 所有，它在 TUI Deferred
Import 后运行；这里若删除 Handler，会让“仅 Import Provider Module”产生永久 Global Logging
Side Effect。
"""

import logging

# litellm 会把 stderr handler 挂到这三个 logger 上（litellm/_logging.py）。
_LITELLM_LOGGERS = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")


def import_litellm():
    """Import litellm，同时禁用 Banner 并抑制 Import-time DEBUG。

    函数保存每个 LiteLLM Logger 原 Level，临时设 WARNING，在 finally 中无条件恢复；Import Error
    仍向 Caller 传播。成功后返回 Module Object，``suppress_debug_info=True`` 保持设置，长期 TTY
    Handler 清理由其他 Owner 完成。
    """
    loggers = [logging.getLogger(name) for name in _LITELLM_LOGGERS]
    prev_levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.WARNING)
    try:
        import litellm

        litellm.suppress_debug_info = True
    finally:
        for lg, prev in zip(loggers, prev_levels):
            lg.setLevel(prev)

    return litellm
