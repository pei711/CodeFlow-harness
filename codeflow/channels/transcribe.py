"""Channel Audio Transcription Helper，是 Groq Provider 的 Thin Wrapper。

真实实现位于 :mod:`codeflow.providers.transcription`。Empty ``api_key`` 传为 `None`，让 Provider 自行回退
``GROQ_API_KEY`` Env；Import、Auth、Network、Decode 等任何 Failure 都 Warning 后返回 ``""``。空结果表示
无可用 Transcript，不能与“音频内容为空”区分。
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger


async def transcribe_audio(file_path: str | Path, api_key: str = "", *, channel: str = "") -> str:
    try:
        from codeflow.providers.transcription import GroqTranscriptionProvider

        provider = GroqTranscriptionProvider(api_key=api_key or None)
        return await provider.transcribe(file_path)
    except Exception as e:
        logger.warning("{}: audio transcription failed: {}", channel or "channel", e)
        return ""
