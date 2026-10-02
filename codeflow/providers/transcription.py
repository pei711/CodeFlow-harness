"""实现使用 Groq Whisper API 的 Voice Transcription Provider。

该 Provider 独立于主 LLM Chat Contract，只把 Local Audio File 作为 Multipart 上传到
``whisper-large-v3``，返回 Transcribed Text。API Key 可由 Constructor 或 GROQ_API_KEY 提供；
缺配置、缺文件、HTTP/JSON Error 都记录并返回空 String，不阻断 Channel Main Flow。
"""

import os
from pathlib import Path

import httpx
from loguru import logger


class GroqTranscriptionProvider:
    """通过 Groq Whisper API 提供快速 Voice-to-text Transcription。

    实例保存 API Key 与固定 Transcriptions Endpoint。Groq 提供 Extremely Fast Transcription 与
    Generous Free Tier，但本类不管理 Quota、Retry 或 Audio Conversion；Caller 必须提供 API 支持
    的真实 File。
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, file_path: str | Path) -> str:
        """把 ``file_path`` 指向的 Audio File 上传 Groq 并返回 Transcribed Text。

        缺 API Key 或 Path 不存在时返回空 String。有效文件以 Binary Multipart ``file`` 与 Model
        ``whisper-large-v3`` POST，Bearer Header 鉴权，Timeout 60 秒；成功读取 JSON ``text`` Field。
        HTTP、File、Decode 任意 Exception 记录 Error 并返回空 String。方法不删除、转换或缓存源文件。
        """
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f),
                        "model": (None, "whisper-large-v3"),
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                    }

                    response = await client.post(self.api_url, headers=headers, files=files, timeout=60.0)

                    response.raise_for_status()
                    data = response.json()
                    return data.get("text", "")

        except Exception as e:
            logger.error("Groq transcription error: {}", e)
            return ""
