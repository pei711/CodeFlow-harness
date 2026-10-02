"""校验并暂存 TUI 为下一次 Turn 附加的本地 image。

``image.attach`` 不会立刻启动 Agent。它解析用户给出的 path，在 worker thread 中读取并
验证 raster 或 SVG 字节，再按 ``session_id`` 放入进程内 pending queue；Turn 提交路径
随后通过 ``pending_images()`` 读取，并在消费或取消时清理。每张图片最多 32 MiB，每个
Turn 最多 8 张且总计最多 64 MiB。

Raster 使用 Pillow 同时执行 ``verify()`` 和 ``load()``，并把 decompression bomb warning
视为失败；SVG 拒绝 ``DOCTYPE``/``ENTITY`` 并要求根元素为 ``svg``。文件扩展推导的 MIME
必须与实际 raster format 一致。generation token 防止异步校验期间 Session 被清理后，
旧结果重新写回。

attach 成功只表示图片字节已通过本地校验并进入内存，尚未提交 Turn、持久化到 Session，
也不证明模型能够理解图片。本模块不负责 UI preview、文件选择器或模型多模态能力判断。
"""

from __future__ import annotations

import asyncio
import mimetypes
import warnings
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

from PIL import Image

from codeflow.spine import Media
from codeflow.tui_rpc.models import ImageAttachParams

if TYPE_CHECKING:
    from codeflow.tui_rpc.dispatcher import Dispatcher

_pending_images: dict[str, list[Media]] = {}
_attachment_generations: dict[str, int] = {}
_attachment_global_generation = 0
_IMAGE_MAX_BYTES = 32 * 1024 * 1024
_IMAGE_MAX_COUNT = 8
_IMAGE_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_RASTER_MIMES = {
    "BMP": "image/bmp",
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "TIFF": "image/tiff",
    "WEBP": "image/webp",
}
_SVG_MIME = "image/svg+xml"


def _attachment_token(session_id: str) -> tuple[int, int]:
    return _attachment_global_generation, _attachment_generations.get(session_id, 0)


def _is_decodable_raster(data: bytes, expected_mime: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if _RASTER_MIMES.get(image.format or "") != expected_mime:
                    return False
                image.verify()
            with Image.open(BytesIO(data)) as image:
                if _RASTER_MIMES.get(image.format or "") != expected_mime:
                    return False
                image.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, OSError, SyntaxError, ValueError):
        return False

    return True


def _is_decodable_svg(data: bytes) -> bool:
    upper = data.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        return False
    try:
        root = ElementTree.fromstring(data)
    except (ElementTree.ParseError, UnicodeDecodeError, ValueError):
        return False
    return isinstance(root.tag, str) and root.tag.rsplit("}", 1)[-1].lower() == "svg"


def _read_validated_image(path: Path) -> tuple[str, bytes] | None:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if mime not in {*_RASTER_MIMES.values(), _SVG_MIME}:
        return None

    with path.open("rb") as source:
        data = source.read(_IMAGE_MAX_BYTES + 1)
    if not data or len(data) > _IMAGE_MAX_BYTES:
        return None

    valid = _is_decodable_svg(data) if mime == _SVG_MIME else _is_decodable_raster(data, mime)
    if not valid:
        return None
    return mime, data


async def image_attach(params: dict[str, Any]) -> dict[str, Any]:
    """校验 ``params`` 指定的图片，并加入对应 Session 的 pending queue。

    参数按 ``ImageAttachParams`` 验证，``session_id`` 和 ``path`` 均必填。路径会去除常见
    引号、展开 ``~``、解析为绝对路径并要求文件存在。读取与解码在 ``asyncio.to_thread``
    中完成，避免阻塞 RPC event loop。非图片、空文件、超限、解码失败、数量/总量超限，
    或校验期间 attachment generation 改变时抛出 ``ValueError``。

    成功返回 ``{"name": path.name, "remainder": ""}``，并把包含原路径、MIME 与字节的
    ``Media`` 存入内存。返回成功不表示图片已经提交给 provider；调用者仍需发起 Turn。
    """
    parsed = ImageAttachParams.model_validate(params)
    if not parsed.session_id:
        raise ValueError("session_id is required")
    if not parsed.path:
        raise ValueError("path is required")

    attachment_token = _attachment_token(parsed.session_id)
    path = Path(parsed.path.strip().strip("`\"'")).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"attachment is not a file: {path}")
    validated = await asyncio.to_thread(_read_validated_image, path)
    if validated is None:
        raise ValueError(f"attachment is not an image: {path.name}")
    mime, content = validated

    if attachment_token != _attachment_token(parsed.session_id):
        raise ValueError("session attachment state changed during validation; attach again")
    pending = _pending_images.setdefault(parsed.session_id, [])
    if len(pending) >= _IMAGE_MAX_COUNT:
        raise ValueError(f"a turn may have at most {_IMAGE_MAX_COUNT} images")
    if sum(len(item.content or b"") for item in pending) + len(content) > _IMAGE_MAX_TOTAL_BYTES:
        raise ValueError("aggregate image size exceeds the per-turn limit")
    pending.append(Media(path=str(path), mime=mime, kind="image", content=content))
    return {"name": path.name, "remainder": ""}


def pending_images(session_id: str) -> tuple[Media, ...]:
    """返回 ``session_id`` 当前待提交图片的只读 tuple snapshot。

    Session 不存在或没有附件时返回空 tuple。``Media.content`` 仍引用内存中的 bytes；
    调用本函数不会消费、复制或持久化附件。
    """
    return tuple(_pending_images.get(session_id, ()))


def consume_pending_images(session_id: str) -> None:
    """在 Turn 接纳附件后清空该 Session 的 pending image。

    该操作与 ``clear_pending_images(session_id)`` 相同，并递增 generation，使仍在后台校验
    的旧 attach 失败。函数无返回值；清空表示 queue 已消费，不证明 Turn 或任务完成。
    """
    clear_pending_images(session_id)


def clear_pending_images(session_id: str | None = None) -> None:
    """清理一个 Session 或全部 Session 的 pending image 状态。

    传入 ``session_id`` 时只清理该队列并递增其 generation；传入 ``None`` 时递增 global
    generation，同时清空所有 per-session generation 和 queue。generation 变化用于拒绝
    清理前启动、清理后才结束的异步校验结果。函数只释放进程内 bytes，不删除源文件。
    """
    global _attachment_global_generation

    if session_id is None:
        _attachment_global_generation += 1
        _attachment_generations.clear()
        _pending_images.clear()
    else:
        _attachment_generations[session_id] = _attachment_generations.get(session_id, 0) + 1
        _pending_images.pop(session_id, None)


def register_image_methods(dispatcher: "Dispatcher") -> None:
    """在 Dispatcher 上注册 ``image.attach``。

    注册不会读取文件或创建 attachment state；重复注册由 Dispatcher 抛出
    ``ValueError``。
    """
    dispatcher.register("image.attach", image_attach)


__all__ = [
    "clear_pending_images",
    "consume_pending_images",
    "image_attach",
    "pending_images",
    "register_image_methods",
]
