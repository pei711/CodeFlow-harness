from __future__ import annotations

import asyncio
import base64
import threading
from io import BytesIO

import pytest
from PIL import Image

from codeflow.context_engine.segments import render
from codeflow.tui_rpc.dispatcher import Dispatcher
from codeflow.tui_rpc.methods import image as image_module
from codeflow.tui_rpc.methods import register_aligned_methods
from codeflow.tui_rpc.methods.image import clear_pending_images, consume_pending_images, image_attach, pending_images
from codeflow.tui_rpc.methods.turn import clear_active, turn_send

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
EMPTY_IDAT_PNG_BYTES = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAElEQVQ1rwYeAAAAAElFTkSuQmCC")


def _raster_bytes(format_name: str, color: str = "red") -> bytes:
    output = BytesIO()
    Image.new("RGB", (1, 1), color).save(output, format=format_name)
    return output.getvalue()


class _Handle:
    def cancel(self) -> None:
        pass

    async def result(self) -> None:
        return None


class _Scheduler:
    def __init__(self) -> None:
        self.submitted = []

    def submit(self, request):
        self.submitted.append(request)
        return _Handle()

    def has_pending_or_running(self, conversation_id: str) -> bool:
        return False


@pytest.fixture(autouse=True)
def _clear_state():
    clear_pending_images()
    clear_active("tui:media")
    yield
    clear_pending_images()
    clear_active("tui:media")


async def test_attached_image_enters_next_turn_request_media(tmp_path) -> None:
    image = tmp_path / "diagram.png"
    image.write_bytes(PNG_BYTES)

    attached = await image_attach({"session_id": "tui:media", "path": str(image)})
    scheduler = _Scheduler()
    await turn_send({"session_key": "tui:media", "content": "explain this"}, scheduler=scheduler, turn_ids={})

    assert attached["name"] == "diagram.png"
    assert [item.path for item in scheduler.submitted[0].media] == [str(image.resolve())]
    assert scheduler.submitted[0].media[0].content == PNG_BYTES
    assert pending_images("tui:media") == ()

    clear_active("tui:media")
    await turn_send({"session_key": "tui:media", "content": "follow up"}, scheduler=scheduler, turn_ids={})

    assert scheduler.submitted[1].media == ()


async def test_attachment_is_discarded_when_no_turn_can_be_submitted(tmp_path) -> None:
    image = tmp_path / "diagram.png"
    image.write_bytes(PNG_BYTES)
    await image_attach({"session_id": "tui:media", "path": str(image)})

    await turn_send({"session_key": "tui:media", "content": "explain this"}, scheduler=None, turn_ids={})

    assert pending_images("tui:media") == ()


async def test_non_image_attachment_is_rejected(tmp_path) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("not an image")

    with pytest.raises(ValueError, match="attachment is not an image"):
        await image_attach({"session_id": "tui:media", "path": str(document)})


async def test_malformed_png_attachment_is_rejected(tmp_path) -> None:
    image = tmp_path / "broken.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nbroken")

    with pytest.raises(ValueError, match="attachment is not an image"):
        await image_attach({"session_id": "tui:media", "path": str(image)})


async def test_malformed_jpeg_attachment_is_rejected(tmp_path) -> None:
    image = tmp_path / "broken.jpg"
    image.write_bytes(b"\xff\xd8\xff")

    with pytest.raises(ValueError, match="attachment is not an image"):
        await image_attach({"session_id": "tui:media", "path": str(image)})


async def test_raster_content_must_match_the_declared_image_type(tmp_path) -> None:
    image = tmp_path / "renamed.jpg"
    image.write_bytes(PNG_BYTES)

    with pytest.raises(ValueError, match="attachment is not an image"):
        await image_attach({"session_id": "tui:media", "path": str(image)})


async def test_crc_valid_png_without_decodable_scanlines_is_rejected(tmp_path) -> None:
    image = tmp_path / "empty-idat.png"
    image.write_bytes(EMPTY_IDAT_PNG_BYTES)

    with pytest.raises(ValueError, match="attachment is not an image"):
        await image_attach({"session_id": "tui:media", "path": str(image)})


async def test_retained_svg_attachment_is_accepted(tmp_path) -> None:
    image = tmp_path / "diagram.svg"
    image.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>')

    attached = await image_attach({"session_id": "tui:media", "path": str(image)})

    assert attached["name"] == "diagram.svg"
    assert pending_images("tui:media")[0].mime == "image/svg+xml"


async def test_malformed_or_active_svg_attachment_is_rejected(tmp_path) -> None:
    malformed = tmp_path / "broken.svg"
    malformed.write_text("<svg>")
    active = tmp_path / "entity.svg"
    active.write_text('<!DOCTYPE svg [<!ENTITY x "x">]><svg>&x;</svg>')

    with pytest.raises(ValueError, match="attachment is not an image"):
        await image_attach({"session_id": "tui:media", "path": str(malformed)})
    with pytest.raises(ValueError, match="attachment is not an image"):
        await image_attach({"session_id": "tui:media", "path": str(active)})


async def test_attachment_uses_validated_snapshot_after_source_changes(tmp_path) -> None:
    image = tmp_path / "mutable.png"
    original = _raster_bytes("PNG", "red")
    image.write_bytes(original)

    await image_attach({"session_id": "tui:media", "path": str(image)})
    image.write_bytes(_raster_bytes("PNG", "blue") + b"x" * (33 * 1024 * 1024))
    media = pending_images("tui:media")[0]
    rendered = render.build_user_content("inspect", [media])

    assert media.content == original
    assert isinstance(rendered, list)
    encoded = rendered[0]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == original


async def test_attachment_count_is_capped_per_turn(tmp_path) -> None:
    image = tmp_path / "bounded.png"
    image.write_bytes(PNG_BYTES)

    for _ in range(8):
        await image_attach({"session_id": "tui:media", "path": str(image)})

    with pytest.raises(ValueError, match="at most 8 images"):
        await image_attach({"session_id": "tui:media", "path": str(image)})
    assert len(pending_images("tui:media")) == 8


async def test_attachment_aggregate_bytes_are_capped_per_turn(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "bounded.png"
    image.write_bytes(PNG_BYTES)
    monkeypatch.setattr(image_module, "_IMAGE_MAX_TOTAL_BYTES", len(PNG_BYTES) + 1, raising=False)

    await image_attach({"session_id": "tui:media", "path": str(image)})

    with pytest.raises(ValueError, match="aggregate image size"):
        await image_attach({"session_id": "tui:media", "path": str(image)})
    assert len(pending_images("tui:media")) == 1


async def test_turn_consumption_rejects_attachment_that_finishes_validation_late(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "late.png"
    image.write_bytes(PNG_BYTES)
    validation_started = threading.Event()
    release_validation = threading.Event()
    original = image_module._read_validated_image

    def blocked_validation(path):
        validation_started.set()
        assert release_validation.wait(1)
        return original(path)

    monkeypatch.setattr(image_module, "_read_validated_image", blocked_validation)
    attach_task = asyncio.create_task(image_attach({"session_id": "tui:media", "path": str(image)}))
    assert await asyncio.to_thread(validation_started.wait, 1)

    consume_pending_images("tui:media")
    release_validation.set()

    with pytest.raises(ValueError, match="attach again"):
        await attach_task
    assert pending_images("tui:media") == ()


async def test_image_attach_is_registered_as_a_real_rpc_method(tmp_path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(_raster_bytes("JPEG"))
    dispatcher = Dispatcher()
    register_aligned_methods(dispatcher)

    response = await dispatcher.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "image.attach",
            "params": {"session_id": "tui:media", "path": str(image)},
        }
    )

    assert response["result"]["name"] == "photo.jpg"
