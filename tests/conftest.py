"""Small SDK doubles for offline behavior tests, not a live-platform substitute."""

from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("smart_tts")
package.__path__ = [str(ROOT)]
sys.modules.setdefault("smart_tts", package)


class Provider:
    def __init__(self, provider_id="chat", provider_type="openai_chat_completion"):
        self.provider_config = {"id": provider_id, "type": provider_type}

    def meta(self):
        return SimpleNamespace(**self.provider_config)

    def get_model(self):
        return self.provider_config.get("model", "test-model")


class TTSProvider:
    def __init__(self, provider_id="voice", provider_type="test_tts"):
        self.provider_config = {"id": provider_id, "type": provider_type}

    def meta(self):
        return SimpleNamespace(**self.provider_config)

    def get_model(self):
        return self.provider_config.get("model", "test-model")


@dataclass
class Record:
    file: str
    url: str | None = None
    text: str | None = None

    @classmethod
    def fromFileSystem(cls, path, **kwargs):
        return cls(file=str(path), url=str(path), **kwargs)


@dataclass
class Plain:
    text: str


@dataclass
class MessageEventResult:
    chain: list


class AstrMessageEvent:
    def __init__(self, umo="test:FriendMessage:one"):
        self.unified_msg_origin = umo
        self.extras = {}
        self.sent = []
        self.send_attempts = 0
        self.send_hook = None

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value

    def chain_result(self, chain):
        return MessageEventResult(chain=list(chain))

    async def send(self, result):
        self.send_attempts += 1
        if self.send_hook is not None:
            await self.send_hook(result)
        self.sent.append(result)


class Context:
    pass


class Star:
    def __init__(self, context):
        self.context = context


def llm_tool(*, name):
    def decorate(function):
        function.tool_name = name
        return function

    return decorate


def install_sdk_doubles():
    modules = {
        name: types.ModuleType(name)
        for name in (
            "astrbot",
            "astrbot.api",
            "astrbot.api.event",
            "astrbot.api.star",
            "astrbot.api.provider",
            "astrbot.api.message_components",
            "astrbot.core",
            "astrbot.core.provider",
            "astrbot.core.provider.provider",
        )
    }
    modules["astrbot"].__path__ = []
    modules["astrbot.api"].__path__ = []
    modules["astrbot.core"].__path__ = []
    modules["astrbot.core.provider"].__path__ = []
    modules["astrbot.api"].logger = logging.getLogger("smart_tts_test")
    modules["astrbot.api"].AstrBotConfig = dict
    modules["astrbot.api.event"].filter = SimpleNamespace(llm_tool=llm_tool)
    modules["astrbot.api.event"].AstrMessageEvent = AstrMessageEvent
    modules["astrbot.api.event"].MessageEventResult = MessageEventResult
    modules["astrbot.api.star"].Star = Star
    modules["astrbot.api.star"].Context = Context
    modules["astrbot.api.provider"].Provider = Provider
    modules["astrbot.core.provider.provider"].TTSProvider = TTSProvider
    modules["astrbot.api.message_components"].Record = Record
    modules["astrbot.api.message_components"].Plain = Plain
    for name, module in modules.items():
        sys.modules[name] = module
        parent_name, _, attribute = name.rpartition(".")
        if parent_name:
            setattr(modules[parent_name], attribute, module)


install_sdk_doubles()


class FakeTTSProvider(TTSProvider):
    def __init__(self, audio_path):
        super().__init__()
        self.audio_path = str(audio_path)
        self.calls = []
        self.audio_hook = None

    async def get_audio(self, text):
        self.calls.append(text)
        if self.audio_hook is not None:
            return await self.audio_hook(text)
        return self.audio_path


class FakeContext(Context):
    def __init__(self, tts):
        self.providers = {
            "voice": tts,
            "chat": Provider(),
            "alternate": Provider("alternate"),
        }
        self.lookups = []
        self.chat_resolutions = []
        self.current_chat = "chat"
        self.llm_calls = []
        self.llm_hook = None
        self.llm_text = "优化后的语音稿。"

    def get_provider_by_id(self, provider_id):
        self.lookups.append(provider_id)
        return self.providers.get(provider_id)

    async def get_current_chat_provider_id(self, umo):
        self.chat_resolutions.append(umo)
        return self.current_chat

    async def llm_generate(self, **kwargs):
        self.llm_calls.append(kwargs)
        if self.llm_hook is not None:
            return await self.llm_hook(**kwargs)
        return SimpleNamespace(completion_text=self.llm_text)


@pytest.fixture
def rig(tmp_path):
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00test")
    tts = FakeTTSProvider(audio_path)
    context = FakeContext(tts)
    return SimpleNamespace(
        context=context,
        tts=tts,
        audio_path=audio_path,
        event=AstrMessageEvent(),
        event_factory=AstrMessageEvent,
        config={"tts_provider_id": "voice"},
    )
