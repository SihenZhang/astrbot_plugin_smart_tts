import asyncio
from copy import deepcopy
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
from astrbot.api.message_components import Plain, Record
from smart_tts import audio_tags
from smart_tts import main as plugin_main
from smart_tts.audio_tags import AudioTagProvider
from smart_tts.audio_tags.elevenlabs import ELEVEN_LABS_ENHANCED, ELEVEN_LABS_LEGACY
from smart_tts.main import SmartTTSPlugin


@pytest.fixture(autouse=True)
def persona_resolver(monkeypatch):
    """Exercise persona gating here; the SDK resolver has dedicated tests."""
    calls = []

    async def resolve(context, event):
        calls.append(event.unified_msg_origin)
        return "PERSONA:" + event.unified_msg_origin

    monkeypatch.setattr(plugin_main, "resolve_persona_prompt", resolve)
    return calls


def make_plugin(rig, **overrides):
    return SmartTTSPlugin(rig.context, {**rig.config, **overrides})


def assert_original_text_sent(event, original):
    assert len(event.sent) == 1
    assert len(event.sent[0].chain) == 1
    assert isinstance(event.sent[0].chain[0], Plain)
    assert event.sent[0].chain[0].text == original


@pytest.mark.parametrize("selection", ["auto", "demo_voice", "demo_old_mode"])
def test_new_provider_integrates_with_configuration_optimization_and_sending(
    rig, monkeypatch, selection
):
    guidance = ElementTree.Element("audio_tag_optimization")
    ElementTree.SubElement(
        guidance, "instructions", lang="zh"
    ).text = "演示声音规则：保留文稿并添加声音标签。"
    voices = ElementTree.SubElement(guidance, "voices")
    ElementTree.SubElement(voices, "voice", tone="calm").text = "[calm]"

    class DemoAudioTagProvider(AudioTagProvider):
        aliases = {"demo_old_mode": "demo_voice"}

        def get_modes(self) -> tuple[str, ...]:
            return ("demo_voice",)

        def detect_mode(self, provider_type: str, model: str) -> str | None:
            if provider_type == "demo_tts" and model == "demo-v1":
                return "demo_voice"
            return None

        def get_guidance(self, mode: str) -> ElementTree.Element:
            return deepcopy({"demo_voice": guidance}[mode])

    monkeypatch.setattr(
        audio_tags, "PROVIDERS", [*audio_tags.PROVIDERS, DemoAudioTagProvider()]
    )
    rig.tts.provider_config.update(type="demo_tts", model="demo-v1")
    plugin = make_plugin(
        rig, audio_tag_optimization_enabled=True, audio_tag_provider=selection
    )
    assert asyncio.run(plugin.send_voice_message(rig.event, "你好。")) is None
    assert len(rig.context.llm_calls) == 1
    prompt = rig.context.llm_calls[0]["system_prompt"]
    audio = ElementTree.fromstring(prompt).find("audio_tag_optimization")
    assert ElementTree.tostring(audio) == ElementTree.tostring(guidance)
    assert rig.tts.calls == [rig.context.llm_text]
    assert isinstance(rig.event.sent[0].chain[0], Record)


def test_success_optimizes_synthesizes_and_sends_one_voice_without_text(
    rig, persona_resolver
):
    original = "  " + "这是准备发送的完整原文。" * 10 + "\n"
    optimized = "  [happy]优化后的语音稿。\n"
    rig.context.llm_text = optimized
    result = asyncio.run(make_plugin(rig).send_voice_message(rig.event, original))
    assert result is None
    assert len(rig.context.llm_calls) == 1
    request = rig.context.llm_calls[0]
    assert request["chat_provider_id"] == "chat"
    assert request.get("tools") is None
    assert request.get("contexts") == []
    assert persona_resolver == [rig.event.unified_msg_origin]
    assert "PERSONA:" + rig.event.unified_msg_origin in request["system_prompt"]
    assert rig.tts.calls == [optimized]
    assert len(rig.event.sent) == 1
    assert len(rig.event.sent[0].chain) == 1
    record = rig.event.sent[0].chain[0]
    assert isinstance(record, Record)
    assert record.text == original
    assert str(rig.audio_path) in record.file
    assert rig.audio_path.exists()


@pytest.mark.parametrize("char_count", [99, 100, 101])
def test_default_length_threshold_counts_original_punctuation_and_whitespace(
    rig, persona_resolver, char_count
):
    original = "原" * (char_count - 2) + "。 "
    assert len(original) == char_count
    result = asyncio.run(make_plugin(rig).send_voice_message(rig.event, original))
    assert result is None
    assert rig.event.sent[0].chain[0].text == original
    if char_count < 100:
        assert rig.tts.calls == [original]
        assert not rig.context.llm_calls
        assert not rig.context.chat_resolutions
        assert not persona_resolver
        assert rig.context.lookups == ["voice"]
    else:
        assert rig.tts.calls == [rig.context.llm_text]
        assert len(rig.context.llm_calls) == 1
        assert "length_optimization" in rig.context.llm_calls[0]["system_prompt"]
        assert rig.context.chat_resolutions == [rig.event.unified_msg_origin]
        assert persona_resolver == [rig.event.unified_msg_origin]


def test_short_default_reply_does_not_resolve_explicit_optimizer(rig, persona_resolver):
    original = " \n短句，原样保留。\n "
    plugin = make_plugin(rig, optimizer_provider_id="missing")
    assert asyncio.run(plugin.send_voice_message(rig.event, original)) is None
    assert rig.tts.calls == [original]
    assert not rig.context.llm_calls
    assert not rig.context.chat_resolutions
    assert not persona_resolver
    assert rig.context.lookups == ["voice"]


@pytest.mark.parametrize(
    ("content_enabled", "audio_enabled", "expected_sections"),
    [
        pytest.param(True, False, {"content_optimization"}, id="content"),
        pytest.param(False, True, {"audio_tag_optimization"}, id="audio"),
        pytest.param(
            True,
            True,
            {"content_optimization", "audio_tag_optimization"},
            id="content-and-audio",
        ),
    ],
)
def test_short_reply_only_runs_enabled_content_or_supported_audio_edits(
    rig, persona_resolver, content_enabled, audio_enabled, expected_sections
):
    original = " \n短句。 "
    rig.context.llm_text = " [happy]" + "优化" * 100 + "\n "
    rig.tts.provider_config.update(type="elevenlabs_tts_api", model="eleven_v3")
    plugin = make_plugin(
        rig,
        content_optimization_enabled=content_enabled,
        audio_tag_optimization_enabled=audio_enabled,
    )
    assert asyncio.run(plugin.send_voice_message(rig.event, original)) is None
    assert len(rig.context.llm_calls) == 1
    system = ElementTree.fromstring(rig.context.llm_calls[0]["system_prompt"])
    optimization_sections = {
        "length_optimization",
        "content_optimization",
        "audio_tag_optimization",
    }
    assert {node.tag for node in system} & optimization_sections == expected_sections
    assert persona_resolver == [rig.event.unified_msg_origin]
    assert rig.tts.calls == [rig.context.llm_text]


@pytest.mark.parametrize("text", ["", " \n ", None, 123])
def test_invalid_input_never_calls_external_services(rig, text):
    result = asyncio.run(make_plugin(rig).send_voice_message(rig.event, text))
    assert isinstance(result, str) and result
    assert not rig.context.llm_calls
    assert not rig.tts.calls
    assert not rig.event.sent


def test_rejected_input_is_remembered_even_if_same_event_retries_valid_text(rig):
    async def scenario():
        plugin = make_plugin(rig)
        rejected = await plugin.send_voice_message(rig.event, "")
        repeated = await plugin.send_voice_message(rig.event, "现在是有效文字。")
        assert rejected == repeated
        assert not rig.context.llm_calls
        assert not rig.tts.calls
        assert not rig.event.sent
        fresh = rig.event_factory("test:FriendMessage:after-rejection")
        assert await plugin.send_voice_message(fresh, "你好") is None

    asyncio.run(scenario())


def test_invalid_configuration_is_rejected_before_external_calls(rig):
    result = asyncio.run(
        make_plugin(rig, tts_provider_id=123).send_voice_message(rig.event, "你好")
    )
    assert isinstance(result, str) and result
    assert not rig.context.llm_calls
    assert not rig.tts.calls
    assert not rig.event.sent


@pytest.mark.parametrize(
    "disabled",
    [
        {
            "optimization_enabled": False,
            "length_optimization_enabled": True,
            "content_optimization_enabled": True,
            "audio_tag_optimization_enabled": True,
        },
        {
            "length_optimization_enabled": False,
            "content_optimization_enabled": False,
            "audio_tag_optimization_enabled": False,
        },
    ],
)
def test_disabled_optimization_sends_exact_original_without_optional_lookups(
    rig, persona_resolver, disabled
):
    original = ' \n**原文**[happy]<break time="1s"/>' + "长" * 4100 + "\n "
    metadata_calls = []

    def forbidden_metadata():
        metadata_calls.append(True)
        raise AssertionError("disabled optimization must not inspect model metadata")

    rig.tts.meta = forbidden_metadata
    rig.tts.get_model = forbidden_metadata
    plugin = make_plugin(rig, optimizer_provider_id="missing", **disabled)
    result = asyncio.run(plugin.send_voice_message(rig.event, original))
    assert result is None
    assert rig.tts.calls == [original]
    assert not rig.context.llm_calls
    assert not rig.context.chat_resolutions
    assert not persona_resolver
    assert rig.context.lookups == ["voice"]
    assert not metadata_calls
    assert rig.event.sent[0].chain[0].text == original


@pytest.mark.parametrize(
    "steps,selection",
    [
        pytest.param({}, "auto", id="length-auto"),
        pytest.param(
            {
                "length_optimization_enabled": False,
                "content_optimization_enabled": True,
            },
            "eleven_labs_enhanced",
            id="content-manual",
        ),
    ],
)
def test_audio_tags_disabled_does_not_inspect_tts_metadata(rig, steps, selection):
    metadata_calls = []

    def forbidden_metadata():
        metadata_calls.append(True)
        raise AssertionError("audio tags are disabled")

    rig.tts.meta = forbidden_metadata
    rig.tts.get_model = forbidden_metadata
    result = asyncio.run(
        make_plugin(rig, audio_tag_provider=selection, **steps).send_voice_message(
            rig.event, "原" * 100
        )
    )
    assert result is None
    assert not metadata_calls
    assert len(rig.context.llm_calls) == 1
    assert rig.tts.calls == [rig.context.llm_text]
    assert "audio_tag_optimization" not in rig.context.llm_calls[0]["system_prompt"]


@pytest.mark.parametrize(
    ("selection", "model", "expected_guidance"),
    [
        ("auto", "eleven_multilingual_v2", ELEVEN_LABS_LEGACY),
        ("auto", "eleven_v3", ELEVEN_LABS_ENHANCED),
        ("elevenlabs", "eleven_v3", ELEVEN_LABS_ENHANCED),
    ],
)
def test_auto_and_existing_elevenlabs_setting_select_rules_from_native_model(
    rig, selection, model, expected_guidance
):
    def provider_meta():
        return SimpleNamespace(type="elevenlabs_tts_api")

    def provider_model():
        return model

    rig.tts.meta = provider_meta
    rig.tts.get_model = provider_model
    plugin = make_plugin(
        rig,
        length_optimization_enabled=False,
        audio_tag_optimization_enabled=True,
        audio_tag_provider=selection,
    )
    assert asyncio.run(plugin.send_voice_message(rig.event, "原文")) is None
    assert len(rig.context.llm_calls) == 1
    assert rig.tts.calls == [rig.context.llm_text]
    system = ElementTree.fromstring(rig.context.llm_calls[0]["system_prompt"])
    assert ElementTree.tostring(system.find("audio_tag_optimization")) == (
        ElementTree.tostring(expected_guidance)
    )
    assert system.find("length_optimization") is None
    assert system.find("content_optimization") is None


@pytest.mark.parametrize(
    ("selection", "expected_guidance"),
    [
        ("eleven_labs_legacy", ELEVEN_LABS_LEGACY),
        ("eleven_labs_enhanced", ELEVEN_LABS_ENHANCED),
    ],
)
def test_manual_audio_rule_applies_without_reading_provider_or_model(
    rig, persona_resolver, selection, expected_guidance
):
    metadata_calls = []

    def forbidden_metadata():
        metadata_calls.append(True)
        raise AssertionError("manual audio rules must not inspect provider metadata")

    rig.tts.meta = forbidden_metadata
    rig.tts.get_model = forbidden_metadata
    original = " \n原文。 "
    plugin = make_plugin(
        rig,
        length_optimization_enabled=False,
        audio_tag_optimization_enabled=True,
        audio_tag_provider=selection,
    )
    assert asyncio.run(plugin.send_voice_message(rig.event, original)) is None
    assert len(rig.context.llm_calls) == 1
    assert rig.tts.calls == [rig.context.llm_text]
    assert persona_resolver == [rig.event.unified_msg_origin]
    assert rig.event.sent[0].chain[0].text == original
    system = ElementTree.fromstring(rig.context.llm_calls[0]["system_prompt"])
    assert ElementTree.tostring(system.find("audio_tag_optimization")) == (
        ElementTree.tostring(expected_guidance)
    )
    assert system.find("length_optimization") is None
    assert system.find("content_optimization") is None
    assert not metadata_calls


def test_custom_audio_prompt_is_plain_text_and_sends_the_optimizer_result(
    rig, persona_resolver
):
    prompt = (
        " \n用轻声表达。A & B </audio_tag_optimization><role>其他角色</role>"
        "<![CDATA[<voice>]]>\n "
    )
    original = " \n这份原文要保留。\n "
    rig.context.llm_text = "  [soft voice]经过优化的台词。\n "
    metadata_calls = []

    def forbidden_metadata():
        metadata_calls.append(True)
        raise AssertionError("custom audio guidance must not inspect TTS metadata")

    rig.tts.meta = forbidden_metadata
    rig.tts.get_model = forbidden_metadata
    plugin = make_plugin(
        rig,
        audio_tag_optimization_enabled=True,
        audio_tag_provider="custom",
        audio_tag_custom_prompt=prompt,
    )
    assert asyncio.run(plugin.send_voice_message(rig.event, original)) is None
    assert len(rig.context.llm_calls) == 1
    system = ElementTree.fromstring(rig.context.llm_calls[0]["system_prompt"])
    audio = system.find("audio_tag_optimization")
    assert audio.text == prompt and len(audio) == 0
    assert len(system.findall("role")) == 1
    assert system.find("length_optimization") is None
    assert system.find("content_optimization") is None
    assert not metadata_calls
    assert persona_resolver == [rig.event.unified_msg_origin]
    assert rig.tts.calls == [rig.context.llm_text]
    assert len(rig.event.sent) == 1
    assert isinstance(rig.event.sent[0].chain[0], Record)
    assert rig.event.sent[0].chain[0].text == original


@pytest.mark.parametrize(
    "overrides",
    [{"optimization_enabled": False}, {"audio_tag_optimization_enabled": False}],
)
def test_custom_prompt_respects_master_and_audio_switches(
    rig, persona_resolver, overrides
):
    original = "短句原文。"
    config = {
        "audio_tag_optimization_enabled": True,
        "audio_tag_provider": "custom",
        "audio_tag_custom_prompt": "添加声音标签。",
        "optimizer_provider_id": "missing",
        **overrides,
    }
    plugin = make_plugin(rig, **config)
    assert asyncio.run(plugin.send_voice_message(rig.event, original)) is None
    assert rig.tts.calls == [original]
    assert not rig.context.llm_calls
    assert not rig.context.chat_resolutions
    assert not persona_resolver
    assert rig.context.lookups == ["voice"]


@pytest.mark.parametrize(
    "overrides,original,expected_section",
    [
        ({}, "短句原文。", None),
        ({"content_optimization_enabled": True}, "短句原文。", "content_optimization"),
        ({}, "原" * 100, "length_optimization"),
    ],
)
def test_blank_custom_prompt_skips_only_audio_optimization(
    rig, persona_resolver, overrides, original, expected_section
):
    config = {
        "audio_tag_optimization_enabled": True,
        "audio_tag_provider": "custom",
        "audio_tag_custom_prompt": " \n\t ",
        **overrides,
    }
    if expected_section is None:
        config["optimizer_provider_id"] = "missing"
    plugin = make_plugin(rig, **config)
    assert asyncio.run(plugin.send_voice_message(rig.event, original)) is None
    if expected_section is None:
        assert not rig.context.llm_calls
        assert not rig.context.chat_resolutions
        assert not persona_resolver
        assert rig.context.lookups == ["voice"]
        assert rig.tts.calls == [original]
    else:
        assert len(rig.context.llm_calls) == 1
        system = ElementTree.fromstring(rig.context.llm_calls[0]["system_prompt"])
        assert system.find(expected_section) is not None
        assert system.find("audio_tag_optimization") is None
        assert persona_resolver == [rig.event.unified_msg_origin]
        assert rig.tts.calls == [rig.context.llm_text]


def test_custom_prompt_optimizer_error_falls_back_to_exact_original(rig, caplog):
    async def fail(**kwargs):
        raise RuntimeError("private-custom-optimizer-key")

    rig.context.llm_hook = fail
    original = ' \n原文 [happy]<break time="1s"/> 123。\n '
    plugin = make_plugin(
        rig,
        audio_tag_optimization_enabled=True,
        audio_tag_provider="custom",
        audio_tag_custom_prompt="按语境添加声音标签。",
    )
    assert asyncio.run(plugin.send_voice_message(rig.event, original)) is None
    assert len(rig.context.llm_calls) == 1
    assert rig.tts.calls == [original]
    assert len(rig.event.sent) == 1
    assert isinstance(rig.event.sent[0].chain[0], Record)
    assert rig.event.sent[0].chain[0].text == original
    assert "private-custom-optimizer-key" not in caplog.text


@pytest.mark.parametrize(
    ("provider_type", "model"),
    [
        ("unknown_vendor", "eleven_v3"),
        ("elevenlabs_tts_api", "unknown_model"),
    ],
)
def test_unknown_audio_guidance_alone_skips_llm_and_persona(
    rig, persona_resolver, provider_type, model
):
    original = " \n原文 [happy]。\n "
    rig.tts.provider_config.update(type=provider_type, model=model)
    plugin = make_plugin(
        rig,
        length_optimization_enabled=False,
        audio_tag_optimization_enabled=True,
    )
    assert asyncio.run(plugin.send_voice_message(rig.event, original)) is None
    assert rig.tts.calls == [original]
    assert not rig.context.llm_calls
    assert not rig.context.chat_resolutions
    assert not persona_resolver


@pytest.mark.parametrize("content_only", [False, True])
def test_unknown_audio_guidance_keeps_other_enabled_optimization_steps(
    rig, content_only
):
    rig.tts.provider_config.update(type="elevenlabs_tts_api", model="unknown_model")
    plugin = make_plugin(
        rig,
        length_optimization_enabled=not content_only,
        content_optimization_enabled=content_only,
        audio_tag_optimization_enabled=True,
    )
    original = "短句" if content_only else "原" * 100
    assert asyncio.run(plugin.send_voice_message(rig.event, original)) is None
    assert len(rig.context.llm_calls) == 1
    system = rig.context.llm_calls[0]["system_prompt"]
    assert ("length_optimization" in system) is not content_only
    assert ("content_optimization" in system) is content_only
    assert "audio_tag_optimization" not in system
    assert rig.tts.calls == [rig.context.llm_text]


@pytest.mark.parametrize("length_enabled", [True, False])
def test_audio_metadata_failure_preserves_independent_optimization_steps(
    rig, persona_resolver, length_enabled
):
    def broken_metadata():
        raise RuntimeError("metadata unavailable")

    rig.tts.meta = broken_metadata
    original = "原" * 100
    plugin = make_plugin(
        rig,
        length_optimization_enabled=length_enabled,
        audio_tag_optimization_enabled=True,
    )
    assert asyncio.run(plugin.send_voice_message(rig.event, original)) is None
    if length_enabled:
        assert len(rig.context.llm_calls) == 1
        assert rig.tts.calls == [rig.context.llm_text]
        assert persona_resolver == [rig.event.unified_msg_origin]
        assert "audio_tag_optimization" not in rig.context.llm_calls[0]["system_prompt"]
    else:
        assert not rig.context.llm_calls
        assert not persona_resolver
        assert rig.tts.calls == [original]


def test_persona_switch_disables_resolver_without_disabling_optimization(
    rig, persona_resolver
):
    plugin = make_plugin(rig, persona_injection_enabled=False)
    assert asyncio.run(plugin.send_voice_message(rig.event, "原" * 100)) is None
    assert not persona_resolver
    assert len(rig.context.llm_calls) == 1
    assert "persona_reference" not in rig.context.llm_calls[0]["system_prompt"]
    assert rig.tts.calls == [rig.context.llm_text]


def test_persona_lookup_failure_still_optimizes_once(rig, monkeypatch, caplog):
    secret = "private-persona-provider-token"

    async def fail(context, event):
        raise RuntimeError(secret)

    monkeypatch.setattr(plugin_main, "resolve_persona_prompt", fail)
    assert (
        asyncio.run(make_plugin(rig).send_voice_message(rig.event, "原" * 100)) is None
    )
    assert len(rig.context.llm_calls) == 1
    assert rig.tts.calls == [rig.context.llm_text]
    assert secret not in caplog.text


def test_target_length_is_a_prompt_goal_and_does_not_retry_or_truncate_output(rig):
    original = "原文比 4000 字符长也可以交给优化器。" + "原" * 4100
    optimized = " \n[happy]" + "优化" * 150 + '<break time="1s"/>\n '
    rig.context.llm_text = optimized
    result = asyncio.run(
        make_plugin(rig, target_max_chars=100).send_voice_message(rig.event, original)
    )
    assert result is None
    assert len(rig.context.llm_calls) == 1
    assert rig.tts.calls == [optimized]
    assert rig.event.sent[0].chain[0].text == original


def test_sequential_duplicate_does_not_synthesize_or_send_again(rig):
    async def scenario():
        plugin = make_plugin(rig, target_max_chars=1)
        first = await plugin.send_voice_message(rig.event, "第一条")
        second = await plugin.send_voice_message(rig.event, "重复调用的文字")
        return first, second

    assert asyncio.run(scenario()) == (None, None)
    assert len(rig.context.llm_calls) == 1
    assert len(rig.tts.calls) == 1
    assert rig.event.send_attempts == 1


def test_concurrent_duplicate_waits_for_same_request(rig):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def synthesize(text):
            entered.set()
            await release.wait()
            return str(rig.audio_path)

        rig.tts.audio_hook = synthesize
        plugin = make_plugin(rig, target_max_chars=1)
        first = asyncio.create_task(plugin.send_voice_message(rig.event, "你好"))
        await asyncio.wait_for(entered.wait(), 1)
        second = asyncio.create_task(plugin.send_voice_message(rig.event, "你好"))
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
        release.set()
        return await asyncio.gather(first, second)

    assert asyncio.run(scenario()) == [None, None]
    assert len(rig.context.llm_calls) == len(rig.tts.calls) == 1
    assert rig.event.send_attempts == 1


def test_distinct_events_resolve_each_sessions_current_llm_and_persona(
    rig, persona_resolver
):
    async def scenario():
        plugin = make_plugin(rig, target_max_chars=1)
        second = rig.event_factory("test:FriendMessage:two")
        await plugin.send_voice_message(rig.event, "你好")
        rig.context.current_chat = "alternate"
        await plugin.send_voice_message(second, "你好")
        return second

    second = asyncio.run(scenario())
    assert [call["chat_provider_id"] for call in rig.context.llm_calls] == [
        "chat",
        "alternate",
    ]
    assert rig.context.chat_resolutions == [
        rig.event.unified_msg_origin,
        second.unified_msg_origin,
    ]
    assert persona_resolver == [rig.event.unified_msg_origin, second.unified_msg_origin]
    assert (
        "PERSONA:" + rig.event.unified_msg_origin
        in rig.context.llm_calls[0]["system_prompt"]
    )
    assert (
        "PERSONA:" + second.unified_msg_origin
        in rig.context.llm_calls[1]["system_prompt"]
    )
    assert len(rig.event.sent) == len(second.sent) == 1


def test_concurrency_limit_returns_busy_without_queuing_and_recovers(rig):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def synthesize(text):
            entered.set()
            await release.wait()
            return str(rig.audio_path)

        rig.tts.audio_hook = synthesize
        plugin = make_plugin(rig, max_concurrent_requests=1, target_max_chars=1)
        first = asyncio.create_task(plugin.send_voice_message(rig.event, "你好"))
        await asyncio.wait_for(entered.wait(), 1)
        rejected_event = rig.event_factory("test:FriendMessage:two")
        busy = await asyncio.wait_for(
            plugin.send_voice_message(rejected_event, "你好"), 0.5
        )
        assert isinstance(busy, str) and busy
        assert not rejected_event.sent
        assert len(rig.context.llm_calls) == 1
        release.set()
        assert await first is None
        repeated_busy = await plugin.send_voice_message(
            rejected_event, "现在可以发了吗"
        )
        assert repeated_busy == busy
        assert not rejected_event.sent
        assert len(rig.context.llm_calls) == 1
        next_event = rig.event_factory("test:FriendMessage:three")
        assert await plugin.send_voice_message(next_event, "你好") is None
        return next_event

    next_event = asyncio.run(scenario())
    assert len(rig.tts.calls) == 2
    assert len(next_event.sent) == 1


def test_optimizer_failure_uses_exact_original_and_keeps_error_private(rig, caplog):
    secret = "sk-test-private-secret"

    async def fail(**kwargs):
        raise RuntimeError(f"provider returned token={secret}")

    original = ' \n**你好**，[happy]<break time="1s"/>' + "长" * 4100 + "\n "
    rig.context.llm_hook = fail
    result = asyncio.run(make_plugin(rig).send_voice_message(rig.event, original))
    assert result is None
    assert len(rig.context.llm_calls) == 1
    assert rig.tts.calls == [original]
    assert rig.event.sent[0].chain[0].text == original
    assert secret not in caplog.text


def test_audio_guidance_construction_error_uses_exact_original_without_llm(
    rig, monkeypatch, caplog
):
    def construct_guidance(*args, **kwargs):
        raise ElementTree.ParseError("private-provider-rule")

    monkeypatch.setattr(plugin_main, "resolve_audio_tag_guidance", construct_guidance)
    original = ' \n原文 [happy]<break time="1s"/> 123。\n '
    plugin = make_plugin(
        rig,
        length_optimization_enabled=False,
        audio_tag_optimization_enabled=True,
        audio_tag_provider="eleven_labs_enhanced",
    )
    assert asyncio.run(plugin.send_voice_message(rig.event, original)) is None
    assert not rig.context.llm_calls
    assert rig.tts.calls == [original]
    assert len(rig.event.sent) == 1
    assert isinstance(rig.event.sent[0].chain[0], Record)
    assert rig.event.sent[0].chain[0].text == original
    assert "private-provider-rule" not in caplog.text


def test_optimizer_error_response_uses_original_without_exposing_body(rig, caplog):
    error_body = "provider failed: sk-test-private-secret"

    async def error_response(**kwargs):
        return SimpleNamespace(role="err", completion_text=error_body)

    original = " \n原文事实 123。\n "
    rig.context.llm_hook = error_response
    result = asyncio.run(
        make_plugin(rig, target_max_chars=1).send_voice_message(rig.event, original)
    )
    assert result is None
    assert len(rig.context.llm_calls) == 1
    assert rig.tts.calls == [original]
    assert isinstance(rig.event.sent[0].chain[0], Record)
    assert rig.event.sent[0].chain[0].text == original
    assert error_body not in caplog.text


@pytest.mark.parametrize(
    "config",
    [
        {"optimizer_provider_id": "missing"},
        {"optimizer_provider_id": "voice"},
    ],
)
def test_unavailable_optimizer_uses_original_tts(rig, config):
    original = " \n[happy]原文的事实 123 不能丢。\n "
    result = asyncio.run(
        make_plugin(rig, target_max_chars=1, **config).send_voice_message(
            rig.event, original
        )
    )
    assert result is None
    assert not rig.context.llm_calls
    assert config["optimizer_provider_id"] in rig.context.lookups
    assert rig.tts.calls == [original]
    assert rig.event.sent[0].chain[0].text == original


def test_current_chat_provider_resolution_failure_uses_original(rig):
    rig.context.current_chat = "missing"
    original = "原文。"
    result = asyncio.run(
        make_plugin(rig, target_max_chars=1).send_voice_message(rig.event, original)
    )
    assert result is None
    assert rig.tts.calls == [original]
    assert not rig.context.llm_calls
    assert rig.context.chat_resolutions == [rig.event.unified_msg_origin]


def test_empty_optimizer_response_falls_back_once_to_original(rig):
    rig.context.llm_text = " \n "
    original = " \n原文。\n "
    result = asyncio.run(
        make_plugin(rig, target_max_chars=1).send_voice_message(rig.event, original)
    )
    assert result is None
    assert len(rig.context.llm_calls) == 1
    assert rig.tts.calls == [original]


def test_tts_failure_sends_original_text_once_and_does_not_expose_secret(rig, caplog):
    secret = "sk-test-private-secret"
    original = " \n**原文**事实123。[happy]\n "

    async def fail(text):
        raise RuntimeError(f"authorization token={secret}")

    rig.tts.audio_hook = fail

    async def scenario():
        plugin = make_plugin(rig, target_max_chars=1)
        first = await plugin.send_voice_message(rig.event, original)
        second = await plugin.send_voice_message(rig.event, "重复调用")
        return first, second

    first, second = asyncio.run(scenario())
    assert first is second is None
    assert secret not in caplog.text
    assert rig.tts.calls == [rig.context.llm_text]
    assert rig.event.send_attempts == 1
    assert_original_text_sent(rig.event, original)


@pytest.mark.parametrize("provider_id", ["", "missing", "chat"])
def test_missing_or_wrong_tts_provider_sends_original_text(rig, provider_id):
    original = " \n**原文**[happy]。\n "
    result = asyncio.run(
        make_plugin(rig, tts_provider_id=provider_id).send_voice_message(
            rig.event, original
        )
    )
    assert result is None
    assert not rig.tts.calls
    assert_original_text_sent(rig.event, original)


@pytest.mark.parametrize("kind", ["missing", "empty", "directory", "none", "url"])
def test_invalid_audio_result_sends_original_text(rig, kind, tmp_path):
    paths = {
        "missing": str(tmp_path / "missing.wav"),
        "empty": str(tmp_path / "empty.wav"),
        "directory": str(tmp_path),
        "none": None,
        "url": "https://example.invalid/audio.wav",
    }
    (tmp_path / "empty.wav").touch()

    async def invalid(text):
        return paths[kind]

    rig.tts.audio_hook = invalid
    original = " \n**原文**[happy]。\n "
    result = asyncio.run(
        make_plugin(rig, target_max_chars=1).send_voice_message(rig.event, original)
    )
    assert result is None
    assert rig.tts.calls == [rig.context.llm_text]
    assert_original_text_sent(rig.event, original)


@pytest.mark.parametrize("send_kind", ["voice", "text_fallback"])
def test_send_failure_is_not_retried_or_reported_as_success(rig, caplog, send_kind):
    secret = "sk-test-private-secret"

    async def fail(result):
        raise RuntimeError(f"request to platform failed: key={secret}")

    rig.event.send_hook = fail
    if send_kind == "text_fallback":
        rig.context.providers.pop("voice")

    async def scenario():
        plugin = make_plugin(rig)
        first = await plugin.send_voice_message(rig.event, "你好")
        second = await plugin.send_voice_message(rig.event, "你好")
        return first, second

    first, second = asyncio.run(scenario())
    assert isinstance(first, str) and first == second
    assert secret not in first and secret not in caplog.text
    assert rig.event.send_attempts == 1
    assert not rig.event.sent


@pytest.mark.parametrize("stage", ["optimizer", "tts", "send", "text_send"])
def test_stage_timeouts_release_capacity_and_do_not_retry_send(rig, stage):
    async def scenario():
        async def hang(*args, **kwargs):
            await asyncio.Event().wait()

        plugin = make_plugin(
            rig,
            target_max_chars=1,
            optimizer_timeout=1,
            tts_timeout=1,
            send_timeout=1,
            max_concurrent_requests=1,
        )
        if stage == "optimizer":
            rig.context.llm_hook = hang
        elif stage == "tts":
            rig.tts.audio_hook = hang
        else:
            rig.event.send_hook = hang
            if stage == "text_send":
                rig.context.providers.pop("voice")
        original = " \n原文不是优化稿。\n "
        result = await asyncio.wait_for(
            plugin.send_voice_message(rig.event, original), 2
        )
        if stage == "optimizer":
            assert result is None
            assert rig.tts.calls == [original]
            assert isinstance(rig.event.sent[0].chain[0], Record)
        elif stage == "tts":
            assert result is None
            assert_original_text_sent(rig.event, original)
        else:
            assert isinstance(result, str) and result
            assert not rig.event.sent
        assert rig.event.send_attempts == 1
        rig.context.llm_hook = None
        rig.tts.audio_hook = None
        rig.context.providers["voice"] = rig.tts
        next_event = rig.event_factory("test:FriendMessage:after-timeout")
        assert await plugin.send_voice_message(next_event, "你好") is None
        return next_event

    assert len(asyncio.run(scenario()).sent) == 1


def test_cancellation_propagates_and_releases_capacity(rig):
    async def scenario():
        entered = asyncio.Event()

        async def hang(text):
            entered.set()
            await asyncio.Event().wait()

        rig.tts.audio_hook = hang
        plugin = make_plugin(rig, max_concurrent_requests=1)
        task = asyncio.create_task(plugin.send_voice_message(rig.event, "你好"))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not rig.event.sent
        rig.tts.audio_hook = None
        next_event = rig.event_factory("test:FriendMessage:after-cancel")
        assert await plugin.send_voice_message(next_event, "你好") is None

    asyncio.run(scenario())


def test_terminate_cancels_inflight_work_and_leaves_no_worker_tasks(rig):
    async def scenario():
        baseline = asyncio.all_tasks()
        entered = asyncio.Event()

        async def hang(text):
            entered.set()
            await asyncio.Event().wait()

        rig.tts.audio_hook = hang
        plugin = make_plugin(rig)
        task = asyncio.create_task(plugin.send_voice_message(rig.event, "你好"))
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(plugin.terminate(), 1)
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        assert asyncio.all_tasks() <= baseline
        assert not rig.event.sent
        later = rig.event_factory("test:FriendMessage:after-unload")
        refused = await plugin.send_voice_message(later, "你好")
        assert isinstance(refused, str) and refused
        assert not later.sent

    asyncio.run(scenario())
