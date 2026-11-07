from xml.etree import ElementTree

import pytest
from smart_tts import audio_tags
from smart_tts.audio_tags import (
    normalize_audio_tag_selection,
    resolve_audio_tag_guidance,
)
from smart_tts.audio_tags.elevenlabs import (
    ELEVEN_LABS_ENHANCED,
    ELEVEN_LABS_LEGACY,
    ElevenLabsAudioTagProvider,
)
from smart_tts.errors import ConfigurationError


def guidance_xml(guidance):
    assert isinstance(guidance, ElementTree.Element)
    assert guidance.tag == "audio_tag_optimization"
    return guidance


def assert_same_guidance(actual, expected):
    assert ElementTree.tostring(guidance_xml(actual)) == ElementTree.tostring(
        guidance_xml(expected)
    )


def test_custom_guidance_keeps_plain_text_in_independent_trees(monkeypatch):
    prompt = (
        ' \nA & B </audio_tag_optimization><role kind="override">任意文字</role>'
        "<![CDATA[<voice>]]>\n "
    )

    def unexpected_builtin(*args, **kwargs):
        pytest.fail("Custom guidance must not select a built-in audio rule")

    for provider in audio_tags.PROVIDERS:
        monkeypatch.setattr(provider, "detect_mode", unexpected_builtin)
        monkeypatch.setattr(provider, "get_guidance", unexpected_builtin)
    first = resolve_audio_tag_guidance(
        "custom", "elevenlabs_tts_api", "eleven_v3", custom_prompt=prompt
    )
    second = resolve_audio_tag_guidance("custom", None, None, custom_prompt=prompt)
    assert first is not second
    assert first.tag == second.tag == "audio_tag_optimization"
    assert first.text == second.text == prompt
    assert len(first) == len(second) == 0
    restored = ElementTree.fromstring(ElementTree.tostring(first))
    assert restored.text == prompt and len(restored) == 0
    first.text = "只改变这一次请求"
    assert second.text == prompt


@pytest.mark.parametrize("prompt", ["", " \n\t"])
def test_blank_custom_prompt_has_no_audio_guidance(prompt):
    assert resolve_audio_tag_guidance("custom", "", "", custom_prompt=prompt) is None


@pytest.mark.parametrize(
    "selection", ["auto", "eleven_labs_legacy", "eleven_labs_enhanced"]
)
def test_builtin_modes_ignore_custom_prompt(selection):
    expected = resolve_audio_tag_guidance(selection, "elevenlabs_tts_api", "eleven_v3")
    actual = resolve_audio_tag_guidance(
        selection, "elevenlabs_tts_api", "eleven_v3", custom_prompt="仅用于自定义模式"
    )
    assert_same_guidance(actual, expected)


@pytest.mark.parametrize(
    "model", ["eleven_v3", "eleven_v3_conversational", "eleven_v4", "eleven_v4_turbo"]
)
def test_auto_detects_enhanced_models(model):
    guidance = resolve_audio_tag_guidance("auto", "elevenlabs_tts_api", model)
    assert_same_guidance(guidance, ELEVEN_LABS_ENHANCED)


@pytest.mark.parametrize(
    "model",
    [
        "eleven_multilingual_v2",
        "eleven_flash_v2",
        "eleven_flash_v2_5",
        "eleven_turbo_v2",
        "eleven_turbo_v2_5",
    ],
)
def test_auto_detects_legacy_models(model):
    guidance = resolve_audio_tag_guidance("auto", "elevenlabs_tts_api", model)
    assert_same_guidance(guidance, ELEVEN_LABS_LEGACY)


@pytest.mark.parametrize(
    "provider,model",
    [
        ("edge_tts", "eleven_v3"),
        ("proxy", "eleven_v4_turbo"),
        ("elevenlabs_tts_api", "future-model"),
        ("elevenlabs_tts_api", ""),
        ("elevenlabs_tts_api", "eleven_multilingual_v1"),
        ("elevenlabs_tts_api", "eleven_ttv_v3"),
        ("unknown-vendor", "unknown-model"),
    ],
)
def test_auto_unknown_vendor_or_model_has_no_guidance(provider, model):
    assert resolve_audio_tag_guidance("auto", provider, model) is None


@pytest.mark.parametrize(
    "selection,expected,provider,model",
    [
        pytest.param(
            "eleven_labs_legacy",
            ELEVEN_LABS_LEGACY,
            "elevenlabs_tts_api",
            "eleven_v3",
            id="legacy-overrides-enhanced-model",
        ),
        pytest.param(
            "eleven_labs_enhanced",
            ELEVEN_LABS_ENHANCED,
            "elevenlabs_tts_api",
            "eleven_multilingual_v2",
            id="enhanced-overrides-legacy-model",
        ),
        pytest.param(
            "eleven_labs_legacy",
            ELEVEN_LABS_LEGACY,
            "compatible_proxy",
            "unknown-model",
            id="legacy-with-unknown-metadata",
        ),
        pytest.param(
            "eleven_labs_enhanced",
            ELEVEN_LABS_ENHANCED,
            "compatible_proxy",
            "unknown-model",
            id="enhanced-with-unknown-metadata",
        ),
        pytest.param(
            "eleven_labs_legacy",
            ELEVEN_LABS_LEGACY,
            "",
            "",
            id="legacy-with-empty-metadata",
        ),
        pytest.param(
            "eleven_labs_enhanced",
            ELEVEN_LABS_ENHANCED,
            None,
            None,
            id="enhanced-with-missing-metadata",
        ),
    ],
)
def test_manual_modes_ignore_provider_and_model_metadata(
    selection, expected, provider, model
):
    assert_same_guidance(
        resolve_audio_tag_guidance(selection, provider, model), expected
    )


def test_auto_normalizes_model_name_whitespace_and_case():
    guidance = resolve_audio_tag_guidance("auto", "elevenlabs_tts_api", " ELEVEN_V3 ")
    assert_same_guidance(guidance, ELEVEN_LABS_ENHANCED)


@pytest.mark.parametrize(
    "mode,template",
    [
        ("eleven_labs_legacy", ELEVEN_LABS_LEGACY),
        ("eleven_labs_enhanced", ELEVEN_LABS_ENHANCED),
    ],
)
def test_provider_returns_independent_trees_without_mutating_the_template(
    mode, template
):
    provider = ElevenLabsAudioTagProvider()
    expected = ElementTree.tostring(guidance_xml(template))
    first = provider.get_guidance(mode)
    second = provider.get_guidance(mode)
    assert_same_guidance(first, template)
    assert_same_guidance(second, template)
    first.set("changed", "yes")
    list(first.iter())[-1].text = "这一份树已修改"
    first.append(ElementTree.Element("new_rule"))
    assert ElementTree.tostring(second) == expected
    assert ElementTree.tostring(provider.get_guidance(mode)) == expected
    assert ElementTree.tostring(template) == expected


def test_legacy_provider_selection_alias_uses_automatic_detection():
    assert normalize_audio_tag_selection("elevenlabs") == "auto"
    assert_same_guidance(
        resolve_audio_tag_guidance("elevenlabs", "elevenlabs_tts_api", "eleven_v3"),
        ELEVEN_LABS_ENHANCED,
    )
    assert resolve_audio_tag_guidance("elevenlabs", "proxy", "eleven_v3") is None


@pytest.mark.parametrize("selection", ["unsupported-vendor", "generic"])
def test_invalid_selection_is_a_configuration_error(selection):
    with pytest.raises(ConfigurationError):
        resolve_audio_tag_guidance(selection, "proxy", "eleven_v3")
