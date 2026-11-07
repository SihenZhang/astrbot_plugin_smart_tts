from copy import deepcopy
from xml.etree import ElementTree

import pytest
from smart_tts import audio_tags
from smart_tts.audio_tags import AudioTagProvider


def sample_guidance(instructions, *, style=None):
    root = ElementTree.Element("audio_tag_optimization")
    parent = ElementTree.SubElement(root, "voice", style=style) if style else root
    ElementTree.SubElement(parent, "instructions").text = instructions
    return root


WARM_GUIDANCE = sample_guidance("使用温暖的声音表现。", style="warm")
PLAIN_GUIDANCE = sample_guidance("保留自然停顿。")


def guidance_xml(selection, provider_type, model):
    guidance = audio_tags.resolve_audio_tag_guidance(selection, provider_type, model)
    assert isinstance(guidance, ElementTree.Element)
    assert guidance.tag == "audio_tag_optimization"
    return ElementTree.tostring(guidance)


class SampleAudioTagProvider(AudioTagProvider):
    aliases = {"sample_old": "sample_warm", "sample_old_auto": "auto"}

    def get_modes(self) -> tuple[str, ...]:
        return ("sample_warm", "sample_plain")

    def detect_mode(self, provider_type: str, model: str) -> str | None:
        if provider_type not in {"sample_tts", "sample_proxy"}:
            return None
        return {
            "sample-v1": "sample_warm",
            "sample-v2": "sample_plain",
        }.get(model.strip().lower())

    def get_guidance(self, mode: str) -> ElementTree.Element:
        return deepcopy(
            {"sample_warm": WARM_GUIDANCE, "sample_plain": PLAIN_GUIDANCE}[mode]
        )


@pytest.fixture
def second_provider(monkeypatch):
    provider = SampleAudioTagProvider()
    monkeypatch.setattr(audio_tags, "PROVIDERS", [*audio_tags.PROVIDERS, provider])
    return provider


def test_subclass_adds_manual_modes_without_changing_existing_options(monkeypatch):
    original_options = audio_tags.get_audio_tag_options()
    provider = SampleAudioTagProvider()

    def unexpected_auto_detection(*args):
        pytest.fail("Manual selection must not inspect provider or model metadata")

    monkeypatch.setattr(provider, "detect_mode", unexpected_auto_detection)
    monkeypatch.setattr(audio_tags, "PROVIDERS", [*audio_tags.PROVIDERS, provider])

    assert audio_tags.get_audio_tag_options() == (
        *original_options,
        "sample_warm",
        "sample_plain",
    )
    assert guidance_xml("sample_warm", None, None) == ElementTree.tostring(
        WARM_GUIDANCE
    )
    assert guidance_xml("sample_plain", "proxy", "unknown") == ElementTree.tostring(
        PLAIN_GUIDANCE
    )


def test_subclass_detects_its_models_and_leaves_unknowns_unmatched(second_provider):
    assert guidance_xml("auto", "sample_tts", "sample-v1") == ElementTree.tostring(
        WARM_GUIDANCE
    )
    assert guidance_xml("auto", "sample_proxy", " SAMPLE-V2 ") == ElementTree.tostring(
        PLAIN_GUIDANCE
    )
    assert (
        audio_tags.resolve_audio_tag_guidance("auto", "sample_tts", "unknown") is None
    )
    assert (
        audio_tags.resolve_audio_tag_guidance("auto", "other_tts", "sample-v1") is None
    )


def test_subclass_aliases_resolve_without_appearing_in_options(second_provider):
    assert audio_tags.normalize_audio_tag_selection(" sample_old ") == "sample_warm"
    assert audio_tags.normalize_audio_tag_selection("sample_old_auto") == "auto"
    assert guidance_xml("sample_old", None, None) == ElementTree.tostring(WARM_GUIDANCE)
    assert guidance_xml(
        "sample_old_auto", "sample_tts", "sample-v2"
    ) == ElementTree.tostring(PLAIN_GUIDANCE)
    assert not set(second_provider.aliases).intersection(
        audio_tags.get_audio_tag_options()
    )
