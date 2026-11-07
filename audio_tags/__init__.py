"""Select guidance from the configured audio-tag provider instances."""

from xml.etree import ElementTree

from ..errors import ConfigurationError
from .base import AudioTagProvider
from .elevenlabs import ElevenLabsAudioTagProvider

PROVIDERS: list[AudioTagProvider] = [ElevenLabsAudioTagProvider()]


def resolve_audio_tag_guidance(
    selection: str, provider_type: str, model: str, *, custom_prompt: str = ""
) -> ElementTree.Element | None:
    selection = normalize_audio_tag_selection(selection)
    if selection == "custom":
        if not custom_prompt.strip():
            return None
        guidance = ElementTree.Element("audio_tag_optimization")
        guidance.text = custom_prompt
        return guidance
    for provider in PROVIDERS:
        if selection == "auto":
            mode = provider.detect_mode(provider_type, model)
            if mode is not None:
                return provider.get_guidance(mode)
        elif selection in provider.get_modes():
            return provider.get_guidance(selection)
    return None


def normalize_audio_tag_selection(value: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError("音频标签模式必须是字符串。")
    value = value.strip()
    for provider in PROVIDERS:
        if value in provider.aliases:
            value = provider.aliases[value]
            break
    if value not in get_audio_tag_options():
        raise ConfigurationError("请选择有效的音频标签模式。")
    return value


def get_audio_tag_options() -> tuple[str, ...]:
    return (
        "auto",
        "custom",
        *(mode for provider in PROVIDERS for mode in provider.get_modes()),
    )


__all__ = [
    "PROVIDERS",
    "AudioTagProvider",
    "get_audio_tag_options",
    "normalize_audio_tag_selection",
    "resolve_audio_tag_guidance",
]
