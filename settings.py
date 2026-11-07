"""Validated, immutable settings for a single speech request."""

from collections.abc import Mapping
from dataclasses import dataclass

from .audio_tags import normalize_audio_tag_selection
from .errors import ConfigurationError


@dataclass(frozen=True)
class Settings:
    tts_provider_id: str = ""
    optimizer_provider_id: str = ""
    optimization_enabled: bool = True
    length_optimization_enabled: bool = True
    target_max_chars: int = 100
    content_optimization_enabled: bool = False
    audio_tag_optimization_enabled: bool = False
    audio_tag_provider: str = "auto"
    audio_tag_custom_prompt: str = ""
    persona_injection_enabled: bool = True
    optimizer_timeout: int = 20
    tts_timeout: int = 20
    send_timeout: int = 10
    max_concurrent_requests: int = 2

    @classmethod
    def from_mapping(cls, config: Mapping) -> "Settings":
        defaults = cls()
        values = {}
        for name in (
            "tts_provider_id",
            "optimizer_provider_id",
            "audio_tag_provider",
            "audio_tag_custom_prompt",
        ):
            value = config.get(name, getattr(defaults, name))
            if not isinstance(value, str):
                raise ConfigurationError(f"配置项 {name} 必须是字符串。")
            values[name] = value if name == "audio_tag_custom_prompt" else value.strip()

        limits = {
            "target_max_chars": (1, 500),
            "optimizer_timeout": (1, 120),
            "tts_timeout": (1, 120),
            "send_timeout": (1, 60),
            "max_concurrent_requests": (1, 8),
        }
        for name, (minimum, maximum) in limits.items():
            # Preserve existing installations' configured length preference.
            default = (
                config.get(
                    "target_chars", config.get("max_chars", defaults.target_max_chars)
                )
                if name == "target_max_chars"
                else getattr(defaults, name)
            )
            value = config.get(name, default)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ConfigurationError(
                    f"配置项 {name} 必须是 {minimum} 到 {maximum} 之间的整数。"
                )
            values[name] = value

        for name in (
            "optimization_enabled",
            "length_optimization_enabled",
            "content_optimization_enabled",
            "audio_tag_optimization_enabled",
            "persona_injection_enabled",
        ):
            value = config.get(name, getattr(defaults, name))
            if type(value) is not bool:
                raise ConfigurationError(f"配置项 {name} 必须是布尔值。")
            values[name] = value
        values["audio_tag_provider"] = normalize_audio_tag_selection(
            values["audio_tag_provider"]
        )
        return cls(**values)

    @property
    def has_optimization_steps(self) -> bool:
        return self.optimization_enabled and (
            self.length_optimization_enabled
            or self.content_optimization_enabled
            or self.audio_tag_optimization_enabled
        )

    def should_optimize_length(self, text: str) -> bool:
        """Gate length editing on the original tool input, including whitespace."""
        return (
            self.optimization_enabled
            and self.length_optimization_enabled
            and len(text) >= self.target_max_chars
        )
