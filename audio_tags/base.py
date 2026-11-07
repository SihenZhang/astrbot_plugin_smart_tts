"""Interface for audio-tag providers."""

from abc import ABC, abstractmethod
from typing import ClassVar
from xml.etree import ElementTree


class AudioTagProvider(ABC):
    aliases: ClassVar[dict[str, str]] = {}

    @abstractmethod
    def get_modes(self) -> tuple[str, ...]:
        """Return the manual modes supported by this provider."""
        ...

    @abstractmethod
    def detect_mode(self, provider_type: str, model: str) -> str | None:
        """Select a mode for known metadata, or return None."""
        ...

    @abstractmethod
    def get_guidance(self, mode: str) -> ElementTree.Element:
        """Return a fresh audio_tag_optimization element with arbitrary children."""
        ...
