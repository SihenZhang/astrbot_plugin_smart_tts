"""Check contracts consumed by AstrBot's config UI and tool registrar."""

import inspect
import json
from dataclasses import asdict
from pathlib import Path

import docstring_parser
from smart_tts.audio_tags import get_audio_tag_options
from smart_tts.main import SmartTTSPlugin
from smart_tts.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def test_webui_defaults_are_valid_and_match_runtime_settings():
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    defaults = {name: item["default"] for name, item in schema.items()}
    assert defaults == asdict(Settings())
    assert Settings.from_mapping(defaults) == Settings()
    assert schema["tts_provider_id"]["_special"] == "select_provider_tts"
    assert schema["optimizer_provider_id"]["_special"] == "select_provider"
    assert schema["audio_tag_provider"]["options"] == list(get_audio_tag_options())
    assert schema["audio_tag_provider"]["options"][:2] == ["auto", "custom"]
    assert schema["audio_tag_custom_prompt"]["type"] == "text"
    assert schema["audio_tag_custom_prompt"]["default"] == ""
    assert (
        schema["optimizer_timeout"]["default"] == schema["tts_timeout"]["default"] == 20
    )
    assert schema["target_max_chars"]["slider"]["max"] == 500
    for selection in schema["audio_tag_provider"]["options"]:
        assert (
            Settings.from_mapping({"audio_tag_provider": selection}).audio_tag_provider
            == selection
        )
    for item in schema.values():
        if "slider" in item:
            assert item["slider"]["min"] <= item["default"] <= item["slider"]["max"]


def test_optimization_settings_visibility_follows_the_relevant_switches():
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    expected = {
        name: {"optimization_enabled": True}
        for name in (
            "optimizer_provider_id",
            "length_optimization_enabled",
            "content_optimization_enabled",
            "audio_tag_optimization_enabled",
            "persona_injection_enabled",
            "optimizer_timeout",
        )
    }
    expected["target_max_chars"] = {
        "optimization_enabled": True,
        "length_optimization_enabled": True,
    }
    expected["audio_tag_provider"] = {
        "optimization_enabled": True,
        "audio_tag_optimization_enabled": True,
    }
    expected["audio_tag_custom_prompt"] = {
        "optimization_enabled": True,
        "audio_tag_optimization_enabled": True,
        "audio_tag_provider": "custom",
    }
    actual = {
        name: item["condition"] for name, item in schema.items() if "condition" in item
    }
    assert actual == expected
    assert all(set(condition) <= schema.keys() for condition in actual.values())


def test_tool_docstring_exposes_the_actual_required_argument():
    # AstrBot 4.28.0 uses docstring_parser, not Python annotations, to produce
    # its tool schema. An empty/misspelled Args block breaks calls at runtime.
    method = SmartTTSPlugin.send_voice_message
    assert method.tool_name == "send_voice_message"
    doc = docstring_parser.parse(inspect.getdoc(method))
    assert [(param.arg_name, param.type_name) for param in doc.params] == [
        ("text", "string")
    ]
    assert doc.description and doc.params[0].description
    assert list(inspect.signature(method).parameters) == ["self", "event", "text"]
