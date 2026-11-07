import pytest
from smart_tts.settings import ConfigurationError, Settings


def test_defaults_enable_length_optimization_and_current_chat_persona():
    settings = Settings.from_mapping({})
    assert settings.optimizer_provider_id == ""
    assert settings.optimization_enabled is True
    assert settings.length_optimization_enabled is True
    assert settings.target_max_chars == 100
    assert settings.content_optimization_enabled is False
    assert settings.audio_tag_optimization_enabled is False
    assert settings.audio_tag_provider == "auto"
    assert settings.audio_tag_custom_prompt == ""
    assert settings.persona_injection_enabled is True
    assert settings.optimizer_timeout == settings.tts_timeout == 20


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("target_max_chars", True),
        ("target_max_chars", "200"),
        ("target_max_chars", 0),
        ("target_max_chars", 501),
        ("optimizer_timeout", 0),
        ("tts_timeout", 121),
        ("send_timeout", 61),
        ("max_concurrent_requests", 0),
        ("max_concurrent_requests", 9),
        ("tts_provider_id", 123),
        ("optimizer_provider_id", None),
        ("audio_tag_provider", "unknown"),
        ("audio_tag_provider", None),
        ("audio_tag_custom_prompt", None),
        ("audio_tag_custom_prompt", 123),
    ],
)
def test_invalid_settings_are_rejected(name, value):
    with pytest.raises(ConfigurationError):
        Settings.from_mapping({name: value})


def test_provider_ids_and_vendor_selection_are_trimmed():
    settings = Settings.from_mapping(
        {
            "tts_provider_id": " voice ",
            "optimizer_provider_id": " chat ",
            "audio_tag_provider": " eleven_labs_enhanced ",
        }
    )
    assert settings.tts_provider_id == "voice"
    assert settings.optimizer_provider_id == "chat"
    assert settings.audio_tag_provider == "eleven_labs_enhanced"


def test_custom_audio_prompt_preserves_all_original_whitespace_and_markup():
    prompt = " \nA & B </audio_tag_optimization><![CDATA[<voice>]]>\n "
    settings = Settings.from_mapping(
        {"audio_tag_provider": " custom ", "audio_tag_custom_prompt": prompt}
    )
    assert settings.audio_tag_provider == "custom"
    assert settings.audio_tag_custom_prompt == prompt


@pytest.mark.parametrize("value", [1, 500])
def test_length_target_accepts_the_full_configured_range(value):
    assert Settings.from_mapping({"target_max_chars": value}).target_max_chars == value


def test_existing_elevenlabs_selection_migrates_to_auto():
    assert (
        Settings.from_mapping({"audio_tag_provider": " elevenlabs "}).audio_tag_provider
        == "auto"
    )


@pytest.mark.parametrize(
    ("name", "invalid"),
    [
        ("optimization_enabled", 1),
        ("length_optimization_enabled", 1),
        ("content_optimization_enabled", 1),
        ("audio_tag_optimization_enabled", 1),
        ("persona_injection_enabled", 1),
        ("optimization_enabled", 0),
        ("optimization_enabled", "true"),
        ("optimization_enabled", None),
    ],
)
def test_optimization_switches_require_actual_booleans(name, invalid):
    with pytest.raises(ConfigurationError):
        Settings.from_mapping({name: invalid})


@pytest.mark.parametrize(
    ("config", "enabled"),
    [
        ({}, True),
        ({"optimization_enabled": False}, False),
        ({"length_optimization_enabled": False}, False),
        (
            {
                "length_optimization_enabled": False,
                "content_optimization_enabled": True,
            },
            True,
        ),
        (
            {
                "length_optimization_enabled": False,
                "audio_tag_optimization_enabled": True,
            },
            True,
        ),
        (
            {"optimization_enabled": False, "content_optimization_enabled": True},
            False,
        ),
    ],
)
def test_master_switch_and_actual_editing_steps_determine_whether_to_optimize(
    config, enabled
):
    assert Settings.from_mapping(config).has_optimization_steps is enabled


def test_existing_length_configuration_migrates_to_target_max_chars():
    assert Settings.from_mapping({"max_chars": 250}).target_max_chars == 250
    assert (
        Settings.from_mapping({"max_chars": 250, "target_chars": 100}).target_max_chars
        == 100
    )
    assert (
        Settings.from_mapping(
            {"max_chars": "invalid", "target_chars": "invalid", "target_max_chars": 120}
        ).target_max_chars
        == 120
    )


def test_invalid_configuration_does_not_echo_secret_values():
    secret = "sk-private-test-secret"
    with pytest.raises(ConfigurationError) as error:
        Settings.from_mapping({"audio_tag_provider": secret})
    assert secret not in str(error.value)
