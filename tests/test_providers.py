import asyncio

import pytest
from smart_tts.providers import resolve_optimizer_provider_id, resolve_tts_provider
from smart_tts.settings import ConfigurationError


def test_tts_resolves_exact_selected_provider(rig):
    assert resolve_tts_provider(rig.context, "voice") is rig.tts
    assert rig.context.lookups == ["voice"]


@pytest.mark.parametrize("provider_id", ["", "missing", "chat"])
def test_tts_missing_or_wrong_type_is_configuration_error(rig, provider_id):
    with pytest.raises(ConfigurationError):
        resolve_tts_provider(rig.context, provider_id)


def test_optimizer_selection_overrides_current_chat(rig):
    result = asyncio.run(
        resolve_optimizer_provider_id(
            rig.context, rig.event.unified_msg_origin, "alternate"
        )
    )
    assert result == "alternate"
    assert rig.context.chat_resolutions == []


def test_optimizer_resolves_current_chat_afresh_for_each_request(rig):
    async def scenario():
        first = await resolve_optimizer_provider_id(rig.context, "one", "")
        rig.context.current_chat = "alternate"
        second = await resolve_optimizer_provider_id(rig.context, "two", "")
        return first, second

    assert asyncio.run(scenario()) == ("chat", "alternate")
    assert rig.context.chat_resolutions == ["one", "two"]


@pytest.mark.parametrize("provider_id", ["missing", "voice"])
def test_optimizer_missing_or_wrong_type_is_configuration_error(rig, provider_id):
    with pytest.raises(ConfigurationError):
        asyncio.run(resolve_optimizer_provider_id(rig.context, "one", provider_id))


def test_missing_current_optimizer_is_configuration_error(rig):
    rig.context.current_chat = "missing"
    with pytest.raises(ConfigurationError):
        asyncio.run(resolve_optimizer_provider_id(rig.context, "one", ""))


def test_provider_exception_does_not_expose_credentials(rig):
    secret = "sk-private-test-secret"

    async def fail(umo):
        raise RuntimeError(f"https://provider.invalid/?key={secret}")

    rig.context.get_current_chat_provider_id = fail
    with pytest.raises(ConfigurationError) as error:
        asyncio.run(resolve_optimizer_provider_id(rig.context, "one", ""))
    assert secret not in str(error.value)
