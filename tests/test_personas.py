"""Exercise the session-scoped contract with AstrBot's persona resolver."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from smart_tts.personas import resolve_persona_prompt


def make_context(persona_id="friendly", persona=None, *, webchat_default=False):
    conversation = SimpleNamespace(persona_id=persona_id)
    manager = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value="conversation-one"),
        get_conversation=AsyncMock(return_value=conversation),
    )
    resolver = AsyncMock(return_value=(persona_id, persona, None, webchat_default))
    context = SimpleNamespace(
        conversation_manager=manager,
        persona_manager=SimpleNamespace(resolve_selected_persona=resolver),
    )
    event = SimpleNamespace(
        unified_msg_origin="qq:FriendMessage:one",
        get_platform_name=lambda: "aiocqhttp",
        get_extra=lambda key: None,
    )
    return context, event


def test_reads_current_conversation_and_returns_persona_prompt_unchanged():
    prompt = "  温柔自然 & 保留 <人物> 个性。\n"
    context, event = make_context(persona={"prompt": prompt})
    assert asyncio.run(resolve_persona_prompt(context, event)) == prompt
    context.conversation_manager.get_curr_conversation_id.assert_awaited_once_with(
        event.unified_msg_origin
    )
    context.conversation_manager.get_conversation.assert_awaited_once_with(
        event.unified_msg_origin, "conversation-one", create_if_not_exists=False
    )
    context.persona_manager.resolve_selected_persona.assert_awaited_once_with(
        umo=event.unified_msg_origin,
        conversation_persona_id="friendly",
        platform_name="aiocqhttp",
    )


@pytest.mark.parametrize("persona_id", [None, "", "[%None]"])
def test_selection_sentinels_are_forwarded_without_default_substitution(persona_id):
    context, event = make_context(persona_id=persona_id)
    assert asyncio.run(resolve_persona_prompt(context, event)) == ""
    arguments = context.persona_manager.resolve_selected_persona.await_args.kwargs
    assert arguments["conversation_persona_id"] == persona_id


def test_without_a_current_conversation_delegates_defaults_without_creating_one():
    context, event = make_context(persona={"prompt": "默认人格。"})
    context.conversation_manager.get_curr_conversation_id.return_value = None
    assert asyncio.run(resolve_persona_prompt(context, event)) == "默认人格。"
    context.conversation_manager.get_conversation.assert_not_awaited()
    args = context.persona_manager.resolve_selected_persona.await_args.kwargs
    assert args["conversation_persona_id"] is None


def test_missing_conversation_delegates_the_none_selection():
    context, event = make_context(persona={"prompt": "默认人格。"})
    context.conversation_manager.get_conversation.return_value = None
    assert asyncio.run(resolve_persona_prompt(context, event)) == "默认人格。"
    args = context.persona_manager.resolve_selected_persona.await_args.kwargs
    assert args["conversation_persona_id"] is None


def test_resolved_forced_persona_takes_precedence_over_conversation_choice():
    context, event = make_context(persona_id="conversation-persona")
    context.persona_manager.resolve_selected_persona.return_value = (
        "forced-persona",
        {"prompt": "会话规则选中的人格。"},
        "forced-persona",
        False,
    )
    assert asyncio.run(resolve_persona_prompt(context, event)) == "会话规则选中的人格。"


def test_personas_are_resolved_afresh_per_session_and_conversation():
    context, event = make_context()

    async def resolve(**kwargs):
        return None, {"prompt": kwargs["umo"]}, None, False

    context.persona_manager.resolve_selected_persona.side_effect = resolve
    other = SimpleNamespace(**vars(event))
    other.unified_msg_origin = "qq:FriendMessage:two"

    async def scenario():
        return await asyncio.gather(
            resolve_persona_prompt(context, event),
            resolve_persona_prompt(context, other),
        )

    assert asyncio.run(scenario()) == [
        event.unified_msg_origin,
        other.unified_msg_origin,
    ]
    assert context.conversation_manager.get_curr_conversation_id.await_count == 2


@pytest.mark.parametrize("persona", [{}, {"prompt": None}, {"prompt": 123}])
def test_empty_or_invalid_persona_body_is_not_stringified(persona):
    context, event = make_context(persona=persona)
    assert asyncio.run(resolve_persona_prompt(context, event)) == ""


@pytest.mark.parametrize("enabled", [None, True, False])
def test_webchat_builtin_persona_honors_default_prompt_opt_out(monkeypatch, enabled):
    module = ModuleType("astrbot.core.astr_main_agent_resources")
    module.CHATUI_SPECIAL_DEFAULT_PERSONA_PROMPT = "WebChat 的专用人格。"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    context, event = make_context(webchat_default=True)
    event.get_platform_name = lambda: "webchat"
    event.get_extra = lambda key: (
        enabled if key == "enable_default_system_prompt" else None
    )
    expected = "" if enabled is False else module.CHATUI_SPECIAL_DEFAULT_PERSONA_PROMPT
    assert asyncio.run(resolve_persona_prompt(context, event)) == expected


def test_persona_lookup_errors_propagate_to_the_callers_fallback():
    context, event = make_context()
    context.conversation_manager.get_conversation.side_effect = RuntimeError("db down")
    with pytest.raises(RuntimeError, match="db down"):
        asyncio.run(resolve_persona_prompt(context, event))
    context.persona_manager.resolve_selected_persona.assert_not_awaited()
