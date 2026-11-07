"""Read the active session's persona using AstrBot's own selection rules."""

from collections.abc import Mapping

from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context


async def resolve_persona_prompt(context: Context, event: AstrMessageEvent) -> str:
    """Resolve session rules, conversation selection and defaults without writes.

    The tuple/Personality contract is from AstrBot 4.28.0's PersonaManager.
    A missing persona is distinct from the platform's built-in WebChat persona.
    """
    umo = event.unified_msg_origin
    manager = context.conversation_manager
    conversation_id = await manager.get_curr_conversation_id(umo)
    conversation = (
        await manager.get_conversation(umo, conversation_id, create_if_not_exists=False)
        if conversation_id
        else None
    )
    (
        _,
        persona,
        _,
        webchat_default,
    ) = await context.persona_manager.resolve_selected_persona(
        umo=umo,
        conversation_persona_id=(
            conversation.persona_id if conversation is not None else None
        ),
        platform_name=event.get_platform_name(),
    )
    if isinstance(persona, Mapping):
        prompt = persona.get("prompt", "")
        return prompt if isinstance(prompt, str) else ""
    if webchat_default and event.get_extra("enable_default_system_prompt") is not False:
        # AstrBot's WebChat default is a dedicated constant rather than a saved
        # PersonaManager entry. Match the same opt-out used by the main agent.
        from astrbot.core.astr_main_agent_resources import (
            CHATUI_SPECIAL_DEFAULT_PERSONA_PROMPT,
        )

        return CHATUI_SPECIAL_DEFAULT_PERSONA_PROMPT
    return ""
