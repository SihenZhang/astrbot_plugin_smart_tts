"""Resolve AstrBot-managed providers without changing their shared configuration."""

from astrbot.api.provider import Provider
from astrbot.api.star import Context
from astrbot.core.provider.provider import TTSProvider

from .settings import ConfigurationError


def resolve_tts_provider(context: Context, provider_id: str) -> TTSProvider:
    if not provider_id:
        raise ConfigurationError("请在插件配置中选择 TTS 供应商。")
    provider = context.get_provider_by_id(provider_id)
    if not isinstance(provider, TTSProvider):
        raise ConfigurationError("所选 TTS 供应商不可用，请检查是否已启用且类型正确。")
    return provider


async def resolve_optimizer_provider_id(
    context: Context, umo: str, configured_id: str
) -> str:
    if configured_id:
        provider_id = configured_id
    else:
        try:
            provider_id = await context.get_current_chat_provider_id(umo=umo)
        except Exception as exc:
            raise ConfigurationError(
                "当前会话没有可用的对话 LLM，请配置优化 LLM。"
            ) from exc
    if not isinstance(provider_id, str) or not provider_id:
        raise ConfigurationError("当前会话没有可用的对话 LLM，请配置优化 LLM。")
    provider = context.get_provider_by_id(provider_id)
    if not isinstance(provider, Provider):
        raise ConfigurationError("优化 LLM 不可用，请检查是否已启用且为聊天模型。")
    return provider_id
