"""Let the conversation model choose when to send a spoken reply."""

import asyncio
from pathlib import Path

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

from .audio_tags import resolve_audio_tag_guidance
from .optimizer import optimize_text
from .personas import resolve_persona_prompt
from .providers import resolve_optimizer_provider_id, resolve_tts_provider
from .settings import ConfigurationError, Settings

_ATTEMPT_KEY = "astrbot_plugin_smart_tts.attempt"
_NO_RETRY = "请不要再次调用 send_voice_message，改用文字回复用户。"


class SmartTTSPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._active_tasks: set[asyncio.Task] = set()
        self._closing = False

    async def initialize(self) -> None:
        """Report configuration issues without preventing users from configuring it."""
        try:
            settings = Settings.from_mapping(self.config)
            if not settings.tts_provider_id:
                logger.warning(
                    "[Smart TTS] 请在插件配置中选择一个已启用的 TTS 供应商。"
                )
        except ConfigurationError as exc:
            logger.warning(f"[Smart TTS] {exc}")

    @filter.llm_tool(name="send_voice_message")
    async def send_voice_message(
        self, event: AstrMessageEvent, text: str
    ) -> str | None:
        """向当前会话发送语音消息，发送完成后结束本轮回复。

        用户要求语音，或简短闲聊、安慰、祝福等适合用声音表达时使用。
        用户要求文字时不要使用；代码、链接、表格和需精确查阅的长内容用文字。
        先完成其他查询，再将最终回复交给本工具，每轮最多调用一次。
        决定使用时直接调用，不要先输出相同文字或“我给你发语音”等前言。
        传入准备说给用户听的完整原文，语音合成失败时插件会将其作为文字消息发送。

        Args:
            text(string): 要说给用户听的完整自然语言回复。
        """
        # Keep the task on the event, not on the session: parallel calls for the
        # same incoming message share one attempt, while later messages can speak.
        attempt = event.get_extra(_ATTEMPT_KEY)
        if isinstance(attempt, str):
            return attempt
        if isinstance(attempt, asyncio.Task):
            if attempt.cancelled():
                return "本轮语音任务已取消。" + _NO_RETRY
            return await attempt
        if self._closing:
            return self._reject(event, "语音插件正在停用。")
        if not isinstance(text, str) or not text.strip():
            return self._reject(event, "语音内容不能为空。")
        try:
            settings = Settings.from_mapping(self.config)
        except ConfigurationError as exc:
            return self._reject(event, f"语音配置错误：{exc}")

        # No queue: the core tool timeout must not be spent waiting behind other
        # conversations. This check and task registration have no await between.
        active_count = sum(not task.done() for task in self._active_tasks)
        if active_count >= settings.max_concurrent_requests:
            return self._reject(event, "语音服务当前繁忙。")
        task = asyncio.create_task(self._speak(event, text, settings))
        event.set_extra(_ATTEMPT_KEY, task)
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        # Cancellation deliberately propagates to the shared task. A core timeout
        # or plugin unload must not leave an orphan task sending audio later.
        return await task

    @staticmethod
    def _reject(event: AstrMessageEvent, reason: str) -> str:
        """Even a rejected attempt is final for this incoming message."""
        result = reason + _NO_RETRY
        event.set_extra(_ATTEMPT_KEY, result)
        return result

    async def _speak(
        self, event: AstrMessageEvent, text: str, settings: Settings
    ) -> str | None:
        stage = "供应商选择"
        try:
            tts = resolve_tts_provider(self.context, settings.tts_provider_id)
        except Exception as exc:
            self._log_failure(stage, exc)
            return await self._send_original(event, text, settings)

        tts_text = text
        if settings.has_optimization_steps:
            try:
                async with asyncio.timeout(settings.optimizer_timeout):
                    tts_text = await self._optimize(event, text, tts, settings)
            except Exception as exc:
                self._log_failure("文本优化", exc)
                tts_text = text

        try:
            async with asyncio.timeout(settings.tts_timeout):
                audio_path = await tts.get_audio(tts_text)
                record = await asyncio.to_thread(self._audio_record, audio_path, text)
        except Exception as exc:
            self._log_failure("语音合成", exc)
            return await self._send_original(event, text, settings)

        try:
            async with asyncio.timeout(settings.send_timeout):
                await event.send(event.chain_result([record]))
            # The provider owns its output file. Do not delete it while an
            # adapter or platform may still be reading it.
            # AstrBot's built-in runner interprets None as a direct tool reply
            # and terminates the loop, preserving the original tool call history.
            return None
        except Exception as exc:
            self._log_failure("语音发送", exc)
            # A timeout can occur after the platform has accepted the audio.
            # Cache this result too; never retry the side effect blindly.
            return "语音发送未能确认完成，可能已送达。" + _NO_RETRY

    async def _optimize(self, event, text, tts, settings: Settings) -> str:
        """Resolve only the contextual inputs needed by the selected edits."""
        audio_guidance = None
        if settings.audio_tag_optimization_enabled:
            try:
                provider_type = ""
                model = ""
                if settings.audio_tag_provider == "auto":
                    provider_type = tts.meta().type
                    model = tts.get_model() or ""
                audio_guidance = resolve_audio_tag_guidance(
                    settings.audio_tag_provider,
                    provider_type,
                    model,
                    custom_prompt=settings.audio_tag_custom_prompt,
                )
            except Exception as exc:
                self._log_failure("音频标签能力读取", exc)

        if not (
            settings.should_optimize_length(text)
            or settings.content_optimization_enabled
            or audio_guidance is not None
        ):
            return text

        persona_prompt = ""
        if settings.persona_injection_enabled:
            try:
                persona_prompt = await resolve_persona_prompt(self.context, event)
            except Exception as exc:
                self._log_failure("人格读取", exc)

        provider_id = await resolve_optimizer_provider_id(
            self.context, event.unified_msg_origin, settings.optimizer_provider_id
        )
        return await optimize_text(
            self.context,
            provider_id,
            text,
            settings,
            audio_guidance=audio_guidance,
            persona_prompt=persona_prompt,
        )

    async def _send_original(
        self, event: AstrMessageEvent, text: str, settings: Settings
    ) -> str | None:
        """Deliver the exact original as the terminal reply after synthesis fails."""
        try:
            async with asyncio.timeout(settings.send_timeout):
                await event.send(event.chain_result([Comp.Plain(text=text)]))
            return None
        except Exception as exc:
            self._log_failure("原文发送", exc)
            return "原文文字消息发送未能确认完成，可能已送达。" + _NO_RETRY

    @staticmethod
    def _audio_record(audio_path: str | Path, original_text: str) -> Comp.Record:
        """Check provider output off the event loop, including path resolution."""
        if not isinstance(audio_path, (str, Path)) or not audio_path:
            raise ValueError("TTS did not return a local audio file")
        audio_file = Path(audio_path)
        if not audio_file.is_file() or audio_file.stat().st_size == 0:
            raise ValueError("TTS returned a missing or empty audio file")
        # Adapters may display Record.text as a caption or text fallback.
        return Comp.Record.fromFileSystem(str(audio_file), text=original_text)

    @staticmethod
    def _log_failure(stage: str, exc: Exception) -> None:
        # Provider errors can include API keys, response bodies and source text.
        # Log the stage and error class, never the raw exception or payload.
        logger.warning(f"[Smart TTS] {stage}失败 ({type(exc).__name__})。")

    async def terminate(self) -> None:
        """Cancel in-flight work on reload/unload without closing shared providers."""
        self._closing = True
        tasks = tuple(self._active_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_tasks.clear()
