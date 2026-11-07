"""Audio-tag rules informed by ElevenLabs' best practices.

https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices
"""

from copy import deepcopy
from xml.etree import ElementTree

from ..errors import ConfigurationError
from .base import AudioTagProvider


def _build_legacy_guidance() -> ElementTree.Element:
    root = ElementTree.Element("audio_tag_optimization")
    ElementTree.SubElement(root, "legacy_pause_rules").text = (
        "标点已能表达自然节奏时，保留文稿即可。只有确有需要的停顿位置才稀疏加入 break，"
        "优先选择 0.2 秒至 0.5 秒这样的短停顿，避免连续插入或每句都加。"
        "时间必须大于 0 秒且不超过 3 秒。3 秒是技术上限，不是推荐时长；"
        "停顿过长会显得不自然。"
        "本规则只指导停顿标记，不能借此删改台词；"
        "已启用的字数或内容优化仍按各自准则处理并保留语义。"
        "不添加方括号音频标签或其他 SSML。"
    )
    examples = ElementTree.SubElement(root, "legacy_examples")
    for category, source, positive, incorrect, reason in (
        (
            "punctuation_only",
            "晚安，祝你睡个好觉。",
            "晚安，祝你睡个好觉。",
            '晚安，<break time="3s" />祝你睡个好觉。',
            "普通标点已经足够，额外的长停顿会打断这句简短祝福。",
        ),
        (
            "brief_thinking_pause",
            "让我想一想。嗯，可以这样做。",
            '让我想一想。<break time="0.3s" />嗯，可以这样做。',
            '让我<break time="2s" />想一想。<break time="3s" />'
            '嗯，<break time="2s" />可以这样做。',
            "需要表现短暂思考时，一个短停顿就够了，不要拆碎词句或堆叠停顿。",
        ),
    ):
        example = ElementTree.SubElement(examples, "legacy_example", category=category)
        for name, text in (
            ("source", source),
            ("positive", positive),
            ("incorrect", incorrect),
            ("reason", reason),
        ):
            ElementTree.SubElement(example, name).text = text
    return root


def _build_enhanced_guidance() -> ElementTree.Element:
    root = ElementTree.Element("audio_tag_optimization")
    rules = ElementTree.SubElement(root, "enhanced_rules")
    for name, text in (
        (
            "purpose",
            "让标签自然地补充文稿的人声表现。标签可有可无，所有规则同时适用。",
        ),
        (
            "text_and_intent",
            "标签本身不能删改词句、替换叙述、添加新台词或扭转意图，"
            "也不能把原有文字装进方括号。"
            "另外启用的字数和内容优化可按各自准则修改文稿，并保持语义。"
            "不要通过标签增加原文没有的敏感内容或虚构细节。",
        ),
        (
            "vocal_tags",
            "根据每行的语境、情绪和音色选择标签，只描述人声的情绪、语调或非语言发声。"
            "新增标签用方括号包围，优先参考 audio_tag_catalog 中的官方人声标签。"
            "确实无法用目录中的标签表达当前声音要求时，才使用清晰的自定义人声描述。"
            "无需强行添加标签。不使用视觉动作、音乐或环境音效标签。"
            "自定义描述的实际效果取决于模型和音色。",
        ),
        (
            "placement_and_emphasis",
            "标签放在相关话语之前、之后或自然停顿处。"
            "普通标点也能形成自然节奏，无需强行增加标签。"
            "标签表达需要强调时，可适度调整字母大小写、问号、叹号或省略号，不改变含义。",
        ),
        (
            "compatibility",
            "v3/v4 使用音频标签，不支持 SSML；不要添加 break 或其他 XML 语音标记。",
        ),
    ):
        ElementTree.SubElement(rules, name).text = text

    catalog = ElementTree.SubElement(root, "audio_tag_catalog")
    for category, tags in (
        (
            "directions",
            (
                "happy",
                "sad",
                "excited",
                "angry",
                "whisper",
                "annoyed",
                "appalled",
                "thoughtful",
                "surprised",
                "sarcastic",
                "curious",
                "mischievously",
            ),
        ),
        (
            "non_verbal",
            (
                "laughing",
                "chuckles",
                "sighs",
                "clears throat",
                "short pause",
                "long pause",
                "exhales sharply",
                "inhales deeply",
                "laughs harder",
                "wheezing",
                "exhales",
                "crying",
                "snorts",
            ),
        ),
    ):
        group = ElementTree.SubElement(catalog, "group", category=category)
        for tag in tags:
            ElementTree.SubElement(group, "tag").text = f"[{tag}]"

    examples = ElementTree.SubElement(root, "enhanced_examples")
    for category, source, positive, incorrect, reason in (
        (
            "narration_preserved",
            "她叹了口气，说还想再试一次。",
            "她叹了口气，[thoughtful]说还想再试一次。",
            "[sighs]她说还想再试一次。",
            "不能用发声标签替换原有叙述；示例中的正例保留了叙述文字。",
        ),
        (
            "dialogue_outside_brackets",
            "谢谢你一直在等我。",
            "[whisper]谢谢你一直在等我。",
            "[谢谢你一直在等我。]",
            "方括号用于声音指令，台词本身应留在括号外。",
        ),
        (
            "audible_delivery",
            "我轻声告诉你，已经没事了。",
            "[whisper]我轻声告诉你，已经没事了。",
            "[nodding]我轻声告诉你，已经没事了。",
            "耳语是可听见的人声表现；点头是视觉动作。",
        ),
        (
            "custom_voice_description",
            "这件事我想清楚了，我们可以慢慢来。",
            "[warm, measured voice]这件事我想清楚了，我们可以慢慢来。",
            "[warm sunset]这件事我想清楚了，我们可以慢慢来。",
            "自定义标签可以描述温暖、从容的声音，但不能只描述画面或氛围。",
        ),
        (
            "voice_only",
            "休息一下吧，明天再继续。",
            "[gentle, quiet voice]休息一下吧，明天再继续。",
            "[music]休息一下吧，明天再继续。[rain]",
            "音乐和雨声不是人声表现，不能作为这里的标签。",
        ),
        (
            "nonverbal_voice",
            "总算忙完了，终于可以休息了。",
            "[sighs]总算忙完了，终于可以休息了。",
            "[door closing]总算忙完了，终于可以休息了。",
            "松一口气是人声发声；关门声是环境音效，与这里的人声标签范围不符。",
        ),
        (
            "intent_preserved",
            "这个消息让我很难过。",
            "[sad]这个消息让我很难过。",
            "[excited]这个消息让我很难过。",
            "标签应与原有情绪一致，不能把难过演绎成兴奋。",
        ),
        (
            "plain_delivery",
            "明天下午三点见。",
            "明天下午三点见。",
            "[excited][laughing][whisper]明天下午三点见。[sighs]",
            "平实的约定可以不加标签；没有语境依据时不要堆叠多种情绪。",
        ),
        (
            "punctuation_and_emphasis",
            "真的吗，我没想到会这样。",
            "[surprised]真的吗？我没想到会这样…",
            '<break time="1s" />真的吗，我没想到会这样。',
            "问号和省略号可以辅助强调，增强模式不使用 SSML break。",
        ),
    ):
        example = ElementTree.SubElement(
            examples, "enhanced_example", category=category
        )
        for name, text in (
            ("source", source),
            ("positive", positive),
            ("incorrect", incorrect),
            ("reason", reason),
        ):
            ElementTree.SubElement(example, name).text = text
    return root


ELEVEN_LABS_LEGACY = _build_legacy_guidance()
ELEVEN_LABS_ENHANCED = _build_enhanced_guidance()

_STANDARD_MODELS = frozenset(
    {
        "eleven_multilingual_v2",
        "eleven_flash_v2",
        "eleven_flash_v2_5",
        "eleven_turbo_v2",
        "eleven_turbo_v2_5",
    }
)
_EXPRESSIVE_MODELS = frozenset(
    {"eleven_v3", "eleven_v3_conversational", "eleven_v4", "eleven_v4_turbo"}
)


class ElevenLabsAudioTagProvider(AudioTagProvider):
    aliases = {"elevenlabs": "auto"}

    def get_modes(self) -> tuple[str, ...]:
        return ("eleven_labs_legacy", "eleven_labs_enhanced")

    def detect_mode(self, provider_type: str, model: str) -> str | None:
        if provider_type != "elevenlabs_tts_api" or not isinstance(model, str):
            return None
        model = model.strip().lower()
        if model in _EXPRESSIVE_MODELS:
            return "eleven_labs_enhanced"
        if model in _STANDARD_MODELS:
            return "eleven_labs_legacy"
        return None

    def get_guidance(self, mode: str) -> ElementTree.Element:
        if mode == "eleven_labs_legacy":
            return deepcopy(ELEVEN_LABS_LEGACY)
        if mode == "eleven_labs_enhanced":
            return deepcopy(ELEVEN_LABS_ENHANCED)
        raise ConfigurationError("请选择有效的音频标签模式。")
