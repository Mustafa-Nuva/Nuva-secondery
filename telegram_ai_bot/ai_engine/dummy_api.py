from __future__ import annotations

import random
import re

from ai_engine.base import BaseAIEngine


class DummyAI(BaseAIEngine):
    """A dynamic dummy AI engine.

    It returns short, polite Kurdish (Sorani) replies selected randomly.
    """

    def generate_reply(self, user_text: str) -> str:
        text = (user_text or "").strip()

        if not text:
            return "تکایە شتێک بنووسە تا یارمەتیت بدەم."

        lang = _detect_language(text)
        kind = _classify_intent(text)

        if lang == "en":
            return _reply_en(kind)
        if lang == "ar":
            return _reply_ar(kind)
        return _reply_ku(kind)


DummyAIEngine = DummyAI


_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _detect_language(text: str) -> str:
    if _ARABIC_RE.search(text):
        return "ar"
    if _LATIN_RE.search(text):
        return "en"
    return "ku"


def _classify_intent(text: str) -> str:
    t = text.strip().lower()
    if any(w in t for w in ("hi", "hello", "hey")) or "سڵاو" in text:
        return "greeting"
    if "?" in text or "؟" in text:
        return "question"
    if len(t) <= 3:
        return "unclear"
    return "general"


def _reply_en(kind: str) -> str:
    if kind == "greeting":
        return random.choice(
            [
                "Hi! How can I help you today?",
                "Hello. What would you like to know?",
            ]
        )
    if kind == "question":
        return "I can help—could you share a bit more detail so I answer accurately?"
    if kind == "unclear":
        return "Could you clarify what you mean?"
    return random.choice(
        [
            "Got it. What’s the main goal you want to achieve?",
            "I’m here to help—tell me what you need.",
        ]
    )


def _reply_ar(kind: str) -> str:
    if kind == "greeting":
        return random.choice(
            [
                "مرحباً. كيف يمكنني مساعدتك؟",
                "أهلاً! ماذا تريد أن تعرف؟",
            ]
        )
    if kind == "question":
        return "أقدر أن أساعدك—هل يمكنك توضيح سؤالك أكثر حتى أجيب بدقة؟"
    if kind == "unclear":
        return "هل يمكنك التوضيح أكثر؟"
    return random.choice(
        [
            "تمام. ما الذي تحتاجه بالضبط؟",
            "أنا جاهز للمساعدة—قل لي ما الموضوع.",
        ]
    )


def _reply_ku(kind: str) -> str:
    if kind == "greeting":
        return random.choice(
            [
                "سڵاو. چۆن دەتوانم یارمەتیت بدەم؟",
                "سڵاو! چی دەتەوێت بزانیت؟",
            ]
        )
    if kind == "question":
        return "دەکرێت زیاتر ڕوون بکەیت؟ بەم شێوەیە وەڵامت بە دروستی دەدەم."
    if kind == "unclear":
        return "تکایە زیاتر ڕوون بکەیت."
    return random.choice(
        [
            "باشە. تکایە بە کورتی بڵێ چی پێویستتە.",
            "من ئامادەم یارمەتیت بدەم—بڵێ کێشەکەت چییە.",
        ]
    )
