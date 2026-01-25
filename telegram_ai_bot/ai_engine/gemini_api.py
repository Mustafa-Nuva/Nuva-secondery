from __future__ import annotations

from typing import Optional

import google.generativeai as genai
from google.api_core.exceptions import NotFound

from ai_engine.base import BaseAIEngine


_SYSTEM_INSTRUCTIONS = (
    "You are a MEDICAL-ONLY AI assistant for a Kurdish Telegram community. "
    "Your ONLY job is to provide general medical and health information. "
    "Do NOT answer questions about programming, technology, school, business, or any non-medical topic. "
    "If the user asks about something non-medical, you must NOT answer the question. Instead, reply briefly that you are only a medical assistant and cannot help with non-medical topics. "
    "Base rules: "
    "Default language: Kurdish (Sorani). "
    "If the user writes in Kurdish, reply in Kurdish. "
    "If the user writes in English, reply in English. "
    "If the user writes in Arabic, reply in Arabic. "
    "If the user writes in any other language, reply in that same language as much as you can. "
    "Never repeat or echo the user's message. "
    "Be clear, friendly, and professional. "
    "If the question is unclear, politely ask for clarification. "
    "If you are not sure, say you are not sure instead of guessing. "
    "Do not give harmful, illegal, or unsafe instructions. "
    "ROLE: You are a medical information assistant, NOT a doctor. "
    "You can explain symptoms, conditions, lifestyle advice, and how treatments generally work, but only as general information. "
    "First, provide clear general information and practical, safe advice. Then remind the user that they should see a qualified healthcare professional. Do not reply only with 'go to the doctor' or similar without any explanation. "
    "Never make a definitive diagnosis. Never prescribe or adjust medication. Never tell the user to stop or change medicine that a doctor has given. "
    "If there is any sign of emergency (for example chest pain, difficulty breathing, signs of stroke, severe injury, suicidal thoughts, or anything very serious), "
    "you MUST tell the user clearly to seek immediate emergency medical help or contact local emergency services. "
    "Tone & style: Friendly, respectful, calm, and reassuring. Use natural, human-like language, not robotic. "
    "Keep answers focused and not too long. Aim for medium-length answers, enough to be clear but not very long essays. "
    "In Sorani, use clear standard writing, not too much slang. "
    "Behavior details: When the user asks for help, first understand their main medical concern, then give simple explanations and practical, safe general advice. "
    "If there are risks, uncertainty, or red flags, mention them and strongly recommend seeing a doctor. "
    "For more complex questions, you can use short sections or bullet points, but do not make the answer very long. "
    "Always adapt your answer to the user's language and level: if they seem beginner, explain more simply. "
    "IMPORTANT: You are NOT a replacement for a real doctor or emergency service. You only give general medical information and always advise users to consult healthcare professionals. "
    "Interaction mode: Assume the user only sends text chat messages. Ignore or politely refuse any request to handle images, voice messages, or other file uploads. "
    "Your task: Read the user's message, detect their language and reply in the same language, focus ONLY on medical and health topics, "
    "and give safe, clear, medium-length answers that follow all rules above."
)


class GeminiAIEngine(BaseAIEngine):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        api_key = (api_key or "").strip()
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY")

        genai.configure(api_key=api_key)
        self._model_name = model
        self._model = self._create_model_with_fallback(model)

    def _create_model_with_fallback(self, model: str) -> genai.GenerativeModel:
        candidates = []

        m = (model or "").strip()
        if m:
            candidates.append(m)
            if not m.startswith("models/"):
                candidates.append(f"models/{m}")

        candidates.extend(
            [
                "gemini-1.5-flash-latest",
                "models/gemini-1.5-flash-latest",
                "gemini-1.5-pro-latest",
                "models/gemini-1.5-pro-latest",
                "gemini-pro",
                "models/gemini-pro",
            ]
        )

        last_err: Optional[Exception] = None
        for candidate in candidates:
            try:
                test_model = genai.GenerativeModel(
                    model_name=candidate,
                    system_instruction=_SYSTEM_INSTRUCTIONS,
                )
                test_model.generate_content("ping")
                self._model_name = candidate
                return test_model
            except NotFound as e:
                last_err = e
                continue

        raise RuntimeError(
            "No supported Gemini model found for generateContent. "
            "Try setting GEMINI_MODEL to one of: gemini-1.5-flash-latest, gemini-1.5-pro-latest, gemini-pro"
        ) from last_err

    def generate_reply(self, user_text: str) -> str:
        text = (user_text or "").strip()
        if not text:
            return ""

        def _detect_language(s: str) -> str:
            s = s.strip()
            has_arabic = any("\u0600" <= ch <= "\u06FF" for ch in s)
            has_latin = any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in s)

            if has_latin and not has_arabic:
                return "en"

            if has_arabic:
                kurdish_chars = "پچژگڵڕڤێۆ"
                if any(ch in kurdish_chars for ch in s):
                    return "ku"
                return "ar"

            return "en"

        def _is_complicated_question(s: str) -> bool:
            lower = s.lower()
            keywords = [
                "explain",
                "step by step",
                "treatment",
                "management",
                "what should i do",
                "recommendation",
                "recommendations",
                "dose",
                "dosage",
                "diagnosis",
                "diagnose",
            ]
            if any(k in lower for k in keywords):
                return True

            word_count = len(s.split())
            if word_count > 40:
                return True

            if s.count("?") > 1:
                return True

            return False

        def _needs_disclaimer(s: str) -> bool:
            lower = s.lower()
            keywords = [
                "what should i do",
                "should i",
                "take this",
                "take it",
                "take the medicine",
                "stop the medicine",
                "start the medicine",
                "treatment",
                "management",
                "dose",
                "dosage",
                "diagnosis",
                "diagnose",
                "recommend",
                "recommendation",
            ]
            if any(k in lower for k in keywords):
                return True

            return False

        def _get_disclaimer(lang: str) -> str:
            if lang == "ku":
                return (
                    "من یارمەتیدەری پزیشکییەکی زیرەکەم، ئەم زانیارییە جێگرەوەی ڕاوێژی پزیشک یان دەرمانساز نییە. "
                    "تکایە بۆ بڕیارە پزیشکییە تایبەتییەکان ڕاوێژی پزیشک یان دەرمانساز بکە."
                )
            if lang == "ar":
                return (
                    "أنا مساعد طبي يعتمد على الذكاء الاصطناعي، وهذه المعلومات لا تُغني عن استشارة طبيب أو صيدلي مختص. "
                    "يُرجى مراجعة مختص صحي لاتخاذ قرارات طبية شخصية."
                )
            return (
                "I am a medical AI assistant, and this information does not replace advice from a real doctor or pharmacist. "
                "Please consult a healthcare professional for personal medical decisions."
            )

        lang = _detect_language(text)

        complicated = _is_complicated_question(text)
        max_tokens = 500 if complicated else 250

        response = self._model.generate_content(
            text,
            generation_config={
                "max_output_tokens": max_tokens,
            },
        )
        reply: Optional[str] = getattr(response, "text", None)
        final_reply = (reply or "").strip()

        if _needs_disclaimer(text) and final_reply:
            disclaimer = _get_disclaimer(lang)
            final_reply = f"{final_reply}\n\n{disclaimer}".strip()

        return final_reply
