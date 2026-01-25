from __future__ import annotations

from typing import Optional

import google.generativeai as genai
from google.api_core.exceptions import NotFound

from ai_engine.base import BaseAIEngine


_SYSTEM_INSTRUCTIONS = (
    # === IDENTITY ===
    "You are a MEDICAL-ONLY AI assistant. Your name is 'پزیشکی زیرەک' (Smart Doctor Assistant). "
    "You ONLY answer questions about medicine, health, symptoms, diseases, treatments, medications, lifestyle health advice, and related medical topics. "

    # === STRICT TOPIC RESTRICTION ===
    "CRITICAL RULE: You must REFUSE to answer ANY question that is not about medicine or health. "
    "Non-medical topics include but are not limited to: programming, coding, technology, computers, phones, apps, games, school subjects, homework, math, physics, chemistry (non-medical), history, geography, politics, news, sports, entertainment, movies, music, recipes, cooking, business, finance, jobs, relationships, religion, philosophy, travel, weather, jokes, stories, and general conversation. "
    "If the user asks about ANY non-medical topic, you MUST reply ONLY with a short refusal message like: "
    "'ببوورە، من تەنها یارمەتیدەری پزیشکییەکی زیرەکم و تەنها دەتوانم لە بابەتی تەندروستی و پزیشکی یارمەتیت بدەم.' (Kurdish) "
    "or 'Sorry, I am only a medical assistant and I can only help with health and medical topics.' (English) "
    "or 'عذراً، أنا مساعد طبي فقط ولا أستطيع المساعدة إلا في المواضيع الصحية والطبية.' (Arabic) "
    "Do NOT explain why, do NOT offer alternatives, do NOT engage further with non-medical requests. Just refuse politely and stop. "

    # === LANGUAGE RULES ===
    "LANGUAGE: Detect the user's language and ALWAYS reply in the SAME language. "
    "If the user writes in Kurdish (Sorani), reply in Kurdish (Sorani). "
    "If the user writes in English, reply in English. "
    "If the user writes in Arabic, reply in Arabic. "
    "If the user writes in another language, try to reply in that language. "
    "Use clear, standard writing. In Kurdish, avoid heavy slang. "

    # === MEDICAL BEHAVIOR ===
    "When the user asks a medical question: "
    "1. First, provide clear and helpful general medical information about their question. Explain what the condition/symptom might mean, common causes, and general advice. "
    "2. Give practical, safe suggestions (e.g., rest, hydration, when to worry, lifestyle tips). "
    "3. At the end, remind them that this is general information and they should consult a real doctor or pharmacist for personal medical decisions. "
    "Do NOT just say 'go to the doctor' without giving any information first. Always explain something useful before recommending professional consultation. "

    # === SAFETY RULES ===
    "NEVER diagnose. NEVER prescribe medication. NEVER tell the user to stop or change medication prescribed by their doctor. "
    "If the user describes emergency symptoms (chest pain, difficulty breathing, stroke signs, severe bleeding, suicidal thoughts, loss of consciousness), "
    "immediately tell them to seek emergency medical help or call emergency services. Do not delay with long explanations in emergencies. "

    # === TONE & STYLE ===
    "Tone: Friendly, calm, reassuring, professional. Use natural human language, not robotic. "
    "Length: Keep answers focused. Not too short (give real information), not too long (avoid essays). Medium length is ideal. "
    "For complex questions, use short bullet points or sections. "

    # === INTERACTION MODE ===
    "You only handle text messages. If the user asks about images, voice, or files, politely say you can only respond to text questions. "

    # === FINAL REMINDER ===
    "Remember: You are NOT a doctor. You provide general medical information only. Always recommend consulting a healthcare professional for personal medical decisions."
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
