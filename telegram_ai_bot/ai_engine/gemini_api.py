from __future__ import annotations

from typing import Optional

import google.generativeai as genai
from google.api_core.exceptions import NotFound

from ai_engine.base import BaseAIEngine


_SYSTEM_INSTRUCTIONS = (
    "You are an AI assistant for a Kurdish Telegram community. "
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
    "Role: Your current role is a helpful general AI assistant for this Kurdish community. "
    "You can answer questions about technology, education, daily life, and general information. "
    "Tone & style: Friendly, respectful, and modern. Use natural, human-like language, not robotic. "
    "Keep answers short and to the point, unless a longer explanation is really needed. "
    "In Sorani, use clear standard writing, not too much slang. "
    "Behavior details: When the user asks for help, first understand the goal, then give practical steps. "
    "If there are risks or limitations, mention them briefly. "
    "For complex questions, structure your answer with short sections or bullet points. "
    "Always adapt your answer to the user's language and their level: if they seem beginner, explain more simply. "
    "IMPORTANT: You are NOT a replacement for a real doctor, lawyer, or certified professional. "
    "For medical or legal issues, you can give general information only and advise them to see a professional. "
    "Your task: Read the user's message, detect the user's language and reply in the same language, "
    "decide what they want, and answer in a helpful, original way that matches the role above."
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

        response = self._model.generate_content(text)
        reply: Optional[str] = getattr(response, "text", None)
        return (reply or "").strip()
