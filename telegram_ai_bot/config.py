import os

from dotenv import load_dotenv

load_dotenv()


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()


AI_ENGINE = os.getenv("AI_ENGINE", "gemini").strip().lower()


def get_ai_engine():
    """Create and return the configured AI engine.

    To switch engines later, change only this file (or the AI_ENGINE env var).
    """

    if AI_ENGINE == "dummy":
        from ai_engine.dummy_api import DummyAI

        return DummyAI()

    if AI_ENGINE == "gemini":
        from ai_engine.gemini_api import GeminiAIEngine

        return GeminiAIEngine(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)

    raise ValueError(f"Unknown AI_ENGINE: {AI_ENGINE}")
