import logging
import time

import asyncio

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import config


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text("سڵاو. تکایە نامەیەک بنێرە تا وەڵامت بدەم.")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    if update.message.text is None:
        await update.message.reply_text("ببورە، تەنها نامەی نووسراو دەتوانم وەربگرم.")
        return

    user = update.effective_user
    if not user:
        return

    user_id = user.id
    now = time.time()

    rl = context.application.bot_data.setdefault("rate_limit", {})
    user_rl = rl.setdefault(
        user_id,
        {
            "last_ts": 0.0,
            "hits": [],
            "warned_short": False,
            "warned_window": False,
        },
    )

    # Short-term: max 1 message every 3 seconds
    if now - user_rl["last_ts"] < 3:
        if not user_rl["warned_short"]:
            await update.message.reply_text(
                "تکایە چەند چرکەیەک چاوەڕێ بکە پێش ئەوەی دووبارە نامە بنێری."
            )
            user_rl["warned_short"] = True
        return

    # 10-minute window: max 20 messages
    window_seconds = 600
    window_limit = 20
    user_rl["hits"] = [t for t in user_rl["hits"] if now - t < window_seconds]
    if len(user_rl["hits"]) >= window_limit:
        if not user_rl["warned_window"]:
            await update.message.reply_text(
                "لە ماوەی کەمدا نامەی زۆرت ناردووە. تکایە ھەندێک پشووی بکە پاشان بەردەوام بە."
            )
            user_rl["warned_window"] = True
        return

    # Update rate-limit state
    user_rl["last_ts"] = now
    user_rl["hits"].append(now)
    user_rl["warned_short"] = False
    user_rl["warned_window"] = False

    ai_engine = context.application.bot_data.get("ai_engine")
    if ai_engine is None:
        await update.message.reply_text("ببورە، سیستەم ئامادە نییە.")
        return

    user_text = update.message.text
    try:
        reply = await asyncio.to_thread(ai_engine.generate_reply, user_text)
    except Exception:
        logger.exception("AI engine generate_reply failed")
        await update.message.reply_text(
            "ببورە، کێشەیەک ڕوویدا لە وەڵامدانەوە. تکایە دواتر هەوڵ بدە."
        )
        return

    if not reply:
        reply = "ببورە، ئێستا نەتوانم وەڵام بدەم."

    await update.message.reply_text(reply)


def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN or config.TELEGRAM_BOT_TOKEN == "PASTE_YOUR_TOKEN_HERE":
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN. Put your token in telegram_ai_bot/.env"
        )

    application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    application.bot_data["ai_engine"] = config.get_ai_engine()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    asyncio.set_event_loop(asyncio.new_event_loop())

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
