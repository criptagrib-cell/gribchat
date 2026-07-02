import os
import logging
from groq import Groq
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant. Respond in the same language the user writes in.")
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "40"))

client = Groq(api_key=GROQ_API_KEY)

# In-memory conversation history: {user_id: [{"role": ..., "content": ...}]}
conversations: dict[int, list] = {}


def get_history(user_id: int) -> list:
    return conversations.setdefault(user_id, [])


def add_message(user_id: int, role: str, content: str):
    history = get_history(user_id)
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY:
        conversations[user_id] = history[-MAX_HISTORY:]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот на базе Groq (Llama 3.3). Пиши мне что угодно.\n\n"
        "/reset — очистить историю диалога"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversations.pop(user_id, None)
    await update.message.reply_text("История очищена. Начинаем заново!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    add_message(user_id, "user", user_text)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + get_history(user_id),
            max_tokens=2048,
        )
        reply = response.choices[0].message.content
        add_message(user_id, "assistant", reply)
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        reply = "Произошла ошибка при обращении к Groq. Попробуй ещё раз."

    await update.message.reply_text(reply)


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
