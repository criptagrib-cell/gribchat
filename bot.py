import os
import base64
import io
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
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Respond in the same language the user writes in."
)
TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "20"))

client = Groq(api_key=GROQ_API_KEY)
conversations: dict[int, list] = {}


def get_history(user_id: int) -> list:
    return conversations.setdefault(user_id, [])


def add_message(user_id: int, role: str, content):
    history = get_history(user_id)
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY:
        conversations[user_id] = history[-MAX_HISTORY:]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я умею:\n"
        "• Отвечать на вопросы\n"
        "• Анализировать фотографии (просто отправь фото)\n\n"
        "/reset — очистить историю диалога"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conversations.pop(update.effective_user.id, None)
    await update.message.reply_text("История очищена. Начинаем заново!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    add_message(user_id, "user", user_text)

    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + get_history(user_id),
            max_tokens=2048,
        )
        reply = response.choices[0].message.content or ""
        add_message(user_id, "assistant", reply)
    except Exception as e:
        logger.error(f"Text error: {e}")
        reply = f"Ошибка: {e}"

    await update.message.reply_text(reply)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await file.download_to_memory(buf)
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    caption = update.message.caption or "Опиши что на этом изображении."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": caption},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            max_tokens=2048,
        )
        reply = response.choices[0].message.content or ""
        add_message(user_id, "user", f"[Фото] {caption}")
        add_message(user_id, "assistant", reply)
    except Exception as e:
        logger.error(f"Vision error: {e}")
        reply = f"Ошибка: {e}"

    await update.message.reply_text(reply)


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Bot started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
