import asyncio
import logging
import os
from telegram import Update
from telegram.ext import ContextTypes, Application, CommandHandler, MessageHandler, filters
from rag_system import RAGSystem
from config import TELEGRAM_TOKEN

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Инициализация Rag системы
rag = RAGSystem()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда start"""
    await update.message.reply_text(
        "Привет! Я RAG-бот. Отправь мне PDF, TXT или RTF-файл, "
        "и я смогу отвечать на вопросы по его содержимому.\n"
        "Напиши /help для справки."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    await update.message.reply_text(
        "Инструкция по использованию:\n"
        "1. Отправьте PDF, TXT или RTF-файл. Я загружу и обработаю его.\n"
        "2. Сразу после отправки файла задайте мне любой вопрос по его содержимому — "
        "я постараюсь ответить.\n"
        "3. Команда /clear — полностью очищает базу данных (удаляет все загруженные документы).\n"
        "4. /start — начать с начала, /help — показать эту справку."
    )

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка команды /clear: очищаем всю базу данных ChromaDB"""
    try:
        rag.clear_database()
        await update.message.reply_text("✅ База данных очищена. Загрузите новые документы.")
    except Exception as e:
        logger.exception("Ошибка при очистке базы данных:")
        await update.message.reply_text(f"Не удалось очистить базу: {e}")

async def upload_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка загрузки документов"""
    if not update.message.document:
        await update.message.reply_text("Пожалуйста, отправьте файл (PDF, TXT или RTF).")
        return

    filename = update.message.document.file_name.lower()
    if not filename.endswith((".pdf", ".txt", ".rtf")):
        await update.message.reply_text("Поддерживаются только PDF, TXT или RTF файлы.")
        return

    os.makedirs("documents", exist_ok=True)
    file_path = f"documents/{filename}"

    try:
        file = await context.bot.get_file(update.message.document.file_id)
        await file.download_to_drive(file_path)
    except Exception as e:
        logger.exception("Ошибка при скачивании файла:")
        await update.message.peply_text(f"Не удалось загрузить файл: {e}")
        return

    # Загружаем в RAG систему
    try:
        rag.load_documents(file_path)
        await update.message.reply_text("Документ успешно загружен и обработан!")
    except Exception as e:
        logger.exception("Ошибка при обработке документа:")
        await update.message.reply_text(f"Ошибка при обработке: {str(e)}")

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка вопросов пользователя"""
    question = update.message.text.strip()
    if not question:
        await update.message.reply_text("Введите, пожалуйста, непустой вопрос.")
        return

    # Уведомляем что бот пишет
    try:
        # Отправляем индикатор печати
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )
    except Exception as e:
        await update.message.reply_text(
            f"Произошла ошибка: {str(e)}"
        )

    # Запускаем тяжелую операцию в фоне
    try:
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, rag.generate_answer, question)
        await update.message.reply_text(answer)
    except Exception as e:
        logger.exception("Ошибка при генерации ответа:")
        await update.message.reply_text(f"Произошла ошибка при ответе: {e}")

def main():
    """Запуск бота"""
    os.makedirs("documents", exist_ok=True)

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Обработчики команд
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))

    app.add_handler(MessageHandler(filters.Document.ALL, upload_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    # Запуск бота
    logger.info("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
