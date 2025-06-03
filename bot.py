import logging
import os
from telegram import Update
from telegram.ext import ContextTypes, Application, CommandHandler, MessageHandler, filters
from rag_system import RAGSystem
from config import TELEGRAM_TOKEN

# Настройка логирования
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация Rag системы
rag = RAGSystem()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда start"""
    await update.message.reply_text(
        "Привет! Я RAG бот. Задавай вопросы по загруженным документам!"
    )

async def upload_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка загрузки документов"""
    filename = update.message.document.file_name.lower()
    if not filename.endswith((".pdf", ".txt", ".rtf")):
        await update.message.reply_text("Поддерживаются только PDF, TXT или RTF файлы.")
        return

    if update.message.document:
        os.makedirs("documents", exist_ok=True)
        file = await context.bot.get_file(update.message.document.file_id)
        file_path = f"documents/{update.message.document.file_name}"
        await file.download_to_drive(file_path)

        # Загружаем в RAG систему
        try:
            rag.load_documents(file_path)
            await update.message.reply_text("Документ успешно загружен и обработан!")
        except Exception as e:
            await update.message.reply_text(f"Ошибка при обработке: {str(e)}")
    else:
        await update.message.reply_text("Пожалуйста, отправьте PDF файл.")

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка вопросов пользователя"""
    question = update.message.text

    try:
        # Отправляем индикатор печати
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        answer = rag.generate_answer(question)
        await update.message.reply_text(answer)

    except Exception as e:
        await update.message.reply_text(
            f"Произошла ошибка: {str(e)}"
        )

def main():
    """Запуск бота"""
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Обработчики команд
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.ALL, upload_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    # Запуск бота
    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
