# 🤖 Telegram RAG Bot

Бот для Telegram, который позволяет загружать документы (PDF, TXT, RTF), извлекать из них содержимое и отвечать на вопросы на основе текста. Используется GPT-3.5-Turbo и векторная база ChromaDB для поиска релевантной информации.

## 🔍 Возможности

- Загрузка документов форматов **PDF**, **TXT**, **RTF**
- Извлечение и разбиение текста на чанки
- Индексация текста с помощью **sentence-transformers**
- Векторный поиск через **ChromaDB**
- Генерация ответа с помощью **GPT-3.5-Turbo (OpenAI API)**
- Telegram-интерфейс через `python-telegram-bot`

## 🛠️ Установка

1. Клонируй репозиторий:

```bash
git clone https://github.com/yourusername/telegram_rag_bot.git
cd telegram_rag_bot
```

2. Создайте и активируйте виртуальное окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Установите зависимости:

```bash
pip install -r requirements.txt
```

4. Создайте файл .env и добавьте следующие переменные:

```bash
TELEGRAM_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
```

# 🚀 Запуск
После настройки выполните следующую команду для запуска бота:
```bash
python bot.py
```