import os

import PyPDF2
import chromadb
import openai
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from config import OPENAI_API_KEY
from striprtf.striprtf import rtf_to_text

# подключить logger
class RAGSystem:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        persist_dir = "chroma_db"
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE
        )
        self.collection = self.client.get_or_create_collection(
            name="documents"
        )
        self.chunk_size = 1000
        self.chunk_overlap = 200

    def _read_pdf(self, path: str) -> str:
        text_acc = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_acc.append(page_text)
        return "\n".join(text_acc)

    def _read_text(self, path: str) -> str:
        ext = path.lower().split('.')[-1]
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        if ext == "rtf":
            return rtf_to_text(raw)
        else:
            return raw


    def load_documents(self, file_path: str):
        '''Загрузка и обработка документов'''
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        ext = file_path.lower().split('.')[-1]
        if ext == 'pdf':
            text = self._read_pdf(file_path)
        elif ext in ('txt', 'rtf'):
            text = self._read_text(file_path)
        else:
            raise ValueError("Поддерживаются только PDF, TXT или RTF")

        # Разбиваем на чанки
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_text(text)

        # Создание эмбеддингов и сохранение
        base = os.path.basename(file_path)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{base}_chunk_{i}"
            existing = self.collection.get(ids=[chunk_id])
            if len(existing["ids"]) > 0:
                continue

            embedding = self.encoder.encode([chunk], convert_to_numpy=True)[0].tolist()
            self.collection.add(
                embeddings = [embedding],
                documents = [chunk],
                ids = [chunk_id],
                metadatas = [{"source": base, "chunk_id": i}]
            )

    def search_relevant_docs(self, query: str, n_results: int=3) -> dict:
        '''Поиск релевантных документов'''
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)[0].tolist()

        results = self.collection.query(
            query_embeddings = [query_embedding],
            n_results = n_results,
            include = ["documents", "metadatas", "distances"]
        )

        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }

    def generate_answer(self, query: str) -> str:
        '''Генерация ответов с помощью Rag'''
        # Поиск релевантных документов
        search_res = self.search_relevant_docs(query, n_results=3)
        docs = search_res["documents"]
        if len(docs) == 0:
            return "К сожалению, в загруженных документах нет информации по вашему запросу."

        # Формирование контекста
        context = '\n---\n'.join(docs)

        # Промпт для LLM
        prompt = (
            "У тебя есть следующий контекст (из документов):\n"
            f"{context}\n"
            "-------\n"
            f"Вопрос: {query}\n"
            "Ответь на вопрос максимально подробно, "
            "опираясь только на приведённый контекст. "
            "Если информации недостаточно, честно скажи об этом."
        )

        # Запрос к OpenAI
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Ты — ассистент, отвечающий на вопросы по документам."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=450
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка при обращении к OpenAI: {e}")

        return response.choices[0].message.content.strip()

    def clear_database(self):
        result = self.collection.get()
        all_ids = result.get("ids", [])
        if all_ids:
            self.collection.delete(ids=all_ids)