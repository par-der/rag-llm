import os

import PyPDF2
import chromadb
from chromadb import Settings
import openai
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from config import OPENAI_API_KEY


class RAGSystem:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        self.client = chromadb.Client(Settings(
            chroma_db_impl = "duckdb+parquet",
            persist_directory = "chroma_db"
        ))
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self._embed
        )
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

    def _read_pdf(self, path):
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    def _read_text(self, path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def load_documents(self, file_path):
        '''Загрузка и обработка документов'''
        # with open(file_path, 'rb') as file:
        #     pdf_reader = PyPDF2.PdfReader(file)
        #     text = ""
        #     for page in pdf_reader.pages:
        #         text += page.extract_text()

        ext = file_path.lower().split('.')[-1]
        if ext == 'pdf':
            text = self._read_pdf(file_path)
        elif ext in ('txt', 'rtf'):
            text = self._read_text(file_path)
        else:
            raise ValueError("Поддерживаются только PDF, TXT или RTF")

        # Разбиваем на чанки
        splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(text)

        # Создание эмбеддингов и сохранение
        base = os.path.basename(file_path)
        for i, chunk in enumerate(chunks):
            embedding = self.encoder.encode([chunk])[0]
            self.collection.add(
                embeddings = [embedding.tolist()],
                documents = [chunk],
                ids = [f"{base}_chunk_{i}"],
                metadatas = [{"source": base, "chunk_id": i}]
            )

    def search_relevant_docs(self, query, n_results=3):
        '''Поиск релевантных документов'''
        query_embedding = self.encoder.encode([query])[0]

        results = self.collection.query(
            query_embeddings = [query_embedding.tolist()],
            n_results = n_results
        )

        return results["documents"][0]

    def generate_answer(self, query):
        '''Генерация ответов с помощью Rag'''
        # Поиск релевантных документов
        relevant_docs = self.search_relevant_docs(query)

        # Формирование контекста
        context = '\n'.join(relevant_docs)

        # Промпт для LLM
        prompt = f"""
        Контекст: {context}
        Вопрос: {query}
        Ответь на вопрос, основываясь на предоставленном контексте. 
        Если информации недостаточно, скажи об этом.
        """

        # Запрос к OpenAI
        response = self.openai_client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": "Ты помощник, отвечающий на вопросы на основе предоставленных документов."},
                {"role": "user", "content": prompt}
            ],
            max_tokens = 500
        )

        return response.choices[0].message.content