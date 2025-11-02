from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from utils.config import Config
import os

class VectorStoreManager:
    """Manage vector store for wellness guides"""

    def __init__(self):
        self.embeddings = CohereEmbeddings(
            cohere_api_key=Config.COHERE_API_KEY,
            model=Config.EMBEDDING_MODEL
        )
        self.persist_directory = Config.VECTOR_STORE_PATH
        self.vectorstore = None

    def load_vectorstore(self):
        """Load existing vector store"""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print("📚 Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("⚠️ Vector store not found. Run build_vectorstore.py first.")
            self.vectorstore = None
        return self.vectorstore

    def get_retriever(self, k: int = 3):
        if not self.vectorstore:
            self.load_vectorstore()
        if not self.vectorstore:
            return None
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, k: int = 3):
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search(query, k=k)
