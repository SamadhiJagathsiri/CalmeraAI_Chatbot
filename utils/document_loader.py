import os
import streamlit as st
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class WellnessDocumentLoader:
    """Load and process wellness guide PDFs for both local and Streamlit deployment."""

    # Base folder: folder where this file resides (utils/)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Default path to data/guides relative to repo root
    GUIDES_PATH = os.path.join(BASE_DIR, "..", "data", "guides")

    def __init__(self, guides_path: str = None, chunk_size: int = 1000, chunk_overlap: int = 200):
        # Use given path or fallback to default
        self.guides_path = guides_path or self.GUIDES_PATH

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

    def load_documents(self) -> List:
        """Load all PDF documents from guides directory."""
        st.write("Looking for PDFs in:", self.guides_path)

        if not os.path.exists(self.guides_path):
            os.makedirs(self.guides_path)
            st.write(f"Folder not found. Created {self.guides_path}. Please add PDFs here.")
            return []

        files = [f for f in os.listdir(self.guides_path) if f.lower().endswith(".pdf")]
        st.write("PDF files found:", files)
        if not files:
            st.write("No PDF files found in folder.")
            return []

        # Load PDFs using DirectoryLoader
        loader = DirectoryLoader(
            self.guides_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )

        try:
            documents = loader.load()
            st.write(f"✓ Loaded {len(documents)} document pages")
            return documents
        except Exception as e:
            st.write(f"Error loading documents: {e}")
            return []

    def split_documents(self, documents: List) -> List:
        """Split documents into chunks for RAG or embeddings."""
        if not documents:
            return []

        chunks = self.text_splitter.split_documents(documents)
        st.write(f"✓ Split into {len(chunks)} chunks")
        return chunks

    def process_documents(self) -> List:
        """Load and split documents (full pipeline)."""
        docs = self.load_documents()
        return self.split_documents(docs)
