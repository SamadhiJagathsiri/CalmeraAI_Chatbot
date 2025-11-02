from chatbot.chains.conversation_chain import ConversationChain
from chatbot.chains.rag_chain import RAGChain
from chatbot.memory.conversation_memory import MindEaseMemory
from chatbot.crisis_detection import CrisisDetector
from chatbot.sentiment_analysis import SentimentAnalyzer
from utils.config import Config
from utils.vectorstore_manager import VectorStoreManager

class MindEaseAI:
    """
    Main orchestrator for MindEase AI
    Coordinates conversation, RAG, crisis detection, and sentiment analysis
    """
    
    def __init__(self, vectorstore_manager: VectorStoreManager = None):
        Config.validate()
        
        self.memory = MindEaseMemory()
        self.conversation_chain = ConversationChain(memory=self.memory)
        
        # Use provided vectorstore manager or create a new one
        self.vectorstore_manager = vectorstore_manager or VectorStoreManager()
        # Ensure the vector store is ready (loads or creates internally)
        self.vectorstore_manager.get_retriever()
        
        self.rag_chain = RAGChain(memory=self.memory)
        
        self.crisis_detector = CrisisDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        self.rag_enabled = self.rag_chain.is_available()
        
        print(" MindEase AI initialized successfully")
        if self.rag_enabled:
            print(" RAG mode: Enabled (wellness guides loaded)")
        else:
            print(" RAG mode: Disabled (add PDFs to data/guides/)")
