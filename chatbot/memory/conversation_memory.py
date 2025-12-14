from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from typing import List, Dict, Any
from utils.config import Config


class MindEaseMemory:
    """Enhanced conversation memory with emotional context tracking"""
    
    def __init__(self, max_length: int = None):
        self.max_length = max_length or Config.MAX_MEMORY_LENGTH
        self.messages: List[BaseMessage] = []
        
        # Track emotional patterns
        self.emotional_history = []
    
    def add_interaction(self, user_input: str, ai_response: str, sentiment: dict = None):
        """Add user-AI interaction to memory"""
        # Add messages to history
        self.messages.append(HumanMessage(content=user_input))
        self.messages.append(AIMessage(content=ai_response))
        
        # Trim to max length (keep last k pairs = 2k messages)
        if len(self.messages) > 2 * self.max_length:
            self.messages = self.messages[-(2 * self.max_length):]
        
        # Track emotional context
        if sentiment:
            self.emotional_history.append({
                "input": user_input,
                "sentiment": sentiment,
                "response": ai_response
            })
            
            if len(self.emotional_history) > self.max_length:
                self.emotional_history = self.emotional_history[-self.max_length:]
    
    def get_chat_history(self):
        """Retrieve formatted chat history"""
        return self.messages
    
    def get_emotional_summary(self):
        """Get summary of emotional patterns"""
        if not self.emotional_history:
            return "No emotional context yet"
        
        recent_sentiments = [entry["sentiment"] for entry in self.emotional_history[-5:]]
        
        avg_polarity = sum(s.get("polarity", 0) for s in recent_sentiments) / len(recent_sentiments)
        
        if avg_polarity > 0.3:
            return "predominantly positive"
        elif avg_polarity < -0.3:
            return "predominantly negative"
        else:
            return "mixed"
    
    def clear(self):
        """Clear conversation memory"""
        self.messages = []
        self.emotional_history = []
    
    def get_memory_object(self):
        """Return the memory object for chains"""
        return self
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables for LangChain compatibility"""
        return {Config.MEMORY_KEY: self.messages}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save context for LangChain compatibility"""
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        
        if user_input:
            self.messages.append(HumanMessage(content=user_input))
        if ai_output:
            self.messages.append(AIMessage(content=ai_output))
        
        # Trim to max length
        if len(self.messages) > 2 * self.max_length:
            self.messages = self.messages[-(2 * self.max_length):]
