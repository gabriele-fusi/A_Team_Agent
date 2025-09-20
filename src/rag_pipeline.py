"""
RAG pipeline implementation using LangChain for A_Team_Agent.
Handles question-answering with context retrieval and response generation.
"""

import logging
from typing import List, Dict, Any, Optional

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from .config import config
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG pipeline for question-answering with document retrieval."""
    
    def __init__(self):
        """Initialize RAG pipeline with vector store and LLM."""
        self.vector_store = VectorStore()
        self.llm = ChatOpenAI(
            openai_api_key=config.OPENAI_API_KEY,
            model_name=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
        self.qa_chain = None
        self._setup_prompt_template()
    
    def _setup_prompt_template(self):
        """Setup the prompt template for the QA chain."""
        self.prompt_template = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context provided, just say that you don't know, 
don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: Please provide a helpful and accurate answer based on the context provided. 
If you reference specific information, mention which document or section it came from.""",
            input_variables=["context", "question"]
        )
    
    def initialize_qa_chain(self) -> bool:
        """
        Initialize the QA chain with the vector store.
        
        Returns:
            True if initialized successfully, False otherwise
        """
        try:
            # Check if vector store is already initialized, if not try to load
            if not self.vector_store.is_initialized():
                if not self.vector_store.load_vector_store():
                    logger.warning("No existing vector store found. Please upload documents first.")
                    return False
            
            # Create retrieval QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.vector_store.as_retriever(
                    search_kwargs={"k": config.TOP_K_RETRIEVAL}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt_template}
            )
            
            logger.info("QA chain initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing QA chain: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to the vector store and reinitialize QA chain.
        
        Args:
            documents: List of documents to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided")
                return False
            
            # Add documents to vector store
            if self.vector_store.is_initialized():
                self.vector_store.add_documents(documents)
            else:
                self.vector_store.create_vector_store(documents)
            
            # Reinitialize QA chain
            return self.initialize_qa_chain()
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.qa_chain:
            return {
                "answer": "Please upload and process documents before asking questions.",
                "source_documents": [],
                "error": "QA chain not initialized"
            }
        
        try:
            # Run the QA chain
            result = self.qa_chain({"query": question})
            
            # Extract answer and sources
            answer = result.get("result", "No answer generated")
            source_docs = result.get("source_documents", [])
            
            # Format source information
            sources = []
            for i, doc in enumerate(source_docs):
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "source_id": i + 1
                }
                sources.append(source_info)
            
            logger.info(f"Query processed successfully. Found {len(sources)} source documents.")
            
            return {
                "answer": answer,
                "source_documents": sources,
                "question": question,
                "model_used": config.LLM_MODEL
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "source_documents": [],
                "error": str(e)
            }
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Perform similarity search without generating an answer.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of similar documents
        """
        if not self.vector_store.is_initialized():
            logger.warning("Vector store not initialized")
            return []
        
        return self.vector_store.similarity_search(query, k)
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        return self.vector_store.get_stats()
    
    def clear_vector_store(self) -> bool:
        """
        Clear the vector store and reset the QA chain.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_store.delete_vector_store()
            self.qa_chain = None
            logger.info("Vector store cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def update_llm_settings(self, model_name: str = None, temperature: float = None, max_tokens: int = None):
        """
        Update LLM settings and reinitialize if needed.
        
        Args:
            model_name: New model name
            temperature: New temperature setting
            max_tokens: New max tokens setting
        """
        try:
            # Update LLM with new settings
            self.llm = ChatOpenAI(
                openai_api_key=config.OPENAI_API_KEY,
                model_name=model_name or config.LLM_MODEL,
                temperature=temperature if temperature is not None else config.TEMPERATURE,
                max_tokens=max_tokens or config.MAX_TOKENS
            )
            
            # Reinitialize QA chain if it exists
            if self.vector_store.is_initialized():
                self.initialize_qa_chain()
            
            logger.info("LLM settings updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating LLM settings: {str(e)}")
    
    def get_conversation_context(self, conversation_history: List[Dict[str, str]], max_context: int = 3) -> str:
        """
        Get conversation context from recent messages.
        
        Args:
            conversation_history: List of conversation messages
            max_context: Maximum number of previous messages to include
            
        Returns:
            Formatted conversation context
        """
        if not conversation_history:
            return ""
        
        # Get recent messages
        recent_messages = conversation_history[-max_context:]
        
        context_parts = []
        for msg in recent_messages:
            if msg.get("role") == "user":
                context_parts.append(f"Previous Question: {msg.get('content', '')}")
            elif msg.get("role") == "assistant":
                context_parts.append(f"Previous Answer: {msg.get('content', '')}")
        
        return "\n".join(context_parts)
    
    def query_with_context(self, question: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Query with conversation context.
        
        Args:
            question: Current question
            conversation_history: Previous conversation messages
            
        Returns:
            Dictionary with answer and source documents
        """
        # Add conversation context if available
        if conversation_history:
            context = self.get_conversation_context(conversation_history)
            if context:
                enhanced_question = f"{context}\n\nCurrent Question: {question}"
            else:
                enhanced_question = question
        else:
            enhanced_question = question
        
        return self.query(enhanced_question)
    
    def is_ready(self) -> bool:
        """
        Check if the RAG pipeline is ready to answer questions.
        
        Returns:
            True if ready, False otherwise
        """
        return self.qa_chain is not None and self.vector_store.is_initialized()