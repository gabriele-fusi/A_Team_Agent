"""
FAISS vector store management for A_Team_Agent RAG application.
Handles vector storage, retrieval, and persistence operations.
"""

import os
import pickle
import logging
from typing import List, Optional, Tuple
from pathlib import Path

import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document

from .config import config

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS vector store manager with persistence."""
    
    def __init__(self):
        """Initialize vector store manager."""
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
            model=config.EMBEDDING_MODEL
        )
        self.vector_store = None
        self.index_path = Path(config.VECTOR_STORE_PATH)
        self.index_file = self.index_path / "faiss_index"
        self.documents_file = self.index_path / "documents.pkl"
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of LangChain Document objects
        """
        if not documents:
            logger.warning("No documents provided for vector store creation")
            return
        
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        try:
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            if config.SAVE_LOCAL:
                self.save_vector_store()
            
            logger.info("Vector store created successfully")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to existing vector store.
        
        Args:
            documents: List of LangChain Document objects to add
        """
        if not documents:
            logger.warning("No documents provided to add")
            return
        
        if self.vector_store is None:
            logger.info("No existing vector store, creating new one")
            self.create_vector_store(documents)
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        try:
            self.vector_store.add_documents(documents)
            
            if config.SAVE_LOCAL:
                self.save_vector_store()
            
            logger.info("Documents added successfully")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """
        Perform similarity search on vector store.
        
        Args:
            query: Search query string
            k: Number of documents to return (default from config)
            
        Returns:
            List of most similar documents
        """
        if self.vector_store is None:
            logger.warning("Vector store not initialized")
            return []
        
        k = k or config.TOP_K_RETRIEVAL
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query string
            k: Number of documents to return
            
        Returns:
            List of tuples (document, score)
        """
        if self.vector_store is None:
            logger.warning("Vector store not initialized")
            return []
        
        k = k or config.TOP_K_RETRIEVAL
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search with score: {str(e)}")
            return []
    
    def save_vector_store(self) -> None:
        """Save vector store to disk."""
        if self.vector_store is None:
            logger.warning("No vector store to save")
            return
        
        try:
            # Ensure directory exists
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            self.vector_store.save_local(str(self.index_path))
            
            logger.info(f"Vector store saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_vector_store(self) -> bool:
        """
        Load vector store from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.index_file.exists():
            logger.info("No existing vector store found")
            return False
        
        try:
            self.vector_store = FAISS.load_local(
                str(self.index_path),
                embeddings=self.embeddings
            )
            
            logger.info("Vector store loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def delete_vector_store(self) -> None:
        """Delete vector store files from disk."""
        try:
            if self.index_file.exists():
                os.remove(self.index_file)
            if self.documents_file.exists():
                os.remove(self.documents_file)
            
            self.vector_store = None
            logger.info("Vector store deleted")
            
        except Exception as e:
            logger.error(f"Error deleting vector store: {str(e)}")
            raise
    
    def get_stats(self) -> dict:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with store statistics
        """
        if self.vector_store is None:
            return {"status": "not_initialized", "document_count": 0}
        
        try:
            # Get document count from FAISS index
            index = self.vector_store.index
            doc_count = index.ntotal if hasattr(index, 'ntotal') else 0
            
            return {
                "status": "initialized",
                "document_count": doc_count,
                "index_type": type(index).__name__,
                "embedding_model": config.EMBEDDING_MODEL
            }
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def is_initialized(self) -> bool:
        """Check if vector store is initialized."""
        return self.vector_store is not None