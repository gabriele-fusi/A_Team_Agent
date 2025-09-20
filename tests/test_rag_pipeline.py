"""
Unit tests for A_Team_Agent RAG pipeline module.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add src to path for testing
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag_pipeline import RAGPipeline
from langchain.schema import Document

class TestRAGPipeline:
    """Test cases for RAG pipeline operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('rag_pipeline.config') as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            mock_config.LLM_MODEL = "gpt-3.5-turbo"
            mock_config.TEMPERATURE = 0.7
            mock_config.MAX_TOKENS = 500
            mock_config.TOP_K_RETRIEVAL = 4
            
            with patch('rag_pipeline.VectorStore') as mock_vector_store_class:
                with patch('rag_pipeline.ChatOpenAI') as mock_chat_openai:
                    self.rag_pipeline = RAGPipeline()
                    self.mock_vector_store = mock_vector_store_class.return_value
                    self.mock_llm = mock_chat_openai.return_value
    
    def test_initialization(self):
        """Test RAG pipeline initialization."""
        assert self.rag_pipeline is not None
        assert self.rag_pipeline.vector_store is not None
        assert self.rag_pipeline.llm is not None
        assert self.rag_pipeline.qa_chain is None
    
    def test_is_ready_false(self):
        """Test is_ready returns False when not initialized."""
        self.rag_pipeline.qa_chain = None
        self.mock_vector_store.is_initialized.return_value = False
        
        assert self.rag_pipeline.is_ready() is False
    
    def test_is_ready_true(self):
        """Test is_ready returns True when properly initialized."""
        self.rag_pipeline.qa_chain = MagicMock()
        self.mock_vector_store.is_initialized.return_value = True
        
        assert self.rag_pipeline.is_ready() is True
    
    def test_add_documents_empty_list(self):
        """Test adding empty document list."""
        result = self.rag_pipeline.add_documents([])
        assert result is False
    
    def test_add_documents_success(self):
        """Test successfully adding documents."""
        documents = [
            Document(page_content="Test content 1", metadata={'source': 'test1.txt'}),
            Document(page_content="Test content 2", metadata={'source': 'test2.txt'})
        ]
        
        self.mock_vector_store.is_initialized.return_value = False
        self.mock_vector_store.create_vector_store.return_value = None
        
        with patch.object(self.rag_pipeline, 'initialize_qa_chain', return_value=True):
            result = self.rag_pipeline.add_documents(documents)
        
        assert result is True
        self.mock_vector_store.create_vector_store.assert_called_once_with(documents)
    
    def test_query_not_initialized(self):
        """Test querying when QA chain is not initialized."""
        self.rag_pipeline.qa_chain = None
        
        result = self.rag_pipeline.query("Test question")
        
        assert "Please upload and process documents" in result["answer"]
        assert result["source_documents"] == []
        assert "error" in result
    
    def test_query_success(self):
        """Test successful query processing."""
        # Set up mock QA chain
        mock_qa_chain = MagicMock()
        mock_result = {
            "result": "This is the answer",
            "source_documents": [
                Document(page_content="Source content", metadata={'filename': 'test.txt'})
            ]
        }
        mock_qa_chain.return_value = mock_result
        self.rag_pipeline.qa_chain = mock_qa_chain
        
        result = self.rag_pipeline.query("Test question")
        
        assert result["answer"] == "This is the answer"
        assert len(result["source_documents"]) == 1
        assert result["question"] == "Test question"
    
    def test_similarity_search_not_initialized(self):
        """Test similarity search when vector store not initialized."""
        self.mock_vector_store.is_initialized.return_value = False
        
        results = self.rag_pipeline.similarity_search("test query")
        
        assert results == []
    
    def test_similarity_search_success(self):
        """Test successful similarity search."""
        self.mock_vector_store.is_initialized.return_value = True
        expected_results = [
            Document(page_content="Result 1", metadata={'source': 'test1.txt'})
        ]
        self.mock_vector_store.similarity_search.return_value = expected_results
        
        results = self.rag_pipeline.similarity_search("test query")
        
        assert results == expected_results
    
    def test_get_vector_store_stats(self):
        """Test getting vector store statistics."""
        expected_stats = {"status": "initialized", "document_count": 10}
        self.mock_vector_store.get_stats.return_value = expected_stats
        
        stats = self.rag_pipeline.get_vector_store_stats()
        
        assert stats == expected_stats
    
    def test_clear_vector_store_success(self):
        """Test successfully clearing vector store."""
        self.mock_vector_store.delete_vector_store.return_value = None
        
        result = self.rag_pipeline.clear_vector_store()
        
        assert result is True
        assert self.rag_pipeline.qa_chain is None
    
    def test_get_conversation_context_empty(self):
        """Test getting conversation context with empty history."""
        context = self.rag_pipeline.get_conversation_context([])
        assert context == ""
    
    def test_get_conversation_context_with_history(self):
        """Test getting conversation context with message history."""
        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"}
        ]
        
        context = self.rag_pipeline.get_conversation_context(history, max_context=2)
        
        assert "Previous Answer: First answer" in context
        assert "Previous Question: Second question" in context
    
    def test_query_with_context(self):
        """Test querying with conversation context."""
        # Set up mock QA chain
        mock_qa_chain = MagicMock()
        mock_result = {
            "result": "Contextual answer",
            "source_documents": []
        }
        mock_qa_chain.return_value = mock_result
        self.rag_pipeline.qa_chain = mock_qa_chain
        
        history = [{"role": "user", "content": "Previous question"}]
        
        result = self.rag_pipeline.query_with_context("Current question", history)
        
        assert result["answer"] == "Contextual answer"
        # Verify the enhanced question was used
        called_query = mock_qa_chain.call_args[0][0]["query"]
        assert "Current Question: Current question" in called_query