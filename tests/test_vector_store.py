"""
Unit tests for A_Team_Agent vector store module.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for testing
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vector_store import VectorStore
from langchain.schema import Document

class TestVectorStore:
    """Test cases for vector store operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config to use temp directory
        with patch('vector_store.config') as mock_config:
            mock_config.VECTOR_STORE_PATH = self.temp_dir
            mock_config.OPENAI_API_KEY = "test-key"
            mock_config.EMBEDDING_MODEL = "text-embedding-ada-002"
            mock_config.TOP_K_RETRIEVAL = 4
            mock_config.SAVE_LOCAL = True
            
            self.vector_store = VectorStore()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test vector store initialization."""
        assert self.vector_store is not None
        assert self.vector_store.embeddings is not None
        assert self.vector_store.vector_store is None
    
    def test_is_initialized_false(self):
        """Test is_initialized returns False when not initialized."""
        assert self.vector_store.is_initialized() is False
    
    @patch('vector_store.FAISS')
    def test_create_vector_store(self, mock_faiss):
        """Test creating vector store from documents."""
        # Mock FAISS.from_documents
        mock_vector_store = MagicMock()
        mock_faiss.from_documents.return_value = mock_vector_store
        
        documents = [
            Document(page_content="Test content 1", metadata={'source': 'test1.txt'}),
            Document(page_content="Test content 2", metadata={'source': 'test2.txt'})
        ]
        
        self.vector_store.create_vector_store(documents)
        
        # Verify FAISS was called correctly
        mock_faiss.from_documents.assert_called_once_with(
            documents=documents,
            embedding=self.vector_store.embeddings
        )
        
        # Verify vector store was set
        assert self.vector_store.vector_store == mock_vector_store
    
    def test_create_vector_store_empty_documents(self):
        """Test creating vector store with empty document list."""
        self.vector_store.create_vector_store([])
        assert self.vector_store.vector_store is None
    
    @patch('vector_store.FAISS')
    def test_add_documents_existing_store(self, mock_faiss):
        """Test adding documents to existing vector store."""
        # Set up existing vector store
        mock_vector_store = MagicMock()
        self.vector_store.vector_store = mock_vector_store
        
        documents = [
            Document(page_content="New content", metadata={'source': 'new.txt'})
        ]
        
        self.vector_store.add_documents(documents)
        
        # Verify add_documents was called
        mock_vector_store.add_documents.assert_called_once_with(documents)
    
    def test_similarity_search_not_initialized(self):
        """Test similarity search when vector store is not initialized."""
        results = self.vector_store.similarity_search("test query")
        assert results == []
    
    @patch('vector_store.FAISS')
    def test_similarity_search_success(self, mock_faiss):
        """Test successful similarity search."""
        # Set up mock vector store
        mock_vector_store = MagicMock()
        mock_results = [
            Document(page_content="Result 1", metadata={'source': 'test1.txt'}),
            Document(page_content="Result 2", metadata={'source': 'test2.txt'})
        ]
        mock_vector_store.similarity_search.return_value = mock_results
        self.vector_store.vector_store = mock_vector_store
        
        results = self.vector_store.similarity_search("test query", k=2)
        
        assert results == mock_results
        mock_vector_store.similarity_search.assert_called_once_with("test query", k=2)
    
    def test_get_stats_not_initialized(self):
        """Test getting stats when not initialized."""
        stats = self.vector_store.get_stats()
        expected = {"status": "not_initialized", "document_count": 0}
        assert stats == expected
    
    @patch('vector_store.FAISS')
    def test_get_stats_initialized(self, mock_faiss):
        """Test getting stats when initialized."""
        # Set up mock vector store with index
        mock_vector_store = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 100
        mock_vector_store.index = mock_index
        self.vector_store.vector_store = mock_vector_store
        
        stats = self.vector_store.get_stats()
        
        assert stats['status'] == 'initialized'
        assert stats['document_count'] == 100
        assert 'embedding_model' in stats
    
    def test_delete_vector_store(self):
        """Test deleting vector store."""
        # Create some fake files
        fake_index = Path(self.temp_dir) / "faiss_index"
        fake_docs = Path(self.temp_dir) / "documents.pkl"
        fake_index.touch()
        fake_docs.touch()
        
        self.vector_store.delete_vector_store()
        
        assert not fake_index.exists()
        assert not fake_docs.exists()
        assert self.vector_store.vector_store is None