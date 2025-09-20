"""
Unit tests for A_Team_Agent document processor module.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for testing
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from document_processor import DocumentProcessor
from langchain.schema import Document

class TestDocumentProcessor:
    """Test cases for document processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
    
    def test_initialization(self):
        """Test document processor initialization."""
        assert self.processor is not None
        assert self.processor.text_splitter is not None
    
    def test_get_supported_file_types(self):
        """Test getting supported file types."""
        supported_types = self.processor.get_supported_file_types()
        expected_types = ['.pdf', '.txt', '.md', '.docx']
        assert supported_types == expected_types
    
    def test_validate_file_type_valid(self):
        """Test file type validation with valid types."""
        assert self.processor.validate_file_type("document.pdf") is True
        assert self.processor.validate_file_type("text.txt") is True
        assert self.processor.validate_file_type("readme.md") is True
        assert self.processor.validate_file_type("doc.docx") is True
    
    def test_validate_file_type_invalid(self):
        """Test file type validation with invalid types."""
        assert self.processor.validate_file_type("image.jpg") is False
        assert self.processor.validate_file_type("video.mp4") is False
        assert self.processor.validate_file_type("archive.zip") is False
    
    def test_process_text(self):
        """Test processing raw text input."""
        text = "This is a test document. " * 100  # Make it long enough to chunk
        documents = self.processor.process_text(text, "test_source")
        
        assert len(documents) > 0
        assert all(isinstance(doc, Document) for doc in documents)
        assert documents[0].metadata['source'] == 'test_source'
        assert 'chunk_id' in documents[0].metadata
    
    def test_get_file_info_empty(self):
        """Test getting file info with empty document list."""
        info = self.processor.get_file_info([])
        expected = {
            'total_documents': 0,
            'total_chunks': 0,
            'files': []
        }
        assert info == expected
    
    def test_get_file_info_with_documents(self):
        """Test getting file info with documents."""
        documents = [
            Document(page_content="Test content 1", metadata={'filename': 'test1.txt', 'file_type': '.txt'}),
            Document(page_content="Test content 2", metadata={'filename': 'test1.txt', 'file_type': '.txt'}),
            Document(page_content="Test content 3", metadata={'filename': 'test2.pdf', 'file_type': '.pdf'})
        ]
        
        info = self.processor.get_file_info(documents)
        
        assert info['total_documents'] == 2
        assert info['total_chunks'] == 3
        assert len(info['files']) == 2
    
    def test_extract_text_preview(self):
        """Test extracting text preview from documents."""
        documents = [
            Document(page_content="First chunk content"),
            Document(page_content="Second chunk content"),
            Document(page_content="Third chunk content")
        ]
        
        preview = self.processor.extract_text_preview(documents, max_length=50)
        assert len(preview) <= 53  # 50 + "..."
        assert "First chunk content" in preview
    
    def test_extract_text_preview_empty(self):
        """Test extracting text preview from empty document list."""
        preview = self.processor.extract_text_preview([])
        assert preview == "No content available"
    
    def test_update_chunk_size(self):
        """Test updating chunk size configuration."""
        original_chunk_size = self.processor.text_splitter._chunk_size
        new_chunk_size = 500
        
        self.processor.update_chunk_size(new_chunk_size)
        
        assert self.processor.text_splitter._chunk_size == new_chunk_size
        assert self.processor.text_splitter._chunk_overlap == new_chunk_size // 5