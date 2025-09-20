"""
Document processing module for A_Team_Agent RAG application.
Handles loading, chunking, and processing of various document types.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from langchain.schema import Document
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Document processor for multiple file types with intelligent chunking."""
    
    def __init__(self):
        """Initialize document processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_uploaded_file(self, uploaded_file, file_type: str) -> List[Document]:
        """
        Process an uploaded file and return chunked documents.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            file_type: File extension (e.g., '.pdf', '.txt')
            
        Returns:
            List of chunked Document objects
        """
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Process the file
            documents = self.load_document(tmp_file_path, file_type)
            
            # Clean up temporary file
            Path(tmp_file_path).unlink()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'filename': uploaded_file.name,
                    'file_type': file_type,
                    'source': uploaded_file.name
                })
            
            logger.info(f"Processed {uploaded_file.name}: {len(documents)} chunks created")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing uploaded file {uploaded_file.name}: {str(e)}")
            # Clean up temporary file if it exists
            if 'tmp_file_path' in locals():
                try:
                    Path(tmp_file_path).unlink()
                except:
                    pass
            raise
    
    def load_document(self, file_path: str, file_type: str) -> List[Document]:
        """
        Load and chunk a document based on its file type.
        
        Args:
            file_path: Path to the document file
            file_type: File extension
            
        Returns:
            List of chunked Document objects
        """
        loader_map = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.docx': Docx2txtLoader
        }
        
        if file_type not in loader_map:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        try:
            # Load document
            loader = loader_map[file_type](file_path)
            raw_documents = loader.load()
            
            # Split into chunks
            documents = self.text_splitter.split_documents(raw_documents)
            
            # Add additional metadata
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(documents),
                    'file_path': file_path,
                    'processing_timestamp': str(Path(file_path).stat().st_mtime)
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def process_text(self, text: str, source: str = "direct_input") -> List[Document]:
        """
        Process raw text input into chunked documents.
        
        Args:
            text: Raw text to process
            source: Source identifier for metadata
            
        Returns:
            List of chunked Document objects
        """
        try:
            # Create a document from text
            doc = Document(
                page_content=text,
                metadata={'source': source}
            )
            
            # Split into chunks
            documents = self.text_splitter.split_documents([doc])
            
            # Add metadata
            for i, chunk_doc in enumerate(documents):
                chunk_doc.metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(documents),
                    'input_type': 'text'
                })
            
            logger.info(f"Processed text input: {len(documents)} chunks created")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing text input: {str(e)}")
            raise
    
    def get_supported_file_types(self) -> List[str]:
        """
        Get list of supported file types.
        
        Returns:
            List of supported file extensions
        """
        return config.SUPPORTED_FILE_TYPES
    
    def validate_file_type(self, filename: str) -> bool:
        """
        Validate if file type is supported.
        
        Args:
            filename: Name of the file to validate
            
        Returns:
            True if file type is supported, False otherwise
        """
        file_extension = Path(filename).suffix.lower()
        return file_extension in config.SUPPORTED_FILE_TYPES
    
    def get_file_info(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get information about processed documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'files': []
            }
        
        # Group by source file
        files_info = {}
        for doc in documents:
            source = doc.metadata.get('filename', doc.metadata.get('source', 'unknown'))
            if source not in files_info:
                files_info[source] = {
                    'chunks': 0,
                    'file_type': doc.metadata.get('file_type', 'unknown'),
                    'total_length': 0
                }
            
            files_info[source]['chunks'] += 1
            files_info[source]['total_length'] += len(doc.page_content)
        
        return {
            'total_documents': len(files_info),
            'total_chunks': len(documents),
            'files': [
                {
                    'name': name,
                    'chunks': info['chunks'],
                    'file_type': info['file_type'],
                    'total_length': info['total_length']
                }
                for name, info in files_info.items()
            ]
        }
    
    def update_chunk_size(self, chunk_size: int, chunk_overlap: int = None):
        """
        Update text splitter configuration.
        
        Args:
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap (optional)
        """
        if chunk_overlap is None:
            chunk_overlap = min(chunk_size // 5, config.CHUNK_OVERLAP)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"Updated chunk size to {chunk_size} with overlap {chunk_overlap}")
    
    def extract_text_preview(self, documents: List[Document], max_length: int = 500) -> str:
        """
        Extract a preview of the document content.
        
        Args:
            documents: List of Document objects
            max_length: Maximum length of preview text
            
        Returns:
            Preview text string
        """
        if not documents:
            return "No content available"
        
        # Get first few chunks of content
        preview_text = ""
        for doc in documents[:3]:  # First 3 chunks
            preview_text += doc.page_content + "\n\n"
            if len(preview_text) > max_length:
                break
        
        # Truncate if too long
        if len(preview_text) > max_length:
            preview_text = preview_text[:max_length] + "..."
        
        return preview_text.strip()