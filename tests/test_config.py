"""
Unit tests for A_Team_Agent configuration module.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src to path for testing
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config

class TestConfig:
    """Test cases for configuration management."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = Config()
        
        # Check default values
        assert config.STREAMLIT_SERVER_PORT == 8501
        assert config.CHUNK_SIZE == 1000
        assert config.CHUNK_OVERLAP == 200
        assert config.TOP_K_RETRIEVAL == 4
        assert config.EMBEDDING_MODEL == "text-embedding-ada-002"
        assert config.LLM_MODEL == "gpt-3.5-turbo"
        assert config.TEMPERATURE == 0.7
        assert config.MAX_TOKENS == 500
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key-123',
        'CHUNK_SIZE': '500',
        'TEMPERATURE': '0.5'
    })
    def test_environment_override(self):
        """Test that environment variables override defaults."""
        config = Config()
        
        assert config.OPENAI_API_KEY == 'test-key-123'
        assert config.CHUNK_SIZE == 500
        assert config.TEMPERATURE == 0.5
    
    def test_supported_file_types(self):
        """Test supported file types configuration."""
        config = Config()
        expected_types = ['.pdf', '.txt', '.md', '.docx']
        assert config.SUPPORTED_FILE_TYPES == expected_types
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('pathlib.Path.mkdir')
    def test_validate_config_success(self, mock_mkdir):
        """Test successful configuration validation."""
        config = Config()
        result = config.validate_config()
        assert result is True
        mock_mkdir.assert_called()
    
    def test_validate_config_missing_api_key(self):
        """Test configuration validation with missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is required"):
                config.validate_config()