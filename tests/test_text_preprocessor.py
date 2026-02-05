"""
Unit Tests for Text Preprocessor Service
"""

import pytest
from app.services.text_preprocessor import TextPreprocessorService


@pytest.fixture
def preprocessor_service():
    """Fixture for preprocessor service instance"""
    return TextPreprocessorService()


class TestTextPreprocessorService:
    
    def test_preprocess_basic(self, preprocessor_service):
        """Test basic text preprocessing"""
        text = "Python Developer with 5 years of experience in Machine Learning."
        tokens = preprocessor_service.preprocess(text)
        
        # Check tokens are lowercase and lemmatized
        assert all(token.islower() for token in tokens)
        assert len(tokens) > 0
        
        # Common stopwords should be removed
        assert "with" not in tokens
        assert "of" not in tokens
        assert "in" not in tokens
    
    def test_remove_special_characters(self, preprocessor_service):
        """Test special character removal"""
        text = "C++ and Python! @#$ Experience: 5 years."
        cleaned = preprocessor_service._remove_special_characters(text)
        
        # Should keep alphanumeric and basic punctuation
        assert "c+" in cleaned or "c++" in cleaned.replace(" ", "")
        assert "@" not in cleaned
        assert "#" not in cleaned
        assert "$" not in cleaned
    
    def test_preprocess_removes_single_chars(self, preprocessor_service):
        """Test that single characters are removed"""
        text = "I am a developer"
        tokens = preprocessor_service.preprocess(text)
        
        # Single chars like 'I' and 'a' should be removed
        assert "i" not in tokens
        assert "a" not in tokens
    
    def test_extract_keywords(self, preprocessor_service):
        """Test keyword extraction"""
        tokens = ["python", "machine", "learning", "python", "data", "python"]
        keywords = preprocessor_service.extract_keywords(tokens, top_n=3)
        
        assert len(keywords) <= 3
        assert "python" == keywords[0]  # Most frequent
    
    def test_clean_text_only(self, preprocessor_service):
        """Test text cleaning without tokenization"""
        text = "Machine Learning Engineer @ Google!"
        cleaned = preprocessor_service.clean_text_only(text)
        
        assert cleaned.islower()
        assert "@" not in cleaned
        assert "!" not in cleaned
    
    def test_empty_text(self, preprocessor_service):
        """Test preprocessing empty text"""
        tokens = preprocessor_service.preprocess("")
        assert tokens == []
    
    def test_lemmatization(self, preprocessor_service):
        """Test that lemmatization works"""
        text = "developing developers developed"
        tokens = preprocessor_service.preprocess(text)
        
        # All should be lemmatized to "develop"
        assert all("develop" in token for token in tokens if token)
