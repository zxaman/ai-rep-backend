"""
Unit Tests for Resume Parser Service
"""

import pytest
from app.services.resume_parser import ResumeParserService


@pytest.fixture
def parser_service():
    """Fixture for parser service instance"""
    return ResumeParserService()


class TestResumeParserService:
    
    def test_validate_file_pdf(self, parser_service):
        """Test PDF file validation"""
        assert parser_service.validate_file("resume.pdf") == True
        assert parser_service.validate_file("resume.PDF") == True
    
    def test_validate_file_docx(self, parser_service):
        """Test DOCX file validation"""
        assert parser_service.validate_file("resume.docx") == True
        assert parser_service.validate_file("resume.doc") == True
    
    def test_validate_file_invalid(self, parser_service):
        """Test invalid file types"""
        assert parser_service.validate_file("resume.txt") == False
        assert parser_service.validate_file("resume.png") == False
        assert parser_service.validate_file("resume") == False
    
    def test_get_file_extension(self, parser_service):
        """Test file extension extraction"""
        assert parser_service._get_file_extension("file.pdf") == ".pdf"
        assert parser_service._get_file_extension("file.DOCX") == ".docx"
        assert parser_service._get_file_extension("file.name.pdf") == ".pdf"
        assert parser_service._get_file_extension("noextension") == ""
    
    def test_parse_unsupported_type(self, parser_service):
        """Test parsing with unsupported file type"""
        with pytest.raises(ValueError, match="Unsupported file type"):
            parser_service.parse(b"dummy content", "file.txt")


# Note: Full PDF/DOCX parsing tests require actual file fixtures
# These would be added in integration tests with sample resumes
