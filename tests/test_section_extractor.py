"""
Unit Tests for Section Extractor Service
"""

import pytest
from app.services.section_extractor import SectionExtractorService


@pytest.fixture
def extractor_service():
    """Fixture for section extractor instance"""
    return SectionExtractorService()


class TestSectionExtractorService:
    
    def test_extract_skills_section(self, extractor_service):
        """Test extracting skills section"""
        resume_text = """
        John Doe
        Software Engineer
        
        SKILLS
        Python, JavaScript, React, Node.js, MongoDB
        Machine Learning, TensorFlow, PyTorch
        
        EXPERIENCE
        Software Engineer at Tech Corp
        """
        
        sections = extractor_service.extract_sections(resume_text)
        
        assert "skills" in sections
        assert "python" in sections["skills"].lower()
        assert "javascript" in sections["skills"].lower()
    
    def test_extract_experience_section(self, extractor_service):
        """Test extracting experience section"""
        resume_text = """
        PROFESSIONAL EXPERIENCE
        
        Senior Developer - ABC Company (2020-2023)
        - Led team of 5 developers
        - Built scalable microservices
        
        EDUCATION
        BSc Computer Science
        """
        
        sections = extractor_service.extract_sections(resume_text)
        
        assert "experience" in sections
        assert "senior developer" in sections["experience"].lower()
        assert "abc company" in sections["experience"].lower()
    
    def test_extract_projects_section(self, extractor_service):
        """Test extracting projects section"""
        resume_text = """
        PROJECTS
        
        E-commerce Platform
        Built a full-stack web application using MERN stack
        
        CERTIFICATIONS
        AWS Certified Developer
        """
        
        sections = extractor_service.extract_sections(resume_text)
        
        assert "projects" in sections
        assert "e-commerce" in sections["projects"].lower()
    
    def test_extract_education_section(self, extractor_service):
        """Test extracting education section"""
        resume_text = """
        EDUCATION
        
        Master of Science in Computer Science
        Stanford University, 2020
        
        Bachelor of Engineering
        MIT, 2018
        """
        
        sections = extractor_service.extract_sections(resume_text)
        
        assert "education" in sections
        assert "stanford" in sections["education"].lower()
    
    def test_get_specific_section(self, extractor_service):
        """Test getting specific section by name"""
        resume_text = """
        SKILLS
        Python, Java, C++
        
        EXPERIENCE
        5 years in software development
        """
        
        skills = extractor_service.get_section(resume_text, "skills")
        
        assert skills != ""
        assert "python" in skills.lower()
    
    def test_list_found_sections(self, extractor_service):
        """Test listing all found sections"""
        resume_text = """
        SKILLS
        Python
        
        EXPERIENCE
        Developer at XYZ
        
        EDUCATION
        BSc Computer Science
        """
        
        found_sections = extractor_service.list_found_sections(resume_text)
        
        assert len(found_sections) >= 3
        assert "skills" in found_sections
        assert "experience" in found_sections
        assert "education" in found_sections
    
    def test_no_sections_found(self, extractor_service):
        """Test when no sections are detected"""
        resume_text = "Just a plain text resume with no clear sections."
        
        sections = extractor_service.extract_sections(resume_text)
        
        # Should return empty dict or minimal sections
        assert isinstance(sections, dict)
    
    def test_looks_like_content_heuristic(self, extractor_service):
        """Test content vs header detection"""
        header_line = "SKILLS"
        content_line = "Python developer with 5 years of experience in building scalable applications."
        
        assert not extractor_service._looks_like_content(header_line)
        assert extractor_service._looks_like_content(content_line)
