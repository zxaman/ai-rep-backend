"""
Unit Tests for Skill Extractor
Phase 3: Feature Engineering Tests
"""

import pytest
from app.features.skill_extractor import SkillExtractor


@pytest.fixture
def skill_extractor():
    """Fixture for skill extractor instance"""
    return SkillExtractor()


class TestSkillExtractor:
    
    def test_extract_skills_basic_match(self, skill_extractor):
        """Test basic skill matching"""
        resume_skills = "Python, Java, SQL, Docker, AWS"
        required_skills = ["Python", "Java", "SQL"]
        
        result = skill_extractor.extract_skills(resume_skills, required_skills)
        
        assert result['matched_count'] == 3
        assert result['skill_match_percent'] == 1.0
        assert len(result['missing_skills']) == 0
    
    def test_extract_skills_partial_match(self, skill_extractor):
        """Test partial skill matching"""
        resume_skills = "Python, Django, PostgreSQL"
        required_skills = ["Python", "Java", "SQL", "Docker"]
        
        result = skill_extractor.extract_skills(resume_skills, required_skills)
        
        assert result['matched_count'] >= 1  # At least Python
        assert result['skill_match_percent'] < 1.0
        assert "Java" in result['missing_skills']
        assert "Docker" in result['missing_skills']
    
    def test_skill_synonyms(self, skill_extractor):
        """Test synonym matching"""
        resume_skills = "JavaScript, Node.js, MongoDB"
        required_skills = ["JS", "NodeJS", "Mongo"]
        
        result = skill_extractor.extract_skills(resume_skills, required_skills)
        
        # Should match due to synonyms
        assert result['matched_count'] >= 2
    
    def test_case_insensitive_matching(self, skill_extractor):
        """Test case-insensitive skill matching"""
        resume_skills = "PYTHON, java, ReAcT"
        required_skills = ["python", "Java", "react"]
        
        result = skill_extractor.extract_skills(resume_skills, required_skills)
        
        assert result['matched_count'] == 3
        assert result['skill_match_percent'] == 1.0
    
    def test_empty_resume_skills(self, skill_extractor):
        """Test with empty resume skills"""
        resume_skills = ""
        required_skills = ["Python", "Java"]
        
        result = skill_extractor.extract_skills(resume_skills, required_skills)
        
        assert result['matched_count'] == 0
        assert result['skill_match_percent'] == 0.0
        assert len(result['missing_skills']) == 2
    
    def test_empty_required_skills(self, skill_extractor):
        """Test with no required skills"""
        resume_skills = "Python, Java, Docker"
        required_skills = []
        
        result = skill_extractor.extract_skills(resume_skills, required_skills)
        
        assert result['skill_match_percent'] == 0.0
        assert result['required_count'] == 0
    
    def test_skill_categories(self, skill_extractor):
        """Test skill categorization"""
        matched_skills = ["Python", "React", "MongoDB", "AWS", "Docker"]
        
        categories = skill_extractor.get_skill_categories(matched_skills)
        
        assert 'programming' in categories
        assert 'frameworks' in categories
        assert 'databases' in categories
        assert 'cloud' in categories
