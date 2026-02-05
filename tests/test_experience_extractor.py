"""
Unit Tests for Experience Extractor
Phase 3: Feature Engineering Tests
"""

import pytest
from app.features.experience_extractor import ExperienceExtractor


@pytest.fixture
def experience_extractor():
    """Fixture for experience extractor instance"""
    return ExperienceExtractor()


class TestExperienceExtractor:
    
    def test_extract_explicit_years(self, experience_extractor):
        """Test extraction of explicit year mentions"""
        text = "Software Engineer with 5 years of experience in Python development."
        
        result = experience_extractor.extract_experience(text)
        
        assert result['experience_years'] == 5.0
        assert result['calculation_method'] == 'explicit'
    
    def test_extract_from_date_ranges(self, experience_extractor):
        """Test calculation from date ranges"""
        text = """
        Senior Developer - Tech Corp (2019 - 2023)
        Junior Developer - StartupX (2017 - 2019)
        """
        
        result = experience_extractor.extract_experience(text)
        
        # Should calculate ~6 years total (4 + 2)
        assert result['experience_years'] >= 5.0
        assert result['calculation_method'] in ['date_calculation', 'combined']
    
    def test_extract_with_present(self, experience_extractor):
        """Test handling of 'Present' or 'Current' in dates"""
        text = "Machine Learning Engineer (2020 - Present)"
        
        result = experience_extractor.extract_experience(text)
        
        # Should calculate from 2020 to current year
        assert result['experience_years'] >= 4.0
    
    def test_experience_match_requirement(self, experience_extractor):
        """Test experience requirement matching"""
        text = "8 years of experience"
        
        result_met = experience_extractor.extract_experience(text, required_years=5.0)
        result_not_met = experience_extractor.extract_experience(text, required_years=10.0)
        
        assert result_met['experience_match'] is True
        assert result_not_met['experience_match'] is False
    
    def test_month_year_format(self, experience_extractor):
        """Test extraction from Month-Year format"""
        text = "January 2020 - December 2022"
        
        result = experience_extractor.extract_experience(text)
        
        # Should be approximately 3 years
        assert 2.5 <= result['experience_years'] <= 3.5
    
    def test_no_experience_found(self, experience_extractor):
        """Test when no experience is detected"""
        text = "Fresh graduate with no work experience"
        
        result = experience_extractor.extract_experience(text)
        
        assert result['experience_years'] == 0.0
    
    def test_get_experience_level(self, experience_extractor):
        """Test experience level categorization"""
        assert experience_extractor.get_experience_level(0.5) == "entry_level"
        assert experience_extractor.get_experience_level(2.0) == "junior"
        assert experience_extractor.get_experience_level(4.0) == "mid_level"
        assert experience_extractor.get_experience_level(7.0) == "senior"
        assert experience_extractor.get_experience_level(10.0) == "lead"
        assert experience_extractor.get_experience_level(15.0) == "expert"
    
    def test_multiple_date_ranges(self, experience_extractor):
        """Test with multiple employment periods"""
        text = """
        2021 - 2023: Senior Engineer
        2019 - 2021: Mid-level Engineer  
        2017 - 2019: Junior Engineer
        """
        
        result = experience_extractor.extract_experience(text)
        
        # Should sum all periods (~6 years)
        assert result['experience_years'] >= 5.0
